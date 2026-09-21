#!/usr/bin/env python3
"""TPDS pairpol — 配對條件式常態行為模型（7_reproduce/lambda_0913/SPEC_pairpol.md）。

重用 lambda_0912/cloud/build_tables.py 產出的決策狀態表與 gbfeat.py 的 Feats/FEAT_BASE，
在其上加：
  trunk f(FEAT_BASE) -> h_t（MLP 256-256-128、GELU、dropout 0.1）
  個人漂移 u[X, phase, q5]（dim 8，初始 0）
  有序配對嵌入 r[X, Y, phase]（dim 8，初始 0），c_t = Σ_{Y∈D_h, Y≠X} r[X,Y,phase]
  加法輸出頭 logits_type = W_f h + W_g(u + c)、logits_size = V_f h + V_g(u + c)

不使用任何勾結標籤；標籤只在 pairpol_accept.py 的驗收段出現。

子命令
  prep  由 tables/ + raw/hands.parquet 建出訓練用扁平陣列（特徵矩陣、u/r 索引、置換對照映射）
  run   在同一個行程內做：6 次訓練（2 半 × 3 個 λ_r）→ 掃 λ → 選 λ → 全量推論 → 產出 P1/P2/manifest
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, os.pardir, os.pardir, 'lambda_0912', 'cloud'))   # (release) gbfeat.py in the repository layout
from gbfeat import Feats, FEAT_BASE, CAT_BASE     # noqa: E402  （重用 09-12 的特徵定義）

T0 = time.time()
PHASES = ['development', 'evaluation']
DEVICE = 'cuda'
EMB = 8                 # u / r 的維度（規格 §2）
CLIP = 30.0             # 標準化後的截斷，避免 bf16 下溢/溢位


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


def amp():
    """A10/A100 上用 bf16 autocast；CPU 自測時退成不做任何事。"""
    import contextlib
    import torch
    if DEVICE.startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cuda_free():
    import torch
    if DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()


def md5_of(path, buf=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ====================================================================== prep
def cmd_prep(args):
    import polars as pl
    T, O = args.tables, args.prep
    os.makedirs(O, exist_ok=True)
    F = Feats(T)
    N_A, N_H, N_S = F.N_A, F.N_H, F.N_S
    hand_ids = np.load(os.path.join(T, 'hand_ids.npy'), allow_pickle=True).astype('U')
    player_ids = np.load(os.path.join(T, 'player_ids.npy'), allow_pickle=True).astype('U')
    n_players = len(player_ids)
    a_type = np.load(os.path.join(T, 'a_type.npy'))
    a_amtb = np.load(os.path.join(T, 'a_amtb.npy'))
    h_grp = np.load(os.path.join(T, 'h_grp.npy'))
    log(f'N_H={N_H:,} N_A={N_A:,} N_S={N_S:,} n_players={n_players:,}')

    # ---------------------------------------------------------------- hands：phase / table / q5
    hp = pl.read_parquet(os.path.join(args.data, 'raw', 'hands.parquet'),
                         columns=['hand_id', 'table_id', 'phase', 'started_at'])
    hp = hp.with_columns(
        (((pl.col('started_at').rank('ordinal').over('table_id', 'phase') - 1) * 5
          // pl.len().over('table_id', 'phase')).cast(pl.Int8).alias('q5'))).drop('started_at')
    order = pl.DataFrame({'hand_id': hand_ids, '_i': np.arange(N_H, dtype=np.int64)})
    hp = order.join(hp, on='hand_id', how='left').sort('_i')
    assert hp.height == N_H and hp['q5'].null_count() == 0, '手表 join 掉列'
    h_q5 = hp['q5'].to_numpy().astype(np.int8)
    h_phase = (hp['phase'].to_numpy() == 'evaluation').astype(np.int8)
    tbl_name = np.sort(hp['table_id'].unique().to_numpy().astype('U'))
    h_tbl = np.searchsorted(tbl_name, hp['table_id'].to_numpy().astype('U')).astype(np.int32)
    log(f'桌數 {len(tbl_name)}；phase 分佈 dev={int((h_phase == 0).sum()):,} eval={int((h_phase == 1).sum()):,}')

    # ---------------------------------------------------------------- 特徵矩陣
    cat_pos = [FEAT_BASE.index(c) for c in CAT_BASE]
    cont_pos = [i for i in range(len(FEAT_BASE)) if i not in cat_pos]
    n_cont = len(cont_pos)
    Xc = np.empty((N_A, n_cont), np.float32)
    Xk = np.empty((N_A, len(cat_pos)), np.int16)
    CH = 1_000_000
    for s in range(0, N_A, CH):
        e = min(s + CH, N_A)
        B = F.base(np.arange(s, e, dtype=np.int64))
        Xc[s:e] = B[:, cont_pos]
        Xk[s:e] = B[:, cat_pos].astype(np.int16)
        if (s // CH) % 5 == 0:
            log(f'  特徵組裝 {e:,}/{N_A:,}')
    assert not np.isnan(Xk.astype(np.float32)).any()
    cat_sizes = [int(Xk[:, j].max()) + 1 for j in range(Xk.shape[1])]
    assert Xk.min() >= 0, '類別欄有負值'
    log(f'類別欄 {CAT_BASE} 基數 {cat_sizes}')

    # 標準化（全母體，無標籤）；只對真的有 NaN 的欄加缺失指示
    ssum = np.zeros(n_cont, np.float64); ssq = np.zeros(n_cont, np.float64)
    cnt = np.zeros(n_cont, np.float64); nanc = np.zeros(n_cont, np.int64)
    for s in range(0, N_A, CH):
        v = Xc[s:min(s + CH, N_A)]
        m = ~np.isnan(v)
        nanc += (~m).sum(0)
        vv = np.where(m, v, 0).astype(np.float64)
        ssum += vv.sum(0); ssq += (vv * vv).sum(0); cnt += m.sum(0)
    mu = ssum / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(ssq / np.maximum(cnt, 1) - mu ** 2, 0)) + 1e-6
    nan_cols = np.where(nanc > 0)[0]
    log(f'連續欄 {n_cont}；有缺失的欄 {len(nan_cols)} 個 -> 加同數量的缺失指示')
    Xm = np.zeros((N_A, len(nan_cols)), np.uint8)
    for s in range(0, N_A, CH):
        e = min(s + CH, N_A)
        v = Xc[s:e]
        Xm[s:e] = np.isnan(v[:, nan_cols]).astype(np.uint8)
        v = (v - mu.astype(np.float32)) / sd.astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        Xc[s:e] = np.clip(v, -CLIP, CLIP)

    # ---------------------------------------------------------------- 目標
    y_type = a_type.astype(np.int8)
    y_size = a_amtb.astype(np.int8)
    size_mask = np.isin(a_type, [3, 4, 5])          # bet / raise / all_in
    n_size = int(a_amtb.max()) + 1
    log(f'尺寸桶數 {n_size}；攻擊性動作 {int(size_mask.sum()):,}/{N_A:,}')

    # ---------------------------------------------------------------- u 索引
    a_hand = F.a_hand.astype(np.int64)
    a_slot = F.a_slot.astype(np.int64)
    s_pcode = F.s_pcode.astype(np.int64)
    srow = a_hand * 6 + a_slot
    pc_act = s_pcode[srow]
    u_idx = ((pc_act * 2 + h_phase[a_hand]) * 5 + h_q5[a_hand]).astype(np.int32)
    n_u = n_players * 10

    # ---------------------------------------------------------------- 有序配對索引
    OTH = np.array([[y for y in range(6) if y != x] for x in range(6)], np.int64)   # [6,5]
    XS = np.repeat(np.arange(6, dtype=np.int64), 5)
    YS = OTH.reshape(-1)
    pc = s_pcode.reshape(N_H, 6)
    key = ((pc[:, XS] * n_players + pc[:, YS]) * 2 + h_phase[:, None].astype(np.int64))
    uniq, inv = np.unique(key.reshape(-1), return_inverse=True)
    n_pairs = int(len(uniq))
    hand_pair30 = inv.reshape(N_H, 30).astype(np.int32)
    del key, inv
    pair_x = ((uniq // 2) // n_players).astype(np.int32)
    pair_y = ((uniq // 2) % n_players).astype(np.int32)
    pair_ph = (uniq % 2).astype(np.int8)
    log(f'有序配對 {n_pairs:,}（理論上限 = 桌數×30×29×2 = {len(tbl_name) * 30 * 29 * 2:,}）')
    pair_idx = hand_pair30[a_hand[:, None], (a_slot[:, None] * 5 + np.arange(5, dtype=np.int64)[None, :])]

    # ---------------------------------------------------------------- 置換對照 π_X
    # 每個 (table, phase, X)：對「其他 29 人」做一個 29-循環（保證無固定點、且 π(Y)≠X）。
    rng = np.random.default_rng(args.perm_seed)
    perm_map = np.full(n_pairs + 1, n_pairs, np.int32)     # 預設指向零列（無嵌入）
    perm_map[n_pairs] = n_pairs
    tp_key = (np.repeat(h_tbl.astype(np.int64), 6) * 2 + np.repeat(h_phase.astype(np.int64), 6))
    tp_pc = np.unique(tp_key * n_players + s_pcode)
    tp_of = tp_pc // n_players
    pc_of = tp_pc % n_players
    bnd = np.searchsorted(tp_of, np.unique(tp_of))
    bnd = np.append(bnd, len(tp_of))
    n_miss_perm = 0
    for gi in range(len(bnd) - 1):
        sl = slice(bnd[gi], bnd[gi + 1])
        pls = pc_of[sl]
        ph = int(tp_of[sl][0] % 2)
        if len(pls) < 3:
            continue
        for X in pls:
            oth = pls[pls != X]
            q = rng.permutation(len(oth))
            src = oth[q]
            dst = oth[np.roll(q, -1)]
            ks = (X * n_players + src) * 2 + ph
            kd = (X * n_players + dst) * 2 + ph
            i_s = np.searchsorted(uniq, ks)
            i_d = np.searchsorted(uniq, kd)
            ok_s = (i_s < n_pairs) & (uniq[np.minimum(i_s, n_pairs - 1)] == ks)
            ok_d = (i_d < n_pairs) & (uniq[np.minimum(i_d, n_pairs - 1)] == kd)
            tgt = np.where(ok_d, i_d, n_pairs).astype(np.int32)
            perm_map[i_s[ok_s]] = tgt[ok_s]
            n_miss_perm += int((~ok_d & ok_s).sum())
    log(f'置換映射建好；映不到有效配對而落到零列的有序對 {n_miss_perm:,}')

    # ---------------------------------------------------------------- 每座位動作數
    seat_nact = np.bincount(srow, minlength=N_S).astype(np.int32)

    # ---------------------------------------------------------------- 存檔
    def sv(name, arr):
        np.save(os.path.join(O, name + '.npy'), arr)
    sv('X_cont', Xc); sv('X_miss', Xm); sv('X_cat', Xk)
    sv('y_type', y_type); sv('y_size', y_size); sv('size_mask', size_mask)
    sv('u_idx', u_idx); sv('pair_idx', pair_idx); sv('perm_map', perm_map)
    sv('hand_pair30', hand_pair30)
    sv('pair_x', pair_x); sv('pair_y', pair_y); sv('pair_ph', pair_ph)
    sv('a_grp', h_grp[a_hand].astype(np.int8))
    sv('a_hand', a_hand.astype(np.int32)); sv('a_slot', a_slot.astype(np.int8))
    sv('h_phase', h_phase); sv('h_q5', h_q5); sv('h_tbl', h_tbl)
    sv('seat_nact', seat_nact); sv('s_pcode', s_pcode.astype(np.int32))
    np.save(os.path.join(O, 'tbl_name.npy'), tbl_name)
    meta = {'N_A': int(N_A), 'N_H': int(N_H), 'N_S': int(N_S), 'n_players': int(n_players),
            'n_pairs': n_pairs, 'n_u': int(n_u), 'n_size': n_size, 'n_cont': n_cont,
            'n_miss': int(len(nan_cols)), 'cat_sizes': cat_sizes, 'cat_cols': CAT_BASE,
            'nan_cols': [FEAT_BASE[cont_pos[i]] for i in nan_cols],
            'feat_mu': mu.tolist(), 'feat_sd': sd.tolist(), 'clip': CLIP,
            'perm_seed': args.perm_seed, 'perm_fallback_zero_rows': n_miss_perm,
            'n_tables': int(len(tbl_name)), 'emb_dim': EMB,
            'action_hands_g0': int((h_grp[a_hand] == 0).sum()),
            'action_hands_g1': int((h_grp[a_hand] == 1).sum())}
    json.dump(meta, open(os.path.join(O, 'prep_meta.json'), 'w'), indent=2)
    log('prep 完成：' + json.dumps({k: v for k, v in meta.items()
                                   if k not in ('feat_mu', 'feat_sd')}, ensure_ascii=False))


# ====================================================================== 模型
def build_model(meta, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class PairPol(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(s, EMB) for s in meta['cat_sizes']])
            d = meta['n_cont'] + meta['n_miss'] + EMB * len(meta['cat_sizes'])
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128))
            self.Wf = nn.Linear(128, 6)
            self.Vf = nn.Linear(128, meta['n_size'])
            self.Wg = nn.Linear(EMB, 6, bias=False)
            self.Vg = nn.Linear(EMB, meta['n_size'], bias=False)
            self.u = nn.Embedding(meta['n_u'], EMB)
            self.r = nn.Embedding(meta['n_pairs'] + 1, EMB)     # 最後一列是永遠為 0 的「無嵌入」列
            # u / r 依規格初始 0；W_g / V_g **不可**也初始 0，否則 (W_g, u, r) 全零是個
            # 死鞍點：dL/dr ∝ W_g = 0、dL/dW_g ∝ (u+c) = 0，加法分支永遠學不動。
            nn.init.zeros_(self.u.weight)
            nn.init.zeros_(self.r.weight)

        def trunk(self, cont, miss, cat):
            e = [self.embs[j](cat[:, j].long()) for j in range(len(self.embs))]
            return self.net(torch.cat([cont, miss.to(cont.dtype)] + e, 1))

        def heads(self, h):
            return self.Wf(h), self.Vf(h)

    return PairPol()


def nll_rows(lt, yt, ls, ys, mask):
    """每列的 NLL＝類型 NLL ＋（攻擊性動作才算的）尺寸 NLL。全部用 float32 算。"""
    import torch.nn.functional as Fn
    a = Fn.cross_entropy(lt.float(), yt, reduction='none')
    b = Fn.cross_entropy(ls.float(), ys, reduction='none')
    return a + b * mask


class Ctx:
    """把 prep 的陣列搬上 GPU（總量約 3.8 GB）。"""

    def __init__(self, prep, dev):
        import torch
        self.meta = json.load(open(os.path.join(prep, 'prep_meta.json')))
        L = lambda n: np.load(os.path.join(prep, n + '.npy'))
        self.np = {n: L(n) for n in ('a_grp', 'a_hand', 'a_slot', 'pair_idx', 'perm_map',
                                     'hand_pair30', 'pair_x', 'pair_y', 'pair_ph', 'seat_nact',
                                     's_pcode', 'h_phase', 'h_q5', 'h_tbl', 'u_idx', 'y_type')}
        self.dev = dev
        t = lambda a, d=None: torch.as_tensor(a if d is None else a.astype(d)).to(dev)
        self.cont = t(L('X_cont'))
        self.miss = t(L('X_miss'))
        self.cat = t(L('X_cat').astype(np.int32))
        self.yt = t(L('y_type').astype(np.int64))
        self.ys = t(L('y_size').astype(np.int64))
        self.sm = t(L('size_mask').astype(np.float32))
        self.uidx = t(L('u_idx').astype(np.int64))
        self.pidx = t(L('pair_idx').astype(np.int64))
        self.pmap = t(L('perm_map').astype(np.int64))
        self.seat5 = t(((self.np['a_hand'].astype(np.int64) * 6 + self.np['a_slot']) * 5))
        self.grp = L('a_grp')
        self.N_A = self.meta['N_A']
        mem = torch.cuda.memory_allocated(dev) / 2 ** 30 if str(dev).startswith('cuda') else 0.0
        log(f'Ctx 上裝置完成：{mem:.2f} GiB')

    def batch(self, ix):
        return self.cont[ix], self.miss[ix], self.cat[ix], self.yt[ix], self.ys[ix], self.sm[ix]


# ====================================================================== 訓練
def train_one(ctx, group, lam_r, args):
    import torch
    dev = ctx.dev
    m = build_model(ctx.meta, args.seed).to(dev)
    emb_params = [m.u.weight, m.r.weight]
    other = [p for n, p in m.named_parameters() if n not in ('u.weight', 'r.weight')]
    opt = torch.optim.AdamW([{'params': other, 'lr': args.lr, 'weight_decay': 0.01},
                             {'params': emb_params, 'lr': args.emb_lr, 'weight_decay': 0.0}])
    tr = np.where(ctx.grp == group)[0]
    ho = np.where(ctx.grp != group)[0]
    ho_s = ho if len(ho) <= args.eval_actions else np.random.default_rng(0).choice(
        ho, args.eval_actions, replace=False)
    ho_t = torch.as_tensor(np.sort(ho_s)).to(dev)
    tr_t = torch.as_tensor(tr).to(dev)
    n = len(tr)
    log(f'  訓練 half={group} λ_r={lam_r:g}：訓練動作 {n:,}、held-out 取樣 {len(ho_s):,}')
    g = torch.Generator(device=dev); g.manual_seed(args.seed + group * 100 + int(-np.log10(lam_r) * 7))
    hist = []
    step = 0
    tlast = time.time()
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(n, device=dev, generator=g)
        tot = 0.0; cnt = 0
        for s in range(0, n, args.bs):
            ix = tr_t[perm[s:s + args.bs]]
            cont, miss, cat, yt, ys, sm = ctx.batch(ix)
            with amp():
                h = m.trunk(cont, miss, cat)
            lt_f, ls_f = m.heads(h.float())
            u = m.u(ctx.uidx[ix])
            rr = m.r(ctx.pidx[ix])                     # [B,5,8]
            gvec = u + rr.sum(1)
            nll = nll_rows(lt_f + m.Wg(gvec), yt, ls_f + m.Vg(gvec), ys, sm)
            B = ix.numel()
            loss = nll.mean() + lam_r * (rr ** 2).sum() / B + args.lam_u * (u ** 2).sum() / B
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                m.r.weight[ctx.meta['n_pairs']].zero_()
            tot += float(nll.detach().mean()) * B; cnt += B
            step += 1
            if step % 200 == 0:
                dt = time.time() - tlast; tlast = time.time()
                log(f'    ep{ep} step{step} train_nll={tot / cnt:.4f} {200 / max(dt, 1e-9):.1f} it/s')
        ev = eval_nll(m, ctx, ho_t, args.bs * 4)
        hist.append({'epoch': ep, 'train_nll': tot / cnt, **ev})
        log(f'    ep{ep} 完：train {tot / cnt:.4f} | held-out N\'={ev["ho_nll_Np"]:.4f} P={ev["ho_nll_P"]:.4f}')
    return m, hist


def eval_nll(m, ctx, idx_t, bs):
    import torch
    m.eval()
    sN = sP = 0.0; c = 0
    with torch.no_grad():
        for s in range(0, idx_t.numel(), bs):
            ix = idx_t[s:s + bs]
            cont, miss, cat, yt, ys, sm = ctx.batch(ix)
            with amp():
                h = m.trunk(cont, miss, cat)
            lt_f, ls_f = m.heads(h.float())
            u = m.u(ctx.uidx[ix])
            gp = u + m.r(ctx.pidx[ix]).sum(1)
            sN += float(nll_rows(lt_f + m.Wg(u), yt, ls_f + m.Vg(u), ys, sm).sum())
            sP += float(nll_rows(lt_f + m.Wg(gp), yt, ls_f + m.Vg(gp), ys, sm).sum())
            c += ix.numel()
    m.train()
    return {'ho_n': c, 'ho_nll_Np': sN / c, 'ho_nll_P': sP / c}


# ====================================================================== 推論／打分
def score_lambda(models, ctx, args, full):
    """兩個半區互打（全部 OOS）。回傳每個有序對的 llr 統計（真實＋置換），以及 held-out NLL。

    full=True 時另外累積每手每有序對的增益（P2 分片與 P1 的手層統計要用）。
    """
    import torch
    dev = ctx.dev
    npair = ctx.meta['n_pairs']
    N_H = ctx.meta['N_H']
    z = lambda k: torch.zeros(k, dtype=torch.float64, device=dev)
    acc = {'sum': z(npair + 1), 'sq': z(npair + 1), 'nact': z(npair + 1),
           'psum': z(npair + 1), 'psq': z(npair + 1)}
    hs = hm = phs = None
    if full:
        hs = torch.zeros(N_H * 30, dtype=torch.float32, device=dev)
        hm = torch.full((N_H * 30,), -np.inf, dtype=torch.float32, device=dev)
        phs = torch.zeros(N_H * 30, dtype=torch.float32, device=dev)
    sN = sP = 0.0; cN = 0
    bs = args.score_bs
    for tgt in (0, 1):
        m = models[1 - tgt]                  # 半區 tgt 由「另一半訓練的模型」打分
        m.eval()
        idx = torch.as_tensor(np.where(ctx.grp == tgt)[0]).to(dev)
        with torch.no_grad():
            for s in range(0, idx.numel(), bs):
                ix = idx[s:s + bs]
                B = ix.numel()
                cont, miss, cat, yt, ys, sm = ctx.batch(ix)
                with amp():
                    h = m.trunk(cont, miss, cat)
                lt_f, ls_f = m.heads(h.float())
                u = m.u(ctx.uidx[ix])
                pi = ctx.pidx[ix]                        # [B,5]
                rr = m.r(pi)                             # [B,5,8]
                bt = lt_f + m.Wg(u); bs_ = ls_f + m.Vg(u)
                nNp = nll_rows(bt, yt, bs_, ys, sm)                       # 基線 N'
                gp = rr.sum(1)
                nP = nll_rows(bt + m.Wg(gp), yt, bs_ + m.Vg(gp), ys, sm)  # 配對模型 P
                sN += float(nNp.sum()); sP += float(nP.sum()); cN += B
                yt5 = yt.repeat_interleave(5); ys5 = ys.repeat_interleave(5)
                sm5 = sm.repeat_interleave(5)
                for tag, pix in (('', pi), ('p', ctx.pmap[pi])):
                    rv = rr if tag == '' else m.r(pix)
                    lt5 = (bt[:, None, :] + m.Wg(rv)).reshape(B * 5, -1)
                    ls5 = (bs_[:, None, :] + m.Vg(rv)).reshape(B * 5, -1)
                    gain = (nNp.repeat_interleave(5) - nll_rows(lt5, yt5, ls5, ys5, sm5))
                    fl = pix.reshape(-1)
                    acc[tag + 'sum'].index_add_(0, fl, gain.double())
                    acc[tag + 'sq'].index_add_(0, fl, (gain * gain).double())
                    if tag == '':
                        acc['nact'].index_add_(0, fl, torch.ones_like(gain, dtype=torch.float64))
                    if full:
                        hidx = (ctx.seat5[ix].unsqueeze(1) + torch.arange(5, device=dev)[None, :]).reshape(-1)
                        if tag == '':
                            hs.index_add_(0, hidx, gain.float())
                            hm.scatter_reduce_(0, hidx, gain.float(), reduce='amax', include_self=True)
                        else:
                            phs.index_add_(0, hidx, gain.float())
                if (s // bs) % 40 == 0:
                    log(f'    打分 half{tgt} {min(s + bs, idx.numel()):,}/{idx.numel():,}')
    out = {k: v.cpu().numpy() for k, v in acc.items()}
    out['ho_nll_Np'] = sN / cN
    out['ho_nll_P'] = sP / cN
    out['ho_n'] = cN
    if full:
        out['hand_sum'] = hs.cpu().numpy().reshape(N_H, 30)
        out['hand_max'] = hm.cpu().numpy().reshape(N_H, 30)
        out['hand_psum'] = phs.cpu().numpy().reshape(N_H, 30)
        del hs, hm, phs
    del acc
    cuda_free()
    return out


def llr_z(sm, sq, nact):
    d = np.sqrt(np.maximum(sq, 0))
    z = np.where((nact > 0) & (d > 0), sm / np.maximum(d, 1e-12), 0.0)
    return z.astype(np.float64)


# ====================================================================== 輸出
def seg_stats(pid, val, valid, npair, dev):
    """依有序對分組的 sum / 手數 / 正值手數 / 最大手 / 前三手平均（在 GPU 上做）。

    top3 用 3 趟 segment-max：每趟把等於該組當前最大值的元素設成 -inf。浮點完全相等的
    並列會被一起扣掉（機率極低），這會讓該組少算一兩個名次，方向偏保守。
    """
    import torch
    p = torch.as_tensor(pid[valid].astype(np.int64)).to(dev)
    w = torch.as_tensor(val[valid].astype(np.float32)).to(dev)
    n = npair + 1
    s = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, p, w.double())
    c = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(
        0, p, torch.ones_like(w, dtype=torch.float64))
    pos = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, p, (w > 0).double())
    tops = []
    for _ in range(3):
        mx = torch.full((n,), -np.inf, dtype=torch.float32, device=dev)
        mx.scatter_reduce_(0, p, w, reduce='amax', include_self=True)
        tops.append(mx.clone())
        w = torch.where(w == mx[p], torch.full_like(w, -np.inf), w)
    T = torch.stack(tops)[:, :npair].cpu().numpy()
    del p, w
    cuda_free()
    fin = np.isfinite(T)
    top3 = np.where(fin.any(0), np.where(fin, T, 0).sum(0) / np.maximum(fin.sum(0), 1), np.nan)
    mx1 = np.where(fin[0], T[0], np.nan)
    return (s.cpu().numpy()[:npair], c.cpu().numpy()[:npair].astype(np.int64),
            pos.cpu().numpy()[:npair], mx1, top3)


def cmd_run(args):
    import torch
    import polars as pl
    global DEVICE
    DEVICE = args.device
    dev = args.device
    assert dev != 'cuda' or torch.cuda.is_available(), '沒有 CUDA'
    gpu = torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'
    log(f'裝置: {gpu}')
    os.makedirs(args.out, exist_ok=True)
    ctx = Ctx(args.prep, dev)
    meta = ctx.meta
    npair = meta['n_pairs']
    lams = [float(x) for x in args.lams.split(',')]

    # ---------------------------------------------------------------- 訓練 + λ 掃描
    trained, sweep = {}, {}
    for lam in lams:
        models, hists = [], []
        for g in (0, 1):
            t0 = time.time()
            m, h = train_one(ctx, g, lam, args)
            models.append(m); hists.append(h)
            log(f'  half={g} λ={lam:g} 訓練耗時 {time.time() - t0:.0f}s')
        sc = score_lambda(models, ctx, args, full=False)
        zr = llr_z(sc['sum'][:npair], sc['sq'][:npair], sc['nact'][:npair])
        zp = llr_z(sc['psum'][:npair], sc['psq'][:npair], sc['nact'][:npair])
        thr = float(np.quantile(zp, 0.999))
        n_tail = int((zr > thr).sum())
        big = sc['nact'][:npair] >= 30
        thr_b = float(np.quantile(zp[big], 0.999)) if big.any() else float('nan')
        n_tail_b = int((zr[big] > thr_b).sum()) if big.any() else 0
        sweep[lam] = {'lambda_r': lam, 'null_z_q999': thr, 'n_real_above_null_q999': n_tail,
                      'n_pairs': npair, 'null_z_q999_nact30': thr_b,
                      'n_real_above_null_q999_nact30': n_tail_b, 'n_pairs_nact30': int(big.sum()),
                      'ho_nll_Np': sc['ho_nll_Np'], 'ho_nll_P': sc['ho_nll_P'],
                      'ho_nll_delta_P_minus_Np': sc['ho_nll_P'] - sc['ho_nll_Np'],
                      'real_z_mean': float(zr.mean()), 'perm_z_mean': float(zp.mean()),
                      'real_z_std': float(zr.std()), 'perm_z_std': float(zp.std()),
                      'llr_sum_mean_real': float(sc['sum'][:npair].mean()),
                      'llr_sum_mean_perm': float(sc['psum'][:npair].mean()),
                      'r_norm_mean': float(np.mean([float(m.r.weight[:npair].detach().norm(dim=1).mean())
                                                    for m in models])),
                      'epochs': [hists[0], hists[1]]}
        log(f'λ={lam:g}: null z q99.9={thr:.3f} → 真實超過的有序對 {n_tail:,}'
            f'（n_act≥30 子集 {n_tail_b:,}/{int(big.sum()):,}）；'
            f'held-out NLL N\'={sc["ho_nll_Np"]:.4f} P={sc["ho_nll_P"]:.4f}')
        trained[lam] = models
        del sc
        cuda_free()

    # ---------------------------------------------------------------- 選 λ
    ok = {l: v for l, v in sweep.items() if v['ho_nll_P'] <= v['ho_nll_Np'] + 1e-9}
    pool = ok if ok else sweep
    best = max(pool, key=lambda l: pool[l]['n_real_above_null_q999'])
    sel = {'selected_lambda_r': best, 'criterion': '真實 llr_z 超過置換 null 99.9 百分位的有序對數量最多',
           'non_degenerate_filter': 'held-out NLL(P) <= held-out NLL(N\')',
           'candidates_passing_filter': sorted(ok.keys()),
           'fell_back_to_all_lambdas': not bool(ok),
           'sweep': {str(k): v for k, v in sweep.items()}}
    json.dump(sel, open(os.path.join(args.out, 'lambda_selection.json'), 'w'), indent=2)
    log(f'選定 λ_r = {best:g}')

    # ---------------------------------------------------------------- 全量推論
    models = trained[best]
    for l in list(trained):
        if l != best:
            del trained[l]
    cuda_free()
    sc = score_lambda(models, ctx, args, full=True)
    rw = [m.r.weight.detach().float().cpu().numpy()[:npair] for m in models]
    uw = [m.u.weight.detach().float().cpu().numpy() for m in models]
    for g, m in enumerate(models):
        torch.save({'state_dict': m.state_dict(), 'meta': meta, 'lambda_r': best, 'group': g},
                   os.path.join(args.out, f'pairpol_g{g}.pt'))
    for a in ('cont', 'miss', 'cat', 'yt', 'ys', 'sm', 'uidx', 'pidx', 'pmap', 'seat5'):
        delattr(ctx, a)
    del models, trained
    cuda_free()

    # ---------------------------------------------------------------- P1
    N_H = meta['N_H']
    hp30 = ctx.np['hand_pair30']
    seat_nact = ctx.np['seat_nact']
    XS = np.repeat(np.arange(6, dtype=np.int64), 5)
    acted = (seat_nact.reshape(N_H, 6)[:, XS] > 0)            # [N_H,30] X 在該手有沒有動作
    pid = hp30.reshape(-1)
    valid = acted.reshape(-1)
    s_r, c_r, pos_r, mx_r, t3_r = seg_stats(pid, sc['hand_sum'].reshape(-1), valid, npair, dev)
    s_p, c_p, pos_p, mx_p, t3_p = seg_stats(pid, sc['hand_psum'].reshape(-1), valid, npair, dev)
    n_dealt = np.bincount(pid, minlength=npair + 1)[:npair].astype(np.int64)
    nact = sc['nact'][:npair]
    zr = llr_z(sc['sum'][:npair], sc['sq'][:npair], nact)
    zp = llr_z(sc['psum'][:npair], sc['psq'][:npair], nact)
    r_avg = (rw[0] + rw[1]) / 2.0
    r_norm = (np.linalg.norm(rw[0], axis=1) + np.linalg.norm(rw[1], axis=1)) / 2.0
    px, py, pph = ctx.np['pair_x'], ctx.np['pair_y'], ctx.np['pair_ph']
    un = [np.linalg.norm(w.reshape(meta['n_players'], 2, 5, EMB), axis=3).mean(2) for w in uw]
    u_norm = ((un[0] + un[1]) / 2.0)[px, pph]
    player_ids = np.load(os.path.join(args.tables, 'player_ids.npy'), allow_pickle=True).astype('U')
    den_h = np.maximum(c_r, 1)
    tab = {'player_id': player_ids[px], 'other_id': player_ids[py],
           'phase': np.array(PHASES)[pph],
           'n_hands_present': n_dealt, 'n_hands_acted': c_r, 'n_actions': nact.astype(np.int64),
           'llr_sum': sc['sum'][:npair], 'llr_mean_hand': s_r / den_h,
           'llr_max_hand': mx_r, 'llr_top3_hand': t3_r, 'llr_pos_frac': pos_r / den_h,
           'llr_z': zr, 'r_norm': r_norm, 'u_norm': u_norm,
           'perm_llr_sum': sc['psum'][:npair], 'perm_llr_mean_hand': s_p / den_h,
           'perm_llr_max_hand': mx_p, 'perm_llr_pos_frac': pos_p / den_h, 'perm_llr_z': zp}
    for j in range(EMB):
        tab[f'r_{j + 1}'] = r_avg[:, j]
    df = pl.DataFrame(tab)
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c, t in df.schema.items()
                          if t == pl.Float64])
    p1 = os.path.join(args.out, 'pairpol_pair.parquet')
    df.write_parquet(p1, compression='zstd', compression_level=9)
    log(f'P1 寫出 {df.height:,} 列 -> {p1}')

    # ---------------------------------------------------------------- P2 分片
    shard_stats = emit_shards(args, ctx, sc, meta)

    # ---------------------------------------------------------------- manifest
    man = {'spec': '7_reproduce/lambda_0913/SPEC_pairpol.md', 'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'gpu': gpu, 'prep_meta': meta,
           'args': {k: v for k, v in vars(args).items()},
           'lambda_selection': sel,
           'final_holdout_nll': {'N_prime': sc['ho_nll_Np'], 'P': sc['ho_nll_P'], 'n_actions': sc['ho_n']},
           'p1_rows': int(df.height), 'shards': shard_stats,
           'elapsed_s': time.time() - T0}
    json.dump(man, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2, default=str)
    np.savez_compressed(os.path.join(args.out, 'pairpol_diag.npz'),
                        real_z=zr.astype(np.float32), perm_z=zp.astype(np.float32),
                        nact=nact.astype(np.float32))
    log('run 完成')


def emit_shards(args, ctx, sc, meta):
    import polars as pl
    N_H = meta['N_H']
    h_tbl = ctx.np['h_tbl']; h_phase = ctx.np['h_phase']
    s_pcode = ctx.np['s_pcode'].reshape(N_H, 6)
    seat_nact = ctx.np['seat_nact'].reshape(N_H, 6)
    tbl_name = np.load(os.path.join(args.prep, 'tbl_name.npy'), allow_pickle=True).astype('U')
    hand_ids = np.load(os.path.join(args.tables, 'hand_ids.npy'), allow_pickle=True).astype('U')
    hsum = sc['hand_sum']; hmax = sc['hand_max']
    cand = {}
    cd = pl.read_parquet(os.path.join(args.data, 'processed', 'cand_pairs_development.parquet'))
    cand['development'] = cd.select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
    ce = pl.read_csv(os.path.join(args.data, 'raw', 'evaluation_pairs.csv'))
    cand['evaluation'] = ce.select('pair_id',
                                   pl.min_horizontal('player_1', 'player_2').alias('a'),
                                   pl.max_horizontal('player_1', 'player_2').alias('b'))
    player_ids = np.load(os.path.join(args.tables, 'player_ids.npy'), allow_pickle=True).astype('U')
    srt = np.argsort(player_ids, kind='stable')
    sp = player_ids[srt]

    def to_code(arr):
        i = np.clip(np.searchsorted(sp, arr), 0, len(sp) - 1)
        return np.where(sp[i] == arr, srt[i], -1).astype(np.int64)

    ordk = np.argsort(h_tbl, kind='stable')
    starts = np.searchsorted(h_tbl[ordk], np.arange(len(tbl_name)))
    ends = np.searchsorted(h_tbl[ordk], np.arange(len(tbl_name)), side='right')
    stats = {}
    for ph_i, ph in enumerate(PHASES):
        outdir = os.path.join(args.shards, ph)
        os.makedirs(outdir, exist_ok=True)
        cp = cand[ph]
        ca = to_code(cp['a'].to_numpy().astype('U'))
        cb = to_code(cp['b'].to_numpy().astype('U'))
        pidv = cp['pair_id'].to_numpy().astype('U')
        assert (ca >= 0).all() and (cb >= 0).all() or args.allow_missing_players, \
            f'{ph} 候選對有玩家不在表裡：{int((ca < 0).sum() + (cb < 0).sum())}'
        # 每個候選對屬於哪張桌：由 a 的座位所在桌決定（每位玩家固定在一張桌）
        p2t = np.full(len(player_ids), -1, np.int32)
        p2t[s_pcode.reshape(-1)] = np.repeat(h_tbl, 6)
        ptab = np.where((ca >= 0) & (cb >= 0), p2t[np.maximum(ca, 0)], -1)
        tot_rows = 0; nfile = 0
        for t in range(len(tbl_name)):
            hs = ordk[starts[t]:ends[t]]
            hs = hs[h_phase[hs] == ph_i]
            sel = np.where(ptab == t)[0]
            if len(hs) == 0 or len(sel) == 0:
                pl.DataFrame(schema={'pair_id': pl.String, 'hand_id': pl.String,
                                     'a': pl.String, 'b': pl.String,
                                     'gain_ab_sum': pl.Float32, 'gain_ab_max': pl.Float32,
                                     'gain_ba_sum': pl.Float32, 'gain_ba_max': pl.Float32,
                                     'n_act_a': pl.Int32, 'n_act_b': pl.Int32}
                             ).write_parquet(os.path.join(outdir, f'{tbl_name[t]}.parquet'),
                                             compression='zstd', compression_level=9)
                nfile += 1
                continue
            # 這張桌的 (候選對 K) × (該期手 H) 全部向量化：slot_of[本地玩家, 手] = 座位或 -1
            pl_loc = np.unique(s_pcode[hs].reshape(-1))
            loc = np.full(len(player_ids), -1, np.int32)
            loc[pl_loc] = np.arange(len(pl_loc))
            slot_of = np.full((len(pl_loc), len(hs)), -1, np.int8)
            hcol = np.tile(np.arange(len(hs)), 6)
            slot_of[loc[s_pcode[hs].T.reshape(-1)], hcol] = np.repeat(np.arange(6), len(hs))
            la = loc[ca[sel]]; lb = loc[cb[sel]]
            keep = (la >= 0) & (lb >= 0)
            sel = sel[keep]; la = la[keep]; lb = lb[keep]
            SA = slot_of[la]; SB = slot_of[lb]                    # [K,H]
            m = (SA >= 0) & (SB >= 0)
            ki, hi = np.nonzero(m)
            hh = hs[hi]
            sa = SA[ki, hi].astype(np.int64); sb = SB[ki, hi].astype(np.int64)
            p_ab = sa * 5 + (sb - (sb > sa))
            p_ba = sb * 5 + (sa - (sa > sb))
            g_ab_m = hmax[hh, p_ab]; g_ba_m = hmax[hh, p_ba]
            d = pl.DataFrame({
                'pair_id': pidv[sel][ki], 'hand_id': hand_ids[hh],
                'a': player_ids[ca[sel]][ki], 'b': player_ids[cb[sel]][ki],
                'gain_ab_sum': hsum[hh, p_ab],
                'gain_ab_max': np.where(np.isfinite(g_ab_m), g_ab_m, np.nan).astype(np.float32),
                'gain_ba_sum': hsum[hh, p_ba],
                'gain_ba_max': np.where(np.isfinite(g_ba_m), g_ba_m, np.nan).astype(np.float32),
                'n_act_a': seat_nact[hh, sa].astype(np.int32),
                'n_act_b': seat_nact[hh, sb].astype(np.int32)})
            d.write_parquet(os.path.join(outdir, f'{tbl_name[t]}.parquet'),
                            compression='zstd', compression_level=9)
            tot_rows += d.height; nfile += 1
            if t % 50 == 0:
                log(f'  {ph} 分片 {t + 1}/{len(tbl_name)}（累計 {tot_rows:,} 列）')
        exp = int(cd['shared'].sum()) if ph == 'development' else int(ce['shared_hands'].sum())
        stats[ph] = {'files': nfile, 'rows': int(tot_rows), 'expected_rows': exp,
                     'match': bool(tot_rows == exp)}
        log(f'{ph} 分片完成：{nfile} 檔、{tot_rows:,} 列（規格期望 {exp:,}，{"符合" if tot_rows == exp else "不符"}）')
    return stats


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--tables', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--prep', required=True)
    p.add_argument('--perm-seed', type=int, default=20260913)
    p.set_defaults(fn=cmd_prep)
    r = sub.add_parser('run')
    r.add_argument('--tables', required=True)
    r.add_argument('--data', required=True)
    r.add_argument('--prep', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--shards', required=True)
    r.add_argument('--lams', default='1e-3,1e-2,1e-1')
    r.add_argument('--epochs', type=int, default=6)
    r.add_argument('--bs', type=int, default=8192)
    r.add_argument('--score-bs', type=int, default=65536)
    r.add_argument('--lr', type=float, default=1e-3)
    r.add_argument('--emb-lr', type=float, default=1e-2)
    r.add_argument('--lam-u', type=float, default=1e-3)
    r.add_argument('--eval-actions', type=int, default=1_000_000)
    r.add_argument('--seed', type=int, default=1234)
    r.add_argument('--device', default='cuda')
    r.add_argument('--allow-missing-players', action='store_true',
                   help='smoke（--limit-hands）時候選對的玩家多半不在子集裡，允許略過')
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
