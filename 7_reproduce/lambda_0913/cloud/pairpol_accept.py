#!/usr/bin/env python3
"""TPDS pairpol — 品質檢查（SPEC_pairpol.md §4）與驗收／冗餘檢查（§5）。

標籤只在 §5 出現，且不用來調任何參數。輸出 pairpol_acceptance.json。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gbfeat import Feats, FEAT_BASE, CAT_BASE          # noqa: E402
from pairpol import build_model, nll_rows, EMB, PHASES, log, CLIP   # noqa: E402


# ====================================================================== 小工具
def rankdata(x):
    """平均名次（處理並列）。"""
    x = np.asarray(x, float)
    n = len(x)
    o = np.argsort(x, kind='stable')
    r = np.empty(n, float)
    xs = x[o]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float('nan')


def desc(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {'n': 0}
    q = np.percentile(v, [25, 50, 75, 90, 99])
    return {'n': int(len(v)), 'mean': float(v.mean()), 'std': float(v.std()),
            'p25': float(q[0]), 'median': float(q[1]), 'p75': float(q[2]),
            'p90': float(q[3]), 'p99': float(q[4])}


# ====================================================================== §4.1 / §4.2
def load_models(out, meta, dev):
    import torch
    ms = []
    for g in (0, 1):
        ck = torch.load(os.path.join(out, f'pairpol_g{g}.pt'), map_location=dev, weights_only=False)
        m = build_model(meta, 0).to(dev)
        m.load_state_dict(ck['state_dict'])
        m.eval()
        ms.append(m)
    return ms


def std_rows(raw, meta, cat_pos, cont_pos, nan_names):
    """把 Feats.base() 的原始列，用 prep 存下的 mu/sd 做成模型輸入（與 prep 完全同一條路）。"""
    mu = np.array(meta['feat_mu'], np.float32)
    sd = np.array(meta['feat_sd'], np.float32)
    cont = raw[:, cont_pos]
    nan_idx = [[FEAT_BASE[cont_pos[i]] for i in range(len(cont_pos))].index(n) for n in nan_names]
    miss = np.isnan(cont[:, nan_idx]).astype(np.uint8)
    v = np.nan_to_num((cont - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, -CLIP, CLIP).astype(np.float32), miss, raw[:, cat_pos].astype(np.int32)


def model_nll(m, ctxdev, cont, miss, cat, yt, ys, sm, uidx, pidx, zero_r=False):
    import torch
    t = lambda a, d: torch.as_tensor(a.astype(d)).to(ctxdev)
    with torch.no_grad():
        h = m.trunk(t(cont, np.float32), t(miss, np.float32), t(cat, np.int32))
        lt, ls = m.heads(h)
        u = m.u(t(uidx, np.int64))
        rr = m.r(t(pidx, np.int64))
        if zero_r:
            rr = torch.zeros_like(rr)
        g = u + rr.sum(1)
        nN = nll_rows(lt + m.Wg(u), t(yt, np.int64), ls + m.Vg(u), t(ys, np.int64), t(sm, np.float32))
        nP = nll_rows(lt + m.Wg(g), t(yt, np.int64), ls + m.Vg(g), t(ys, np.int64), t(sm, np.float32))
    return nN.cpu().numpy(), nP.cpu().numpy()


def check_zero_embed(A, m, n_hands=100, seed=0):
    """§4.1 嵌入歸零一致性：r≡0 時 P 與 N' 逐位元相同。"""
    rng = np.random.default_rng(seed)
    hs = rng.choice(A['N_H'], min(n_hands, A['N_H']), replace=False)
    idx = np.concatenate([np.arange(A['h_aoff'][h], A['h_aoff'][h] + A['h_nact'][h]) for h in hs])
    idx = idx[:200000]
    if len(idx) == 0:
        return {'pass': False, 'reason': '抽到的手沒有動作'}
    cont = A['X_cont'][idx]; miss = A['X_miss'][idx]; cat = A['X_cat'][idx]
    nN, nP0 = model_nll(m, A['dev'], cont, miss, cat, A['y_type'][idx], A['y_size'][idx],
                        A['size_mask'][idx].astype(np.float32), A['u_idx'][idx], A['pair_idx'][idx],
                        zero_r=True)
    _, nP = model_nll(m, A['dev'], cont, miss, cat, A['y_type'][idx], A['y_size'][idx],
                      A['size_mask'][idx].astype(np.float32), A['u_idx'][idx], A['pair_idx'][idx])
    return {'n_hands': int(len(hs)), 'n_actions': int(len(idx)),
            'bitwise_identical_when_r_zero': bool(np.array_equal(nN, nP0)),
            'max_abs_diff_when_r_zero': float(np.abs(nN - nP0).max()),
            'real_r_did_change_output': bool(not np.array_equal(nN, nP)),
            'mean_abs_gain_with_real_r': float(np.abs(nN - nP).mean()),
            'pass': bool(np.array_equal(nN, nP0) and not np.array_equal(nN, nP))}


def check_causal(A, m, data, tables, n_hands=100, seed=1):
    """§4.2 因果性：把第 t 個動作之後的動作打亂，重跑 build_tables 的因果掃描，
    第 ≤t 個動作的決策狀態、特徵列與 NLL 必須逐位元不變。"""
    import build_tables as BT
    rng = np.random.default_rng(seed)
    hand_ids = A['hand_ids']
    cands = np.where(A['h_nact'] >= 4)[0]
    hs = rng.choice(cands, min(n_hands, len(cands)), replace=False)
    picked = hand_ids[hs]
    a = pl.read_parquet(os.path.join(data, 'raw', 'actions.parquet'),
                        columns=['hand_id', 'action_no', 'player_id', 'amount']).filter(
        pl.col('hand_id').is_in(pl.Series(picked).implode()))
    h = pl.read_parquet(os.path.join(data, 'raw', 'hands.parquet'),
                        columns=['hand_id', 'button_seat', 'small_blind', 'big_blind']).filter(
        pl.col('hand_id').is_in(pl.Series(picked).implode()))
    hmap = {r[0]: (r[1], r[2], r[3]) for r in h.iter_rows()}
    amt_of = {(r[0], r[1]): r[3] for r in a.iter_rows()}

    F = A['F']
    nact = A['h_nact'][hs].astype(np.int64)
    aoff_l = np.zeros(len(hs), np.int64)
    np.cumsum(nact[:-1], out=aoff_l[1:])
    gidx = np.concatenate([np.arange(A['h_aoff'][hh], A['h_aoff'][hh] + A['h_nact'][hh]) for hh in hs])
    a_actno = np.load(os.path.join(tables, 'a_actno.npy'))[gidx]
    slot = F.a_slot[gidx].astype(np.int8)
    sidx = F.a_sidx[gidx].astype(np.int8)
    isfold = np.load(os.path.join(tables, 'a_isfold.npy'))[gidx]
    isagg = np.load(os.path.join(tables, 'a_isagg.npy'))[gidx]
    y4 = F.a_y4[gidx].astype(np.int8)
    amount = np.array([amt_of[(hand_ids[hh], int(an))] for hh, an in
                       zip(np.repeat(hs, A['h_nact'][hs]), a_actno)], np.int64)
    btn = np.array([hmap[hand_ids[hh]][0] for hh in hs], np.int64)
    sb_slot = ((btn + 1) % 6).astype(np.int64)
    bb_slot = ((btn + 2) % 6).astype(np.int64)
    sb_amt = np.array([hmap[hand_ids[hh]][1] for hh in hs], np.int64)
    bb_amt = np.array([hmap[hand_ids[hh]][2] for hh in hs], np.int64)

    def run(sl, si, isf, isa, yy, am):
        N = len(sl)
        o = [np.zeros(N, np.uint8), np.zeros(N, np.uint8), np.zeros(N, np.int8),
             np.zeros(N, np.int16), np.zeros(N, np.int8), np.zeros(N, np.int8),
             np.zeros(N, np.int16), np.zeros(N, np.int8), np.zeros(N, np.int8),
             np.zeros((N, 6), np.int64)]
        BT._scan(aoff_l, nact, sl, si, isf, isa, yy, am, sb_slot, bb_slot, sb_amt, bb_amt, *o)
        return o

    base = run(slot, sidx, isfold, isagg, y4, amount)
    t_local = np.array([rng.integers(0, n - 1) for n in nact], np.int64)   # 保證 t 之後至少一個動作
    keep = np.zeros(len(slot), bool)
    perm = np.arange(len(slot))
    for k in range(len(hs)):
        s, e = aoff_l[k], aoff_l[k] + nact[k]
        tt = s + t_local[k]
        keep[s:tt + 1] = True
        seg = np.arange(tt + 1, e)
        perm[tt + 1:e] = seg[rng.permutation(len(seg))]
    pert = run(slot[perm], sidx[perm], isfold[perm], isagg[perm], y4[perm], amount[perm])
    ok_scan = all(np.array_equal(b[keep], p[keep]) for b, p in zip(base, pert))
    changed = any(not np.array_equal(b, p) for b, p in zip(base, pert))

    # 特徵列與 NLL：把擾動後的掃描結果貼回 Feats 的對應位置再重算
    cat_pos = [FEAT_BASE.index(c) for c in CAT_BASE]
    cont_pos = [i for i in range(len(FEAT_BASE)) if i not in cat_pos]
    raw0 = F.base(gidx[keep])
    saved = {k: getattr(F, k).copy() for k in ('a_foldbm', 'a_actedbm', 'a_lastaggr', 'a_naggb',
                                               'a_naggs', 'a_nacts', 'a_apn', 'a_apt', 'a_apa')}
    names = ['a_foldbm', 'a_actedbm', 'a_lastaggr', 'a_naggb', 'a_naggs', 'a_nacts',
             'a_apn', 'a_apt', 'a_apa']
    for nm, arr in zip(names, pert[:9]):
        getattr(F, nm)[gidx] = arr.astype(getattr(F, nm).dtype)
    raw1 = F.base(gidx[keep])
    for nm in names:
        setattr(F, nm, saved[nm])
    same_feat = bool(np.array_equal(np.nan_to_num(raw0, nan=-9e9), np.nan_to_num(raw1, nan=-9e9)))
    c0 = std_rows(raw0, A['meta'], cat_pos, cont_pos, A['meta']['nan_cols'])
    c1 = std_rows(raw1, A['meta'], cat_pos, cont_pos, A['meta']['nan_cols'])
    gi = gidx[keep]
    n0 = model_nll(m, A['dev'], *c0, A['y_type'][gi], A['y_size'][gi],
                   A['size_mask'][gi].astype(np.float32), A['u_idx'][gi], A['pair_idx'][gi])[1]
    n1 = model_nll(m, A['dev'], *c1, A['y_type'][gi], A['y_size'][gi],
                   A['size_mask'][gi].astype(np.float32), A['u_idx'][gi], A['pair_idx'][gi])[1]
    return {'n_hands': int(len(hs)), 'n_actions_checked': int(keep.sum()),
            'scan_outputs_bitwise_identical_up_to_t': bool(ok_scan),
            'shuffle_did_change_after_t': bool(changed),
            'feature_rows_bitwise_identical': same_feat,
            'nll_bitwise_identical': bool(np.array_equal(n0, n1)),
            'max_abs_nll_diff': float(np.abs(n0 - n1).max()),
            'pass': bool(ok_scan and changed and same_feat and np.array_equal(n0, n1))}


# ====================================================================== 主程式
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--prep', required=True)
    ap.add_argument('--tables', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--extra', required=True, help='presence_contrast / v5x OOF 所在目錄')
    ap.add_argument('--check-hands', type=int, default=100)
    ap.add_argument('--skip-model-checks', action='store_true')
    args = ap.parse_args()
    res = {'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    meta = json.load(open(os.path.join(args.prep, 'prep_meta.json')))

    # ---------------------------------------------------------------- §4.1 / §4.2
    if not args.skip_model_checks:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        M = lambda n: np.load(os.path.join(args.prep, n + '.npy'), mmap_mode='r')
        A = {'meta': meta, 'dev': dev, 'N_H': meta['N_H'],
             'X_cont': M('X_cont'), 'X_miss': M('X_miss'), 'X_cat': M('X_cat'),
             'y_type': M('y_type'), 'y_size': M('y_size'), 'size_mask': M('size_mask'),
             'u_idx': M('u_idx'), 'pair_idx': M('pair_idx'),
             'h_aoff': np.load(os.path.join(args.tables, 'h_aoff.npy')).astype(np.int64),
             'h_nact': np.load(os.path.join(args.tables, 'h_nact.npy')).astype(np.int64),
             'hand_ids': np.load(os.path.join(args.tables, 'hand_ids.npy'),
                                 allow_pickle=True).astype('U'),
             'F': Feats(args.tables)}
        models = load_models(args.out, meta, dev)
        log('檢查 1：嵌入歸零一致性')
        res['check1_zero_embedding'] = check_zero_embed(A, models[0], args.check_hands)
        log('檢查 2：因果性')
        try:
            res['check2_causality'] = check_causal(A, models[0], args.data, args.tables,
                                                   args.check_hands)
        except Exception as e:                              # noqa: BLE001
            res['check2_causality'] = {'pass': False, 'error': f'{type(e).__name__}: {e}'}
        del A, models
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 讀 P1 與標籤
    p1 = pl.read_parquet(os.path.join(args.out, 'pairpol_pair.parquet'))
    sel = json.load(open(os.path.join(args.out, 'lambda_selection.json')))
    res['check4_lambda_curve'] = {
        k: {'ho_nll_Np': v['ho_nll_Np'], 'ho_nll_P': v['ho_nll_P'],
            'delta_P_minus_Np': v['ho_nll_delta_P_minus_Np'],
            'P_not_worse_than_Nprime': bool(v['ho_nll_P'] <= v['ho_nll_Np']),
            'null_z_q999': v['null_z_q999'],
            'n_real_above_null_q999': v['n_real_above_null_q999'],
            'n_real_above_null_q999_nact30': v['n_real_above_null_q999_nact30'],
            'r_norm_mean': v['r_norm_mean']}
        for k, v in sel['sweep'].items()}
    res['selected_lambda_r'] = sel['selected_lambda_r']

    dev_p1 = p1.filter(pl.col('phase') == 'development')
    cols = ['llr_mean_hand', 'llr_z', 'llr_max_hand', 'llr_top3_hand', 'llr_pos_frac',
            'r_norm', 'u_norm', 'perm_llr_mean_hand', 'perm_llr_z', 'n_hands_acted', 'n_actions']
    ab = dev_p1.select([pl.col('player_id').alias('a'), pl.col('other_id').alias('b')]
                       + [pl.col(c).alias(c + '_ab') for c in cols])
    ba = dev_p1.select([pl.col('player_id').alias('b'), pl.col('other_id').alias('a')]
                       + [pl.col(c).alias(c + '_ba') for c in cols])
    cand = pl.read_parquet(os.path.join(args.data, 'processed', 'cand_pairs_development.parquet')) \
             .select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8), 'label', 'shared')
    j = cand.join(ab, on=['a', 'b'], how='left').join(ba, on=['a', 'b'], how='left')
    ex = []
    for c in cols:
        ex += [((pl.col(c + '_ab') + pl.col(c + '_ba')) / 2).alias(c + '_mean'),
               pl.min_horizontal(c + '_ab', c + '_ba').alias(c + '_min'),
               pl.max_horizontal(c + '_ab', c + '_ba').alias(c + '_max')]
    j = j.with_columns(ex)
    res['p1_join'] = {'cand_pairs': cand.height,
                      'matched_both_directions': int(j.filter(
                          pl.col('llr_mean_hand_ab').is_not_null()
                          & pl.col('llr_mean_hand_ba').is_not_null()).height)}

    lab = pl.read_csv(os.path.join(args.data, 'raw', 'development_labels.csv')) \
            .select('pair_id', 'label', 'label_status')
    ev = j.join(lab.rename({'label': 'lab'}), on='pair_id', how='inner')
    y = ev['lab'].to_numpy()
    res['acceptance_set'] = {'n_positive': int((y == 1).sum()), 'n_negative': int((y == 0).sum())}

    # ---------------------------------------------------------------- §5 驗收 AUC
    pc = pl.read_parquet(os.path.join(args.extra, 'presence_contrast_development.parquet'))
    ev = ev.join(pc.select('pair_id', 'pc_nll_sum_tg_mean', 'pc_nll_sum_tg_min',
                           'pc_nll_sum_dg_mean'), on='pair_id', how='left')
    aucs = {}
    for name in ['llr_mean_hand_mean', 'llr_mean_hand_min', 'llr_mean_hand_max',
                 'llr_z_mean', 'llr_z_min', 'llr_max_hand_mean', 'llr_top3_hand_mean',
                 'llr_pos_frac_mean', 'r_norm_mean', 'r_norm_min', 'u_norm_mean',
                 'perm_llr_mean_hand_mean', 'perm_llr_mean_hand_min', 'perm_llr_z_mean',
                 'pc_nll_sum_tg_mean', 'pc_nll_sum_tg_min', 'pc_nll_sum_dg_mean']:
        if name in ev.columns:
            aucs[name] = auc(y, ev[name].to_numpy())
    res['acceptance_auc'] = aucs

    # ---------------------------------------------------------------- §4.3 置換對照
    allz = p1['perm_llr_sum'].to_numpy().astype(float)
    _s = sel['sweep'][str(sel['selected_lambda_r'])]
    tail_n = _s['n_real_above_null_q999']
    tail_exp = _s['n_pairs'] * 0.001
    res['check3_permutation'] = {
        'perm_llr_sum_全母體': desc(allz), 'real_llr_sum_全母體': desc(p1['llr_sum'].to_numpy()),
        'perm_llr_z_全母體': desc(p1['perm_llr_z'].to_numpy()),
        'real_llr_z_全母體': desc(p1['llr_z'].to_numpy()),
        'auc_real_llr_mean_hand_mean': aucs.get('llr_mean_hand_mean'),
        'auc_perm_llr_mean_hand_mean': aucs.get('perm_llr_mean_hand_mean'),
        'auc_perm_llr_mean_hand_min': aucs.get('perm_llr_mean_hand_min'),
        'perm_llr_z_median': float(np.nanmedian(p1['perm_llr_z'].to_numpy())),
        'real_llr_z_median': float(np.nanmedian(p1['llr_z'].to_numpy())),
        'tail_real_above_null_q999': int(tail_n), 'tail_expected_under_null': float(tail_exp),
        'note': ('置換後放進去的是「錯的」嵌入，平均會讓 NLL 變大，所以 llr 的中心本來就略為負；'
                 '決定性的判準是 AUC 掉回 0.5 與真實尾部厚於 null 尾部。'),
        'pass': bool(abs((aucs.get('perm_llr_mean_hand_mean') or 0.5) - 0.5) < 0.10
                     and abs((aucs.get('perm_llr_mean_hand_min') or 0.5) - 0.5) < 0.10
                     and tail_n > tail_exp)}

    # ---------------------------------------------------------------- §5 名次帶
    oof = pl.read_parquet(os.path.join(args.extra, 'pair_oof_v5x_head_s1.parquet'))
    jj = j.join(oof, on='pair_id', how='inner').join(
        lab.rename({'label': 'lab'}), on='pair_id', how='left').sort('oof', descending=True)
    bands = {'1-150': (0, 150), '151-300': (150, 300), '301-600': (300, 600), '601-1000': (600, 1000)}
    res['rank_bands'] = {}
    for nm, (s, e) in bands.items():
        sub = jj.slice(s, e - s)
        res['rank_bands'][nm] = {
            'n': sub.height,
            'n_labeled_positive': int((sub['lab'].fill_null(-1).to_numpy() == 1).sum()),
            'llr_mean_hand_mean': desc(sub['llr_mean_hand_mean'].to_numpy()),
            'llr_z_mean': desc(sub['llr_z_mean'].to_numpy()),
            'r_norm_mean': desc(sub['r_norm_mean'].to_numpy())}

    # ---------------------------------------------------------------- §5 冗餘檢查
    top = jj.head(1000).join(pc.select('pair_id', 'pc_nll_sum_tg_mean'), on='pair_id', how='left')
    unl = top.filter(pl.col('lab').is_null())
    red = {'n_top1000': top.height, 'n_top1000_unlabeled': unl.height}
    for c in ('llr_mean_hand_mean', 'llr_z_mean', 'r_norm_mean'):
        red[f'spearman_{c}_vs_pc_nll_sum_tg_mean'] = spearman(
            unl[c].to_numpy(), unl['pc_nll_sum_tg_mean'].to_numpy())
    x = ev['pc_nll_sum_tg_mean'].fill_null(0).to_numpy().astype(float)
    for c in ('llr_mean_hand_mean', 'llr_z_mean', 'r_norm_mean'):
        v = ev[c].to_numpy().astype(float)
        ok = np.isfinite(v) & np.isfinite(x)
        if int(ok.sum()) < 10:
            red[f'residual_auc_{c}'] = float('nan')
            red[f'residual_beta_{c}'] = None
            continue
        Xd = np.stack([np.ones(int(ok.sum())), x[ok]], 1)
        beta, *_ = np.linalg.lstsq(Xd, v[ok], rcond=None)
        r_ = np.full(len(v), np.nan)
        r_[ok] = v[ok] - Xd @ beta
        red[f'residual_auc_{c}'] = auc(y, r_)
        red[f'residual_beta_{c}'] = beta.tolist()
    red['verdict'] = ('換了個算法算同一件事'
                      if (red.get('spearman_llr_mean_hand_mean_vs_pc_nll_sum_tg_mean') or 0) > 0.9
                      and (red.get('residual_auc_llr_mean_hand_mean') or 0) < 0.6
                      else '不是冗餘（至少其中一項未達冗餘判準）')
    res['redundancy'] = red

    # ---------------------------------------------------------------- 採用門檻
    a_main = aucs.get('llr_mean_hand_mean') or 0.0
    a_min = aucs.get('llr_mean_hand_min') or 0.0
    res['adoption_gate'] = {
        'permutation_ok': res['check3_permutation']['pass'],
        'redundancy_ok': red['verdict'].startswith('不是冗餘'),
        'auc_ge_0.95': bool(max(a_main, a_min) >= 0.95),
        'best_llr_auc': max(a_main, a_min),
        'baseline_rough_auc': aucs.get('pc_nll_sum_tg_mean')}
    res['adoption_gate']['all_pass'] = bool(res['adoption_gate']['permutation_ok']
                                            and res['adoption_gate']['redundancy_ok']
                                            and res['adoption_gate']['auc_ge_0.95'])
    json.dump(res, open(os.path.join(args.out, 'pairpol_acceptance.json'), 'w'),
              indent=2, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
