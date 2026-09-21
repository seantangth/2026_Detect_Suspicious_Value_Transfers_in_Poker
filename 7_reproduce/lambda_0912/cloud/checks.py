#!/usr/bin/env python3
"""TPDS seqnll — SPEC §4 的五項實作品質檢查。全部在 eval + fp32 + math SDPA 下跑，
要求「逐位元不變」的三項用 torch.equal 判定。"""
import argparse, contextlib, json, os, time
import numpy as np
import torch
from model import SeqNLL, Tables
from engine import flat_ctx, head_nll

T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


@contextlib.contextmanager
def math_sdpa():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            yield
    except Exception:
        yield


# --------------------------------------------------------------- 檢查 1
def check_reconstruct(data_dir, n_hands, seed=0):
    """獨立重放 pot／to_call 演變，與 action_ctx 對照。"""
    import polars as pl
    rng = np.random.default_rng(seed)
    h = pl.read_parquet(os.path.join(data_dir, 'raw', 'hands.parquet'),
                        columns=['hand_id', 'button_seat', 'small_blind', 'big_blind'])
    _ids = h['hand_id'].to_numpy().astype('U')
    pick = pl.Series('hand_id', _ids[rng.choice(h.height, min(n_hands, h.height), replace=False)])
    h = h.filter(pl.col('hand_id').is_in(pick))
    a = (pl.scan_parquet(os.path.join(data_dir, 'raw', 'actions.parquet'))
           .filter(pl.col('hand_id').is_in(pick)).collect())
    ac = pl.read_parquet(os.path.join(data_dir, 'processed', 'action_ctx.parquet'),
                         columns=['hand_id', 'action_no', 'to_call_bb', 'pot_before_bb']).filter(pl.col('hand_id').is_in(pick))
    s = pl.read_parquet(os.path.join(data_dir, 'raw', 'seats.parquet'),
                        columns=['hand_id', 'player_id', 'seat_no']).filter(pl.col('hand_id').is_in(pick))
    a = a.join(ac, on=['hand_id', 'action_no']).join(h, on='hand_id').join(s, on=['hand_id', 'player_id'])
    a = a.sort(['hand_id', 'action_no'])
    bad = {'pot': [], 'to_call': [], 'amount_to': []}
    n_act = 0
    for _key, grp in a.group_by('hand_id', maintain_order=True):
        g = grp.to_dicts()
        btn, sb, bb = g[0]['button_seat'], g[0]['small_blind'], g[0]['big_blind']
        contrib = {i: 0 for i in range(6)}
        contrib[(btn + 1) % 6] = sb
        contrib[(btn + 2) % 6] = bb
        pot = sb + bb
        street = 'preflop'
        for r in g:
            n_act += 1
            if r['street'] != street:
                street = r['street']
                contrib = {i: 0 for i in range(6)}
            lvl = max(contrib.values())
            # to_call 以行動者當下的籌碼為上限（本機實測 91,853 個動作：不加這個上限
            # 有 530 個不符，加了之後 0 個不符；不符的全是 raw > stack_before 的短碼情形）
            tc = min(max(lvl - contrib[r['seat_no']], 0), r['stack_before'])
            if pot != r['pot_before']:
                bad['pot'].append((r['hand_id'], r['action_no'], pot, r['pot_before']))
            if tc != r['to_call']:
                bad['to_call'].append((r['hand_id'], r['action_no'], tc, r['to_call']))
            if contrib[r['seat_no']] + r['amount'] != r['amount_to']:
                bad['amount_to'].append((r['hand_id'], r['action_no'],
                                         contrib[r['seat_no']] + r['amount'], r['amount_to']))
            contrib[r['seat_no']] = r['amount_to']
            pot += r['amount']
    # 與 action_ctx 的 bb 正規化欄位對照
    d1 = (a['pot_before'] / a['big_blind'] - a['pot_before_bb']).abs().max()
    d2 = (a['to_call'] / a['big_blind'] - a['to_call_bb']).abs().max()
    return {'n_hands': n_hands, 'n_actions': n_act,
            'mismatch_pot_before': len(bad['pot']), 'mismatch_to_call': len(bad['to_call']),
            'mismatch_amount_to': len(bad['amount_to']),
            'to_call_rule': 'to_call = min(本街最高投注額 - 行動者本街已投注, stack_before)',
            'examples': {k: v[:5] for k, v in bad.items() if v},
            'max_abs_diff_pot_before_bb': float(d1), 'max_abs_diff_to_call_bb': float(d2),
            'pass': len(bad['pot']) == 0 and len(bad['to_call']) == 0 and len(bad['amount_to']) == 0}


# --------------------------------------------------------------- 檢查 2
POSKEYS = ['sidx', 'c_pos', 'potb', 'tcb', 'sprb', 'nactive', 'nvis', 'p_type', 'p_amtb', 'p_pos']


@torch.no_grad()
def check_causal(model, tb, hb, seed=0):
    """(a) 把位置 t 之後的 token 內容全部打亂 -> 位置 t 的編碼輸出逐位元不變。
       (b) 把「當時尚未發出的公牌」換成隨機牌 -> 全部輸出逐位元不變（無未來公牌洩漏）。"""
    g = torch.Generator(device=hb.device); g.manual_seed(seed)
    f, aidx, valid, T = tb.enc_feats(hb)
    base = model.enc(f)
    B = hb.numel()
    nact = tb.h_nact[hb]
    t = (torch.rand(B, device=hb.device, generator=g) * nact.float()).long().clamp(max=T - 1)
    ar = torch.arange(T, device=hb.device)
    after = ar[None, :] > t[:, None]
    f2 = dict(f)
    for k in POSKEYS:
        v = f[k]
        hi = max(int(v.max().item()) + 1, 1)
        perm = torch.randint(0, hi, v.shape, device=v.device, generator=g, dtype=v.dtype)
        f2[k] = torch.where(after, perm, v)
    pert = model.enc(f2)
    bi = torch.arange(B, device=hb.device)
    ok_a = torch.equal(base[bi, t], pert[bi, t])
    changed_a = not torch.equal(base, pert)

    # (b) 未來公牌
    maxvis = torch.where(valid, f['nvis'], torch.zeros_like(f['nvis'])).max(1).values     # [B]
    slot = torch.arange(5, device=hb.device)[None, :]
    fut = slot >= maxvis[:, None]
    f3 = dict(f)
    f3['brank'] = torch.where(fut, torch.randint(0, 13, f['brank'].shape, device=hb.device,
                                                 generator=g, dtype=f['brank'].dtype), f['brank'])
    f3['bsuit'] = torch.where(fut, torch.randint(0, 4, f['bsuit'].shape, device=hb.device,
                                                 generator=g, dtype=f['bsuit'].dtype), f['bsuit'])
    pert_b = model.enc(f3)
    ok_b = torch.equal(base, pert_b)
    n_fut = int((fut & (f['brank'] >= 0)).sum())
    return {'n_hands': int(B), 'shuffle_after_t_bitwise_identical': bool(ok_a),
            'max_abs_diff_at_t': float((base[bi, t] - pert[bi, t]).abs().max()),
            'shuffle_did_change_other_positions': bool(changed_a),
            'future_board_randomised_bitwise_identical': bool(ok_b),
            'n_future_board_cards_randomised': n_fut,
            'max_abs_diff_future_board': float((base - pert_b).abs().max()),
            'pass': bool(ok_a and changed_a and ok_b)}


# --------------------------------------------------------------- 檢查 3
@torch.no_grad()
def check_private(model, tb, hb, seed=0):
    """逐一把某個座位 s 的私有資料換成隨機值：N 模式下，其他座位的動作輸出必須逐位元不變。"""
    g = torch.Generator(device=hb.device); g.manual_seed(seed)
    c = flat_ctx(tb, model, hb)
    nt0, ns0, _, _, _ = head_nll(model, tb, c, None, None)
    priv = ['s_hole_rank', 's_hole_suit', 's_strpct', 's_strcat', 's_chen',
            's_pfhi', 's_pflo', 's_pfpair', 's_pfsuited']
    orig = {k: getattr(tb, k) for k in priv}
    res, worst = [], 0.0
    try:
        for s in range(6):
            rows = hb * 6 + s
            for k in priv:
                v = orig[k].clone()
                sub = v[rows]
                if sub.is_floating_point():
                    v[rows] = torch.rand(sub.shape, device=sub.device, generator=g, dtype=sub.dtype)
                else:
                    hi = int(orig[k].max().item()) + 1
                    v[rows] = torch.randint(0, max(hi, 1), sub.shape, device=sub.device,
                                            generator=g, dtype=sub.dtype)
                setattr(tb, k, v)
            c2 = flat_ctx(tb, model, hb)
            nt1, ns1, _, _, _ = head_nll(model, tb, c2, None, None)
            other = c['slot'] != s
            same = torch.equal(nt0[other], nt1[other]) and torch.equal(ns0[other], ns1[other])
            selfchg = not torch.equal(nt0[~other], nt1[~other]) if (~other).any() else True
            worst = max(worst, float((nt0[other] - nt1[other]).abs().max()) if other.any() else 0.0)
            res.append({'seat': s, 'others_bitwise_identical': bool(same),
                        'own_actions_changed': bool(selfchg), 'n_other_actions': int(other.sum())})
            for k in priv:
                setattr(tb, k, orig[k])
    finally:
        for k in priv:
            setattr(tb, k, orig[k])
    return {'per_seat': res, 'max_abs_diff_others': worst,
            'pass': all(r['others_bitwise_identical'] and r['own_actions_changed'] for r in res)}


# --------------------------------------------------------------- 檢查 4
@torch.no_grad()
def check_symmetry(model, tb, hb):
    """交換 X 兩張底牌的順序，輸出必須逐位元不變（N 與 I 兩種模式都測）。"""
    c = flat_ctx(tb, model, hb)
    ys = (c['slot'] + 1) % 6
    on = torch.ones(c['n'], dtype=torch.bool, device=hb.device)
    a0 = head_nll(model, tb, c, None, None)[:2]
    b0 = head_nll(model, tb, c, ys, on)[:2]
    o1, o2 = tb.s_hole_rank, tb.s_hole_suit
    try:
        tb.s_hole_rank = o1.flip(1).contiguous()
        tb.s_hole_suit = o2.flip(1).contiguous()
        c2 = flat_ctx(tb, model, hb)
        a1 = head_nll(model, tb, c2, None, None)[:2]
        b1 = head_nll(model, tb, c2, ys, on)[:2]
    finally:
        tb.s_hole_rank, tb.s_hole_suit = o1, o2
    okN = torch.equal(a0[0], a1[0]) and torch.equal(a0[1], a1[1])
    okI = torch.equal(b0[0], b1[0]) and torch.equal(b0[1], b1[1])
    return {'n_actions': c['n'], 'N_mode_bitwise_identical': bool(okN),
            'I_mode_bitwise_identical': bool(okI),
            'max_abs_diff_N': float((a0[0] - a1[0]).abs().max()),
            'pass': bool(okN and okI)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--ckpt', default='/home/ubuntu/tpds/ckpt')
    ap.add_argument('--data', default='/home/ubuntu/tpds/data')
    ap.add_argument('--out', default='/home/ubuntu/tpds/out')
    ap.add_argument('--recon-hands', type=int, default=1000)
    ap.add_argument('--causal-hands', type=int, default=100)
    ap.add_argument('--other-hands', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-recon', action='store_true')
    ap.add_argument('--prefix', default='seqnll')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False       # 位元一致性優先
    torch.backends.cudnn.allow_tf32 = False
    dev = args.device
    out = {}
    if not args.skip_recon:
        log('檢查 1：動作重建')
        out['check1_action_reconstruction'] = check_reconstruct(args.data, args.recon_hands)
        log('  ->', json.dumps({k: v for k, v in out['check1_action_reconstruction'].items() if k != 'examples'}))

    tb = Tables(args.tables, dev)
    model = SeqNLL().to(dev)
    model.load_state_dict(torch.load(os.path.join(args.ckpt, 'g0_best.pt'), map_location=dev))
    model.eval()
    pool = (tb.h_grp == 1).nonzero(as_tuple=True)[0]      # g0 的 out-of-sample 半區
    gsel = torch.Generator(device=dev); gsel.manual_seed(7)
    idx = torch.randperm(pool.numel(), device=dev, generator=gsel)
    with math_sdpa():
        log('檢查 2：因果性')
        out['check2_causality'] = check_causal(model, tb, pool[idx[:args.causal_hands]])
        log('  ->', json.dumps(out['check2_causality']))
        log('檢查 3：私有資訊隔離')
        out['check3_private_isolation'] = check_private(model, tb, pool[idx[:args.other_hands]])
        log('  ->', json.dumps({k: v for k, v in out['check3_private_isolation'].items() if k != 'per_seat'}))
        log('檢查 4：底牌順序對稱性')
        out['check4_hole_card_symmetry'] = check_symmetry(model, tb, pool[idx[:args.other_hands]])
        log('  ->', json.dumps(out['check4_hole_card_symmetry']))

    p = os.path.join(args.out, f'{args.prefix}_infer_summary.json')
    if os.path.exists(p):
        s = json.load(open(p))
        mg = s.get('mean_gain_sum')
        out['check5_reveal_slot'] = {'mean_gain_sum': mg, 'mean_gain_max': s.get('mean_gain_max'),
                                     'pass': bool(mg is not None and mg >= 0 and mg < 0.5)}
        log('  ->', json.dumps(out['check5_reveal_slot']))
    else:
        out['check5_reveal_slot'] = {'pass': None, 'note': 'infer_summary.json 尚未產生'}
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, f'{args.prefix}_quality_checks.json'), 'w'), indent=2, ensure_ascii=False)
    log('五項檢查結果：', {k: v.get('pass') for k, v in out.items()})


if __name__ == '__main__':
    main()
