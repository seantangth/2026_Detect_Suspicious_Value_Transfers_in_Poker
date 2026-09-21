#!/usr/bin/env python3
"""TPDS gbnll — LightGBM 版的 N／I 常態行為模型（SPEC_gbnll.md）。
與 seqnll 共用同一份決策狀態表與同一條交叉擬合雜湊規則（crc32(hand_id)&1）。
訓練不使用任何勾結標籤。輸出 schema 與 seqnll 的 T1/T2/T3 完全相同，只換前綴。"""
import argparse, json, os, time
import numpy as np
import lightgbm as lgb
from gbfeat import Feats, hand_actions, FEAT_BASE, FEAT_I
from outputs import write_all, nanify

T0 = time.time()
NEG = float('-inf')


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


def params(nclass, lr, leaves, jobs, seed):
    return dict(objective='multiclass', num_class=nclass, learning_rate=lr, num_leaves=leaves,
                min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=5.0, verbose=-1, num_threads=jobs, seed=seed, metric='multi_logloss')


def fit(X, y, nclass, rounds, args, tag, val_frac=0.03):
    n = len(y)
    rng = np.random.default_rng(7)
    vm = np.zeros(n, bool)
    vm[rng.choice(n, max(int(n * val_frac), 1000), replace=False)] = True
    ds = lgb.Dataset(X[~vm], label=y[~vm], free_raw_data=True)
    dv = lgb.Dataset(X[vm], label=y[vm], reference=ds, free_raw_data=True)
    ev = {}
    t = time.time()
    b = lgb.train(params(nclass, args.lr, args.leaves, args.jobs, args.seed), ds,
                  num_boost_round=rounds, valid_sets=[dv], valid_names=['val'],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(ev),
                             lgb.log_evaluation(200)])
    vl = ev['val']['multi_logloss']
    log(f'  {tag}: n={n:,} 最佳輪 {b.best_iteration}/{rounds} '
        f'val_logloss {vl[b.best_iteration-1]:.4f} ({time.time()-t:.0f}s)')
    return b, {'n_train': int(n), 'best_iteration': int(b.best_iteration),
               'val_logloss': float(vl[b.best_iteration - 1]), 'rounds_cap': rounds,
               'fit_sec': round(time.time() - t, 1)}


def nll_of(booster, X, tgt):
    p = booster.predict(X, num_iteration=booster.best_iteration)
    p = np.clip(p, 1e-12, 1.0)
    take = p[np.arange(len(tgt)), tgt]
    return -np.log(take).astype(np.float32), take.astype(np.float32), \
        (-(p * np.log(p)).sum(1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--out', default='/home/ubuntu/tpds/out')
    ap.add_argument('--prefix', default='gbnll')
    ap.add_argument('--train-rows', type=int, default=6_000_000)
    ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--leaves', type=int, default=127)
    ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk-hands', type=int, default=100_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    F = Feats(args.tables)
    N_A, N_H, N_S = F.N_A, F.N_H, F.N_S
    log(f'表載入 N_H={N_H:,} N_A={N_A:,} 特徵 base={len(FEAT_BASE)} I={len(FEAT_I)}')

    T1 = {k: np.zeros(N_A, np.float32) for k in ['nll_type_N', 'nll_size_N', 'p_taken_N', 'entropy_N']}
    T1['nll_size_N'][:] = np.nan
    t2 = {'n_act': np.zeros(N_S, np.int32), 'nll_N_sum': np.zeros(N_S, np.float32),
          'nll_size_N_sum': np.zeros(N_S, np.float32)}
    for k in ['nll_N_max', 'nll_N_fold_max', 'nll_N_call_max', 'nll_N_agg_max']:
        t2[k] = np.full(N_S, NEG, np.float32)
    t2_street = np.zeros(N_S * 4, np.float32)
    t3_sum = np.zeros(N_S * 6, np.float32)
    t3_max = np.full(N_S * 6, NEG, np.float32)
    t3_gmax = np.full(N_S * 6, NEG, np.float32)

    by_n = np.zeros(6); by_i = np.zeros(6); by_c = np.zeros(6)
    a_type_all = np.load(os.path.join(args.tables, 'a_type.npy'))
    a_isagg = np.load(os.path.join(args.tables, 'a_isagg.npy'))
    a_iscall = np.load(os.path.join(args.tables, 'a_iscall.npy'))
    a_isfold = np.load(os.path.join(args.tables, 'a_isfold.npy'))
    info = {'feature_base': FEAT_BASE, 'feature_I': FEAT_I, 'models': {}, 'args': vars(args),
            'crossfit_hash': 'zlib.crc32(hand_id.encode()) & 1（與 seqnll 相同）'}

    for g in (0, 1):
        log(f'=== 半區 {g}：訓練 ===')
        tr_h = np.flatnonzero(F.h_grp == g)
        sc_h = np.flatnonzero(F.h_grp == (1 - g))
        tr_a = hand_actions(F.h_aoff, F.h_nact, tr_h)
        rng = np.random.default_rng(1000 + g)
        pick = tr_a if len(tr_a) <= args.train_rows else rng.choice(tr_a, args.train_rows, replace=False)
        pick = np.sort(pick)
        y4 = F.a_y4[pick].astype(np.int32)
        Xb = F.base(pick)
        mN, iN = fit(Xb, y4, 4, args.rounds, args, f'g{g} N-type')
        aggm = y4 == 3
        mNs, iNs = fit(Xb[aggm], F.a_amtb[pick][aggm].astype(np.int32), 13, args.rounds, args, f'g{g} N-size')
        del Xb
        yk = rng.integers(1, 6, len(pick))
        ysl = (F.a_slot[pick] + yk) % 6
        Xi = F.with_y(pick, ysl)
        mI, iI = fit(Xi, y4, 4, args.rounds, args, f'g{g} I-type')
        mIs, iIs = fit(Xi[aggm], F.a_amtb[pick][aggm].astype(np.int32), 13, args.rounds, args, f'g{g} I-size')
        del Xi
        info['models'][f'g{g}'] = {'N_type': iN, 'N_size': iNs, 'I_type': iI, 'I_size': iIs}

        log(f'=== 半區 {g} 的模型替半區 {1-g} 打分（{len(sc_h):,} 手）===')
        t_s = time.time()
        for c0 in range(0, len(sc_h), args.chunk_hands):
            hb = sc_h[c0:c0 + args.chunk_hands]
            idx = hand_actions(F.h_aoff, F.h_nact, hb)
            if not len(idx):
                continue
            slot = F.a_slot[idx]
            srow = F.a_hand[idx] * 6 + slot
            sidx = F.a_sidx[idx]
            y = F.a_y4[idx].astype(np.int32)
            am = F.a_amtb[idx].astype(np.int32)
            agg = y == 3
            Xb = F.base(idx)
            nt, pt, en = nll_of(mN, Xb, y)
            ns = np.zeros(len(idx), np.float32)
            if agg.any():
                ns[agg] = nll_of(mNs, Xb[agg], am[agg])[0]
            del Xb
            tot_N = nt + ns
            T1['nll_type_N'][idx] = nt
            T1['nll_size_N'][idx] = np.where(agg, ns, np.nan)
            T1['p_taken_N'][idx] = pt
            T1['entropy_N'][idx] = en
            # ---- 依 (hand, seat) 分組：同一手的動作一定落在同一個 chunk，所以可直接指派
            order = np.argsort(srow, kind='stable')
            sr = srow[order]
            st = np.flatnonzero(np.r_[True, sr[1:] != sr[:-1]])
            grp = sr[st]
            red = lambda v: (np.add.reduceat(v[order], st), np.maximum.reduceat(v[order], st))
            s_, m_ = red(tot_N)
            t2['n_act'][grp] = np.diff(np.r_[st, len(sr)]).astype(np.int32)
            t2['nll_N_sum'][grp] = s_
            t2['nll_N_max'][grp] = m_
            t2['nll_size_N_sum'][grp] = np.add.reduceat(ns[order], st)
            for nm, msk in (('nll_N_fold_max', a_isfold[idx]), ('nll_N_call_max', a_iscall[idx]),
                            ('nll_N_agg_max', a_isagg[idx])):
                t2[nm][grp] = np.maximum.reduceat(np.where(msk, tot_N, NEG)[order], st)
            for k4 in range(4):
                v = np.where(sidx == k4, tot_N, 0.0).astype(np.float32)
                t2_street[grp * 4 + k4] = np.add.reduceat(v[order], st)
            ty = a_type_all[idx]
            by_n += np.bincount(ty, weights=tot_N.astype(np.float64), minlength=6)
            by_c += np.bincount(ty, minlength=6)
            # ---- I 模式：每個動作 × 該手其他 5 位
            for k in range(1, 6):
                ys = (slot + k) % 6
                Xi = F.with_y(idx, ys)
                it, _, _ = nll_of(mI, Xi, y)
                is_ = np.zeros(len(idx), np.float32)
                if agg.any():
                    is_[agg] = nll_of(mIs, Xi[agg], am[agg])[0]
                del Xi
                tot_I = it + is_
                by_i += np.bincount(ty, weights=tot_I.astype(np.float64), minlength=6) / 5.0
                gidx = grp * 6 + ((grp % 6) + k) % 6
                t3_sum[gidx] = np.add.reduceat(tot_I[order], st)
                t3_max[gidx] = np.maximum.reduceat(tot_I[order], st)
                t3_gmax[gidx] = np.maximum.reduceat((tot_N - tot_I)[order], st)
            if (c0 // args.chunk_hands) % 2 == 0:
                done = c0 + len(hb)
                el = time.time() - t_s
                log(f'  打分 {done:,}/{len(sc_h):,} 手 {done/max(el,1e-9):.0f} hands/s '
                    f'eta={(len(sc_h)-done)/max(done/max(el,1e-9),1e-9)/60:.1f}min')
        del mN, mNs, mI, mIs

    log('組表輸出')
    T2 = {'n_act': t2['n_act'], 'nll_N_sum': t2['nll_N_sum'],
          'nll_N_max': nanify(t2['nll_N_max']), 'nll_N_fold_max': nanify(t2['nll_N_fold_max']),
          'nll_N_call_max': nanify(t2['nll_N_call_max']), 'nll_N_agg_max': nanify(t2['nll_N_agg_max']),
          'nll_size_N_sum': t2['nll_size_N_sum']}
    st4 = t2_street.reshape(-1, 4)
    for k, nm in enumerate(['pre', 'flop', 'turn', 'river']):
        T2[f'nll_N_{nm}'] = st4[:, k].copy()
    summ = write_all(args.out, args.prefix, args.tables, T1, T2,
                     t3_sum, nanify(t3_max), nanify(t3_gmax), log=log)
    ACT = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    summ['by_action_type'] = {ACT[i]: {'n': int(by_c[i]), 'nll_N': by_n[i] / max(by_c[i], 1),
                                       'nll_I_mean_over_Y': by_i[i] / max(by_c[i], 1),
                                       'mean_gain': (by_n[i] - by_i[i]) / max(by_c[i], 1)}
                              for i in range(6)}
    json.dump(summ, open(os.path.join(args.out, f'{args.prefix}_infer_summary.json'), 'w'), indent=2)
    info['summary'] = summ
    json.dump(info, open(os.path.join(args.out, f'{args.prefix}_train_info.json'), 'w'),
              indent=2, ensure_ascii=False)
    log('gbnll 完成', json.dumps(summ))


if __name__ == '__main__':
    main()
