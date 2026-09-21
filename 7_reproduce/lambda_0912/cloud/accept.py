#!/usr/bin/env python3
"""TPDS seqnll — SPEC §5 驗收數字。只在這裡用到勾結標籤（訓練完全沒用）。"""
import argparse, json, os, time
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


def collect_stream(lf):
    """polars 各版本的 streaming collect API 名稱不同，逐一退回。"""
    for kw in ({'engine': 'streaming'}, {'streaming': True}, {}):
        try:
            return lf.collect(**kw)
        except TypeError:
            continue
    return lf.collect()


def auc(pos, neg):
    """Mann-Whitney U 的 AUC。NaN 一律丟掉。"""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return None, len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind='mergesort')
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = r[:len(pos)].sum()
    a = (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(a), len(pos), len(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/ubuntu/tpds/data')
    ap.add_argument('--out', default='/home/ubuntu/tpds/out')
    ap.add_argument('--tables', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--ckpt', default='/home/ubuntu/tpds/ckpt')
    ap.add_argument('--strict', action='store_true', help='全量模式：共享手太少就直接失敗')
    ap.add_argument('--prefix', default='seqnll')
    args = ap.parse_args()
    RAW = os.path.join(args.data, 'raw')
    res = {}

    # ---------------------------------------------------- held-out NLL（全 out-of-sample）
    log('讀 T1 算 held-out NLL')
    hands = pl.read_parquet(os.path.join(RAW, 'hands.parquet'), columns=['hand_id', 'phase'])
    dev_hands = hands.filter(pl.col('phase') == 'development').select('hand_id')
    t1 = pl.read_parquet(os.path.join(args.out, f'{args.prefix}_action.parquet'),
                         columns=['hand_id', 'nll_type_N', 'nll_size_N', 'p_taken_N', 'entropy_N'])
    t1 = t1.with_columns(pl.col('hand_id').cast(pl.String))
    n_all = t1.height
    d = t1.join(dev_hands.with_columns(pl.col('hand_id').cast(pl.String)), on='hand_id', how='inner')
    nt = d['nll_type_N'].to_numpy(); ns = d['nll_size_N'].to_numpy()
    agg = np.isfinite(ns)
    res['heldout_nll'] = {
        'scope': 'development 期全部動作，全部 out-of-sample（另一半模型打分）',
        'n_actions': int(len(nt)), 'n_agg_actions': int(agg.sum()),
        'type': float(nt.mean()),
        'size_per_agg_action': float(ns[agg].mean()),
        'size_per_action': float(np.nan_to_num(ns, nan=0.0).mean()),
        'total_per_action': float(nt.mean() + np.nan_to_num(ns, nan=0.0).mean()),
        'mean_p_taken': float(d['p_taken_N'].to_numpy().mean()),
        'mean_entropy': float(d['entropy_N'].to_numpy().mean()),
        'all_2M_hands_type': float(t1['nll_type_N'].to_numpy().mean()),
        'n_actions_all': int(n_all)}
    del t1
    # 邊際熵（dev 期）
    meta = json.load(open(os.path.join(args.tables, 'meta.json')))
    res['reference'] = {'marginal_type_entropy_nats_recomputed': meta['marginal_type_entropy_nats'],
                        'spec_marginal_entropy': 1.198, 'spec_local_gbdt_type_nll': 0.501}
    res['prefix'] = args.prefix
    for g in (0, 1):
        p = os.path.join(args.ckpt, f'g{g}_hist.json')
        if os.path.exists(p):
            h = json.load(open(p))
            res[f'train_g{g}'] = {'best_epoch': h['best_epoch'], 'best_mix_nll': h['best_mix'],
                                  'history': [{k: v for k, v in e.items()
                                               if k in ('epoch', 'train_loss', 'N_type', 'N_total',
                                                        'I_total', 'MIX_total', 'epoch_sec')}
                                              for e in h['history']]}

    # ---------------------------------------------------- 配對驗收
    log('組配對的共享手')
    lab = pl.read_csv(os.path.join(RAW, 'development_labels.csv'))
    evi = pl.read_csv(os.path.join(RAW, 'development_evidence.csv'))
    seats = pl.read_parquet(os.path.join(RAW, 'seats.parquet'), columns=['hand_id', 'player_id'])
    seats = seats.join(dev_hands, on='hand_id', how='inner')
    pairs = lab.select(['pair_id', 'player_1', 'player_2', 'label', 'label_status'])
    P = pairs.rename({'player_1': 'pA', 'player_2': 'pB'})
    sh = (seats.rename({'player_id': 'pA'}).join(P, on='pA')
                .join(seats.rename({'player_id': 'pB'}), on=['hand_id', 'pB'])
                .select(['pair_id', 'hand_id', 'pA', 'pB', 'label']))
    log(f'共享手 {sh.height:,} 列，配對 {sh["pair_id"].n_unique()}')
    if args.strict:
        assert sh.height > 10000, f'共享手只有 {sh.height} 列，遠低於預期'
    if sh.height == 0:
        log('⚠ 沒有任何共享手（smoke 子集太小），跳過配對驗收段')
        json.dump(res, open(os.path.join(args.out, f'{args.prefix}_acceptance.json'), 'w'), indent=2, ensure_ascii=False)
        return
    ev = evi.select(['pair_id', 'hand_id']).with_columns(pl.lit(1, pl.Int8).alias('is_evi'))
    sh = sh.join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('is_evi').fill_null(0))

    log('接 T2 的 nll_N_max')
    t2 = pl.read_parquet(os.path.join(args.out, f'{args.prefix}_player.parquet'),
                         columns=['hand_id', 'player_id', 'n_act', 'nll_N_max']).with_columns(
        pl.col('hand_id').cast(pl.String), pl.col('player_id').cast(pl.String))
    need = pl.concat([sh.select(pl.col('hand_id'), pl.col('pA').alias('player_id')),
                      sh.select(pl.col('hand_id'), pl.col('pB').alias('player_id'))]).unique()
    t2s = t2.join(need, on=['hand_id', 'player_id'], how='semi')
    del t2
    sh = (sh.join(t2s.rename({'player_id': 'pA', 'nll_N_max': 'mA', 'n_act': 'nA'}), on=['hand_id', 'pA'], how='left')
            .join(t2s.rename({'player_id': 'pB', 'nll_N_max': 'mB', 'n_act': 'nB'}), on=['hand_id', 'pB'], how='left'))
    sh = sh.with_columns(pl.max_horizontal(pl.col('mA').fill_nan(None),
                                           pl.col('mB').fill_nan(None)).alias('stat_nllmax'))

    log('接 T3 的 gain_max')
    t3 = pl.scan_parquet(os.path.join(args.out, f'{args.prefix}_pair.parquet')).select(
        pl.col('hand_id').cast(pl.String), pl.col('player_id').cast(pl.String),
        pl.col('other_id').cast(pl.String), pl.col('gain_max'))
    keyX = pl.concat([sh.select(pl.col('hand_id'), pl.col('pA').alias('player_id')),
                      sh.select(pl.col('hand_id'), pl.col('pB').alias('player_id'))]).unique()
    t3s = collect_stream(t3.join(keyX.lazy(), on=['hand_id', 'player_id'], how='semi'))
    log(f'T3 相關列 {t3s.height:,}')
    gm = t3s.rename({'player_id': 'X', 'other_id': 'Y'})
    # 該 (hand, X) 對所有 Y 的 gain_max 總和與計數 -> 用來算「其他 Y 的平均」
    tot = gm.group_by(['hand_id', 'X']).agg(pl.col('gain_max').sum().alias('gsum'),
                                            pl.col('gain_max').count().alias('gcnt'))
    def attach(df, xcol, ycol, sfx):
        o = (df.join(gm.rename({'X': xcol, 'Y': ycol, 'gain_max': f'g_{sfx}'}),
                     on=['hand_id', xcol, ycol], how='left')
               .join(tot.rename({'X': xcol, 'gsum': f'gs_{sfx}', 'gcnt': f'gc_{sfx}'}),
                     on=['hand_id', xcol], how='left'))
        return o.with_columns(
            (pl.col(f'g_{sfx}').fill_nan(None)
             - (pl.col(f'gs_{sfx}').fill_nan(None) - pl.col(f'g_{sfx}').fill_nan(None))
             / (pl.col(f'gc_{sfx}') - 1)).alias(f'adj_{sfx}'))
    sh = attach(sh, 'pA', 'pB', 'AB')
    sh = attach(sh, 'pB', 'pA', 'BA')
    sh = sh.with_columns(pl.max_horizontal(pl.col('adj_AB').fill_nan(None),
                                           pl.col('adj_BA').fill_nan(None)).alias('stat_gainmax'))

    posp = sh.filter(pl.col('label') == 1)
    a1 = auc(posp.filter(pl.col('is_evi') == 1)['stat_nllmax'].to_numpy(),
             posp.filter(pl.col('is_evi') == 0)['stat_nllmax'].to_numpy())
    a2 = auc(posp.filter(pl.col('is_evi') == 1)['stat_gainmax'].to_numpy(),
             posp.filter(pl.col('is_evi') == 0)['stat_gainmax'].to_numpy())
    res['auc'] = {
        'design': '372 個公開正例配對：證據手為正、該配對其餘共享手為負，全部 pooled 成一個 AUC；無任何調參',
        'n_pos_pairs': int(posp['pair_id'].n_unique()),
        'n_evidence_hands': int((posp['is_evi'] == 1).sum()),
        'n_other_shared_hands': int((posp['is_evi'] == 0).sum()),
        'auc_T2_nll_N_max': a1[0], 'auc_T2_n_pos': a1[1], 'auc_T2_n_neg': a1[2],
        'auc_T3_gain_max_minus_otherY': a2[0], 'auc_T3_n_pos': a2[1], 'auc_T3_n_neg': a2[2]}

    ctl = sh.filter(pl.col('label') == 0)
    def dist(x):
        x = x[np.isfinite(x)]
        if not len(x):
            return None
        return {'n': int(len(x)), 'mean': float(x.mean()), 'p50': float(np.percentile(x, 50)),
                'p90': float(np.percentile(x, 90)), 'p99': float(np.percentile(x, 99))}
    res['control_1488_non_target'] = {
        'n_pairs': int(ctl['pair_id'].n_unique()), 'n_shared_hands': int(ctl.height),
        'nll_N_max': dist(ctl['stat_nllmax'].to_numpy()),
        'gain_max_adj': dist(ctl['stat_gainmax'].to_numpy())}
    res['positive_pairs_all_shared'] = {
        'nll_N_max': dist(posp['stat_nllmax'].to_numpy()),
        'gain_max_adj': dist(posp['stat_gainmax'].to_numpy())}
    res['positive_evidence_hands'] = {
        'nll_N_max': dist(posp.filter(pl.col('is_evi') == 1)['stat_nllmax'].to_numpy()),
        'gain_max_adj': dist(posp.filter(pl.col('is_evi') == 1)['stat_gainmax'].to_numpy())}
    # 額外對照：配對層級（正例配對 vs 確認非目標配對），用該配對共享手的最大值
    pl_max = sh.group_by('pair_id').agg(pl.col('label').first(),
                                        pl.col('stat_nllmax').max().alias('m1'),
                                        pl.col('stat_gainmax').max().alias('m2'))
    p1 = pl_max.filter(pl.col('label') == 1); p0 = pl_max.filter(pl.col('label') == 0)
    res['pair_level_control_auc'] = {
        'note': '補充對照，非 SPEC 要求的兩個數字',
        'auc_nll_N_max': auc(p1['m1'].to_numpy(), p0['m1'].to_numpy())[0],
        'auc_gain_max_adj': auc(p1['m2'].to_numpy(), p0['m2'].to_numpy())[0]}

    json.dump(res, open(os.path.join(args.out, f'{args.prefix}_acceptance.json'), 'w'), indent=2, ensure_ascii=False)
    log(json.dumps(res['heldout_nll'], indent=2))
    log(json.dumps(res['auc'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
