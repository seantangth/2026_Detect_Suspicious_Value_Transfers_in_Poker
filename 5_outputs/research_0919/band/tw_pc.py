"""截短世界第 2 部分：夥伴在場對照兩族（pc_ 熵校正版 54 欄＝presence_contrast_x；pc2_ 推廣版 120 欄）依時間窗重算。
配方逐式照抄 5_outputs/eqx_0913/build_presence_contrast.py（PC_NLLX=1）與 build_presence_contrast_v2.py，
唯一差別：只取 dev 期、hands 依 t_rank∈[lo,hi) 過濾、時間五分位 q5 在窗口內重算。
用法: python tw_pc.py <pc|pc2> ；一次跑 FULL（一致性檢查）、W1、W2。"""
import os, sys, time
os.environ.setdefault('POLARS_MAX_THREADS', '6')
import polars as pl, numpy as np
R = (str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/')
raw = R + '1_data/raw/detect-suspicious-value-transfers-in-poker/'; EQ = R + '5_outputs/eqx_0913/'; SQ = R + '5_outputs/seqnll_0912/'; PROC = R + '1_data/processed/'
OUT = R + '5_outputs/research_0919/band/tw/'
WINDOWS = [('FULL', 0, 3000), ('W1', 0, 2000), ('W2', 1000, 3000)]
which = sys.argv[1]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
t0 = time.time()
h = pl.read_parquet(PROC + 'hands_l1.parquet', columns=['hand_id', 'table_id', 'phase', 't_rank']).filter(pl.col('phase') == 'development').drop('phase')
hid = h.select('hand_id')
if which == 'pc':
    MET = ['nll_sum', 'nll_max', 'n_act', 'folded', 'sd', 'contrib']; PRE = 'pc_'
    s = pl.read_parquet(raw + 'seats.parquet', columns=['hand_id', 'player_id', 'starting_stack', 'total_contribution', 'folded', 'went_to_showdown']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8)).join(h, on='hand_id')
    g = pl.read_parquet(SQ + 'gbnll_player.parquet', columns=['hand_id', 'player_id', 'n_act', 'nll_N_sum', 'nll_N_max']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8)).join(hid, on='hand_id', how='semi')
    s = s.join(g, on=['hand_id', 'player_id'], how='left'); del g
    x = pl.read_parquet(EQ + 'nllx_hand_player.parquet').join(hid, on='hand_id', how='semi'); s = s.join(x, on=['hand_id', 'player_id'], how='left'); del x
    s = s.with_columns(pl.col('nllx_sum').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_N_sum'))
    s = s.select('hand_id', 'player_id', 'table_id', 't_rank',
                 pl.col('nll_N_sum').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_sum'), pl.col('nll_N_max').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_max'),
                 pl.col('n_act').fill_null(0).cast(pl.Float64).alias('n_act'), pl.col('folded').cast(pl.Float64).alias('folded'), pl.col('went_to_showdown').cast(pl.Float64).alias('sd'),
                 (pl.col('total_contribution') / pl.col('starting_stack').clip(1)).cast(pl.Float64).alias('contrib'))
else:
    MET = ['nllx_s0', 'nllx_s1', 'nllx_s2', 'nllx_s3', 'nllsz_sum', 'nll_N_fold_max', 'nll_N_call_max', 'nll_N_agg_max', 'vpip', 'pfr', 'n_agg', 'n_call', 'n_fold', 'net_frac', 'won_share']; PRE = 'pc2_'
    act = pl.scan_parquet(raw + 'actions.parquet').select('hand_id', 'player_id', 'street', 'action').with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8), pl.col('action').cast(pl.Utf8), pl.col('street').cast(pl.Utf8)).join(hid.lazy(), on='hand_id', how='semi')
    is_agg = pl.col('action').is_in(['bet', 'raise', 'all_in']); is_call = pl.col('action') == 'call'; is_fold = pl.col('action') == 'fold'; pre = pl.col('street').str.to_lowercase().str.starts_with('pre')
    a1 = act.group_by('hand_id', 'player_id').agg((is_agg).sum().alias('n_agg'), is_call.sum().alias('n_call'), is_fold.sum().alias('n_fold'), ((is_agg | is_call) & pre).any().cast(pl.Float64).alias('vpip'), (is_agg & pre).any().cast(pl.Float64).alias('pfr')).collect(engine='streaming')
    log(f'actions agg rows {a1.height:,}')
    ga = pl.scan_parquet(SQ + 'gbnll_action.parquet').select('hand_id', 'player_id', 'sidx', 'nll_type_N', 'entropy_N', 'nll_size_N').with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8), (pl.col('nll_type_N') - pl.col('entropy_N')).alias('x')).join(hid.lazy(), on='hand_id', how='semi')
    a2 = ga.group_by('hand_id', 'player_id').agg(*[pl.col('x').filter(pl.col('sidx') == k).sum().alias(f'nllx_s{k}') for k in range(4)], pl.col('nll_size_N').fill_null(0.0).sum().alias('nllsz_sum')).collect(engine='streaming')
    log(f'gbnll_action agg rows {a2.height:,}')
    gp = pl.read_parquet(SQ + 'gbnll_player.parquet', columns=['hand_id', 'player_id', 'nll_N_fold_max', 'nll_N_call_max', 'nll_N_agg_max']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8)).join(hid, on='hand_id', how='semi')
    s = pl.read_parquet(raw + 'seats.parquet', columns=['hand_id', 'player_id', 'starting_stack', 'net_chips', 'won_share']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8)).join(h, on='hand_id').join(a1, on=['hand_id', 'player_id'], how='left').join(a2, on=['hand_id', 'player_id'], how='left').join(gp, on=['hand_id', 'player_id'], how='left')
    del a1, a2, gp
    s = s.with_columns((pl.col('net_chips') / pl.col('starting_stack').clip(1)).alias('net_frac')).select('hand_id', 'player_id', 'table_id', 't_rank', *[pl.col(m).cast(pl.Float64).fill_null(0.0).fill_nan(0.0) for m in MET])
log(f'{which}: dev seat rows {s.height:,} ({time.time() - t0:.0f}s)')
def stats(df, keys, sfx):
    return df.group_by(keys).agg([pl.len().alias('n' + sfx)] + [pl.col(m).sum().alias(f'{m}_s{sfx}') for m in MET] + [(pl.col(m) ** 2).sum().alias(f'{m}_q{sfx}') for m in MET])
def welch(d, sp, sa, tag):
    ex = []
    for m in MET:
        np_, na = pl.col('n' + sp), pl.col('n' + sa)
        mp = pl.col(f'{m}_s{sp}') / np_; ma = pl.col(f'{m}_s{sa}') / na.clip(1)
        vp = (pl.col(f'{m}_q{sp}') / np_ - mp ** 2).clip(0); va = (pl.col(f'{m}_q{sa}') / na.clip(1) - ma ** 2).clip(0)
        diff = pl.when(na > 0).then(mp - ma).otherwise(0.0); se2 = vp / np_ + va / na.clip(1) + 1e-6
        ex += [diff.alias(f'{m}_d{tag}'), (diff / se2.sqrt()).alias(f'{m}_t{tag}'), se2.alias(f'{m}_v{tag}')]
    return d.with_columns(ex)
parts = s.partition_by('table_id', as_dict=True); del s
cp = pl.read_parquet(PROC + 'cand_pairs_development.parquet').select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
for wtag, lo, hi in WINDOWS:
    res = []; t1 = time.time()
    for key, st in parts.items():
        x = st.filter((pl.col('t_rank') >= lo) & (pl.col('t_rank') < hi)).with_columns((((pl.col('t_rank') - lo) * 5) // (hi - lo)).cast(pl.Int8).alias('q5')).drop('table_id', 't_rank')
        tot = stats(x, ['player_id'], '_all'); totq = stats(x, ['player_id', 'q5'], '_allq')
        pr = x.join(x.select('hand_id', pl.col('player_id').alias('other')), on='hand_id').filter(pl.col('player_id') != pl.col('other'))
        p = stats(pr, ['player_id', 'other'], '_p').join(tot, on='player_id')
        for m in MET: p = p.with_columns((pl.col(f'{m}_s_all') - pl.col(f'{m}_s_p')).alias(f'{m}_s_a'), (pl.col(f'{m}_q_all') - pl.col(f'{m}_q_p')).alias(f'{m}_q_a'))
        p = welch(p.with_columns((pl.col('n_all') - pl.col('n_p')).alias('n_a')), '_p', '_a', 'g')
        pq = stats(pr, ['player_id', 'other', 'q5'], '_pq').join(totq, on=['player_id', 'q5'])
        for m in MET: pq = pq.with_columns((pl.col(f'{m}_s_allq') - pl.col(f'{m}_s_pq')).alias(f'{m}_s_aq'), (pl.col(f'{m}_q_allq') - pl.col(f'{m}_q_pq')).alias(f'{m}_q_aq'))
        pq = welch(pq.with_columns((pl.col('n_allq') - pl.col('n_pq')).alias('n_aq')), '_pq', '_aq', 'q'); wsum = pl.col('n_pq').sum()
        lq = pq.group_by('player_id', 'other').agg([((pl.col(f'{m}_dq') * pl.col('n_pq')).sum() / wsum).alias(f'{m}_dl') for m in MET] +
                                                   [(((pl.col(f'{m}_dq') * pl.col('n_pq')).sum() / wsum) / (((pl.col('n_pq') / wsum) ** 2 * pl.col(f'{m}_vq')).sum() + 1e-12).sqrt()).alias(f'{m}_tl') for m in MET])
        p = p.join(lq, on=['player_id', 'other'], how='left')
        keep = ['player_id', 'other'] + (['n_p', 'n_a'] if which == 'pc' else []) + [c for c in p.columns if c[-3:] in ('_dg', '_tg', '_dl', '_tl')]
        res.append(p.select(keep))
    d = pl.concat(res); fc = [c for c in d.columns if c not in ('player_id', 'other')]
    ab = d.rename({'player_id': 'a', 'other': 'b'}).rename({c: c + '_ab' for c in fc}); ba = d.rename({'player_id': 'b', 'other': 'a'}).rename({c: c + '_ba' for c in fc})
    j = cp.join(ab, on=['a', 'b'], how='left').join(ba, on=['a', 'b'], how='left')
    if which == 'pc':
        ex = [pl.min_horizontal('n_a_ab', 'n_a_ba').alias('pc_n_absent_min'), pl.col('n_p_ab').alias('pc_n_present')]
        for c in fc:
            if c in ('n_p', 'n_a'): continue
            ex += [((pl.col(c + '_ab') + pl.col(c + '_ba')) / 2).alias(f'pc_{c}_mean'), pl.min_horizontal(c + '_ab', c + '_ba').alias(f'pc_{c}_min')]
            if c.startswith('nll_sum'): ex.append(pl.max_horizontal(c + '_ab', c + '_ba').alias(f'pc_{c}_max'))
        o = j.select(['pair_id'] + ex).fill_null(0.0); ref_file = EQ + 'presence_contrast_x_development.parquet'; name = 'presence_contrast_x.parquet'
    else:
        ex = []
        for c in fc: ex += [((pl.col(c + '_ab') + pl.col(c + '_ba')) / 2).alias(f'pc2_{c}_mean'), pl.min_horizontal(c + '_ab', c + '_ba').alias(f'pc2_{c}_min')]
        o = j.select(['pair_id'] + ex).fill_null(0.0).fill_nan(0.0); ref_file = EQ + 'presence_contrast_v2_development.parquet'; name = 'presence_contrast_v2.parquet'
    os.makedirs(OUT + wtag, exist_ok=True); o.write_parquet(OUT + wtag + '/' + name)
    log(f'{which} {wtag}: pairs {o.height:,} cols {o.width - 1} ({time.time() - t1:.0f}s)')
    if wtag == 'FULL':
        ref = pl.read_parquet(ref_file).sort('pair_id'); oo = o.sort('pair_id'); assert ref['pair_id'].to_list() == oo['pair_id'].to_list()
        assert set(ref.columns) == set(oo.columns), (sorted(set(ref.columns) ^ set(oo.columns))[:10])
        worst = sorted(((float(np.nanmax(np.abs(oo[c].to_numpy() - ref[c].to_numpy()))), c) for c in ref.columns if c != 'pair_id'), reverse=True)
        log(f'FULL 一致性：欄數 {len(ref.columns) - 1}，最大絕對差前 5 {worst[:5]}')
log(f'ALL DONE {time.time() - t0:.0f}s')
