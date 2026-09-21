"""Partner-presence behavioural contrast (new pair-axis signal, 2026-09-13).
For every ordered pair (X, Y) at the same table and phase: X's per-hand behaviour statistics in hands where Y is dealt
(present) vs X's own hands where Y is not dealt (absent, same phase = own baseline). Global diff + Welch t, and a
time-local version (5 chronological quintiles of the table-phase, absent baseline matched per quintile).
Per-hand NLL comes from the 09-12 cross-fitted gbnll normal-behaviour model (5_outputs/seqnll_0912/gbnll_player.parquet,
recipe 7_reproduce/lambda_0912/cloud/gbnll.py; out-of-sample by hand-hash halves within table). No labels are used.
Output: 5_outputs/eqx_0913/presence_contrast_{phase}.parquet keyed by pair_id (cand_pairs a/b -> _ab/_ba -> mean/min/max)."""
import polars as pl, numpy as np, time, sys, os
TAG = os.environ.get('PC_TAG', ''); NLLX = os.environ.get('PC_NLLX', '0') == '1'
R = (str(__import__('pathlib').Path(__file__).resolve().parents[2]) + '/')
raw = R + '1_data/raw/detect-suspicious-value-transfers-in-poker/'; OUT = R + '5_outputs/eqx_0913/'; PROC = R + '1_data/processed/'
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
MET = ['nll_sum', 'nll_max', 'n_act', 'folded', 'sd', 'contrib']
h = pl.read_parquet(raw + 'hands.parquet', columns=['hand_id', 'table_id', 'phase', 'started_at'])
h = h.with_columns(((pl.col('started_at').rank('ordinal').over('table_id', 'phase') - 1) * 5 // pl.len().over('table_id', 'phase')).cast(pl.Int8).alias('q5')).drop('started_at')
s = pl.read_parquet(raw + 'seats.parquet', columns=['hand_id', 'player_id', 'starting_stack', 'total_contribution', 'folded', 'went_to_showdown']).join(h, on='hand_id')
g = pl.read_parquet(R + '5_outputs/seqnll_0912/gbnll_player.parquet', columns=['hand_id', 'player_id', 'n_act', 'nll_N_sum', 'nll_N_max']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8))
s = s.join(g, on=['hand_id', 'player_id'], how='left')
if NLLX:
    x = pl.read_parquet(OUT + 'nllx_hand_player.parquet'); s = s.join(x, on=['hand_id', 'player_id'], how='left')
    s = s.with_columns(pl.col('nllx_sum').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_N_sum')); log('nll_sum replaced by entropy-adjusted nllx_sum')
log(f'seat rows {s.height:,} | nll null {s["nll_N_sum"].null_count():,} nan {int(s["nll_N_sum"].is_nan().sum())} | nll_max nan {int(s["nll_N_max"].is_nan().sum())} | n_act null {s["n_act"].null_count():,}')
s = s.select('hand_id', 'player_id', 'table_id', 'phase', 'q5',
             pl.col('nll_N_sum').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_sum'), pl.col('nll_N_max').cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias('nll_max'),
             pl.col('n_act').fill_null(0).cast(pl.Float64).alias('n_act'), pl.col('folded').cast(pl.Float64).alias('folded'), pl.col('went_to_showdown').cast(pl.Float64).alias('sd'),
             (pl.col('total_contribution') / pl.col('starting_stack').clip(1)).cast(pl.Float64).alias('contrib'))
del g
def stats(df, keys, sfx):
    return df.group_by(keys).agg([pl.len().alias('n' + sfx)] + [pl.col(m).sum().alias(f'{m}_s{sfx}') for m in MET] + [(pl.col(m) ** 2).sum().alias(f'{m}_q{sfx}') for m in MET])
def welch(d, sfx_p, sfx_a, tag):
    ex = []
    for m in MET:
        np_, na = pl.col('n' + sfx_p), pl.col('n' + sfx_a)
        mp = pl.col(f'{m}_s{sfx_p}') / np_; ma = pl.col(f'{m}_s{sfx_a}') / na.clip(1)
        vp = (pl.col(f'{m}_q{sfx_p}') / np_ - mp ** 2).clip(0); va = (pl.col(f'{m}_q{sfx_a}') / na.clip(1) - ma ** 2).clip(0)
        diff = pl.when(na > 0).then(mp - ma).otherwise(0.0)
        se2 = vp / np_ + va / na.clip(1) + 1e-6
        ex += [diff.alias(f'{m}_d{tag}'), (diff / se2.sqrt()).alias(f'{m}_t{tag}'), se2.alias(f'{m}_v{tag}')]
    return d.with_columns(ex)
res = {'development': [], 'evaluation': []}
tables = s['table_id'].unique().to_list(); t0 = time.time()
for i, tb in enumerate(tables):
    st = s.filter(pl.col('table_id') == tb)
    for ph in ('development', 'evaluation'):
        x = st.filter(pl.col('phase') == ph).drop('table_id', 'phase')
        tot = stats(x, ['player_id'], '_all'); totq = stats(x, ['player_id', 'q5'], '_allq')
        pr = x.join(x.select('hand_id', pl.col('player_id').alias('other')), on='hand_id').filter(pl.col('player_id') != pl.col('other'))
        p = stats(pr, ['player_id', 'other'], '_p').join(tot, on='player_id')
        for m in MET: p = p.with_columns((pl.col(f'{m}_s_all') - pl.col(f'{m}_s_p')).alias(f'{m}_s_a'), (pl.col(f'{m}_q_all') - pl.col(f'{m}_q_p')).alias(f'{m}_q_a'))
        p = welch(p.with_columns((pl.col('n_all') - pl.col('n_p')).alias('n_a')), '_p', '_a', 'g')
        # time-local: per quintile present vs absent, weighted by present count
        pq = stats(pr, ['player_id', 'other', 'q5'], '_pq').join(totq, on=['player_id', 'q5'])
        for m in MET: pq = pq.with_columns((pl.col(f'{m}_s_allq') - pl.col(f'{m}_s_pq')).alias(f'{m}_s_aq'), (pl.col(f'{m}_q_allq') - pl.col(f'{m}_q_pq')).alias(f'{m}_q_aq'))
        pq = welch(pq.with_columns((pl.col('n_allq') - pl.col('n_pq')).alias('n_aq')), '_pq', '_aq', 'q')
        wsum = pl.col('n_pq').sum()
        lq = pq.group_by('player_id', 'other').agg([((pl.col(f'{m}_dq') * pl.col('n_pq')).sum() / wsum).alias(f'{m}_dl') for m in MET] + [(((pl.col(f'{m}_dq') * pl.col('n_pq')).sum() / wsum) / (((pl.col('n_pq') / wsum) ** 2 * pl.col(f'{m}_vq')).sum() + 1e-12).sqrt()).alias(f'{m}_tl') for m in MET])
        p = p.join(lq, on=['player_id', 'other'], how='left')
        keep = ['player_id', 'other', 'n_p', 'n_a'] + [c for c in p.columns if c[-3:] in ('_dg', '_tg', '_dl', '_tl')]
        res[ph].append(p.select(keep))
    if i % 50 == 49: log(f'{i + 1}/{len(tables)} tables ({time.time() - t0:.0f}s)')
for ph in ('development', 'evaluation'):
    d = pl.concat(res[ph]); d.write_parquet(OUT + f'presence_contrast{TAG}_ordered_{ph}.parquet')
    cp = pl.read_parquet(PROC + f'cand_pairs_{ph}.parquet').select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
    fc = [c for c in d.columns if c not in ('player_id', 'other')]
    ab = d.rename({'player_id': 'a', 'other': 'b'}).rename({c: c + '_ab' for c in fc}); ba = d.rename({'player_id': 'b', 'other': 'a'}).rename({c: c + '_ba' for c in fc})
    j = cp.join(ab, on=['a', 'b'], how='left').join(ba, on=['a', 'b'], how='left')
    ex = [pl.min_horizontal('n_a_ab', 'n_a_ba').alias('pc_n_absent_min'), pl.col('n_p_ab').alias('pc_n_present')]
    for c in fc:
        if c in ('n_p', 'n_a'): continue
        ex += [((pl.col(c + '_ab') + pl.col(c + '_ba')) / 2).alias(f'pc_{c}_mean'), pl.min_horizontal(c + '_ab', c + '_ba').alias(f'pc_{c}_min')]
        if c.startswith('nll_sum'): ex.append(pl.max_horizontal(c + '_ab', c + '_ba').alias(f'pc_{c}_max'))
    o = j.select(['pair_id'] + ex).fill_null(0.0); o.write_parquet(OUT + f'presence_contrast{TAG}_{ph}.parquet')
    log(f'{ph}: pairs {o.height:,} cols {o.width - 1} | unmatched {int((j["n_p_ab"].is_null()).sum())}')
log('ALL DONE')
