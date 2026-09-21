"""截短世界第 3 部分：pairpol 21 欄依時間窗重新聚合。
逐手分片（1_data/processed/pairpol/development/T*.parquet：gain_ab_sum／gain_ba_sum／n_act_a／n_act_b）可重算
llr_mean_hand／llr_max_hand／llr_top3_hand／llr_pos_frac；llr_z 需逐動作平方和（分片沒有）→ 用全期 sq＝(llr_sum/llr_z)² 依動作數比例縮放；
r_norm／u_norm 是雲端模型的配對嵌入範數，無法依窗重算 → 沿用全期值（與生產同樣的不匹配，不更差）。
用法: python tw_pairpol.py ；一次跑 FULL（一致性檢查）、W1、W2。"""
import os, sys, time, glob
import polars as pl, numpy as np
R = (str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/')
PROC = R + '1_data/processed/'; OUT = R + '5_outputs/research_0919/band/tw/'
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
P1 = pl.read_parquet(R + '5_outputs/pairpol_0913/pairpol_pair.parquet').filter(pl.col('phase') == 'development').select(
    pl.col('player_id').cast(pl.Utf8).alias('x'), pl.col('other_id').cast(pl.Utf8).alias('y'), 'n_actions', 'llr_sum', 'llr_z', 'r_norm', 'u_norm')
P1 = P1.with_columns(pl.when(pl.col('llr_z').abs() > 1e-9).then((pl.col('llr_sum') / pl.col('llr_z')) ** 2).otherwise(0.0).alias('sq'))
h = pl.read_parquet(PROC + 'hands_l1.parquet', columns=['hand_id', 't_rank'])
cp = pl.read_parquet(PROC + 'cand_pairs_development.parquet').select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
files = sorted(glob.glob(PROC + 'pairpol/development/*.parquet'))
def agg_dir(d, g, n, tag):
    x = d.filter(pl.col(n) > 0)
    return x.group_by('pair_id').agg(pl.col(g).mean().alias(f'mean_{tag}'), pl.col(g).max().alias(f'max_{tag}'), pl.col(g).top_k(3).mean().alias(f'top3_{tag}'),
                                     (pl.col(g) > 0).mean().alias(f'pos_{tag}'), pl.col(g).sum().alias(f'sum_{tag}'), pl.col(n).sum().alias(f'nact_{tag}'))
for wtag, lo, hi in [('FULL', 0, 3000), ('W1', 0, 2000), ('W2', 1000, 3000)]:
    t0 = time.time(); res = []
    for f in files:
        d = pl.read_parquet(f).join(h, on='hand_id').filter((pl.col('t_rank') >= lo) & (pl.col('t_rank') < hi))
        res.append(agg_dir(d, 'gain_ab_sum', 'n_act_a', 'ab').join(agg_dir(d, 'gain_ba_sum', 'n_act_b', 'ba'), on='pair_id', how='full', coalesce=True))
    g = cp.join(pl.concat(res), on='pair_id', how='left')
    g = g.join(P1.rename({'x': 'a', 'y': 'b', **{c: c + '_ab' for c in ('n_actions', 'llr_sum', 'llr_z', 'r_norm', 'u_norm', 'sq')}}), on=['a', 'b'], how='left')
    g = g.join(P1.rename({'x': 'b', 'y': 'a', **{c: c + '_ba' for c in ('n_actions', 'llr_sum', 'llr_z', 'r_norm', 'u_norm', 'sq')}}), on=['a', 'b'], how='left')
    for t in ('ab', 'ba'):
        sqw = pl.col(f'sq_{t}') * pl.col(f'nact_{t}') / pl.col(f'n_actions_{t}').clip(1)
        g = g.with_columns(pl.when(sqw > 0).then(pl.col(f'sum_{t}') / sqw.sqrt()).otherwise(0.0).alias(f'z_{t}'))
    ex = []
    for name, src in (('llr_mean_hand', 'mean'), ('llr_max_hand', 'max'), ('llr_top3_hand', 'top3'), ('llr_pos_frac', 'pos'), ('llr_z', 'z'), ('r_norm', 'r_norm'), ('u_norm', 'u_norm')):
        A, B = pl.col(f'{src}_ab'), pl.col(f'{src}_ba')
        ex += [((A + B) / 2).alias(f'pp_{name}_mean'), pl.min_horizontal(A, B).alias(f'pp_{name}_min'), pl.max_horizontal(A, B).alias(f'pp_{name}_max')]
    o = g.select(['pair_id'] + ex).fill_null(0.0).fill_nan(0.0)
    os.makedirs(OUT + wtag, exist_ok=True); o.write_parquet(OUT + wtag + '/pairpol_feats.parquet'); log(f'{wtag}: {o.height:,} pairs {o.width - 1} cols ({time.time() - t0:.0f}s)')
    if wtag == 'FULL':
        ref = pl.read_parquet(R + '5_outputs/eqx_0913/pairpol_feats_development.parquet').sort('pair_id'); oo = o.sort('pair_id').select(ref.columns)
        for c in ref.columns[1:]:
            a, b = oo[c].to_numpy(), ref[c].to_numpy(); log(f'  {c:28s} 最大絕對差 {np.abs(a - b).max():.5f}  相關 {np.corrcoef(a, b)[0, 1]:.5f}')
