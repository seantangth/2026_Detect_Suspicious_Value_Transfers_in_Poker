"""Head-rerank helpers (context / EXTRA / P2 / lg / neutralise / place) frozen 2026-09-13 as a verbatim slice of
5_outputs/seqnll_0912/ens_head_rerank.py (v022). pair_chain_v5x.py / ens_head_rerank_p150.py exec() that slice at run time;
for a clean rebuild import this module instead (needs R, np, pd, pl, lgb in scope as in the chains)."""
import numpy as np, pandas as pd, polars as pl, lightgbm as lgb
R = (str(__import__('pathlib').Path(__file__).resolve().parents[2]) + '/')
def context(phase, pids, score, hs_file, dom_file):
    c = pl.read_parquet(R + f'1_data/processed/cand_pairs_{phase}.parquet', columns=['pair_id', 'a', 'b']).join(pl.DataFrame({'pair_id': pids, 'oof': score}), on='pair_id')
    long = pl.concat([c.select('pair_id', pl.col('a').alias('pl'), 'oof'), c.select('pair_id', pl.col('b').alias('pl'), 'oof')])
    long = long.with_columns(pl.col('oof').rank(method='ordinal', descending=True).over('pl').alias('rk_in_pl'))
    t1 = long.filter(pl.col('rk_in_pl') == 1).select('pl', pl.col('pair_id').alias('q1'), pl.col('oof').alias('o1'))
    t2 = long.filter(pl.col('rk_in_pl') == 2).select('pl', pl.col('pair_id').alias('q2'), pl.col('oof').alias('o2'))
    long = long.join(t1, on='pl', how='left').join(t2, on='pl', how='left').with_columns(
        other=pl.when(pl.col('q1') != pl.col('pair_id')).then(pl.col('o1')).otherwise(pl.col('o2')).fill_null(0.0),
        q=pl.when(pl.col('q1') != pl.col('pair_id')).then(pl.col('q1')).otherwise(pl.col('q2')))
    ctx = long.group_by('pair_id').agg(pl.col('other').max().alias('ctx_other_max'), pl.col('other').min().alias('ctx_other_min'),
                                       pl.col('rk_in_pl').max().alias('ctx_rk_max'), pl.col('rk_in_pl').min().alias('ctx_rk_min'))
    hsall = pl.scan_parquet(hs_file).select('pair_id', 'hand_id', 's_gen')
    prof = hsall.group_by('pair_id').agg(pl.col('s_gen').top_k(3).alias('tk')).collect().with_columns(
        t1=pl.col('tk').list.get(0), t3=pl.col('tk').list.get(2, null_on_oob=True)).drop('tk')
    hs = hsall.sort('s_gen', descending=True).group_by('pair_id').head(5).collect()
    mm = long.filter(pl.col('q').is_not_null() & (pl.col('other') > pl.col('oof'))).select('pair_id', 'q')
    ov = mm.join(hs.select('pair_id', 'hand_id'), on='pair_id').join(hs.select(pl.col('pair_id').alias('q'), 'hand_id'), on=['q', 'hand_id'])
    ovl = ov.group_by('pair_id', 'q').len().group_by('pair_id').agg((pl.col('len').max() / 5.0).alias('ovl'))
    dom = pl.read_parquet(dom_file).select('pair_id', 'dom5', 'dommass', 'ratio5')
    x = pl.DataFrame({'pair_id': pids, 'oof': score}).join(ctx, on='pair_id', how='left').join(ovl, on='pair_id', how='left') \
          .join(dom, on='pair_id', how='left').join(prof, on='pair_id', how='left').with_columns(pl.col('ovl').fill_null(0.0), pl.col('dom5').fill_null(0.0))
    x = x.with_columns((pl.col('oof') - pl.col('ctx_other_max')).alias('ctx_gap'), (pl.col('oof') / (pl.col('ctx_other_max') + 1e-6)).alias('ctx_ratio'))
    return x.to_pandas().set_index('pair_id').loc[pids]
EXTRA = ['oof', 'ctx_other_max', 'ctx_other_min', 'ctx_rk_max', 'ctx_rk_min', 'ctx_gap', 'ctx_ratio', 'ovl', 'dom5', 'dommass', 'ratio5']
P2 = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, n_jobs=8)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def neutralise(X, f2, base):
    Xn = X.copy(); ix = {f: f2.index(f) for f in EXTRA}
    Xn[:, ix['ctx_other_max']] = 0.0; Xn[:, ix['ctx_other_min']] = 0.0; Xn[:, ix['ctx_gap']] = base; Xn[:, ix['ctx_ratio']] = base / 1e-6
    Xn[:, ix['ctx_rk_max']] = 1.0; Xn[:, ix['ctx_rk_min']] = 1.0; Xn[:, ix['ovl']] = 0.0; return Xn
def place(base, head, adj):
    r = pd.Series(base).rank(method='average').to_numpy() / (len(base) + 1); z = np.log(r / (1 - r))
    fh = pd.Series(np.where(head, z + adj, -np.inf)).rank().to_numpy()
    return np.where(head, 2.0 + fh / (len(base) + 1), pd.Series(base).rank().to_numpy() / (len(base) + 1))
