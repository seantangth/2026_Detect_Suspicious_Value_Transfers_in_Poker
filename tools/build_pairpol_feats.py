"""Rebuild pairpol_feats_{phase}.parquet (21 pair columns) from the cloud pairpol output pairpol_pair.parquet.
For each candidate pair (a, b): the two directed rows (a->b, b->a) of 7 statistics, combined as mean / min / max."""
import sys
from pathlib import Path
import polars as pl
R = str(Path(__file__).resolve().parents[1]) + '/'
OUT = sys.argv[1] if len(sys.argv) > 1 else R + '5_outputs/eqx_0913'
P = pl.read_parquet(R + '5_outputs/pairpol_0913/pairpol_pair.parquet')
ST = ['llr_mean_hand', 'llr_max_hand', 'llr_top3_hand', 'llr_pos_frac', 'llr_z', 'r_norm', 'u_norm']
for ph in ('development', 'evaluation'):
    d = P.filter(pl.col('phase') == ph).select(pl.col('player_id').cast(pl.Utf8).alias('x'), pl.col('other_id').cast(pl.Utf8).alias('y'), *[pl.col(c).cast(pl.Float64) for c in ST])
    cp = pl.read_parquet(R + f'1_data/processed/cand_pairs_{ph}.parquet').select('pair_id', pl.col('a').cast(pl.Utf8), pl.col('b').cast(pl.Utf8))
    g = cp.join(d.rename({'x': 'a', 'y': 'b', **{c: c + '_ab' for c in ST}}), on=['a', 'b'], how='left') \
          .join(d.rename({'x': 'b', 'y': 'a', **{c: c + '_ba' for c in ST}}), on=['a', 'b'], how='left')
    ex = []
    for c in ST:
        A, B = pl.col(c + '_ab'), pl.col(c + '_ba')
        ex += [((A + B) / 2).alias(f'pp_{c}_mean'), pl.min_horizontal(A, B).alias(f'pp_{c}_min'), pl.max_horizontal(A, B).alias(f'pp_{c}_max')]
    o = g.select(['pair_id'] + ex).fill_null(0.0).fill_nan(0.0)
    out = OUT.rstrip('/') + f'/pairpol_feats_{ph}.parquet'; o.write_parquet(out); print(ph, o.shape, '->', out)
