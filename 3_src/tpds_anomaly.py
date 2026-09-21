"""Unsupervised within-table pair anomaly (for other_coordination, which never appears in the public labels).
For every pair, standardise its Marginal-Impact statistics against the other 434 pairs at the SAME table
(robust z = (x - median)/IQR). A pair that is extreme relative to its own table's population is anomalous
regardless of which of the four mechanisms produced it.
Writes 1_data/processed/anomaly_{phase}.parquet keyed (a, b).
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_paths import vpath

STATS = ['flow_asym', 'flow_rate', 'both_vpip_rate', 'agg_a_contrast', 'agg_b_contrast', 'sum_agg_contrast',
         'ab_d_fold', 'ba_d_fold', 'sum_d_fold', 'max_d_fold', 'ab_d_raise', 'ba_d_raise', 'sum_d_raise', 'min_d_raise',
         'ab_d_call', 'ba_d_call', 'ab_d_surp', 'ba_d_surp', 'sum_d_surp', 'max_resp_surp',
         'ab_surp_mean', 'ba_surp_mean', 'ab_surp_max', 'ba_surp_max', 'flow_a2b', 'flow_b2a']


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(phase):
    ps = pl.read_parquet(vpath(f'pairstats_{phase}.parquet'))
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select(['pair_id', 'a', 'b', 'table_id'])
    # table id for every pair in the pool (not only the candidate ones) so the null population is the whole table
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['player_id', 'table_id']).unique()
    ps = ps.join(l0.rename({'player_id': 'a'}), on='a', how='left')
    cols = [c for c in STATS if c in ps.columns]
    z = ps.select(['a', 'b', 'table_id'] + cols)
    exprs = []
    for c in cols:
        med = pl.col(c).median().over('table_id')
        q1 = pl.col(c).quantile(0.25).over('table_id')
        q3 = pl.col(c).quantile(0.75).over('table_id')
        exprs.append(((pl.col(c) - med) / (q3 - q1 + 1e-6)).cast(pl.Float32).alias(f'z_{c}'))
    z = z.with_columns(exprs)
    zc = [f'z_{c}' for c in cols]
    z = z.with_columns(
        pl.max_horizontal([pl.col(c).abs() for c in zc]).alias('z_max'),
        pl.sum_horizontal([pl.col(c).abs().clip(0, 20) for c in zc]).alias('z_sum'),
        pl.sum_horizontal([(pl.col(c).abs() > 3).cast(pl.Int8) for c in zc]).alias('z_n3'),
        pl.sum_horizontal([(pl.col(c).abs() > 5).cast(pl.Int8) for c in zc]).alias('z_n5'),
    )
    out = cand.join(z.select(['a', 'b'] + zc + ['z_max', 'z_sum', 'z_n3', 'z_n5']), on=['a', 'b'], how='left')
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in out.columns if c.startswith('z')])
    out.drop('table_id').write_parquet(vpath(f'anomaly_{phase}.parquet'))
    log(f"anomaly {phase}: {out.height:,} pairs, {len(zc)+4} features")


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph)
