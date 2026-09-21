"""Directed-transfer direction invariant.

The generator applied ONE recipe per pair, and for `directed_transfer` that recipe has an EXACT
invariant: across all 148 dev DT pairs the five planted hands move chips in the SAME direction -
purity 1.000, 148/148, versus 0.643 for five random shared hands of the same pairs (2026-09-09 probe).
No other family/property pair in the data has this shape: soft_play 0.662 and coordinated_isolation
0.683 sit at their baselines, and every other candidate property scores BELOW baseline.

The evidence ranker scores each hand independently, so nothing stops its top-5 from mixing directions -
36.7% of its top-5 false positives run opposite to the pair's true direction, versus 17.5% of its hits.
This module supplies the per-row direction so the submission step can enforce the constraint on pairs
PREDICTED as directed_transfer (the direction itself is inferred from the model's own top-K, never
from a label, so it is available on the evaluation phase).
"""
import sys, time, glob
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(phase):
    out = PROC / f'direction_{phase}.parquet'
    if out.exists():
        return pl.read_parquet(out)
    parts = []
    for f in sorted(glob.glob(str(PROC / 'l2v2' / phase / '*.parquet'))):
        t = pl.read_parquet(f, columns=['pair_id', 'hand_id', 'A_net_bb', 'B_net_bb'])
        # +1 = A gained relative to B, -1 = B gained, 0 = neutral (check-downs and split pots)
        t = t.with_columns(pl.when(pl.col('A_net_bb') - pl.col('B_net_bb') > 0).then(1)
                             .when(pl.col('A_net_bb') - pl.col('B_net_bb') < 0).then(-1)
                             .otherwise(0).cast(pl.Int8).alias('dir'))
        parts.append(t.select(['pair_id', 'hand_id', 'dir']))
    d = pl.concat(parts)
    d.write_parquet(out)
    log(f'{phase}: {d.height:,} rows -> {out.name}')
    return d


def apply_constraint(rows, phase, dt_pairs, K=5, penalty=1e6, score_col='ev_score'):
    """rows: (pair_id, hand_id, ev_score). dt_pairs: pair_ids predicted directed_transfer.
    Infers each pair's direction from its own top-K scoring hands, then demotes hands running the
    other way. Applied ONLY to DT pairs, where the invariant is exact."""
    d = build(phase)
    r = rows.join(d, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('dir').fill_null(0))
    r = r.with_columns(pl.col(score_col).rank(method='ordinal', descending=True).over('pair_id').alias('_r'))
    inf = (r.filter(pl.col('_r') <= K).group_by('pair_id')
             .agg(pl.col('dir').sum().sign().cast(pl.Int8).alias('_idir')))
    r = r.join(inf, on='pair_id', how='left').with_columns(pl.col('_idir').fill_null(0))
    isdt = pl.col('pair_id').is_in(dt_pairs)
    bad = isdt & (pl.col('_idir') != 0) & (pl.col('dir') == -pl.col('_idir'))
    n_bad = r.filter(bad).height
    log(f'{phase}: direction constraint on {len(dt_pairs):,} DT pairs, demoted {n_bad:,} opposite-direction rows')
    return r.with_columns(pl.when(bad).then(pl.col(score_col) - penalty).otherwise(pl.col(score_col)).alias(score_col)) \
            .select(['pair_id', 'hand_id', score_col])


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        build(ph)
