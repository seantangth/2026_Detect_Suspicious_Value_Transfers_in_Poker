"""Commit-side decision value: CALL REGRET against the partner's ACTUAL hand.

Why: the fold side is covered (`fold_ev_loss`, `folder_eq_vs_partner`, `A_eq_fold_any`) but the pipeline has no
call-side twin. The 508 planted hands the ranker misses (2026-09-09 23:40) are big pots where the eventual fold is
CORRECT - the value left through the calls that built the pot. A call facing a bet C into pot P is +EV only if the
caller's equity e exceeds C/(P+C); with every hole card known, e against the partner is computable at the decision
street, so `regret = max(0, C - e*(P+C))` is the chips the caller knowingly gave up (Mazrooei-style value attribution,
restricted to the closed heads-up decision; third players and future betting are ignored, and flagged).

Only calls by a candidate-pair member FACING THE PARTNER'S aggression are scored (last_aggr == partner), so the feature
is opponent-conditional by construction. Equity source per street: preflop pfeq_{phase} (MC-200), flop equity 36-runout,
turn/river exact (tpds_equity.py) - the tables the rankers already use, so no new equity computation.

Writes 1_data/processed/callvalue_{phase}.parquet keyed (hand_id, a, b).  Enable in read_l2 with TPDS_CALLVAL=1.
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_paths import eqpath


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(phase):
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'phase']).filter(pl.col('phase') == phase).drop('phase')
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet', columns=['a', 'b'])
    calls = (pl.scan_parquet(PROC / 'action_ctx.parquet')
               .filter(pl.col('is_call') & pl.col('last_aggr').is_not_null())
               .select(['hand_id', 'sidx', 'player_id', 'last_aggr', 'to_call_bb', 'pot_before_bb', 'players_active'])
               .collect().join(l1, on='hand_id'))
    calls = calls.with_columns(pl.min_horizontal('player_id', 'last_aggr').alias('a'), pl.max_horizontal('player_id', 'last_aggr').alias('b'),
                               (pl.col('player_id') < pl.col('last_aggr')).alias('caller_is_a'))
    calls = calls.join(cand, on=['a', 'b'], how='inner')     # only pair members calling the PARTNER
    log(f'{phase}: {calls.height:,} calls facing the partner across candidate pairs')
    eq = pl.read_parquet(eqpath(f'equity_{phase}.parquet'), columns=['hand_id', 'a', 'b', 'eq_flop_A', 'eq_turn_A', 'eq_river_A'])
    pf = pl.read_parquet(eqpath(f'pfeq_{phase}.parquet'))
    d = calls.join(pf, on=['hand_id', 'a', 'b'], how='left').join(eq, on=['hand_id', 'a', 'b'], how='left')
    # equity of A vs B at the street of the call, then flip for B
    eqA = (pl.when(pl.col('sidx') == 0).then(pl.col('pf_eq_A')).when(pl.col('sidx') == 1).then(pl.col('eq_flop_A'))
             .when(pl.col('sidx') == 2).then(pl.col('eq_turn_A')).otherwise(pl.col('eq_river_A')))
    d = d.with_columns(eqA.alias('_eqA')).with_columns(
        pl.when(pl.col('caller_is_a')).then(pl.col('_eqA')).otherwise(1.0 - pl.col('_eqA')).cast(pl.Float32).alias('e'),
        (pl.col('to_call_bb') / (pl.col('pot_before_bb') + pl.col('to_call_bb') + 1e-3)).cast(pl.Float32).alias('need'))
    d = d.filter(pl.col('e').is_not_null() & pl.col('e').is_not_nan())
    d = d.with_columns(
        (pl.col('e') - pl.col('need')).cast(pl.Float32).alias('edge'),
        (pl.col('to_call_bb') - pl.col('e') * (pl.col('pot_before_bb') + pl.col('to_call_bb'))).clip(lower_bound=0).cast(pl.Float32).alias('regret'),
        (pl.col('players_active') == 2).alias('hu'),
    )
    def side(mask, pfx):
        s = d.filter(mask)
        return s.group_by(['hand_id', 'a', 'b']).agg(
            pl.len().cast(pl.Int8).alias(f'{pfx}_n_call_vs_p'),
            pl.col('regret').sum().alias(f'{pfx}_call_regret_sum'),
            pl.col('regret').max().alias(f'{pfx}_call_regret_max'),
            pl.col('regret').filter(pl.col('hu')).max().alias(f'{pfx}_call_regret_hu'),
            pl.col('regret').filter(pl.col('sidx') >= 2).max().alias(f'{pfx}_call_regret_late'),
            pl.col('edge').min().alias(f'{pfx}_call_edge_min'),
            pl.col('e').min().alias(f'{pfx}_call_eq_min'),
            pl.col('to_call_bb').max().alias(f'{pfx}_call_C_max'),
            (pl.col('edge') < -0.1).sum().cast(pl.Int8).alias(f'{pfx}_n_bad_call'),
        )
    A = side(pl.col('caller_is_a'), 'A'); B = side(~pl.col('caller_is_a'), 'B')
    out = A.join(B, on=['hand_id', 'a', 'b'], how='full', coalesce=True)
    fill0 = [c for c in out.columns if c.endswith(('_n_call_vs_p', '_call_regret_sum', '_call_regret_max', '_n_bad_call'))]
    out = out.with_columns([pl.col(c).fill_null(0) for c in fill0])
    out = out.with_columns(
        pl.max_horizontal('A_call_regret_max', 'B_call_regret_max').alias('call_regret_max'),
        (pl.col('A_call_regret_sum') + pl.col('B_call_regret_sum')).alias('call_regret_sum'),
        pl.min_horizontal('A_call_edge_min', 'B_call_edge_min').alias('call_edge_min'),
    )
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in out.columns if c not in ('hand_id', 'a', 'b') and out.schema[c] != pl.Int8])
    p = eqpath(f'callvalue_{phase}.parquet'); out.write_parquet(p)
    # ---- pair-level aggregate (order-invariant, keyed a<b): "how much did either partner knowingly hand over"
    sh = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet', columns=['a', 'b', 'shared'])
    g = out.group_by(['a', 'b']).agg(
        pl.col('A_call_regret_sum').sum().alias('_ab'), pl.col('B_call_regret_sum').sum().alias('_ba'),
        pl.col('call_regret_max').max().alias('cv_regret_max'),
        (pl.col('A_n_bad_call') + pl.col('B_n_bad_call')).sum().alias('_nbad'),
        pl.col('call_edge_min').min().alias('cv_edge_min'),
        pl.len().alias('_nh'),
    ).join(sh, on=['a', 'b'], how='inner')
    g = g.with_columns(
        (pl.col('_ab') + pl.col('_ba')).alias('cv_regret_sum'),
        pl.max_horizontal('_ab', '_ba').alias('cv_regret_dir_max'), pl.min_horizontal('_ab', '_ba').alias('cv_regret_dir_min'),
        ((pl.col('_ab') - pl.col('_ba')).abs() / (pl.col('_ab') + pl.col('_ba') + 1.0)).alias('cv_regret_asym'),
        ((pl.col('_ab') + pl.col('_ba')) / pl.col('shared')).alias('cv_regret_rate'),
        (pl.col('_nbad') / pl.col('shared')).alias('cv_bad_rate'), (pl.col('_nh') / pl.col('shared')).alias('cv_call_hand_rate'),
    ).select(['a', 'b', 'cv_regret_sum', 'cv_regret_dir_max', 'cv_regret_dir_min', 'cv_regret_asym', 'cv_regret_rate', 'cv_regret_max', 'cv_bad_rate', 'cv_call_hand_rate', 'cv_edge_min'])
    g = g.with_columns([pl.col(c).cast(pl.Float32) for c in g.columns if c not in ('a', 'b')])
    g.write_parquet(eqpath(f'callvalue_pair_{phase}.parquet'))
    log(f'{phase}: pair aggregate {g.height:,} pairs -> callvalue_pair_{phase}.parquet')
    log(f'{phase}: {out.height:,} pair-hands with a call facing the partner -> {p.name} ({len(out.columns)} cols); '
        f'regret>0 in {(out["call_regret_max"] > 0).sum():,} rows, median regret_max among those '
        f'{out.filter(pl.col("call_regret_max") > 0)["call_regret_max"].median():.2f} bb')


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph)
