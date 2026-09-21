"""Opponent-conditional behaviour profiles = the Marginal Impact differential (Mazrooei et al. 2013).
For every ordered (responder, aggressor) inside a pool and phase, compare the responder's behaviour toward that
specific opponent against the same responder's behaviour toward everyone else in the same period.
Also: pairwise value flow vs the rest of the field, and joint-presence aggression contrast.
Writes 1_data/processed/pairstats_{phase}.parquet keyed (a, b) with a<b.
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_paths import vpath


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(phase):
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id', 'phase', 'big_blind']).filter(pl.col('phase') == phase)
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'table_id', 'phase', 'net_bb', 'vpip', 'n_agg', 'n_act', 'folded', 'sd', 'contrib_bb']).filter(pl.col('phase') == phase)
    resp = pl.read_parquet(PROC / 'responses.parquet').join(l1.select('hand_id'), on='hand_id')
    sur = pl.read_parquet(vpath('surprise_resp.parquet')).join(l1.select('hand_id'), on='hand_id')
    resp = resp.join(sur, on=['hand_id', 'responder', 'aggressor'], how='left')

    # --- 1. responder x aggressor profile
    ra = resp.group_by(['responder', 'aggressor']).agg(
        pl.len().alias('n_ev'), pl.col('r_fold').sum().alias('n_fold'), pl.col('r_call').sum().alias('n_call'),
        pl.col('r_raise').sum().alias('n_raise'), pl.col('rsurp_max').mean().alias('surp_mean'), pl.col('rsurp_max').max().alias('surp_max'))
    tot = ra.group_by('responder').agg(pl.col('n_ev').sum().alias('t_ev'), pl.col('n_fold').sum().alias('t_fold'),
                                       pl.col('n_call').sum().alias('t_call'), pl.col('n_raise').sum().alias('t_raise'),
                                       (pl.col('surp_mean') * pl.col('n_ev')).sum().alias('t_surp'))
    ra = ra.join(tot, on='responder').with_columns(
        (pl.col('n_fold') / pl.col('n_ev')).alias('fold_to'),
        (pl.col('n_raise') / pl.col('n_ev')).alias('raise_to'),
        (pl.col('n_call') / pl.col('n_ev')).alias('call_to'),
        ((pl.col('t_fold') - pl.col('n_fold')) / pl.max_horizontal(pl.col('t_ev') - pl.col('n_ev'), 1)).alias('fold_oth'),
        ((pl.col('t_raise') - pl.col('n_raise')) / pl.max_horizontal(pl.col('t_ev') - pl.col('n_ev'), 1)).alias('raise_oth'),
        ((pl.col('t_call') - pl.col('n_call')) / pl.max_horizontal(pl.col('t_ev') - pl.col('n_ev'), 1)).alias('call_oth'),
        ((pl.col('t_surp') - pl.col('surp_mean') * pl.col('n_ev')) / pl.max_horizontal(pl.col('t_ev') - pl.col('n_ev'), 1)).alias('surp_oth'),
    ).with_columns(
        (pl.col('fold_to') - pl.col('fold_oth')).alias('d_fold'),
        (pl.col('raise_to') - pl.col('raise_oth')).alias('d_raise'),
        (pl.col('call_to') - pl.col('call_oth')).alias('d_call'),
        (pl.col('surp_mean') - pl.col('surp_oth')).alias('d_surp'),
    ).select(['responder', 'aggressor', 'n_ev', 'fold_to', 'raise_to', 'call_to', 'd_fold', 'd_raise', 'd_call', 'd_surp', 'surp_mean', 'surp_max'])

    # --- 2. value flow between players (chips A lost while B won, per shared hand) vs field
    from itertools import combinations
    g = l0.sort(['hand_id', 'player_id']).group_by('hand_id').agg(pl.col('player_id').alias('pl'), pl.col('net_bb').alias('net'),
                                                                  pl.col('vpip').alias('vp'), pl.col('n_agg').alias('ag'),
                                                                  pl.col('folded').alias('fo'), pl.col('sd').alias('sd'),
                                                                  pl.col('contrib_bb').alias('co'), pl.col('table_id').first())
    frames = []
    for i, j in combinations(range(6), 2):
        frames.append(g.select('hand_id', 'table_id',
                               pl.col('pl').list.get(i).alias('x'), pl.col('pl').list.get(j).alias('y'),
                               pl.col('net').list.get(i).alias('nx'), pl.col('net').list.get(j).alias('ny'),
                               pl.col('vp').list.get(i).alias('vx'), pl.col('vp').list.get(j).alias('vy'),
                               pl.col('ag').list.get(i).alias('ax'), pl.col('ag').list.get(j).alias('ay'),
                               pl.col('co').list.get(i).alias('cx'), pl.col('co').list.get(j).alias('cy'),
                               pl.col('sd').list.get(i).alias('sx'), pl.col('sd').list.get(j).alias('sy')))
    P = pl.concat(frames).with_columns(pl.min_horizontal('x', 'y').alias('a'), pl.max_horizontal('x', 'y').alias('b'))
    P = P.with_columns(
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('nx')).otherwise(pl.col('ny')).alias('net_a'),
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('ny')).otherwise(pl.col('nx')).alias('net_b'),
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('vx')).otherwise(pl.col('vy')).alias('vp_a'),
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('vy')).otherwise(pl.col('vx')).alias('vp_b'),
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('ax')).otherwise(pl.col('ay')).alias('ag_a'),
        pl.when(pl.col('x') == pl.col('a')).then(pl.col('ay')).otherwise(pl.col('ax')).alias('ag_b'),
    )
    pair = P.group_by(['a', 'b']).agg(
        pl.len().alias('shared'),
        pl.min_horizontal((-pl.col('net_a')).clip(lower_bound=0), pl.col('net_b').clip(lower_bound=0)).sum().alias('flow_a2b'),
        pl.min_horizontal((-pl.col('net_b')).clip(lower_bound=0), pl.col('net_a').clip(lower_bound=0)).sum().alias('flow_b2a'),
        ((pl.col('vp_a') == 1) & (pl.col('vp_b') == 1)).mean().alias('both_vpip_rate'),
        (pl.col('ag_a') * ((pl.col('vp_b') == 1).cast(pl.Float64))).sum().alias('agg_a_when_b_in'),
        ((pl.col('vp_b') == 1).cast(pl.Float64)).sum().alias('n_b_in'),
        (pl.col('ag_a') * ((pl.col('vp_b') == 0).cast(pl.Float64))).sum().alias('agg_a_when_b_out'),
        ((pl.col('vp_b') == 0).cast(pl.Float64)).sum().alias('n_b_out'),
        (pl.col('ag_b') * ((pl.col('vp_a') == 1).cast(pl.Float64))).sum().alias('agg_b_when_a_in'),
        ((pl.col('vp_a') == 1).cast(pl.Float64)).sum().alias('n_a_in'),
        (pl.col('ag_b') * ((pl.col('vp_a') == 0).cast(pl.Float64))).sum().alias('agg_b_when_a_out'),
        ((pl.col('vp_a') == 0).cast(pl.Float64)).sum().alias('n_a_out'),
    ).with_columns(
        ((pl.col('flow_a2b') - pl.col('flow_b2a')).abs() / (pl.col('flow_a2b') + pl.col('flow_b2a') + 1.0)).alias('flow_asym'),
        ((pl.col('flow_a2b') + pl.col('flow_b2a')) / pl.col('shared')).alias('flow_rate'),
        (pl.col('agg_a_when_b_in') / pl.max_horizontal('n_b_in', 1) - pl.col('agg_a_when_b_out') / pl.max_horizontal('n_b_out', 1)).alias('agg_a_contrast'),
        (pl.col('agg_b_when_a_in') / pl.max_horizontal('n_a_in', 1) - pl.col('agg_b_when_a_out') / pl.max_horizontal('n_a_out', 1)).alias('agg_b_contrast'),
    ).select(['a', 'b', 'shared', 'flow_a2b', 'flow_b2a', 'flow_asym', 'flow_rate', 'both_vpip_rate', 'agg_a_contrast', 'agg_b_contrast'])

    # --- merge the two directional response profiles onto the unordered pair
    ab = ra.rename({'responder': 'a', 'aggressor': 'b', **{c: f'ab_{c}' for c in ['n_ev', 'fold_to', 'raise_to', 'call_to', 'd_fold', 'd_raise', 'd_call', 'd_surp', 'surp_mean', 'surp_max']}})
    ba = ra.rename({'responder': 'b', 'aggressor': 'a', **{c: f'ba_{c}' for c in ['n_ev', 'fold_to', 'raise_to', 'call_to', 'd_fold', 'd_raise', 'd_call', 'd_surp', 'surp_mean', 'surp_max']}})
    out = pair.join(ab, on=['a', 'b'], how='left').join(ba, on=['a', 'b'], how='left')
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in out.columns if c.startswith(('ab_', 'ba_'))])
    out = out.with_columns(
        pl.max_horizontal('ab_d_fold', 'ba_d_fold').alias('max_d_fold'),
        (pl.col('ab_d_fold') + pl.col('ba_d_fold')).alias('sum_d_fold'),
        pl.min_horizontal('ab_d_raise', 'ba_d_raise').alias('min_d_raise'),
        (pl.col('ab_d_raise') + pl.col('ba_d_raise')).alias('sum_d_raise'),
        (pl.col('ab_d_surp') + pl.col('ba_d_surp')).alias('sum_d_surp'),
        pl.max_horizontal('ab_surp_max', 'ba_surp_max').alias('max_resp_surp'),
        (pl.col('agg_a_contrast') + pl.col('agg_b_contrast')).alias('sum_agg_contrast'),
    )
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in out.columns if c not in ('a', 'b')])
    out.write_parquet(vpath(f'pairstats_{phase}.parquet'))
    log(f"pairstats {phase}: {out.height:,} pairs, {len(out.columns)} cols")


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph)
