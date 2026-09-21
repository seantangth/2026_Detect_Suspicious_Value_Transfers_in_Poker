"""TPDS feature pipeline v1 (vectorised, polars).
Stages (cached in 1_data/processed/):
  L1  hands_l1.parquet          per hand: time index, session, pot_bb, board, per-street activity stats
  L0  seat_l0.parquet           per (hand, player): seat info, action aggregates, card strength (phevaluator)
  RESP responses.parquet        per (hand, responder, aggressor): fold/call/raise counts when facing aggression
  BASE player_baselines.parquet per (player, phase in {development, evaluation, all}): tendencies
  PAIRS cand_pairs_{phase}.parquet   candidate pairs (dev: shared>=57 with labels; eval: evaluation_pairs.csv)
  L2  l2/{phase}/{table_id}.parquet  per (pair, hand) features, filtered to at-least-one-VPIP
Rules: IDs are join keys only; players.parquet metadata never used; started_at only for ordering/session.
"""
import sys, time, math
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
PROC.mkdir(exist_ok=True, parents=True)
STREETS = ['preflop', 'flop', 'turn', 'river']
RANK_MAP = {r: i + 2 for i, r in enumerate('23456789TJQKA')}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- L1
def build_l1(force=False):
    out = PROC / 'hands_l1.parquet'
    if out.exists() and not force:
        return pl.read_parquet(out)
    h = pl.read_parquet(RAW / 'hands.parquet')
    h = h.sort(['table_id', 'started_at']).with_columns(
        pl.int_range(pl.len()).over('table_id').cast(pl.Int32).alias('t_rank'),
        pl.col('started_at').diff().over('table_id').dt.total_seconds().alias('gap_s'),
        (pl.col('final_pot') / pl.col('big_blind')).cast(pl.Float32).alias('pot_bb'),
        pl.when(pl.col('board_cards').str.len_chars() > 0)
          .then((pl.col('board_cards').str.len_chars() + 1) // 3).otherwise(0).cast(pl.Int8).alias('n_board'),
    ).with_columns(
        ((pl.col('gap_s') > 1800) | pl.col('gap_s').is_null()).cast(pl.Int32).cum_sum().over('table_id').alias('session_id'),
    )
    # per-street activity stats from actions
    a = pl.scan_parquet(RAW / 'actions.parquet').select(['hand_id', 'street', 'action', 'players_active'])
    st = a.group_by(['hand_id', 'street']).agg(
        pl.col('players_active').min().alias('min_active'),
        pl.col('players_active').max().alias('max_active'),
        (pl.col('action') == 'check').all().alias('all_check'),
        pl.len().alias('n_act'),
    ).collect()
    wide = None
    for k, s in enumerate(STREETS):
        w = st.filter(pl.col('street') == s).select(
            'hand_id',
            pl.col('min_active').cast(pl.Int8).alias(f's{k}_min_active'),
            pl.col('max_active').cast(pl.Int8).alias(f's{k}_max_active'),
            pl.col('all_check').cast(pl.Int8).alias(f's{k}_all_check'),
            pl.col('n_act').cast(pl.Int8).alias(f's{k}_n_act'),
        )
        wide = w if wide is None else wide.join(w, on='hand_id', how='full', coalesce=True)
    h = h.join(wide, on='hand_id', how='left')
    h = h.with_columns([pl.col(c).fill_null(0) for c in h.columns if c.startswith('s') and c[1].isdigit()])
    h = h.select(['hand_id', 'table_id', 'phase', 't_rank', 'session_id', 'button_seat', 'big_blind', 'board_cards',
                  'n_board', 'pot_bb', 'players_at_showdown'] + [c for c in h.columns if c[0] == 's' and c[1].isdigit()])
    h.write_parquet(out)
    return h


# ----------------------------------------------------------------------------- strength
PH_CATS = [(10, 9), (166, 8), (322, 7), (1599, 6), (1609, 5), (2467, 4), (3325, 3), (6185, 2), (7462, 1)]


def ph_category(rank):
    for hi, cat in PH_CATS:
        if rank <= hi:
            return cat
    return 1


def compute_strength(l0, l1):
    """phevaluator ranks at flop/turn/river for every (hand, player) whose hand has that many board cards."""
    from phevaluator import evaluate_cards
    j = l0.select(['hand_id', 'player_id', 'hole_card_1', 'hole_card_2']).join(
        l1.select(['hand_id', 'board_cards', 'n_board']), on='hand_id')
    res = {}
    for k, ncards in [(1, 3), (2, 4), (3, 5)]:
        sub = j.filter(pl.col('n_board') >= ncards)
        c1 = sub['hole_card_1'].to_list(); c2 = sub['hole_card_2'].to_list(); bs = sub['board_cards'].to_list()
        t0 = time.time()
        ranks = np.empty(len(c1), dtype=np.int32)
        for i in range(len(c1)):
            b = bs[i].split(' ')[:ncards]
            ranks[i] = evaluate_cards(c1[i], c2[i], *b)
        log(f"strength street {k}: {len(c1):,} evals in {time.time()-t0:.1f}s")
        cats = np.select([ranks <= hi for hi, _ in PH_CATS], [cat for _, cat in PH_CATS], 1).astype(np.int8)
        res[k] = sub.select(['hand_id', 'player_id']).with_columns(
            pl.Series(f'str_pct_{k}', (1.0 - (ranks - 1) / 7461.0).astype(np.float32)),
            pl.Series(f'str_cat_{k}', cats),
        )
    out = l0.select(['hand_id', 'player_id'])
    for k in (1, 2, 3):
        out = out.join(res[k], on=['hand_id', 'player_id'], how='left')
    return out


# ----------------------------------------------------------------------------- L0
def build_l0(l1, force=False):
    out = PROC / 'seat_l0.parquet'
    if out.exists() and not force:
        return pl.read_parquet(out)
    seats = pl.read_parquet(RAW / 'seats.parquet')
    hmeta = l1.select(['hand_id', 'table_id', 'phase', 'big_blind', 'button_seat', 'n_board', 'pot_bb'])
    s = seats.join(hmeta, on='hand_id')
    r1 = pl.col('hole_card_1').str.slice(0, 1).replace_strict(RANK_MAP, return_dtype=pl.Int8)
    r2 = pl.col('hole_card_2').str.slice(0, 1).replace_strict(RANK_MAP, return_dtype=pl.Int8)
    hi = pl.max_horizontal(r1, r2); lo = pl.min_horizontal(r1, r2)
    suited = (pl.col('hole_card_1').str.slice(1, 1) == pl.col('hole_card_2').str.slice(1, 1))
    gap = hi - lo - 1
    base = (pl.when(hi == 14).then(10.0).when(hi == 13).then(8.0).when(hi == 12).then(7.0).when(hi == 11).then(6.0)
              .otherwise(hi.cast(pl.Float32) / 2.0))
    chen = (pl.when(r1 == r2).then(pl.max_horizontal(pl.lit(5.0), base * 2))
              .otherwise(base + pl.when(suited).then(2.0).otherwise(0.0)
                         - pl.when(gap == 0).then(0.0).when(gap == 1).then(1.0).when(gap == 2).then(2.0).when(gap == 3).then(4.0).otherwise(5.0)
                         + pl.when((gap <= 1) & (hi < 12)).then(1.0).otherwise(0.0)))
    s = s.with_columns(
        (pl.col('starting_stack') / pl.col('big_blind')).cast(pl.Float32).alias('stack_bb'),
        (pl.col('total_contribution') / pl.col('big_blind')).cast(pl.Float32).alias('contrib_bb'),
        (pl.col('net_chips') / pl.col('big_blind')).cast(pl.Float32).alias('net_bb'),
        ((pl.col('seat_no') - pl.col('button_seat')) % 6).cast(pl.Int8).alias('rel_pos'),   # 0=BTN 1=SB 2=BB 3=UTG 4=HJ 5=CO
        hi.alias('pf_hi'), lo.alias('pf_lo'), (r1 == r2).cast(pl.Int8).alias('pf_pair'), suited.cast(pl.Int8).alias('pf_suited'),
        chen.cast(pl.Float32).alias('chen'),
        (pl.col('won_share') > 0).cast(pl.Int8).alias('won'),
        pl.col('folded').cast(pl.Int8), pl.col('went_to_showdown').cast(pl.Int8).alias('sd'),
    )
    # ---- action aggregates
    a = pl.read_parquet(RAW / 'actions.parquet').join(l1.select(['hand_id', 'big_blind']), on='hand_id')
    a = a.sort(['hand_id', 'action_no']).with_columns(
        pl.col('street').replace_strict({s_: i for i, s_ in enumerate(STREETS)}, return_dtype=pl.Int8).alias('sidx'),
        (pl.col('action').is_in(['bet', 'raise']) | ((pl.col('action') == 'all_in') & (pl.col('amount') > pl.col('to_call')))).alias('is_agg'),
        ((pl.col('action') == 'call') | ((pl.col('action') == 'all_in') & (pl.col('amount') <= pl.col('to_call')))).alias('is_call'),
        (pl.col('action') == 'check').alias('is_check'), (pl.col('action') == 'fold').alias('is_fold'),
        (pl.col('action') == 'all_in').alias('is_allin'),
        (pl.col('amount') / pl.col('big_blind')).cast(pl.Float32).alias('amount_bb'),
        (pl.col('to_call') / pl.col('big_blind')).cast(pl.Float32).alias('to_call_bb'),
        (pl.col('pot_before') / pl.col('big_blind')).cast(pl.Float32).alias('pot_before_bb'),
        (pl.col('amount') / pl.max_horizontal(pl.col('pot_before'), pl.col('big_blind'))).clip(0, 20).cast(pl.Float32).alias('amt_pot'),
        (pl.col('to_call') > 0).alias('facing'),
    ).with_columns(
        pl.when(pl.col('is_agg')).then(pl.col('player_id')).otherwise(None).alias('_ag'),
    ).with_columns(
        pl.col('_ag').shift(1).over(['hand_id', 'sidx']).forward_fill().over(['hand_id', 'sidx']).alias('last_aggr'),
    )
    a.select(['hand_id', 'action_no', 'sidx', 'player_id', 'is_agg', 'is_call', 'is_fold', 'facing', 'last_aggr',
              'to_call_bb', 'pot_before_bb', 'amt_pot', 'players_active']).write_parquet(PROC / 'action_ctx.parquet')
    aggs = [
        pl.len().cast(pl.Int8).alias('n_act'),
        pl.col('is_agg').sum().cast(pl.Int8).alias('n_agg'),
        pl.col('is_call').sum().cast(pl.Int8).alias('n_call'),
        pl.col('is_check').sum().cast(pl.Int8).alias('n_check'),
        pl.col('is_allin').sum().cast(pl.Int8).alias('n_allin'),
        (pl.col('is_call') | pl.col('is_agg')).any().cast(pl.Int8).alias('vpip'),
        ((pl.col('sidx') == 0) & pl.col('is_agg')).any().cast(pl.Int8).alias('pfr'),
        ((pl.col('sidx') == 0) & pl.col('is_agg') & pl.col('facing') & (pl.col('to_call_bb') > 1.0)).any().cast(pl.Int8).alias('pf_3bet'),
        pl.col('amount_bb').max().cast(pl.Float32).alias('max_amount_bb'),
        pl.col('to_call_bb').max().cast(pl.Float32).alias('max_to_call_bb'),
        pl.col('amt_pot').max().cast(pl.Float32).alias('max_amt_pot'),
        pl.col('sidx').max().alias('last_street'),
        pl.col('facing').sum().cast(pl.Int8).alias('n_facing'),
        (pl.col('facing') & pl.col('is_fold')).sum().cast(pl.Int8).alias('n_fold_facing'),
        (pl.col('facing') & pl.col('is_call')).sum().cast(pl.Int8).alias('n_call_facing'),
        (pl.col('facing') & pl.col('is_agg')).sum().cast(pl.Int8).alias('n_raise_facing'),
        pl.col('sidx').filter(pl.col('is_fold')).first().alias('fold_street'),
        pl.col('to_call_bb').filter(pl.col('is_fold')).first().alias('fold_to_call_bb'),
        pl.col('pot_before_bb').filter(pl.col('is_fold')).first().alias('fold_pot_bb'),
        pl.col('last_aggr').filter(pl.col('is_fold')).first().alias('fold_last_aggr'),
        pl.col('facing').filter(pl.col('is_fold')).first().cast(pl.Int8).alias('fold_facing'),
        pl.col('amt_pot').filter(pl.col('is_agg')).mean().cast(pl.Float32).alias('mean_agg_amt_pot'),
        ((pl.col('sidx') > 0) & pl.col('is_check')).sum().cast(pl.Int8).alias('post_check'),
        (pl.col('sidx') > 0).sum().cast(pl.Int8).alias('post_act'),
    ]
    for k in range(4):
        aggs += [
            (pl.col('sidx') == k).sum().cast(pl.Int8).alias(f'acted_{k}'),
            ((pl.col('sidx') == k) & pl.col('is_agg')).sum().cast(pl.Int8).alias(f'agg_{k}'),
            ((pl.col('sidx') == k) & pl.col('is_call')).sum().cast(pl.Int8).alias(f'call_{k}'),
            ((pl.col('sidx') == k) & pl.col('is_check')).sum().cast(pl.Int8).alias(f'check_{k}'),
        ]
    ag = a.group_by(['hand_id', 'player_id']).agg(aggs)
    s = s.join(ag, on=['hand_id', 'player_id'], how='left')
    fill0 = ['n_act', 'n_agg', 'n_call', 'n_check', 'n_allin', 'vpip', 'pfr', 'pf_3bet', 'n_facing', 'n_fold_facing',
             'n_call_facing', 'n_raise_facing', 'post_check', 'post_act', 'max_amount_bb', 'max_to_call_bb', 'max_amt_pot'] + \
            [f'{p}_{k}' for k in range(4) for p in ('acted', 'agg', 'call', 'check')]
    s = s.with_columns([pl.col(c).fill_null(0) for c in fill0]).with_columns(
        pl.col('last_street').fill_null(-1).cast(pl.Int8), pl.col('fold_facing').fill_null(0).cast(pl.Int8))
    # ---- strength
    strength = compute_strength(s, l1)
    s = s.join(strength, on=['hand_id', 'player_id'], how='left')
    s = s.select(['hand_id', 'player_id', 'table_id', 'phase', 'seat_no', 'rel_pos', 'stack_bb', 'contrib_bb', 'net_bb', 'folded', 'sd', 'won',
                  'hole_card_1', 'hole_card_2', 'pf_hi', 'pf_lo', 'pf_pair', 'pf_suited', 'chen', 'n_board', 'pot_bb',
                  'n_act', 'n_agg', 'n_call', 'n_check', 'n_allin', 'vpip', 'pfr', 'pf_3bet', 'max_amount_bb', 'max_to_call_bb', 'max_amt_pot',
                  'last_street', 'n_facing', 'n_fold_facing', 'n_call_facing', 'n_raise_facing', 'fold_street', 'fold_to_call_bb', 'fold_pot_bb',
                  'fold_last_aggr', 'fold_facing', 'mean_agg_amt_pot', 'post_check', 'post_act'] +
                 [f'{p}_{k}' for k in range(4) for p in ('acted', 'agg', 'call', 'check')] +
                 ['str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3'])
    s.write_parquet(out)
    return s


# ----------------------------------------------------------------------------- responses (directional)
def build_responses(force=False):
    out = PROC / 'responses.parquet'
    if out.exists() and not force:
        return pl.read_parquet(out)
    a = pl.scan_parquet(PROC / 'action_ctx.parquet').filter(pl.col('facing') & pl.col('last_aggr').is_not_null())
    r = a.group_by(['hand_id', 'player_id', 'last_aggr']).agg(
        pl.col('is_fold').sum().cast(pl.Int8).alias('r_fold'),
        pl.col('is_call').sum().cast(pl.Int8).alias('r_call'),
        pl.col('is_agg').sum().cast(pl.Int8).alias('r_raise'),
        pl.col('to_call_bb').max().cast(pl.Float32).alias('r_max_to_call'),
    ).rename({'player_id': 'responder', 'last_aggr': 'aggressor'}).collect()
    r.write_parquet(out)
    return r


# ----------------------------------------------------------------------------- baselines
def build_baselines(l0, force=False):
    out = PROC / 'player_baselines.parquet'
    if out.exists() and not force:
        return pl.read_parquet(out)
    def agg(df):
        return df.group_by(['player_id']).agg(
            pl.len().alias('b_hands'),
            pl.col('vpip').mean().alias('b_vpip'), pl.col('pfr').mean().alias('b_pfr'),
            (pl.col('n_agg').sum() / pl.max_horizontal(pl.col('n_act').sum(), 1)).alias('b_agg_rate'),
            (pl.col('n_fold_facing').sum() / pl.max_horizontal(pl.col('n_facing').sum(), 1)).alias('b_fold_facing'),
            (pl.col('n_call_facing').sum() / pl.max_horizontal(pl.col('n_facing').sum(), 1)).alias('b_call_facing'),
            (pl.col('n_raise_facing').sum() / pl.max_horizontal(pl.col('n_facing').sum(), 1)).alias('b_raise_facing'),
            (pl.col('sd').sum() / pl.max_horizontal(pl.col('vpip').sum(), 1)).alias('b_sd_rate'),
            (pl.col('post_check').sum() / pl.max_horizontal(pl.col('post_act').sum(), 1)).alias('b_post_check'),
            pl.col('contrib_bb').mean().alias('b_contrib'), pl.col('net_bb').mean().alias('b_net'),
            pl.col('n_allin').mean().alias('b_allin'),
            (pl.col('last_street') >= 1).mean().alias('b_saw_flop'),
        )
    parts = []
    for ph in ['development', 'evaluation']:
        parts.append(agg(l0.filter(pl.col('phase') == ph)).with_columns(pl.lit(ph).alias('bphase')))
    parts.append(agg(l0).with_columns(pl.lit('all').alias('bphase')))
    b = pl.concat(parts)
    b = b.with_columns([pl.col(c).cast(pl.Float32) for c in b.columns if c.startswith('b_') and c != 'b_hands'])
    b.write_parquet(out)
    return b


# ----------------------------------------------------------------------------- candidate pairs
def build_cand_pairs(l0, force=False):
    outd, oute = PROC / 'cand_pairs_development.parquet', PROC / 'cand_pairs_evaluation.parquet'
    if outd.exists() and oute.exists() and not force:
        return pl.read_parquet(outd), pl.read_parquet(oute)
    lab = pl.read_csv(RAW / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('a'), pl.max_horizontal('player_1', 'player_2').alias('b'))
    # dev: all pairs with shared dev hands >= 57
    ph = l0.filter(pl.col('phase') == 'development').select(['hand_id', 'table_id', 'player_id'])
    pairs = pair_rows(ph).group_by(['a', 'b']).agg(pl.len().cast(pl.Int32).alias('shared'), pl.col('table_id').first())
    dev = pairs.filter(pl.col('shared') >= 57).join(lab.select(['a', 'b', 'pair_id', 'label', 'behavior_family']), on=['a', 'b'], how='left')
    dev = dev.with_columns(
        pl.when(pl.col('pair_id').is_null()).then(pl.concat_str([pl.lit('U_'), pl.col('a'), pl.lit('_'), pl.col('b')])).otherwise(pl.col('pair_id')).alias('pair_id'),
        pl.col('label').fill_null(-1).cast(pl.Int8),   # -1 unknown, 0 confirmed non-target, 1 positive
        pl.col('behavior_family').fill_null('unknown'),
        pl.lit('development').alias('phase'),
    )
    ep = pl.read_csv(RAW / 'evaluation_pairs.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('a'), pl.max_horizontal('player_1', 'player_2').alias('b'))
    p2t = l0.select(['player_id', 'table_id']).unique()
    ev = ep.join(p2t, left_on='a', right_on='player_id').select(
        'pair_id', 'a', 'b', pl.col('shared_hands').cast(pl.Int32).alias('shared'), 'table_id',
        pl.lit(-1).cast(pl.Int8).alias('label'), pl.lit('unknown').alias('behavior_family'), pl.lit('evaluation').alias('phase'))
    dev = dev.select(ev.columns)
    dev.write_parquet(outd); ev.write_parquet(oute)
    return dev, ev


def pair_rows(seat_min):
    """(hand_id, table_id, player_id) -> one row per unordered pair per hand (a<b)."""
    g = seat_min.sort(['hand_id', 'player_id']).group_by('hand_id').agg(pl.col('player_id').alias('pl'), pl.col('table_id').first())
    frames = []
    for i in range(6):
        for j in range(i + 1, 6):
            frames.append(g.select('hand_id', 'table_id', pl.col('pl').list.get(i).alias('a'), pl.col('pl').list.get(j).alias('b')))
    return pl.concat(frames)


# ----------------------------------------------------------------------------- L2 per table
A_COLS = ['rel_pos', 'stack_bb', 'contrib_bb', 'net_bb', 'folded', 'sd', 'won', 'pf_hi', 'pf_lo', 'pf_pair', 'pf_suited', 'chen',
          'n_act', 'n_agg', 'n_call', 'n_check', 'n_allin', 'vpip', 'pfr', 'pf_3bet', 'max_amount_bb', 'max_to_call_bb', 'max_amt_pot',
          'last_street', 'n_facing', 'n_fold_facing', 'n_call_facing', 'n_raise_facing', 'fold_street', 'fold_to_call_bb', 'fold_pot_bb',
          'fold_last_aggr', 'fold_facing', 'mean_agg_amt_pot', 'post_check', 'post_act'] + \
         [f'{p}_{k}' for k in range(4) for p in ('acted', 'agg', 'call', 'check')] + \
         ['str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3']
B_COLS_BASE = ['b_vpip', 'b_pfr', 'b_agg_rate', 'b_fold_facing', 'b_call_facing', 'b_raise_facing', 'b_sd_rate', 'b_post_check', 'b_contrib', 'b_net', 'b_allin', 'b_saw_flop']


def build_l2_table(table_id, phase, l0_t, l1_t, resp_t, base_other, cand_t):
    """cand_t: candidate pairs of this table & phase (a<b). l0_t/l1_t/resp_t restricted to the table+phase."""
    seat_min = l0_t.select(['hand_id', 'table_id', 'player_id'])
    ph = pair_rows(seat_min).join(cand_t.select(['a', 'b', 'pair_id']), on=['a', 'b'], how='inner')
    A = l0_t.select(['hand_id', pl.col('player_id').alias('a')] + [pl.col(c).alias(f'A_{c}') for c in A_COLS])
    B = l0_t.select(['hand_id', pl.col('player_id').alias('b')] + [pl.col(c).alias(f'B_{c}') for c in A_COLS])
    d = ph.join(A, on=['hand_id', 'a']).join(B, on=['hand_id', 'b'])
    d = d.join(l1_t.select(['hand_id', 't_rank', 'session_id', 'n_board', 'pot_bb', 'players_at_showdown'] +
                           [f's{k}_{m}' for k in range(4) for m in ('min_active', 'max_active', 'all_check')]), on='hand_id')
    # responses: A responding to B, B responding to A, and totals per responder
    rab = resp_t.select(['hand_id', pl.col('responder').alias('a'), pl.col('aggressor').alias('b'),
                         pl.col('r_fold').alias('rAB_fold'), pl.col('r_call').alias('rAB_call'), pl.col('r_raise').alias('rAB_raise'), pl.col('r_max_to_call').alias('rAB_max_to_call')])
    rba = resp_t.select(['hand_id', pl.col('responder').alias('b'), pl.col('aggressor').alias('a'),
                         pl.col('r_fold').alias('rBA_fold'), pl.col('r_call').alias('rBA_call'), pl.col('r_raise').alias('rBA_raise'), pl.col('r_max_to_call').alias('rBA_max_to_call')])
    tot = resp_t.group_by(['hand_id', 'responder']).agg(pl.col('r_fold').sum().alias('rt_fold'), pl.col('r_call').sum().alias('rt_call'), pl.col('r_raise').sum().alias('rt_raise'))
    d = d.join(rab, on=['hand_id', 'a', 'b'], how='left').join(rba, on=['hand_id', 'a', 'b'], how='left')
    d = d.join(tot.rename({'responder': 'a', 'rt_fold': 'rtA_fold', 'rt_call': 'rtA_call', 'rt_raise': 'rtA_raise'}), on=['hand_id', 'a'], how='left')
    d = d.join(tot.rename({'responder': 'b', 'rt_fold': 'rtB_fold', 'rt_call': 'rtB_call', 'rt_raise': 'rtB_raise'}), on=['hand_id', 'b'], how='left')
    rcols = [c for c in d.columns if c.startswith('rAB_') or c.startswith('rBA_') or c.startswith('rtA_') or c.startswith('rtB_')]
    d = d.with_columns([pl.col(c).fill_null(0) for c in rcols])
    # --- outsiders' responses to the pair's aggression (isolation success) and the pair's responses to outsiders
    om = ph.select(['hand_id', 'a', 'b']).join(resp_t, on='hand_id')
    out_to_pair = (om.filter(((pl.col('aggressor') == pl.col('a')) | (pl.col('aggressor') == pl.col('b'))) &
                             (pl.col('responder') != pl.col('a')) & (pl.col('responder') != pl.col('b')))
                     .group_by(['hand_id', 'a', 'b']).agg(
                         pl.col('r_fold').sum().cast(pl.Int8).alias('out_fold_to_pair'),
                         pl.col('r_call').sum().cast(pl.Int8).alias('out_call_to_pair'),
                         pl.col('r_raise').sum().cast(pl.Int8).alias('out_raise_to_pair'),
                         pl.col('responder').n_unique().cast(pl.Int8).alias('out_responders')))
    pair_to_out = (om.filter(((pl.col('responder') == pl.col('a')) | (pl.col('responder') == pl.col('b'))) &
                             (pl.col('aggressor') != pl.col('a')) & (pl.col('aggressor') != pl.col('b')))
                     .group_by(['hand_id', 'a', 'b']).agg(
                         pl.col('r_fold').sum().cast(pl.Int8).alias('pair_fold_to_out'),
                         pl.col('r_call').sum().cast(pl.Int8).alias('pair_call_to_out'),
                         pl.col('r_raise').sum().cast(pl.Int8).alias('pair_raise_to_out')))
    d = d.join(out_to_pair, on=['hand_id', 'a', 'b'], how='left').join(pair_to_out, on=['hand_id', 'a', 'b'], how='left')
    ocols = ['out_fold_to_pair', 'out_call_to_pair', 'out_raise_to_pair', 'out_responders', 'pair_fold_to_out', 'pair_call_to_out', 'pair_raise_to_out']
    d = d.with_columns([pl.col(c).fill_null(0) for c in ocols])
    # baselines from the other phase
    bo = base_other.select(['player_id'] + B_COLS_BASE)
    d = d.join(bo.rename({c: f'A{c}' for c in B_COLS_BASE}), left_on='a', right_on='player_id', how='left')
    d = d.join(bo.rename({c: f'B{c}' for c in B_COLS_BASE}), left_on='b', right_on='player_id', how='left')
    # ---- interaction features
    hu = [((pl.col(f'A_acted_{k}') > 0) & (pl.col(f'B_acted_{k}') > 0) & (pl.col(f's{k}_min_active') == 2)) for k in range(4)]
    war = [((pl.col(f'A_agg_{k}') > 0) & (pl.col(f'B_agg_{k}') > 0) & (pl.col(f's{k}_max_active') > 2)) for k in range(4)]
    d = d.with_columns(
        (pl.col('A_fold_last_aggr') == pl.col('b')).fill_null(False).cast(pl.Int8).alias('A_fold_to_B'),
        (pl.col('B_fold_last_aggr') == pl.col('a')).fill_null(False).cast(pl.Int8).alias('B_fold_to_A'),
        sum(h.cast(pl.Int8) for h in hu).alias('hu_streets'),
        sum((h & (pl.col(f's{k}_all_check') == 1)).cast(pl.Int8) for k, h in enumerate(hu)).alias('both_check_hu'),
        sum(w.cast(pl.Int8) for w in war).alias('war_streets'),
        ((pl.col('A_vpip') == 1) & (pl.col('B_vpip') == 1)).cast(pl.Int8).alias('both_vpip'),
        ((pl.col('A_last_street') >= 1) & (pl.col('B_last_street') >= 1)).cast(pl.Int8).alias('both_flop'),
        ((pl.col('A_sd') == 1) & (pl.col('B_sd') == 1)).cast(pl.Int8).alias('both_sd'),
        pl.min_horizontal((-pl.col('A_net_bb')).clip(lower_bound=0), pl.col('B_net_bb').clip(lower_bound=0)).alias('tr_A_to_B'),
        pl.min_horizontal((-pl.col('B_net_bb')).clip(lower_bound=0), pl.col('A_net_bb').clip(lower_bound=0)).alias('tr_B_to_A'),
        (pl.col('A_net_bb') - pl.col('B_net_bb')).abs().alias('net_gap'),
        (pl.col('A_contrib_bb') + pl.col('B_contrib_bb')).alias('pair_contrib'),
        (pl.col('A_contrib_bb') - pl.col('B_contrib_bb')).abs().alias('contrib_gap'),
        (pl.col('A_chen') - pl.col('B_chen')).alias('chen_diff'),
        pl.max_horizontal('A_chen', 'B_chen').alias('chen_max'), pl.min_horizontal('A_chen', 'B_chen').alias('chen_min'),
        (pl.col('rAB_fold') + pl.col('rAB_call') + pl.col('rAB_raise')).alias('rAB_n'),
        (pl.col('rBA_fold') + pl.col('rBA_call') + pl.col('rBA_raise')).alias('rBA_n'),
    )
    d = d.with_columns(
        (pl.col('out_fold_to_pair') - pl.col('pair_fold_to_out')).alias('iso_balance'),
        (pl.col('out_fold_to_pair') * (pl.col('war_streets') > 0).cast(pl.Int8)).alias('war_then_out_folds'),
        ((pl.col('s0_max_active') - pl.col('s1_max_active')).clip(lower_bound=0)).alias('pf_dropouts'),
        ((pl.col('A_fold_to_B') == 1) & (pl.col('A_fold_facing') == 1) & (pl.col('war_streets') > 0)).cast(pl.Int8).alias('A_fold_to_B_after_war'),
        ((pl.col('B_fold_to_A') == 1) & (pl.col('B_fold_facing') == 1) & (pl.col('war_streets') > 0)).cast(pl.Int8).alias('B_fold_to_A_after_war'),
        ((pl.col('A_fold_facing') == 1) & (pl.col('A_fold_to_B') == 0)).cast(pl.Int8).alias('A_fold_to_other'),
        ((pl.col('B_fold_facing') == 1) & (pl.col('B_fold_to_A') == 0)).cast(pl.Int8).alias('B_fold_to_other'),
        pl.max_horizontal('tr_A_to_B', 'tr_B_to_A').alias('tr_any'),
        (pl.col('tr_A_to_B') - pl.col('tr_B_to_A')).alias('tr_dir'),
        # strength margin at A's fold street (A - B); preflop -> chen diff scaled
        pl.when(pl.col('A_fold_street') == 1).then(pl.col('A_str_pct_1') - pl.col('B_str_pct_1'))
          .when(pl.col('A_fold_street') == 2).then(pl.col('A_str_pct_2') - pl.col('B_str_pct_2'))
          .when(pl.col('A_fold_street') == 3).then(pl.col('A_str_pct_3') - pl.col('B_str_pct_3'))
          .when(pl.col('A_fold_street') == 0).then((pl.col('A_chen') - pl.col('B_chen')) / 20.0).otherwise(None).cast(pl.Float32).alias('A_fold_margin'),
        pl.when(pl.col('B_fold_street') == 1).then(pl.col('B_str_pct_1') - pl.col('A_str_pct_1'))
          .when(pl.col('B_fold_street') == 2).then(pl.col('B_str_pct_2') - pl.col('A_str_pct_2'))
          .when(pl.col('B_fold_street') == 3).then(pl.col('B_str_pct_3') - pl.col('A_str_pct_3'))
          .when(pl.col('B_fold_street') == 0).then((pl.col('B_chen') - pl.col('A_chen')) / 20.0).otherwise(None).cast(pl.Float32).alias('B_fold_margin'),
        (pl.col('A_str_pct_3') - pl.col('B_str_pct_3')).cast(pl.Float32).alias('river_margin_AB'),
        # deviations from own baseline (other phase)
        (pl.col('A_vpip') - pl.col('Ab_vpip')).cast(pl.Float32).alias('A_dev_vpip'),
        (pl.col('B_vpip') - pl.col('Bb_vpip')).cast(pl.Float32).alias('B_dev_vpip'),
        (pl.col('A_n_agg') / pl.max_horizontal(pl.col('A_n_act'), 1) - pl.col('Ab_agg_rate')).cast(pl.Float32).alias('A_dev_agg'),
        (pl.col('B_n_agg') / pl.max_horizontal(pl.col('B_n_act'), 1) - pl.col('Bb_agg_rate')).cast(pl.Float32).alias('B_dev_agg'),
        (pl.col('A_fold_facing') - pl.col('Ab_fold_facing')).cast(pl.Float32).alias('A_dev_fold_facing'),
        (pl.col('B_fold_facing') - pl.col('Bb_fold_facing')).cast(pl.Float32).alias('B_dev_fold_facing'),
        (pl.col('A_post_check') / pl.max_horizontal(pl.col('A_post_act'), 1) - pl.col('Ab_post_check')).cast(pl.Float32).alias('A_dev_post_check'),
        (pl.col('B_post_check') / pl.max_horizontal(pl.col('B_post_act'), 1) - pl.col('Bb_post_check')).cast(pl.Float32).alias('B_dev_post_check'),
    )
    d = d.with_columns(
        ((pl.col('A_fold_to_B') == 1) & (pl.col('A_fold_margin') > 0)).cast(pl.Int8).alias('A_folded_better_to_B'),
        ((pl.col('B_fold_to_A') == 1) & (pl.col('B_fold_margin') > 0)).cast(pl.Int8).alias('B_folded_better_to_A'),
        # pot odds the folder was getting, and the EV given up by folding (the literal transferred value)
        (pl.col('A_fold_to_call_bb') / (pl.col('A_fold_pot_bb') + pl.col('A_fold_to_call_bb') + 1e-3)).cast(pl.Float32).alias('A_fold_pot_odds'),
        (pl.col('B_fold_to_call_bb') / (pl.col('B_fold_pot_bb') + pl.col('B_fold_to_call_bb') + 1e-3)).cast(pl.Float32).alias('B_fold_pot_odds'),
        (pl.col('A_fold_margin') * pl.col('A_fold_pot_bb')).cast(pl.Float32).alias('A_fold_ev_loss'),
        (pl.col('B_fold_margin') * pl.col('B_fold_pot_bb')).cast(pl.Float32).alias('B_fold_ev_loss'),
        # aggression-then-fold in the same hand (the isolation squeeze: build the pot, then get out)
        ((pl.col('A_n_agg') > 0) & (pl.col('A_folded') == 1)).cast(pl.Int8).alias('A_agg_then_fold'),
        ((pl.col('B_n_agg') > 0) & (pl.col('B_folded') == 1)).cast(pl.Int8).alias('B_agg_then_fold'),
        # showdown with the weaker hand while the partner is also at showdown
        ((pl.col('A_sd') == 1) & (pl.col('B_sd') == 1) & (pl.col('river_margin_AB') < 0)).cast(pl.Int8).alias('A_sd_worse'),
        ((pl.col('A_sd') == 1) & (pl.col('B_sd') == 1) & (pl.col('river_margin_AB') > 0)).cast(pl.Int8).alias('B_sd_worse'),
        # passivity with a made-hand advantage: checked/called down on a street where the partner was the only opponent
        ((pl.col('hu_streets') > 0) & (pl.col('A_n_agg') == 0) & (pl.col('A_last_street') >= 1)).cast(pl.Int8).alias('A_passive_hu'),
        ((pl.col('hu_streets') > 0) & (pl.col('B_n_agg') == 0) & (pl.col('B_last_street') >= 1)).cast(pl.Int8).alias('B_passive_hu'),
    )
    d = d.with_columns(
        pl.max_horizontal('A_fold_ev_loss', 'B_fold_ev_loss').alias('fold_ev_loss_max'),
        ((pl.col('A_agg_then_fold') == 1) & (pl.col('B_folded') == 0)).cast(pl.Int8).alias('A_built_pot_for_B'),
        ((pl.col('B_agg_then_fold') == 1) & (pl.col('A_folded') == 0)).cast(pl.Int8).alias('B_built_pot_for_A'),
    )
    d = d.filter((pl.col('A_vpip') == 1) | (pl.col('B_vpip') == 1))
    d = d.drop(['A_fold_last_aggr', 'B_fold_last_aggr'])
    return d


def build_l2(phase, l0, l1, resp, base, cand, tables=None, force=False):
    outdir = PROC / 'l2' / phase
    outdir.mkdir(parents=True, exist_ok=True)
    other = 'evaluation' if phase == 'development' else 'development'
    base_other = base.filter(pl.col('bphase') == other)
    l0p = l0.filter(pl.col('phase') == phase)
    l1p = l1.filter(pl.col('phase') == phase)
    hands_phase = l1p.select('hand_id')
    respp = resp.join(hands_phase, on='hand_id')
    tabs = sorted(cand['table_id'].unique().to_list()) if tables is None else tables
    t0 = time.time()
    for i, t in enumerate(tabs):
        out = outdir / f'{t}.parquet'
        if out.exists() and not force:
            continue
        l0_t = l0p.filter(pl.col('table_id') == t)
        hid = l0_t.select('hand_id').unique()
        l1_t = l1p.filter(pl.col('table_id') == t)
        resp_t = respp.join(hid, on='hand_id')
        cand_t = cand.filter(pl.col('table_id') == t)
        d = build_l2_table(t, phase, l0_t, l1_t, resp_t, base_other, cand_t)
        d.write_parquet(out)
        if i % 50 == 0:
            log(f"L2 {phase}: {i+1}/{len(tabs)} tables, last rows={d.height:,}, {time.time()-t0:.0f}s")
    log(f"L2 {phase} done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    log('L1'); l1 = build_l1()
    log('L0'); l0 = build_l0(l1)
    log('responses'); resp = build_responses()
    log('baselines'); base = build_baselines(l0)
    log('cand pairs'); dev, ev = build_cand_pairs(l0)
    log(f"dev cand {dev.height:,} (pos {int((dev['label']==1).sum())}, neg {int((dev['label']==0).sum())}); eval cand {ev.height:,}")
    if stage in ('all', 'l2'):
        tables = None
        if len(sys.argv) > 2:
            tables = sorted(dev['table_id'].unique().to_list())[:int(sys.argv[2])]
        build_l2('development', l0, l1, resp, base, dev, tables=tables)
        build_l2('evaluation', l0, l1, resp, base, ev, tables=tables)
    log('done')
