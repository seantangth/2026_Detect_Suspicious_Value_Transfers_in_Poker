"""Transparent per-pair context statistics for the case reviews, computed directly from the raw tables (evaluation period only).
For every hand both players were dealt:
  * folds: each fold facing a bet/raise, attributed to the last aggressor; the folder's heads-up equity against that aggressor at the moment
    of the fold (exact enumeration of the remaining board; pre-flop: 2,000-sample Monte Carlo with a fixed seed);
  * net chips of each player; heads-up streets where both players only checked; pre-flop raise wars (3-bet or more) involving the pair.
Usage: case_stats.py <pair_id> [...]  -> prints one block per pair."""
import sys, itertools, random
from pathlib import Path
import polars as pl
from phevaluator import evaluate_cards

RAW = Path(__file__).resolve().parents[1] / '1_data/raw/detect-suspicious-value-transfers-in-poker'
RANKS = '23456789TJQKA'; SUITS = 'cdhs'; DECK = [r + s for r in RANKS for s in SUITS]
AGG = {'bet', 'raise', 'all_in'}


def hu_equity(h1, h2, board, n_mc=2000, seed=0):
    used = set(board) | set(h1) | set(h2); rest = [c for c in DECK if c not in used]; need = 5 - len(board)
    if need == 0:
        a, b = evaluate_cards(*(board + h1)), evaluate_cards(*(board + h2)); return 1.0 if a < b else 0.5 if a == b else 0.0
    rng = random.Random(seed)
    combos = itertools.combinations(rest, need) if need <= 2 else (rng.sample(rest, need) for _ in range(n_mc))
    w = t = 0.0
    for extra in combos:
        full = board + list(extra); a, b = evaluate_cards(*(full + h1)), evaluate_cards(*(full + h2))
        w += 1.0 if a < b else 0.5 if a == b else 0.0; t += 1
    return w / t


def stats(pair_id):
    ep = pl.read_csv(RAW / 'evaluation_pairs.csv').filter(pl.col('pair_id') == pair_id).row(0, named=True)
    A, B = ep['player_1'], ep['player_2']
    s = pl.scan_parquet(RAW / 'seats.parquet').filter(pl.col('player_id').is_in([A, B])).collect()
    hs = s.group_by('hand_id').len().filter(pl.col('len') == 2)['hand_id'].to_list()
    h = pl.scan_parquet(RAW / 'hands.parquet').filter(pl.col('hand_id').is_in(hs) & (pl.col('phase') == 'evaluation')).collect()
    hids = h['hand_id'].to_list()
    seats = pl.scan_parquet(RAW / 'seats.parquet').filter(pl.col('hand_id').is_in(hids)).collect()
    acts = pl.scan_parquet(RAW / 'actions.parquet').filter(pl.col('hand_id').is_in(hids)).collect().sort(['hand_id', 'action_no'])
    board_of = dict(zip(h['hand_id'], h['board_cards'].fill_null('')))
    nb = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
    hole = {(r['hand_id'], r['player_id']): [r['hole_card_1'], r['hole_card_2']] for r in seats.iter_rows(named=True)}
    net = {(r['hand_id'], r['player_id']): r['net_chips'] for r in seats.iter_rows(named=True)}
    role = {A: 'A', B: 'B'}
    folds = {('A', 'partner'): [0, 0], ('A', 'other'): [0, 0], ('B', 'partner'): [0, 0], ('B', 'other'): [0, 0]}
    wars = {'A': 0, 'B': 0}; both_check_hu = 0; hu_hands = 0; bets_into = {'A': 0, 'B': 0}
    for hid, g in acts.group_by('hand_id', maintain_order=True):
        hid = hid[0]; board = board_of[hid].split() if board_of[hid] else []
        last_aggr = None; live = set(seats.filter(pl.col('hand_id') == hid)['player_id'].to_list()); street = None
        pre_raises = 0; raisers = []; street_checks = {}
        for r in g.iter_rows(named=True):
            if r['street'] != street:
                street = r['street']
                if street != 'preflop':
                    last_aggr = None
            p = r['player_id']
            if r['action'] == 'fold' and p in role and last_aggr is not None:
                partner = B if p == A else A
                kind = 'partner' if last_aggr == partner else 'other'
                eq = hu_equity(hole[(hid, p)], hole[(hid, last_aggr)], board[:nb[street]])
                folds[(role[p], kind)][0] += 1; folds[(role[p], kind)][1] += eq > 0.5
            if r['action'] in AGG:
                if street == 'preflop':
                    pre_raises += 1; raisers.append(p)
                if p in role and (B if p == A else A) in live:
                    bets_into[role[p]] += 1
                last_aggr = p
            if r['action'] == 'fold':
                live.discard(p)
            if street != 'preflop' and live == {A, B}:
                street_checks.setdefault(street, set()).add(r['action'])
        if pre_raises >= 3:
            for x in (A, B):
                if x in raisers: wars[role[x]] += 1
        hu = [st for st, acts_ in street_checks.items()]
        if hu:
            hu_hands += 1; both_check_hu += sum(1 for st, a in street_checks.items() if a == {'check'})
    netA = sum(net[(x, A)] for x in hids); netB = sum(net[(x, B)] for x in hids)
    print(f'pair {pair_id}: A={A} B={B} | shared evaluation hands {len(hids)}')
    for who in ('A', 'B'):
        fp, fo = folds[(who, 'partner')], folds[(who, 'other')]
        print(f'  {who}: folds facing the partner\'s bet/raise {fp[0]} (holding the better hand vs the partner: {fp[1]}) | '
              f'folds facing another player\'s bet/raise {fo[0]} (better hand vs that bettor: {fo[1]}) | bets/raises while the partner was still in: {bets_into[who]} | '
              f'pre-flop raise wars (3+ raises) it raised in: {wars[who]}')
    print(f'  net chips over the shared hands: A {netA:+d}, B {netB:+d} | hands that became heads-up between them: {hu_hands}, of which streets checked through by both: {both_check_hu}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        stats(p)
