"""Render hands in a human-readable form: all hole cards, board, every action, and the equity of each player still in at the start of
every street (exact enumeration after the flop, 400-sample Monte Carlo before it). Memory: lazy scans filtered by hand_id / player_id."""
import itertools, random, sys
import numpy as np, polars as pl
from phevaluator import evaluate_cards
from pathlib import Path
R = str(Path(__file__).resolve().parents[1] / '1_data/raw/detect-suspicious-value-transfers-in-poker') + '/'; OUT = str(Path(__file__).resolve().parent) + '/'
rng = random.Random(20260917)
RANKS = '23456789TJQKA'; SUITS = 'cdhs'; DECK = [r + s for r in RANKS for s in SUITS]

def equities(holes, board, n_mc=400):
    """holes: {name: [c1,c2]} of players still in; board: list of known board cards. exact for flop/turn/river, MC preflop."""
    names = list(holes); used = set(board) | {c for h in holes.values() for c in h}
    rest = [c for c in DECK if c not in used]; need = 5 - len(board)
    wins = dict.fromkeys(names, 0.0); tot = 0
    combos = itertools.combinations(rest, need) if need <= 2 else (rng.sample(rest, need) for _ in range(n_mc))
    for extra in combos:
        full = board + list(extra)
        scores = {n: evaluate_cards(*(full + holes[n])) for n in names}
        best = min(scores.values()); ws = [n for n in names if scores[n] == best]
        for n in ws: wins[n] += 1.0 / len(ws)
        tot += 1
    return {n: wins[n] / max(tot, 1) for n in names}

def load(hand_ids):
    hs = list(hand_ids)
    h = pl.scan_parquet(R + 'hands.parquet').filter(pl.col('hand_id').is_in(hs)).collect()
    s = pl.scan_parquet(R + 'seats.parquet').filter(pl.col('hand_id').is_in(hs)).collect()
    a = pl.scan_parquet(R + 'actions.parquet').filter(pl.col('hand_id').is_in(hs)).collect()
    return h, s, a

def render(hid, h, s, a, A, B, tag, pos):
    hr = h.filter(pl.col('hand_id') == hid).row(0, named=True)
    sr = s.filter(pl.col('hand_id') == hid).sort('seat_no'); ar = a.filter(pl.col('hand_id') == hid).sort('action_no')
    board = hr['board_cards'].split() if hr['board_cards'] else []
    role = {A: 'A', B: 'B'}
    name = {}
    for r in sr.iter_rows(named=True):
        name[r['player_id']] = role.get(r['player_id'], f"o{r['seat_no']}")
    L = [f"#### {hid}  [{tag}]  {pos}", f"table {hr['table_id']} | {hr['phase']} | {hr['started_at']} | button seat {hr['button_seat']} | blinds {hr['small_blind']}/{hr['big_blind']} | board: {' '.join(board) or '-'} | final_pot {hr['final_pot']} | at showdown {hr['players_at_showdown']}"]
    for r in sr.iter_rows(named=True):
        L.append(f"  seat{r['seat_no']} {name[r['player_id']]:>3} {r['hole_card_1']}{r['hole_card_2']} stack {r['starting_stack']:>4} | contrib {r['total_contribution']:>4} net {r['net_chips']:>+5} | folded {int(r['folded'])} sd {int(r['went_to_showdown'])} won {r['won_share']:.2f}")
    holes = {name[r['player_id']]: [r['hole_card_1'], r['hole_card_2']] for r in sr.iter_rows(named=True)}
    folded = set(); nb = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}; cur = None
    for r in ar.iter_rows(named=True):
        if r['street'] != cur:
            cur = r['street']; live = {n: holes[n] for n in holes if n not in folded}
            if len(live) >= 2 and cur in nb and len(board) >= nb[cur]:
                eq = equities(live, board[:nb[cur]])
                L.append(f"  -- {cur} (board {' '.join(board[:nb[cur]]) or '-'}) equity: " + ', '.join(f"{n}={eq[n]:.2f}" for n in live))
            else:
                L.append(f"  -- {cur}")
        n = name.get(r['player_id'], r['player_id'][:6])
        L.append(f"     {n:>3} {r['action']:<6} amt {r['amount']:>4} to {r['amount_to']:>4} | pot_before {r['pot_before']:>4} to_call {r['to_call']:>4} stack {r['stack_before']:>4} active {r['players_active']}")
        if r['action'] == 'fold': folded.add(n)
    return '\n'.join(L)

def shared_hands(pairs, phase):
    """pairs: list of (pair_id, A, B) -> dict pair_id -> sorted list of shared hand_ids in phase (chronological)."""
    players = list({p for _, x, y in pairs for p in (x, y)})
    s = pl.scan_parquet(R + 'seats.parquet').filter(pl.col('player_id').is_in(players)).select('hand_id', 'player_id').collect()
    h = pl.scan_parquet(R + 'hands.parquet').filter(pl.col('phase') == phase).select('hand_id', 'started_at').collect()
    s = s.join(h, on='hand_id')
    out = {}
    for pid, x, y in pairs:
        hx = s.filter(pl.col('player_id') == x).select('hand_id', 'started_at'); hy = s.filter(pl.col('player_id') == y).select('hand_id')
        out[pid] = hx.join(hy, on='hand_id').sort('started_at')['hand_id'].to_list()
    return out

def write_pack(fname, title, intro, items, phase):
    """items: list of dict(pair_id, A, B, header, hands=[(hid, tag)])"""
    sh = shared_hands([(it['pair_id'], it['A'], it['B']) for it in items], phase)
    allh = {hid for it in items for hid, _ in it['hands']}
    h, s, a = load(allh)
    L = [f"# {title}", intro, '']
    for it in items:
        order = {hid: i for i, hid in enumerate(sh[it['pair_id']])}
        L.append(f"## pair {it['pair_id']} — {it['header']} | shared hands in {phase}: {len(order)}  (A={it['A']}, B={it['B']})")
        for hid, tag in sorted(it['hands'], key=lambda t: order.get(t[0], 10**9)):
            pos = f"shared-hand #{order.get(hid, -1) + 1}/{len(order)}"
            L.append(render(hid, h, s, a, it['A'], it['B'], tag, pos)); L.append('')
    open(OUT + fname, 'w').write('\n'.join(L)); print('wrote', fname, len(L), 'lines', flush=True)

