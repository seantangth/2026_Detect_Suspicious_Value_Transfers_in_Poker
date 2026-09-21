"""Equity of A vs B (the pair) at flop (fixed 40-runout design) and turn (exact 44 runouts), for L2 rows where both reached the flop.
Output: 1_data/processed/equity_{phase}.parquet with (hand_id, a, b, eq_flop_A, eq_turn_A, A_eq_at_fold, B_eq_at_fold, eq_river_A)
"""
import sys, time, glob
from pathlib import Path
import numpy as np, polars as pl
from phevaluator.card import Card
from phevaluator.evaluator import evaluate_7cards

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
CID = {r + s: Card(r + s).id_ for r in '23456789TJQKA' for s in 'cdhs'}
rng = np.random.default_rng(42)
FLOP_PAIRS = [tuple(x) for x in rng.choice(45, size=(40, 2), replace=True) if x[0] != x[1]][:36]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def eq_flop(a1, a2, b1, b2, board3):
    known = {a1, a2, b1, b2, *board3}
    deck = [c for c in range(52) if c not in known]
    w = 0.0
    for i, j in FLOP_PAIRS:
        t, r = deck[i], deck[j]
        ra = evaluate_7cards(a1, a2, board3[0], board3[1], board3[2], t, r)
        rb = evaluate_7cards(b1, b2, board3[0], board3[1], board3[2], t, r)
        w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
    return w / len(FLOP_PAIRS)


def eq_turn(a1, a2, b1, b2, board4):
    known = {a1, a2, b1, b2, *board4}
    w = 0.0; n = 0
    for r in range(52):
        if r in known:
            continue
        ra = evaluate_7cards(a1, a2, board4[0], board4[1], board4[2], board4[3], r)
        rb = evaluate_7cards(b1, b2, board4[0], board4[1], board4[2], board4[3], r)
        w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0); n += 1
    return w / n


def run(phase):
    out = PROC / f'equity_{phase}.parquet'
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'hole_card_1', 'hole_card_2'])
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'board_cards', 'n_board'])
    files = sorted(glob.glob(str(PROC / 'l2' / phase / '*.parquet')))
    parts = []; t0 = time.time(); n_rows = 0
    for i, f in enumerate(files):
        d = pl.read_parquet(f, columns=['hand_id', 'a', 'b', 'A_last_street', 'B_last_street', 'A_fold_street', 'B_fold_street'])
        d = d.filter((pl.col('A_last_street') >= 1) & (pl.col('B_last_street') >= 1))
        if d.height == 0:
            continue
        d = d.join(l0.rename({'player_id': 'a', 'hole_card_1': 'a1', 'hole_card_2': 'a2'}), on=['hand_id', 'a'])
        d = d.join(l0.rename({'player_id': 'b', 'hole_card_1': 'b1', 'hole_card_2': 'b2'}), on=['hand_id', 'b'])
        d = d.join(l1, on='hand_id')
        a1 = [CID[x] for x in d['a1'].to_list()]; a2 = [CID[x] for x in d['a2'].to_list()]
        b1 = [CID[x] for x in d['b1'].to_list()]; b2 = [CID[x] for x in d['b2'].to_list()]
        boards = [[CID[c] for c in s.split(' ')] for s in d['board_cards'].to_list()]
        als = d['A_last_street'].to_list(); bls = d['B_last_street'].to_list()
        afs = d['A_fold_street'].fill_null(-1).to_list(); bfs = d['B_fold_street'].fill_null(-1).to_list()
        ef = np.full(d.height, np.nan, np.float32); et = np.full(d.height, np.nan, np.float32); er = np.full(d.height, np.nan, np.float32)
        for k in range(d.height):
            bd = boards[k]
            if len(bd) >= 3:
                ef[k] = eq_flop(a1[k], a2[k], b1[k], b2[k], bd[:3])
            if len(bd) >= 4 and als[k] >= 2 and bls[k] >= 2:
                et[k] = eq_turn(a1[k], a2[k], b1[k], b2[k], bd[:4])
            if len(bd) == 5 and als[k] >= 3 and bls[k] >= 3:
                ra = evaluate_7cards(a1[k], a2[k], *bd); rb = evaluate_7cards(b1[k], b2[k], *bd)
                er[k] = 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
        afs_a = np.array(afs); bfs_a = np.array(bfs)
        a_fold_eq = np.where(afs_a == 1, ef, np.where(afs_a == 2, et, np.where(afs_a == 3, er, np.nan)))
        b_fold_eq = np.where(bfs_a == 1, 1 - ef, np.where(bfs_a == 2, 1 - et, np.where(bfs_a == 3, 1 - er, np.nan)))
        parts.append(d.select(['hand_id', 'a', 'b']).with_columns(
            pl.Series('eq_flop_A', ef), pl.Series('eq_turn_A', et), pl.Series('eq_river_A', er),
            pl.Series('A_eq_at_fold', a_fold_eq.astype(np.float32)), pl.Series('B_eq_at_fold', b_fold_eq.astype(np.float32))))
        n_rows += d.height
        if i % 50 == 0:
            log(f"equity {phase}: {i+1}/{len(files)} tables, rows so far {n_rows:,}, {time.time()-t0:.0f}s")
    res = pl.concat(parts)
    res.write_parquet(out)
    log(f"equity {phase} done: {res.height:,} rows in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph)
