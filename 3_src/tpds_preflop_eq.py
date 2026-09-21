"""Preflop head-to-head equity A vs B (Monte Carlo, cached by the 4-card combination).
Covers the evidence hands where one partner folds preflop - the case the postflop equity table misses.
Writes 1_data/processed/pfeq_{phase}.parquet: (hand_id, a, b, pf_eq_A)
"""
import sys, time, glob
from pathlib import Path
import numpy as np, polars as pl
from phevaluator.card import Card
from phevaluator.evaluator import evaluate_7cards

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
CID = {r + s: Card(r + s).id_ for r in '23456789TJQKA' for s in 'cdhs'}
NMC = 200
rng = np.random.default_rng(7)
# fixed board samples: index into the 48 remaining cards
BOARDS = rng.integers(0, 48, size=(NMC, 5))
for i in range(NMC):                       # make each board's 5 indices distinct
    seen = set(); out = []
    j = 0
    for v in BOARDS[i]:
        while v in seen:
            v = (v + 1) % 48
        seen.add(v); out.append(v)
    BOARDS[i] = out


def eq_preflop(a1, a2, b1, b2, cache):
    if a1 > a2:
        a1, a2 = a2, a1
    if b1 > b2:
        b1, b2 = b2, b1
    flip = (b1, b2) < (a1, a2)
    key = (b1, b2, a1, a2) if flip else (a1, a2, b1, b2)
    v = cache.get(key)
    if v is None:
        known = {a1, a2, b1, b2}
        deck = [c for c in range(52) if c not in known]
        w = 0.0
        for k in range(NMC):
            b = BOARDS[k]
            c0, c1, c2, c3, c4 = deck[b[0]], deck[b[1]], deck[b[2]], deck[b[3]], deck[b[4]]
            ra = evaluate_7cards(key[0], key[1], c0, c1, c2, c3, c4)
            rb = evaluate_7cards(key[2], key[3], c0, c1, c2, c3, c4)
            w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
        v = w / NMC
        cache[key] = v
    return (1.0 - v) if flip else v


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(phase, cache):
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'hole_card_1', 'hole_card_2'])
    files = sorted(glob.glob(str(PROC / 'l2' / phase / '*.parquet')))
    parts = []; t0 = time.time(); n = 0
    for i, f in enumerate(files):
        d = pl.read_parquet(f, columns=['hand_id', 'a', 'b'])
        d = d.join(l0.rename({'player_id': 'a', 'hole_card_1': 'a1', 'hole_card_2': 'a2'}), on=['hand_id', 'a'])
        d = d.join(l0.rename({'player_id': 'b', 'hole_card_1': 'b1', 'hole_card_2': 'b2'}), on=['hand_id', 'b'])
        A1 = [CID[x] for x in d['a1'].to_list()]; A2 = [CID[x] for x in d['a2'].to_list()]
        B1 = [CID[x] for x in d['b1'].to_list()]; B2 = [CID[x] for x in d['b2'].to_list()]
        e = np.empty(d.height, np.float32)
        for k in range(d.height):
            e[k] = eq_preflop(A1[k], A2[k], B1[k], B2[k], cache)
        parts.append(d.select(['hand_id', 'a', 'b']).with_columns(pl.Series('pf_eq_A', e)))
        n += d.height
        if i % 50 == 0:
            log(f"pfeq {phase}: {i+1}/{len(files)} rows {n:,} cache {len(cache):,} {time.time()-t0:.0f}s")
    pl.concat(parts).write_parquet(PROC / f'pfeq_{phase}.parquet')
    log(f"pfeq {phase} done: {n:,} rows, cache {len(cache):,}, {time.time()-t0:.0f}s")


if __name__ == '__main__':
    cache = {}
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph, cache)
