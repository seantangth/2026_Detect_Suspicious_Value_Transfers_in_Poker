"""Exact / high-precision heads-up equity of A vs B (the candidate pair) - the 2026-09-13 audit F08 fix.

Replaces the NUMBERS of tpds_equity.py + tpds_preflop_eq.py, keeping their schema, row coverage and tie = 0.5 semantics:
  flop    exact enumeration of all C(45,2) = 990 turn/river runouts        (old: 36 fixed random runouts; MAE 0.048, p95 0.129,
                                                                              value changed under suit renaming in 8/96 sampled hands)
  turn    exact 44 rivers                                                    (unchanged)
  river   showdown                                                           (unchanged)
  preflop suit-canonical key, NMC boards drawn WITHOUT replacement with a    (old: 200 boards, biased "shift duplicates" scheme,
          key-derived seed -> identical value for suit-isomorphic matchups     keyed on raw cards -> unstable under suit renaming)
Postflop enumeration is suit-invariant by construction (the board fixes the suits).

Writes  1_data/processed/equity_{phase}_<tag>.parquet   (hand_id, a, b, eq_flop_A, eq_turn_A, eq_river_A, A_eq_at_fold, B_eq_at_fold)
        1_data/processed/pfeq_{phase}_<tag>.parquet     (hand_id, a, b, pf_eq_A)
with <tag> = TPDS_EQTAG (required, so that the fixed-path tables behind the submitted chains are never overwritten).
Usage: TPDS_EQTAG=x tpds_equity_exact.py [development evaluation] [workers=4] [nmc=10000] [pf_only=0] [post_only=0]
"""
import os, sys, time, glob, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
EQTAG = os.environ.get('TPDS_EQTAG', '').strip()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- card ids (phevaluator convention)
# phevaluator Card id = rank_index * 4 + suit_index, ranks '23456789TJQKA', suits 'cdhs'
RANKS = '23456789TJQKA'; SUITS = 'cdhs'
CID = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
PAIRS_990 = np.array(list(itertools.combinations(range(45), 2)), dtype=np.int64)   # index pairs into the 45-card remainder


# ----------------------------------------------------------------------------- worker side (postflop)
def _post_worker(task):
    """task: (key, a1, a2, b1, b2, board(n,5; -1 pad), nb, als, bls). Returns (key, ef, et, er) float32 arrays."""
    from phevaluator.evaluator import evaluate_7cards
    key, a1, a2, b1, b2, board, nb, als, bls = task
    n = len(a1)
    ef = np.full(n, np.nan, np.float32); et = np.full(n, np.nan, np.float32); er = np.full(n, np.nan, np.float32)
    comb = itertools.combinations
    for k in range(n):
        x1, x2, y1, y2 = int(a1[k]), int(a2[k]), int(b1[k]), int(b2[k])
        bd = [int(c) for c in board[k, :int(nb[k])]]
        if len(bd) >= 3:
            f0, f1, f2 = bd[0], bd[1], bd[2]
            known = {x1, x2, y1, y2, f0, f1, f2}
            deck = [c for c in range(52) if c not in known]
            w = 0.0
            for t, r in comb(deck, 2):
                ra = evaluate_7cards(x1, x2, f0, f1, f2, t, r); rb = evaluate_7cards(y1, y2, f0, f1, f2, t, r)
                w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
            ef[k] = w / 990.0
        if len(bd) >= 4 and als[k] >= 2 and bls[k] >= 2:
            f0, f1, f2, f3 = bd[0], bd[1], bd[2], bd[3]
            known = {x1, x2, y1, y2, f0, f1, f2, f3}
            w = 0.0; m = 0
            for r in range(52):
                if r in known:
                    continue
                ra = evaluate_7cards(x1, x2, f0, f1, f2, f3, r); rb = evaluate_7cards(y1, y2, f0, f1, f2, f3, r)
                w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0); m += 1
            et[k] = w / m
        if len(bd) == 5 and als[k] >= 3 and bls[k] >= 3:
            ra = evaluate_7cards(x1, x2, *bd); rb = evaluate_7cards(y1, y2, *bd)
            er[k] = 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
    return key, ef, et, er


def run_postflop(phase, workers):
    import polars as pl
    from multiprocessing import get_context
    out = PROC / f'equity_{phase}_{EQTAG}.parquet'
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'hole_card_1', 'hole_card_2'])
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'board_cards', 'n_board'])
    files = sorted(glob.glob(str(PROC / 'l2' / phase / '*.parquet')))
    t0 = time.time(); tasks = []; frames = {}
    for f in files:
        d = pl.read_parquet(f, columns=['hand_id', 'a', 'b', 'A_last_street', 'B_last_street', 'A_fold_street', 'B_fold_street'])
        d = d.filter((pl.col('A_last_street') >= 1) & (pl.col('B_last_street') >= 1))
        if d.height == 0:
            continue
        d = d.join(l0.rename({'player_id': 'a', 'hole_card_1': 'a1', 'hole_card_2': 'a2'}), on=['hand_id', 'a'])
        d = d.join(l0.rename({'player_id': 'b', 'hole_card_1': 'b1', 'hole_card_2': 'b2'}), on=['hand_id', 'b'])
        d = d.join(l1, on='hand_id')
        key = Path(f).stem
        a1 = np.array([CID[x] for x in d['a1'].to_list()], np.int8); a2 = np.array([CID[x] for x in d['a2'].to_list()], np.int8)
        b1 = np.array([CID[x] for x in d['b1'].to_list()], np.int8); b2 = np.array([CID[x] for x in d['b2'].to_list()], np.int8)
        board = np.full((d.height, 5), -1, np.int8)
        for i, s in enumerate(d['board_cards'].to_list()):
            if s:
                cs = [CID[c] for c in s.split(' ')]
                board[i, :len(cs)] = cs
        nb = d['n_board'].to_numpy().astype(np.int8)
        als = d['A_last_street'].to_numpy().astype(np.int8); bls = d['B_last_street'].to_numpy().astype(np.int8)
        tasks.append((key, a1, a2, b1, b2, board, nb, als, bls))
        frames[key] = d.select(['hand_id', 'a', 'b', 'A_fold_street', 'B_fold_street'])
    n_rows = sum(len(t[1]) for t in tasks)
    log(f"equity {phase}: {len(tasks)} tables, {n_rows:,} both-saw-flop rows; enumerating with {workers} workers")
    del l0, l1
    results = {}
    done = 0; done_rows = 0
    with get_context('spawn').Pool(workers) as pool:
        for key, ef, et, er in pool.imap_unordered(_post_worker, tasks, chunksize=1):
            results[key] = (ef, et, er); done += 1; done_rows += len(ef)
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                log(f"equity {phase}: {done}/{len(tasks)} tables, {done_rows:,} rows, {el:.0f}s elapsed, eta {el / done_rows * (n_rows - done_rows):.0f}s")
    parts = []
    for key, _a1, *_ in tasks:
        d = frames[key]; ef, et, er = results[key]
        afs = d['A_fold_street'].fill_null(-1).to_numpy(); bfs = d['B_fold_street'].fill_null(-1).to_numpy()
        a_fold_eq = np.where(afs == 1, ef, np.where(afs == 2, et, np.where(afs == 3, er, np.nan)))
        b_fold_eq = np.where(bfs == 1, 1 - ef, np.where(bfs == 2, 1 - et, np.where(bfs == 3, 1 - er, np.nan)))
        parts.append(d.select(['hand_id', 'a', 'b']).with_columns(
            pl.Series('eq_flop_A', ef), pl.Series('eq_turn_A', et), pl.Series('eq_river_A', er),
            pl.Series('A_eq_at_fold', a_fold_eq.astype(np.float32)), pl.Series('B_eq_at_fold', b_fold_eq.astype(np.float32))))
    res = pl.concat(parts)
    res.write_parquet(out)
    log(f"equity {phase} done: {res.height:,} rows in {time.time()-t0:.0f}s -> {out.name}")


# ----------------------------------------------------------------------------- preflop: suit-canonical keys
_PERMS = np.array(list(itertools.permutations(range(4))), dtype=np.int64)    # 24 suit relabelings


def _canon_codes(cards):
    """cards: (n,4) int card ids [a1,a2,b1,b2]. Returns for each row the minimal code over the 24 suit relabelings,
    where within each player the two cards are sorted by (rank desc, relabelled suit asc). Code packs 4 cards (6 bits each)."""
    r = cards // 4; s = cards % 4                                               # (n,4)
    n = cards.shape[0]
    best = None
    for perm in _PERMS:
        s2 = perm[s]                                                            # (n,4)
        c2 = r * 4 + s2
        # sort within player: descending rank, then ascending suit  (key = -(rank*4) + suit ... use (13-1-r)*4 + s ascending)
        ka = (12 - r[:, :2]) * 4 + s2[:, :2]; kb = (12 - r[:, 2:]) * 4 + s2[:, 2:]
        oa = np.argsort(ka, axis=1, kind='stable'); ob = np.argsort(kb, axis=1, kind='stable')
        ca = np.take_along_axis(c2[:, :2], oa, axis=1); cb = np.take_along_axis(c2[:, 2:], ob, axis=1)
        code = (((ca[:, 0].astype(np.int64) * 64 + ca[:, 1]) * 64 + cb[:, 0]) * 64 + cb[:, 1])
        best = code if best is None else np.minimum(best, code)
    return best


def _decode(code):
    c = []
    for _ in range(4):
        c.append(int(code % 64)); code //= 64
    return c[::-1]   # [a1, a2, b1, b2]


def _pf_worker(task):
    """task: (codes array, nmc). Returns equity of A vs B for each canonical code (MC without replacement, key-derived seed)."""
    from phevaluator.evaluator import evaluate_7cards
    codes, nmc = task
    out = np.empty(len(codes), np.float32)
    for i, code in enumerate(codes):
        a1, a2, b1, b2 = _decode(int(code))
        deck = np.array([c for c in range(52) if c not in (a1, a2, b1, b2)], np.int64)     # 48 cards
        rng = np.random.default_rng(1_000_003 + int(code))
        boards = deck[np.argsort(rng.random((nmc, 48)), axis=1)[:, :5]]                  # rows = uniform 5-subsets, no replacement
        w = 0.0
        for j in range(nmc):
            c0, c1, c2, c3, c4 = (int(x) for x in boards[j])
            ra = evaluate_7cards(a1, a2, c0, c1, c2, c3, c4); rb = evaluate_7cards(b1, b2, c0, c1, c2, c3, c4)
            w += 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
        out[i] = w / nmc
    return codes, out


def run_preflop(phases, workers, nmc):
    import polars as pl
    from multiprocessing import get_context
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'hole_card_1', 'hole_card_2'])
    t0 = time.time()
    per_phase = {}
    all_codes = []; all_flip = []
    for phase in phases:
        files = sorted(glob.glob(str(PROC / 'l2' / phase / '*.parquet')))
        d = pl.concat([pl.read_parquet(f, columns=['hand_id', 'a', 'b']) for f in files])
        d = d.join(l0.rename({'player_id': 'a', 'hole_card_1': 'a1', 'hole_card_2': 'a2'}), on=['hand_id', 'a'])
        d = d.join(l0.rename({'player_id': 'b', 'hole_card_1': 'b1', 'hole_card_2': 'b2'}), on=['hand_id', 'b'])
        cards = np.stack([np.array([CID[x] for x in d[c].to_list()], np.int64) for c in ('a1', 'a2', 'b1', 'b2')], axis=1)
        # exact (ordered) key -> canonical code + flip, computed once per distinct exact key
        ex = ((cards[:, 0] * 52 + cards[:, 1]) * 52 + cards[:, 2]) * 52 + cards[:, 3]
        uex, inv = np.unique(ex, return_inverse=True)
        ucards = np.stack([(uex // 52 ** 3) % 52, (uex // 52 ** 2) % 52, (uex // 52) % 52, uex % 52], axis=1)
        code_ab = _canon_codes(ucards); code_ba = _canon_codes(ucards[:, [2, 3, 0, 1]])
        flip_u = code_ba < code_ab
        code_u = np.where(flip_u, code_ba, code_ab)
        per_phase[phase] = (d.select(['hand_id', 'a', 'b']), code_u[inv], flip_u[inv])
        all_codes.append(code_u); all_flip.append(flip_u)
        log(f"pfeq {phase}: {d.height:,} rows, {len(uex):,} distinct exact keys, {len(np.unique(code_u)):,} canonical classes ({time.time()-t0:.0f}s)")
    codes = np.unique(np.concatenate(all_codes))
    log(f"pfeq: {len(codes):,} canonical classes to evaluate with {nmc:,} boards each, {workers} workers")
    chunks = [codes[i:i + 200] for i in range(0, len(codes), 200)]
    eq_map = {}
    done = 0
    with get_context('spawn').Pool(workers) as pool:
        for cs, es in pool.imap_unordered(_pf_worker, [(c, nmc) for c in chunks], chunksize=1):
            for c, e in zip(cs, es):
                eq_map[int(c)] = float(e)
            done += 1
            if done % 20 == 0 or done == len(chunks):
                el = time.time() - t0
                log(f"pfeq: {done}/{len(chunks)} chunks, {el:.0f}s, eta {el / done * (len(chunks) - done):.0f}s")
    keys = np.array(sorted(eq_map)); vals = np.array([eq_map[k] for k in keys], np.float32)
    for phase in phases:
        d, code, flip = per_phase[phase]
        e = vals[np.searchsorted(keys, code)]
        e = np.where(flip, 1.0 - e, e).astype(np.float32)
        out = PROC / f'pfeq_{phase}_{EQTAG}.parquet'
        d.with_columns(pl.Series('pf_eq_A', e)).write_parquet(out)
        log(f"pfeq {phase} done: {d.height:,} rows -> {out.name}")


if __name__ == '__main__':
    assert EQTAG, 'set TPDS_EQTAG=<tag>: this module never overwrites the fixed-path equity tables'
    phases = [a for a in sys.argv[1:] if '=' not in a] or ['development', 'evaluation']
    kw = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
    workers = int(kw.get('workers', 4)); nmc = int(kw.get('nmc', 10000))
    if not int(kw.get('pf_only', 0)):
        for ph in phases:
            run_postflop(ph, workers)
    if not int(kw.get('post_only', 0)):
        run_preflop(phases, workers, nmc)
