"""Eval-side prompts for the top-K v032 pairs: same serialization as build_prompts.py (dev), restricted to the pair's
candidate hands in evidence_eval_onscvx4.parquet and to pair-member decisions. Usage: build_prompts_eval.py [K=8000]"""
import sys; from pathlib import Path
import polars as pl
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; PROC = ROOT / '1_data/processed'; RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
POS = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'HJ', 5: 'CO'}
RANK = {'2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine', 'T': 'Ten', 'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'A': 'Ace'}
SUIT = {'h': 'Heart', 'd': 'Diamond', 'c': 'Club', 's': 'Spade'}
def card(c, of='Of'): return f'{RANK[c[0]]} {of} {SUIT[c[1]]}'
def fmt(x):
    x = round(float(x), 1); return str(int(x)) if abs(x - int(x)) < 1e-9 else f'{x:.1f}'
sub = pl.read_csv(ROOT / '5_outputs/submissions/submission_v032cand_v5xcp2_eqx4.csv').select(['pair_id', 'risk_score']).sort('risk_score', descending=True).head(K)
cand = pl.scan_parquet(ROOT / '5_outputs/models/v5nb/evidence_eval_onscvx4.parquet')
cols = cand.collect_schema().names(); keep = ['pair_id', 'hand_id'] + [c for c in ('a', 'b') if c in cols]
d = cand.select(keep).join(sub.lazy().select('pair_id'), on='pair_id', how='semi').collect().unique()
if 'a' not in d.columns:
    ep = pl.read_csv(RAW / 'evaluation_pairs.csv'); pc = [c for c in ep.columns if c.startswith('player')][:2]
    d = d.join(ep.select(['pair_id', pl.col(pc[0]).alias('a'), pl.col(pc[1]).alias('b')]), on='pair_id', how='inner')
print(f'top-{K} pairs: {d.select("pair_id").n_unique():,} pairs, {d.height:,} pair-hands, {d.select("hand_id").n_unique():,} unique hands', flush=True)
hids = d.select('hand_id').unique()
acts = pl.read_parquet(RAW / 'actions.parquet').join(hids, on='hand_id', how='semi').sort(['hand_id', 'action_no'])
hands = pl.read_parquet(RAW / 'hands.parquet', columns=['hand_id', 'small_blind', 'board_cards']).join(hids, on='hand_id', how='semi')
seats = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'rel_pos', 'hole_card_1', 'hole_card_2']).join(hids, on='hand_id', how='semi')
bb = dict(zip(hands['hand_id'].to_list(), (hands['small_blind'] * 2).to_list())); board = dict(zip(hands['hand_id'].to_list(), hands['board_cards'].fill_null('').to_list()))
pos_of = {(r['hand_id'], r['player_id']): POS.get(int(r['rel_pos'])) for r in seats.iter_rows(named=True)}
hole = {(r['hand_id'], r['player_id']): (r['hole_card_1'], r['hole_card_2']) for r in seats.iter_rows(named=True)}
members = {}
for r in d.iter_rows(named=True): members.setdefault(r['hand_id'], []).append((r['pair_id'], r['a'], r['b']))
STREETS = ['preflop', 'flop', 'turn', 'river']; rows = []; uniq = {}
for hid, g in acts.group_by('hand_id', maintain_order=True):
    hid = hid[0]; B = bb[hid]; bc = board[hid].split(); pos = lambda pid: pos_of.get((hid, pid))
    seq = g.select(['action_no', 'street', 'player_id', 'action', 'amount', 'amount_to', 'pot_before', 'to_call']).rows()
    for k, (ano, street, pid, act, amt, amt_to, pot, to_call) in enumerate(seq):
        pairs_here = [(p, a, b) for p, a, b in members.get(hid, []) if pid in (a, b)]
        if not pairs_here or pos(pid) is None: continue
        hc = hole.get((hid, pid))
        if not hc or not hc[0] or not hc[1]: continue
        taken = {'fold': 'fold', 'check': 'check', 'call': 'call', 'bet': 'bet', 'raise': 'raise', 'all_in': 'allin'}[act]; legal = 'fold,call,raise' if to_call > 0 else 'check,bet'
        if (hid, ano) not in uniq:
            holding = f'[{card(hc[0], "of")} and {card(hc[1], "of")}]'; by_st = {s: [] for s in STREETS}
            for (ano2, st2, pid2, act2, amt2, amt_to2, pot2, tc2) in seq[:k]:
                p2 = pos(pid2) or 'a player'
                if act2 == 'fold': by_st[st2].append(f'{p2} fold')
                elif act2 == 'check': by_st[st2].append(f'{p2} check')
                elif act2 == 'call': by_st[st2].append(f'{p2} call')
                elif act2 == 'bet': by_st[st2].append(f'{p2} bet {fmt(amt2 / B)} chips')
                elif act2 == 'raise': by_st[st2].append(f'{p2} raise {fmt((amt_to2 if amt_to2 else amt2) / B)} chips')
                elif act2 == 'all_in': by_st[st2].append(f'{p2} allin')
            def joinlist(xs): return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ', and ' + xs[-1]
            parts = []; pf = by_st['preflop']; parts.append('Before the flop, ' + (joinlist(pf) if pf else 'nobody has acted yet') + '. Assume that all other players that is not mentioned folded.')
            si = STREETS.index(street)
            if si >= 1 and len(bc) >= 3: fl = by_st['flop']; parts.append('The flop comes ' + joinlist([card(c) for c in bc[:3]]) + (', then ' + joinlist(fl) if fl else '') + '.')
            if si >= 2 and len(bc) >= 4: tu = by_st['turn']; parts.append('The turn comes ' + card(bc[3]) + (', then ' + joinlist(tu) if tu else '') + '.')
            if si >= 3 and len(bc) >= 5: rv = by_st['river']; parts.append('The river comes ' + card(bc[4]) + (', then ' + joinlist(rv) if rv else '') + '.')
            text = ('You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision. '
                    'Here is a game summary: The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. '
                    'The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB. '
                    f'In this hand, your position is {pos(pid)}, and your holding is {holding}. ' + ' '.join(parts) +
                    f' Now it is your turn to make a move. To remind you, the current pot size is {fmt(pot / B)} chips, and your holding is {holding}. '
                    'Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.')
            uniq[(hid, ano)] = (pid, street, taken, legal, text)
        for (p, a, b) in pairs_here: rows.append((p, hid, int(ano), pid, 'A' if pid == a else 'B', street, taken, legal))
out = pl.DataFrame(rows, schema=['pair_id', 'hand_id', 'action_no', 'player_id', 'role', 'street', 'taken', 'legal'], orient='row')
out.write_parquet(HERE / f'prompts_eval_top{K}.parquet')
u = pl.DataFrame([(h, a, *v) for (h, a), v in uniq.items()], schema=['hand_id', 'action_no', 'player_id', 'street', 'taken', 'legal', 'prompt'], orient='row')
u.write_parquet(HERE / f'prompts_eval_unique_top{K}.parquet')
print(f'decision rows {out.height:,}; unique prompts {u.height:,}; parquet {(HERE / f"prompts_eval_unique_top{K}.parquet").stat().st_size / 1e6:.1f} MB', flush=True)
