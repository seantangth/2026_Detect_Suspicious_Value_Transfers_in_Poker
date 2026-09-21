"""Preflop isolation features from the raw action log (symmetric in A/B; 2026-09-19).
For every pair-member preflop raise: who was the previous aggressor (none / outsider / partner), how many outsiders had
voluntarily entered and were still live (victims), hand quality of the raiser, sizing; plus what outsiders / the partner did next.
build(keys) : keys = DataFrame(pair_id, hand_id, a, b) -> DataFrame(pair_id, hand_id, cif_*)"""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
RANK = {r: i for i, r in enumerate('23456789TJQKA', 2)}


def chen(c1, c2):
    if c1 is None or c2 is None:
        return np.nan
    r1, r2 = RANK[c1[0]], RANK[c2[0]]; hi, lo = max(r1, r2), min(r1, r2)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2.0)
    if hi == lo:
        return max(base * 2, 5)
    s = base + (2 if c1[1] == c2[1] else 0); gap = hi - lo - 1
    s -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        s += 1
    return s


def build(keys):
    hs = keys['hand_id'].unique().to_list()
    acts = (pl.scan_parquet(RAW / 'actions.parquet').filter(pl.col('hand_id').is_in(hs) & (pl.col('street') == 'preflop'))
              .select(['hand_id', 'action_no', 'player_id', 'action', 'amount_to', 'to_call', 'pot_before', 'players_active']).collect().sort(['hand_id', 'action_no']))
    seats = pl.scan_parquet(RAW / 'seats.parquet').filter(pl.col('hand_id').is_in(hs)).select(['hand_id', 'player_id', 'hole_card_1', 'hole_card_2']).collect()
    bbm = dict(zip(*[c.to_list() for c in pl.read_parquet(RAW / 'hands.parquet', columns=['hand_id', 'big_blind']).filter(pl.col('hand_id').is_in(hs)).get_columns()]))
    ch = {(h, p): chen(a, b) for h, p, a, b in seats.iter_rows()}
    by = {h: g for (h,), g in acts.group_by(['hand_id'], maintain_order=True)}
    rows = []
    for pid, h, a, b in keys.select(['pair_id', 'hand_id', 'a', 'b']).iter_rows():
        g = by.get(h); bb = bbm[h]
        f = dict(pair_id=pid, hand_id=h, cif_n_raise=0, cif_min_chen=np.nan, cif_max_level=0, cif_open_trash=np.nan, cif_vs_out_chen=np.nan, cif_vs_partner_chen=np.nan,
                 cif_squeeze=0, cif_squeeze_chen=np.nan, cif_partner_war_noout=0, cif_max_victims=0, cif_out_fold_after=0, cif_out_cont_after=0,
                 cif_partner_fold_after=0, cif_partner_fold_clean=0, cif_size_ratio_max=np.nan, cif_both_raise=0, cif_first_in_member=0, cif_n_out_vpip=0, cif_limp_member=0,
                 cif_first_mem_idx=np.nan, cif_active_at_first=np.nan, cif_nfold_before=np.nan, cif_noutvol_before=np.nan, cif_first_mem_raise=np.nan, cif_first_mem_chen=np.nan,
                 cif_first_mem_level=np.nan, cif_other_mem_resp=np.nan, cif_active_at_first_raise=np.nan, cif_nfold_before_raise=np.nan)
        if g is None:
            rows.append(f); continue
        folded = set(); vpip = set(); level = 0; last_aggr = None; last_to = 1.0; raisers = set(); first_vol = None
        ev = list(g.iter_rows(named=True)); last_member_raise_i = -1
        nf = 0; nov = 0; first_mem = None
        for i, x in enumerate(ev):
            p = x['player_id']
            if p in (a, b) and x['action'] != 'fold' and first_mem is None:
                first_mem = p; f['cif_first_mem_idx'] = i; f['cif_active_at_first'] = x['players_active']; f['cif_nfold_before'] = nf; f['cif_noutvol_before'] = nov
                f['cif_first_mem_raise'] = int(x['action'] in ('raise', 'bet', 'all_in')); f['cif_first_mem_chen'] = ch.get((h, p), np.nan)
                f['cif_first_mem_level'] = sum(1 for y in ev[:i] if y['action'] in ('raise', 'bet', 'all_in'))
                other = b if p == a else a; resp = [y['action'] for y in ev[i + 1:] if y['player_id'] == other]
                f['cif_other_mem_resp'] = {'fold': 0, 'call': 1, 'check': 1, 'raise': 2, 'bet': 2, 'all_in': 2}.get(resp[0], -1) if resp else (-2 if any(y['player_id'] == other for y in ev[:i]) else -1)
            if x['action'] == 'fold': nf += 1
            elif p not in (a, b): nov += 1
        nf = 0
        for i, x in enumerate(ev):
            if x['player_id'] in (a, b) and x['action'] in ('raise', 'bet', 'all_in'):
                f['cif_active_at_first_raise'] = x['players_active']; f['cif_nfold_before_raise'] = nf; break
            if x['action'] == 'fold': nf += 1
        for i, x in enumerate(ev):
            p = x['player_id']; act = x['action']; mem = p in (a, b)
            if act == 'fold':
                folded.add(p)
                if mem and last_aggr is not None and last_aggr in (a, b) and last_aggr != p:
                    f['cif_partner_fold_after'] = 1
                    if p not in vpip:
                        f['cif_partner_fold_clean'] = 1
                continue
            if act in ('call',):
                if first_vol is None: first_vol = p
                vpip.add(p)
                if mem and level == 0: f['cif_limp_member'] = 1
                continue
            if act in ('raise', 'bet', 'all_in') and x['amount_to'] / bb > last_to + 1e-9:
                if first_vol is None: first_vol = p
                level += 1; to = x['amount_to'] / bb
                if mem:
                    partner = b if p == a else a
                    victims = [q for q in vpip if q not in (a, b) and q not in folded]
                    c = ch.get((h, p), np.nan); f['cif_n_raise'] += 1; raisers.add(p)
                    f['cif_min_chen'] = np.nanmin([f['cif_min_chen'], c]); f['cif_max_level'] = max(f['cif_max_level'], level)
                    f['cif_max_victims'] = max(f['cif_max_victims'], len(victims)); f['cif_size_ratio_max'] = np.nanmax([f['cif_size_ratio_max'], to / max(last_to, 1.0)])
                    if last_aggr is None and not victims: f['cif_open_trash'] = np.nanmin([f['cif_open_trash'], c])
                    if (last_aggr is not None and last_aggr not in (a, b)) or victims: f['cif_vs_out_chen'] = np.nanmin([f['cif_vs_out_chen'], c])
                    if last_aggr == partner:
                        f['cif_vs_partner_chen'] = np.nanmin([f['cif_vs_partner_chen'], c])
                        if victims: f['cif_squeeze'] = 1; f['cif_squeeze_chen'] = np.nanmin([f['cif_squeeze_chen'], c])
                        else: f['cif_partner_war_noout'] = 1
                    last_member_raise_i = i
                vpip.add(p); last_aggr = p; last_to = to
        f['cif_both_raise'] = int(len(raisers) == 2); f['cif_first_in_member'] = int(first_vol in (a, b)); f['cif_n_out_vpip'] = len([q for q in vpip if q not in (a, b)])
        if last_member_raise_i >= 0:
            for x in ev[last_member_raise_i + 1:]:
                if x['player_id'] in (a, b): continue
                if x['action'] == 'fold': f['cif_out_fold_after'] += 1
                else: f['cif_out_cont_after'] += 1
        rows.append(f)
    return pl.DataFrame(rows).with_columns([pl.col(c).cast(pl.Float32) for c in rows[0] if c.startswith('cif_')])


if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    g = pl.read_parquet(ROOT / '5_outputs/revise_0915/gate_frame.parquet', columns=['pair_id', 'hand_id', 'a', 'b'])
    d = build(g); d.write_parquet(Path(__file__).resolve().parent / 'cif_dev.parquet'); print(d.shape, f'{time.time()-t0:.0f}s'); print(d.describe())
