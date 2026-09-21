"""Submission hygiene checks (STRATEGY §5.7). Usage: validate_submission.py path/to/submission.csv"""
import sys
from pathlib import Path
import pandas as pd, polars as pl, numpy as np
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
sys.path.insert(0, str(ROOT / '3_src'))
from metric import score, ALLOWED_BEHAVIORS, EVIDENCE_COLUMNS


def validate(path):
    sub = pd.read_csv(path)
    sample = pd.read_csv(RAW / 'sample_submission.csv')
    errs = []
    if list(sub.columns) != list(sample.columns): errs.append(f'columns differ: {list(sub.columns)}')
    if len(sub) != len(sample): errs.append(f'row count {len(sub)} != {len(sample)}')
    if set(sub.pair_id) != set(sample.pair_id): errs.append('pair_id set mismatch')
    if sub.pair_id.duplicated().any(): errs.append('duplicate pair_id')
    if sub.isna().any().any(): errs.append(f'NaN cells: {sub.isna().sum().to_dict()}')
    r = pd.to_numeric(sub.risk_score, errors='coerce')
    if r.isna().any() or not r.between(0, 1).all(): errs.append('risk_score out of [0,1] or non-numeric')
    ties = r.duplicated().sum()
    bad_beh = set(sub.predicted_behavior) - ALLOWED_BEHAVIORS
    if bad_beh: errs.append(f'bad behaviors {bad_beh}')
    # evidence checks
    ev_cols = list(EVIDENCE_COLUMNS)
    long = sub.melt(id_vars=['pair_id'], value_vars=ev_cols, value_name='hand_id')
    long = long[long.hand_id != 'NO_EVIDENCE']
    dup = long.duplicated(['pair_id', 'hand_id']).sum()
    if dup: errs.append(f'{dup} repeated evidence within pair')
    hands = pl.read_parquet(RAW / 'hands.parquet', columns=['hand_id', 'phase']).to_pandas().set_index('hand_id').phase
    ph = long.hand_id.map(hands)
    if ph.isna().any(): errs.append(f'{int(ph.isna().sum())} unknown hand ids')
    if (ph == 'development').any(): errs.append(f'{int((ph=="development").sum())} development-period evidence hands')
    # both players present
    ep = pd.read_csv(RAW / 'evaluation_pairs.csv').set_index('pair_id')
    seats = pl.read_parquet(RAW / 'seats.parquet', columns=['hand_id', 'player_id'])
    lp = pl.from_pandas(long[['pair_id', 'hand_id']]).join(pl.from_pandas(ep.reset_index()[['pair_id', 'player_1', 'player_2']]), on='pair_id')
    s1 = lp.join(seats.rename({'player_id': 'player_1'}), on=['hand_id', 'player_1'], how='inner').height
    s2 = lp.join(seats.rename({'player_id': 'player_2'}), on=['hand_id', 'player_2'], how='inner').height
    if s1 != lp.height or s2 != lp.height: errs.append(f'evidence hands without both players: p1 missing {lp.height-s1}, p2 missing {lp.height-s2}')
    n_ev = (sub[ev_cols] != 'NO_EVIDENCE').sum(axis=1)
    print(f"rows {len(sub)} | risk ties {ties} | behaviors {sub.predicted_behavior.value_counts().to_dict()} | evidence per row: mean {n_ev.mean():.2f}, rows with 5 = {(n_ev==5).mean():.3f}, rows with 0 = {(n_ev==0).mean():.3f}")
    # run the official metric against a dummy solution to trigger its format checks
    dummy = sample.copy(); dummy['risk_score'] = 0; dummy.loc[dummy.index[:10], 'risk_score'] = 1
    dummy.loc[dummy.index[:10], 'predicted_behavior'] = 'soft_play'
    try:
        score(dummy, sub, 'pair_id')
    except Exception as e:
        errs.append(f'metric raised: {e}')
    if errs:
        print('INVALID:'); [print(' -', e) for e in errs]; return False
    print('VALID'); return True


if __name__ == '__main__':
    ok = validate(sys.argv[1]); sys.exit(0 if ok else 1)
