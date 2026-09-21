"""Opponent-conditional BET SIZE residuals.

The global policy model (tpds_policy.py) already conditions on hand strength, so "took an action that
does not match my cards" is covered by `surprise`, and `d_surp` already differences it by opponent.
But its target is the action CLASS (fold / check / call / aggressive) - it never models HOW MUCH.
Size is the direct carrier of value transfer:
  soft_play              - bet SMALLER than the spot warrants while the partner is in the pot
  coordinated_isolation  - both bet LARGER to price the outsiders out
  directed_transfer      - inflate the pot, then surrender it

So: regress bet size on the same state (including hand strength), take the residual, and difference it
by whether the partner is in the pot - the Marginal Impact construction applied to size instead of rate.

Writes 1_data/processed/betsize_{phase}.parquet keyed (a, b) with a < b.
Usage: tpds_betsize.py [rounds=400]
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_policy import build_state, FEATS

PARAMS = dict(objective='regression', metric='l2', learning_rate=0.05, num_leaves=63,
              min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, n_jobs=8, seed=42)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def residuals(rounds=400):
    a = build_state()
    agg = a.filter(pl.col('is_agg') & pl.col('amt_pot').is_not_null() & (pl.col('amt_pot') > 0))
    log(f'aggressive actions with a size: {agg.height:,}')
    X = agg.select(FEATS).with_columns(pl.col('facing').cast(pl.Int8)).to_numpy().astype(np.float32)
    y = np.log1p(agg['amt_pot'].to_numpy().astype(np.float64))
    # a plain in-sample fit would let the model absorb the very anomalies we are looking for, so hold
    # out half at random and score each half with the model that did not see it
    rng = np.random.default_rng(42)
    half = rng.random(len(y)) < 0.5
    pred = np.empty(len(y))
    for m_ in (half, ~half):
        mdl = lgb.train(PARAMS, lgb.Dataset(X[m_], y[m_], feature_name=FEATS), num_boost_round=rounds)
        pred[~m_] = mdl.predict(X[~m_])
    res = (y - pred).astype(np.float32)
    log(f'size residual: sd {res.std():.4f}, mean {res.mean():+.4f}')
    return agg.select(['hand_id', 'player_id', 'phase']).with_columns(pl.Series('sz_res', res))


def run(rounds=400):
    r = residuals(rounds)
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'table_id', 'phase', 'vpip'])
    for phase in ('development', 'evaluation'):
        rp = r.filter(pl.col('phase') == phase).drop('phase')
        seats = l0.filter(pl.col('phase') == phase).drop('phase')
        # one row per (aggressor's action, other player at the table in that hand)
        act = rp.join(seats.select(['hand_id', 'player_id', 'table_id']), on=['hand_id', 'player_id'])
        oth = seats.select(['hand_id', pl.col('player_id').alias('o'), pl.col('vpip').alias('o_vpip')])
        d = act.join(oth, on='hand_id').filter(pl.col('player_id') != pl.col('o'))
        log(f'{phase}: {d.height:,} (action, other-player) rows')
        # mean residual when the other player entered the pot vs when they did not
        g = d.group_by(['player_id', 'o']).agg(
            pl.when(pl.col('o_vpip') == 1).then(pl.col('sz_res')).mean().alias('sz_with'),
            pl.when(pl.col('o_vpip') == 0).then(pl.col('sz_res')).mean().alias('sz_without'),
            (pl.col('o_vpip') == 1).sum().alias('n_with'),
            pl.when(pl.col('o_vpip') == 1).then(pl.col('sz_res')).max().alias('sz_with_max'),
        ).with_columns(
            (pl.col('sz_with').fill_null(0) - pl.col('sz_without').fill_null(0)).alias('d_sz'))
        # fold to the unordered pair: A->B and B->A both matter, keep min/max so it is order-invariant
        x = g.rename({'player_id': 'a', 'o': 'b'})
        y_ = g.rename({'player_id': 'b', 'o': 'a'})
        j = x.join(y_, on=['a', 'b'], suffix='_r').filter(pl.col('a') < pl.col('b'))
        out = j.select(
            'a', 'b',
            pl.min_horizontal('d_sz', 'd_sz_r').alias('bs_d_min'),
            pl.max_horizontal('d_sz', 'd_sz_r').alias('bs_d_max'),
            (pl.col('d_sz') + pl.col('d_sz_r')).alias('bs_d_sum'),
            (pl.col('d_sz') - pl.col('d_sz_r')).abs().alias('bs_d_asym'),
            pl.min_horizontal('sz_with', 'sz_with_r').alias('bs_with_min'),
            pl.max_horizontal('sz_with', 'sz_with_r').alias('bs_with_max'),
            pl.max_horizontal('sz_with_max', 'sz_with_max_r').alias('bs_with_peak'),
            pl.min_horizontal('n_with', 'n_with_r').cast(pl.Float32).alias('bs_n_min'),
        )
        out = out.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in out.columns if c not in ('a', 'b')])
        p = PROC / f'betsize_{phase}.parquet'
        out.write_parquet(p)
        log(f'wrote {p} ({out.height:,} rows)')


if __name__ == '__main__':
    kw = dict(x.split('=') for x in sys.argv[1:])
    run(int(kw.get('rounds', 400)))
