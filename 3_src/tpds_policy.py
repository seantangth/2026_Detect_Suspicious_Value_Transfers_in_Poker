"""Policy-deviation ("surprise") scoring.
Train a global action model P(action | decision state, player tendencies) on a sample of all actions, score every action,
surprise = -log P(taken action). Aggregate to (hand, player) and (hand, responder, aggressor).
Outputs: 1_data/processed/surprise_player.parquet, surprise_resp.parquet
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
SEED = 42
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_paths import vpath


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_state():
    a = pl.read_parquet(PROC / 'action_ctx.parquet')          # hand_id, action_no, sidx, player_id, is_agg, is_call, is_fold, facing, last_aggr, to_call_bb, pot_before_bb, amt_pot, players_active
    raw = pl.read_parquet(RAW / 'actions.parquet', columns=['hand_id', 'action_no', 'stack_before', 'action'])
    a = a.join(raw, on=['hand_id', 'action_no'])
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'rel_pos', 'stack_bb', 'chen', 'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo',
                                                            'str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3'])
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'big_blind', 'n_board', 'phase'])
    base = pl.read_parquet(PROC / 'player_baselines.parquet').filter(pl.col('bphase') == 'all').drop('bphase')
    a = a.join(l0, on=['hand_id', 'player_id']).join(l1, on='hand_id').join(base, on='player_id', how='left')
    a = a.sort(['hand_id', 'action_no']).with_columns(
        (pl.col('stack_before') / pl.col('big_blind')).cast(pl.Float32).alias('stack_before_bb'),
        pl.col('is_agg').cast(pl.Int32).cum_sum().over('hand_id').alias('_cum_agg'),
    ).with_columns(
        (pl.col('_cum_agg') - pl.col('is_agg').cast(pl.Int32)).alias('n_agg_before'),
        (pl.col('stack_bb') - pl.col('stack_before') / pl.col('big_blind')).cast(pl.Float32).alias('invested_bb'),
        (pl.col('to_call_bb') / (pl.col('pot_before_bb') + pl.col('to_call_bb') + 1e-3)).cast(pl.Float32).alias('pot_odds'),
        (pl.col('stack_before') / pl.col('big_blind') / (pl.col('pot_before_bb') + 1e-3)).clip(0, 200).cast(pl.Float32).alias('spr'),
        pl.when(pl.col('sidx') == 1).then(pl.col('str_pct_1')).when(pl.col('sidx') == 2).then(pl.col('str_pct_2')).when(pl.col('sidx') == 3).then(pl.col('str_pct_3')).otherwise(None).cast(pl.Float32).alias('str_now'),
        pl.when(pl.col('sidx') == 1).then(pl.col('str_cat_1')).when(pl.col('sidx') == 2).then(pl.col('str_cat_2')).when(pl.col('sidx') == 3).then(pl.col('str_cat_3')).otherwise(None).cast(pl.Int8).alias('cat_now'),
        pl.when(pl.col('is_fold')).then(0).when(pl.col('action') == 'check').then(1).when(pl.col('is_call')).then(2).otherwise(3).cast(pl.Int8).alias('y'),
    )
    return a


FEATS = ['sidx', 'rel_pos', 'players_active', 'to_call_bb', 'pot_before_bb', 'pot_odds', 'stack_before_bb', 'spr', 'invested_bb', 'n_agg_before',
         'facing', 'chen', 'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo', 'str_now', 'cat_now', 'n_board',
         'b_vpip', 'b_pfr', 'b_agg_rate', 'b_fold_facing', 'b_call_facing', 'b_raise_facing', 'b_sd_rate', 'b_post_check', 'b_contrib', 'b_allin', 'b_saw_flop']
PARAMS = dict(objective='multiclass', num_class=4, learning_rate=0.1, num_leaves=127, min_child_samples=200, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=8, seed=SEED)


def main(n_train=3_000_000, rounds=250, drop_n_board=0, crossfit=0):
    # crossfit=1: two table groups (fold parity of 5_outputs/models/v5/folds_by_table.json); each group's actions are scored
    # by the model trained on the OTHER group, so no action's surprise comes from a model that saw it (default 0 = unchanged).
    # `n_board` is the FINAL number of board cards of the hand (hands_l1, one value per hand_id), so every
    # action in the hand would see how far the hand eventually went: a preflop fold in a hand that ended
    # preflop, or a call in a hand that reached the flop, becomes 'expected' and its surprise is deflated.
    # That is look-ahead, not the player's decision state. drop_n_board=1 removes it from the policy
    # features ONLY (the module-level FEATS is unchanged so tpds_betsize keeps its historical behaviour).
    feats = [f for f in FEATS if not (drop_n_board and f == 'n_board')]
    if drop_n_board:
        log("drop_n_board=1: 'n_board' removed from the policy features (look-ahead: final board size joined per hand)")
    log(f"policy features ({len(feats)}): {feats}")
    log('building state table'); a = build_state()
    log(f"actions {a.height:,}")
    X_all = a.select(feats).with_columns(pl.col('facing').cast(pl.Int8))
    y_all = a['y'].to_numpy()
    if crossfit:
        import json
        if int(crossfit) == 2:
            # crossfit=2, by HAND: seeded random 50/50 over the sorted unique hand_ids. Every table (= one 30-player pool) is in
            # both halves, so the scoring model knows the same players from their other hands and only never saw THIS hand.
            # (crossfit=1 splits by table and so also removes all player-level knowledge; EXPERIMENT_LOG 2026-09-10 22:21.)
            uh = a.select(pl.col('hand_id').unique().sort()).to_series()
            gm = pl.DataFrame({'hand_id': uh, 'g': np.random.default_rng(SEED + 7).integers(0, 2, len(uh))})
            grp = a.select('hand_id').with_row_index('ri').join(gm, on='hand_id', how='left').sort('ri')['g'].to_numpy()
            assert len(grp) == a.height and not np.isnan(grp.astype(float)).any(), 'crossfit=2: every action needs a hand group'
            log(f'crossfit=2 (by hand): group action counts {np.bincount(grp)} over {len(uh):,} hands')
        else:
            folds = json.load(open(ROOT / '5_outputs/models/v5/folds_by_table.json'))
            tb = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id'])
            tid = a.select('hand_id').with_row_index('ri').join(tb, on='hand_id', how='left').sort('ri')['table_id']
            assert tid.null_count() == 0 and len(tid) == a.height, 'crossfit: every action needs a table_id'
            grp = tid.replace_strict(folds, return_dtype=pl.Int64).to_numpy() % 2
            log(f'crossfit: table-group action counts {np.bincount(grp)}')
        surp = np.empty(a.height, dtype=np.float32); ptaken = np.empty(a.height, dtype=np.float32)
        rng = np.random.default_rng(SEED)
        for g in (0, 1):
            pool = np.flatnonzero(grp == g)
            idx = rng.choice(pool, size=min(n_train, len(pool)), replace=False)
            log(f'crossfit: training policy model on table group {g} ({len(pool):,} actions, sample {len(idx):,})')
            m = lgb.train(PARAMS, lgb.Dataset(X_all[idx].to_numpy().astype(np.float32), y_all[idx], feature_name=feats), num_boost_round=rounds)
            m.save_model(str(vpath(f'policy_model_g{g}.txt')))
            tgt = np.flatnonzero(grp != g)
            for s0 in range(0, len(tgt), 2_000_000):
                ii = tgt[s0:s0 + 2_000_000]
                p = m.predict(X_all[ii].to_numpy().astype(np.float32))
                yt = y_all[ii]; pt = p[np.arange(len(ii)), yt]
                ptaken[ii] = pt; surp[ii] = -np.log(np.clip(pt, 1e-6, 1))
            log(f'  scored {len(tgt):,} actions of the other group')
        log(f"crossfit: mean surprise {surp.mean():.4f}; class dist {np.bincount(y_all)/len(y_all)}")
    else:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(a.height, size=min(n_train, a.height), replace=False)
        log('training policy model')
        Xtr = X_all[idx].to_numpy().astype(np.float32)
        ds = lgb.Dataset(Xtr, y_all[idx], feature_name=feats)
        m = lgb.train(PARAMS, ds, num_boost_round=rounds)
        m.save_model(str(vpath('policy_model.txt')))
        log('scoring all actions')
        surp = np.empty(a.height, dtype=np.float32); ptaken = np.empty(a.height, dtype=np.float32)
        chunk = 2_000_000
        for s in range(0, a.height, chunk):
            Xc = X_all[s:s + chunk].to_numpy().astype(np.float32)
            p = m.predict(Xc)
            yt = y_all[s:s + chunk]
            pt = p[np.arange(len(yt)), yt]
            ptaken[s:s + chunk] = pt; surp[s:s + chunk] = -np.log(np.clip(pt, 1e-6, 1))
            log(f"  scored {min(s+chunk, a.height):,}")
        # held-out sanity: logloss on non-train rows
        mask = np.ones(a.height, bool); mask[idx] = False
        log(f"held-out mean surprise {surp[mask].mean():.4f} (train {surp[~mask].mean():.4f}); class dist {np.bincount(y_all)/len(y_all)}")
    a = a.select(['hand_id', 'player_id', 'sidx', 'is_agg', 'is_call', 'is_fold', 'facing', 'last_aggr', 'pot_before_bb']).with_columns(
        pl.Series('surp', surp), pl.Series('p_taken', ptaken))
    a = a.with_columns((pl.col('surp') * (pl.col('pot_before_bb') + 1.0).log1p()).alias('surp_w'))
    per_player = a.group_by(['hand_id', 'player_id']).agg(
        pl.col('surp').sum().cast(pl.Float32).alias('surp_sum'), pl.col('surp').max().cast(pl.Float32).alias('surp_max'),
        pl.col('surp_w').sum().cast(pl.Float32).alias('surpw_sum'), pl.col('surp_w').max().cast(pl.Float32).alias('surpw_max'),
        pl.col('surp').filter(pl.col('is_fold')).max().cast(pl.Float32).alias('surp_fold'),
        pl.col('surp').filter(pl.col('is_call')).max().cast(pl.Float32).alias('surp_call'),
        pl.col('surp').filter(pl.col('is_agg')).max().cast(pl.Float32).alias('surp_agg'),
        pl.col('surp').filter(pl.col('sidx') > 0).max().cast(pl.Float32).alias('surp_post'),
        pl.col('p_taken').min().cast(pl.Float32).alias('p_min'),
    )
    per_player.write_parquet(vpath('surprise_player.parquet'))
    resp = a.filter(pl.col('facing') & pl.col('last_aggr').is_not_null()).group_by(['hand_id', 'player_id', 'last_aggr']).agg(
        pl.col('surp').max().cast(pl.Float32).alias('rsurp_max'), pl.col('surp').sum().cast(pl.Float32).alias('rsurp_sum'),
        pl.col('surp_w').max().cast(pl.Float32).alias('rsurpw_max'),
    ).rename({'player_id': 'responder', 'last_aggr': 'aggressor'})
    resp.write_parquet(vpath('surprise_resp.parquet'))
    log(f"done -> {vpath('surprise_player.parquet').name}, {vpath('surprise_resp.parquet').name}, {vpath('policy_model.txt').name}")


if __name__ == '__main__':
    kw = {}
    for arg in sys.argv[1:]:
        k, v = arg.split('='); kw[k] = int(v)
    main(**kw)
