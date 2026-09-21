"""Dedicated evidence ranker (LambdaRank@5).
Key difference from the hand models in tpds_model.py: trained ONLY on positive pairs' rows, so it learns
"which shared hand of this colluding pair is the planted one" rather than "is this pair colluding".
Features: raw L2 features + per-pair percentile ranks + predicted-family one-hot.
Usage: tpds_evidence.py <run> [use_equity=2] [use_surprise=2] [rounds=600] [topk_pairs=0]
Writes: 5_outputs/models/<run>/evidence_{dev,eval}.parquet  (pair_id, hand_id, ev_score)
"""
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import read_l2, l2_files, feature_cols, swap_ab, KEY_COLS, FAMS, log

PCT_BASE = ['pot_bb', 'pair_contrib', 'tr_any', 'net_gap', 'hu_streets', 'both_check_hu', 'war_streets', 'contrib_gap',
            'A_contrib_bb', 'B_contrib_bb', 'A_net_bb', 'B_net_bb', 'chen_max', 'chen_min', 'A_n_agg', 'B_n_agg',
            'A_stack_bb', 'B_stack_bb', 'A_fold_margin', 'B_fold_margin', 'surp_max_pair', 'surpw_max_pair', 'rsurp_max_pair',
            'A_surp_max', 'B_surp_max', 'eq_flop_abs', 'folder_eq_vs_partner', 'A_max_amt_pot', 'B_max_amt_pot',
            'pf_eq_abs', 'pf_fold_ev_loss', 'fold_ev_loss_max', 'A_fold_ev_loss', 'B_fold_ev_loss', 'A_eq_fold_any', 'B_eq_fold_any',
            'A_fold_pot_odds', 'B_fold_pot_odds', 'call_regret_max', 'call_regret_sum', 'call_edge_min', 'pass_val_max', 'pass_val_sum',
            'nll_max_pair', 'nll_sum_pair', 'gain_max_pair', 'gain_sum_pair', 'excess_max_pair', 'excess_sum_pair']


def add_pair_pct(d):
    cols = [c for c in PCT_BASE if c in d.columns]
    return d.with_columns([(pl.col(c).rank(method='average').over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias(f'q_{c}') for c in cols])


def load_rows(phase, pair_filter, use_equity, use_surprise, onset=0, out=None):
    parts = []
    for f in l2_files(phase):
        d = read_l2(f, phase, use_equity, use_surprise)
        if pair_filter is not None:
            d = d.join(pair_filter, on='pair_id', how='inner')
        if d.height:
            parts.append(add_pair_pct(d))
    d = pl.concat(parts)

    # NOTE: Transformer features disabled — proven to hurt (0.5283 < baseline 0.5421, 2026-09-11)
    # tf_path = Path('5_outputs/action_full_0911/transformer_unified.parquet')
    # if phase == 'development' and tf_path.exists():
    #     import numpy as np
    #     tf_df = pl.read_parquet(tf_path).unique(subset=['pair_id', 'hand_id'])
    #     d = d.join(tf_df, on=['pair_id', 'hand_id'], how='left')
    #     for i in range(256):
    #         d = d.with_columns(pl.col(f"tf_emb_{i}").fill_null(0.0))
    #     d = d.with_columns(pl.col("transformer_score").fill_null(-10.0))

    if onset:
        # The label `is_ev` is TIME-CONTAMINATED: it marks the 5 EARLIEST manipulated hands, so later
        # manipulated hands of the same episode are labelled 0 while looking identical on every feature
        # the ranker has (measured 2026-09-09: their feature means match true evidence at ratio 1.04, and
        # they differ only in time - quantile 0.608 vs 0.361). Without an "how much came before" feature
        # that boundary is not learnable at stage 1; the re-ranker was the only stage that could see it.
        from tpds_rerank import onset_cached
        d = d.join(onset_cached(out, phase, norm_only=(onset == 3), pair_filter=pair_filter), on=['pair_id', 'hand_id'], how='left')
    return d


RANK_PARAMS = dict(objective='lambdarank', metric='map', eval_at=[5], lambdarank_truncation_level=15,
                   learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.6,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=8, label_gain=[0, 1])
GRADED_GAIN = [0, 1, 3, 7, 15, 31]
# Objective-diversity variant. LambdaRank optimises the within-pair ORDER; plain log-loss optimises a
# calibrated per-hand probability. On the in-window contrast (evidence vs the hands the earliest-5
# mechanism certifies as clean) the two disagree enough to be worth fusing: measured 2026-09-10,
# rank-blending the two lifts in-window AP 0.8122 -> 0.8240 while neither model alone exceeds 0.8122.
BIN_PARAMS = dict(objective='binary', metric='average_precision', learning_rate=0.05, num_leaves=63,
                  min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, n_jobs=8)


def map5(pair_ids, y, s):
    df = pd.DataFrame({'p': pair_ids, 'y': y, 's': s})
    out = []
    for _, g in df.groupby('p', sort=False):
        rel = g.sort_values('s', ascending=False, kind='mergesort').y.to_numpy()[:5]
        nrel = min(int(g.y.sum()), 5)
        if nrel == 0:
            continue
        hits = np.cumsum(rel)
        out.append(float((hits / np.arange(1, len(rel) + 1) * rel).sum() / nrel))
    return float(np.mean(out)) if out else 0.0


def add_hand_scores(d, out, phase):
    """Stack the hand-model scores (OOF for dev) as ranker features, plus their within-pair percentile."""
    f = out / ('hand_scores_dev.parquet' if phase == 'development' else 'hand_scores_eval.parquet')
    if not f.exists():
        return d
    hs = pl.read_parquet(f)
    d = d.join(hs, on=['pair_id', 'hand_id'], how='left')
    sc = [c for c in hs.columns if c.startswith('s_')]
    d = d.with_columns([pl.col(c).fill_null(0.0) for c in sc])
    return d.with_columns([(pl.col(c).rank(method='average').over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias(f'q_{c}') for c in sc])


def main(run, use_equity=2, use_surprise=2, rounds=600, seeds=2, stack=1, lr=0.05, leaves=63, per_family=0, tag='', trunc=15, ff=0.6, mcs=20, graded=0, onset=0, binary=0, famonly='', seed0=42):
    # famonly='<substr>:<family>[,<substr>:<family>...]' (per_family=1 only; default '' = unchanged): columns whose name contains
    # <substr> are ZEROED in the training matrix of every family ranker EXCEPT <family>. A constant column has a single bin, so
    # LightGBM never splits on it and the saved model ignores that column at apply/gauge time - no downstream change needed.
    # Purpose (擱置清單 2026-09-12 rescan): family-specific signal only in that family's branch, e.g. nbr_:coordinated_isolation.
    fam_mask = [tuple(kv.split(':')) for kv in str(famonly).split(',') if kv]
    for _sub, _fam in fam_mask:
        assert _fam in FAMS, f'famonly: unknown family {_fam!r}'
    if fam_mask and not per_family:
        raise SystemExit('famonly requires per_family=1')
    out = ROOT / '5_outputs/models' / run
    folds = json.load(open(out / 'folds_by_table.json'))
    lab = pl.read_csv(RAW / 'development_labels.csv')
    pos = lab.filter(pl.col('label') == 1).select(['pair_id', 'behavior_family'])
    evd = pl.read_csv(RAW / 'development_evidence.csv')
    ev = evd.select(['pair_id', 'hand_id', 'evidence_rank']).with_columns(pl.lit(1).cast(pl.Int8).alias('is_ev'))
    log('loading positive-pair rows')
    d = load_rows('development', pos.select('pair_id'), use_equity, use_surprise, onset, out)
    d = d.join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('is_ev').fill_null(0), pl.col('evidence_rank').fill_null(99)).join(pos, on='pair_id')
    if stack:
        d = add_hand_scores(d, out, 'development')
    log(f"rows {d.height:,}, evidence {int(d['is_ev'].sum())}, pairs {d['pair_id'].n_unique()}")
    feats = [c for c in feature_cols(d.drop(['is_ev', 'evidence_rank'])) if c not in ('is_ev', 'evidence_rank')]
    fam_true = d['behavior_family'].to_numpy()
    # family one-hot (true family at train time; predicted family at inference)
    for i, f in enumerate(FAMS):
        d = d.with_columns(pl.lit(0.0).alias(f'fam_{i}'))
    feats_all = feats + [f'fam_{i}' for i in range(3)]
    Xbase = d.select(feats).to_numpy().astype(np.float32)
    Xs_base = swap_ab(d).select(feats).to_numpy().astype(np.float32)
    onehot = np.zeros((d.height, 3), np.float32)
    for i, f in enumerate(FAMS):
        onehot[:, i] = (fam_true == f).astype(np.float32)
    X = np.hstack([Xbase, onehot]); Xs = np.hstack([Xs_base, onehot])
    y = d['is_ev'].to_numpy().astype(int)
    if graded:
        er = d['evidence_rank'].to_numpy()
        y = np.where(er <= 5, 6 - er, 0).astype(int)   # rank1 -> 5 ... rank5 -> 1, non-evidence -> 0
        log(f"graded relevance: {np.bincount(y)}")
    pairs = d['pair_id'].to_numpy(); tables = d['table_id'].to_numpy()
    fold_id = np.array([folds[t] for t in tables])
    oof = np.zeros(d.height, np.float32)
    models = []
    fam_sel = np.ones(d.height, bool)
    fam_list = [None] if not per_family else FAMS
    for fam_only in fam_list:
      zero_cols = []
      if fam_only is not None:
        fam_sel = fam_true == fam_only
        log(f"--- per-family ranker: {fam_only} ({fam_sel.sum():,} rows, {len(set(pairs[fam_sel]))} pairs)")
        zero_cols = [j for j, f in enumerate(feats_all) if any(sub in f and fam != fam_only for sub, fam in fam_mask)]
        if zero_cols:
            log(f"    famonly: zeroing {len(zero_cols)} columns for {fam_only}: {[feats_all[j] for j in zero_cols]}")
      for fo in range(5):
          tr_mask = fold_id != fo; va_mask = fold_id == fo
          tr_idx = np.flatnonzero(tr_mask & fam_sel); tr_idx = tr_idx[np.argsort(pairs[tr_idx], kind='mergesort')]
          va_idx = np.flatnonzero(va_mask & fam_sel); va_idx = va_idx[np.argsort(pairs[va_idx], kind='mergesort')]
          # augment training with the A/B swapped copy (same groups, appended)
          gtr = pd.Series(pairs[tr_idx]).value_counts(sort=False)
          gtr = pd.Series(pairs[tr_idx]).groupby(pd.Series(pairs[tr_idx]), sort=False).size().to_numpy()
          gva = pd.Series(pairs[va_idx]).groupby(pd.Series(pairs[va_idx]), sort=False).size().to_numpy()
          Xtr = np.vstack([X[tr_idx], Xs[tr_idx]]); ytr = np.concatenate([y[tr_idx], y[tr_idx]]); gtr2 = np.concatenate([gtr, gtr])
          if zero_cols:
              Xtr[:, zero_cols] = 0.0
          preds = []
          for s in range(seeds):
              if binary:
                  P = dict(BIN_PARAMS, seed=seed0 + s, learning_rate=lr, num_leaves=leaves, feature_fraction=ff, min_child_samples=mcs)
                  ds = lgb.Dataset(Xtr, (ytr > 0).astype(int), feature_name=feats_all)
              else:
                  P = dict(RANK_PARAMS, seed=seed0 + s, learning_rate=lr, num_leaves=leaves, lambdarank_truncation_level=trunc, feature_fraction=ff, min_child_samples=mcs, **({'label_gain': GRADED_GAIN} if graded else {}))
                  ds = lgb.Dataset(Xtr, ytr, group=gtr2, feature_name=feats_all)
              m = lgb.train(P, ds, num_boost_round=rounds)
              preds.append(0.5 * (m.predict(X[va_idx]) + m.predict(Xs[va_idx])))
              models.append(m)
          oof[va_idx] = np.mean(preds, axis=0)
          log(f"  fold {fo}: MAP@5 {map5(pairs[va_idx], (y[va_idx] > 0).astype(int), oof[va_idx]):.4f}")
    ybin = d['is_ev'].to_numpy().astype(int)
    log(f"OOF Evidence MAP@5 (positive pairs): {map5(pairs, ybin, oof):.4f}")
    for f in FAMS:
        m_ = fam_true == f
        log(f"   {f:24s} {map5(pairs[m_], ybin[m_], oof[m_]):.4f}")
    suffix = f'_{tag}' if tag else ''
    d.select(['pair_id', 'hand_id']).with_columns(pl.Series('ev_score', oof)).write_parquet(out / f'evidence_dev{suffix}.parquet')
    for i, m in enumerate(models):
        m.save_model(str(out / f'evrank{suffix}_{i}.txt'))
    json.dump(feats, open(out / f'evidence_features{suffix}.json', 'w'))
    import os as _os
    json.dump({'stack': int(stack), 'per_family': int(per_family), 'binary': int(binary), 'famonly': str(famonly), 'seed0': int(seed0), 'seeds': int(seeds),
               'rounds': int(rounds), 'lr': lr, 'leaves': int(leaves), 'onset': int(onset), 'use_equity': int(use_equity), 'use_surprise': int(use_surprise),
               'n_feats': len(feats), 'env': {k: v for k, v in _os.environ.items() if k.startswith('TPDS_')}}, open(out / f'evidence_cfg{suffix}.json', 'w'))
    log('done training; use tpds_evidence_apply.py to score evaluation rows')


if __name__ == '__main__':
    run = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        kw[k] = v if k in ('tag', 'famonly') else (float(v) if '.' in v else int(v))
    main(run, **kw)
