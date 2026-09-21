"""Second-stage evidence re-ranker.
Diagnosis: the first-stage ranker already puts 89% of planted hands in the top 10 and 98.5% in the top 20;
only the ordering inside that shortlist limits MAP@5. So: retrieve top-K with stage 1, then re-rank the
shortlist with a model trained only on shortlists, with features contrasted WITHIN the shortlist.
Usage: tpds_rerank.py <run> [K=20] [rounds=400] [seeds=3] [lr=0.05] [leaves=31] [ev_tag=pf] [ev2_tag=base] [apply=0]
"""
import sys, json, time, glob
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import read_l2, l2_files, feature_cols, swap_ab, FAMS, log, family_from_pf
from tpds_evidence import add_pair_pct, map5

SHORT_PCT = ['pot_bb', 'pair_contrib', 'tr_any', 'net_gap', 'hu_streets', 'both_check_hu', 'war_streets',
             'surp_max_pair', 'surpw_max_pair', 'rsurp_max_pair', 'A_surp_max', 'B_surp_max', 'A_surp_fold', 'B_surp_fold',
             'eq_flop_abs', 'folder_eq_vs_partner', 'pf_eq_abs', 'fold_ev_loss_max', 'A_fold_ev_loss', 'B_fold_ev_loss',
             'A_eq_fold_any', 'B_eq_fold_any', 'chen_max', 'chen_min', 'A_n_agg', 'B_n_agg', 'A_contrib_bb', 'B_contrib_bb',
             'out_fold_to_pair', 'iso_balance', 'A_fold_pot_odds', 'B_fold_pot_odds']

PARAMS = dict(objective='lambdarank', metric='map', eval_at=[5], lambdarank_truncation_level=20,
              learning_rate=0.05, num_leaves=31, min_child_samples=10, feature_fraction=0.7,
              bagging_fraction=0.9, bagging_freq=1, lambda_l2=3.0, verbose=-1, n_jobs=8, label_gain=[0, 1])


def add_short_contrast(d, prefix='sh_'):
    """rank / z of each feature WITHIN the shortlist of its pair"""
    cols = [c for c in SHORT_PCT if c in d.columns]
    exprs = []
    for c in cols:
        exprs.append((pl.col(c).rank().over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias(f'{prefix}r_{c}'))
        exprs.append(((pl.col(c) - pl.col(c).mean().over('pair_id')) / (pl.col(c).std().over('pair_id') + 1e-6)).cast(pl.Float32).alias(f'{prefix}z_{c}'))
    return d.with_columns(exprs)


def load_stage1(out, phase, ev_tag, ev2_tag, w1=0.65, w2=0.35):
    sfx = f'_{ev_tag}' if ev_tag and ev_tag != 'base' else ''
    sfx2 = f'_{ev2_tag}' if ev2_tag and ev2_tag != 'base' else ''
    kind = 'dev' if phase == 'development' else 'eval'
    a = pl.read_parquet(out / f'evidence_{kind}{sfx}.parquet').rename({'ev_score': 's1a'})
    if ev2_tag:
        b = pl.read_parquet(out / f'evidence_{kind}{sfx2}.parquet').rename({'ev_score': 's1b'})
        a = a.join(b, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('s1b').fill_null(pl.col('s1a')))
    else:
        a = a.with_columns(pl.col('s1a').alias('s1b'))
    a = a.with_columns(
        (pl.col('s1a').rank().over('pair_id') / pl.len().over('pair_id')).alias('r1a'),
        (pl.col('s1b').rank().over('pair_id') / pl.len().over('pair_id')).alias('r1b'))
    return a.with_columns((w1 * pl.col('r1a') + w2 * pl.col('r1b')).alias('s1'))


def build_shortlist(out, phase, pair_filter, K, ev_tag, ev2_tag, extra_time=0, full_time=None, no_time=0, multi_k=(), onset=0):
    s1 = load_stage1(out, phase, ev_tag, ev2_tag)
    short = (s1.sort(['pair_id', 's1'], descending=[False, True]).group_by('pair_id', maintain_order=True)
               .head(K).with_columns(pl.int_range(pl.len()).over('pair_id').alias('s1_pos')))
    parts = []
    for f in l2_files(phase):
        d = read_l2(f, phase, 2, 2)
        if pair_filter is not None:
            d = d.join(pair_filter, on='pair_id', how='inner')
        d = d.join(short.select(['pair_id', 'hand_id', 's1', 's1a', 's1b', 'r1a', 'r1b', 's1_pos']), on=['pair_id', 'hand_id'], how='inner')
        if d.height:
            parts.append(d)
    d = pl.concat(parts)
    hs = pl.read_parquet(out / ('hand_scores_dev.parquet' if phase == 'development' else 'hand_scores_eval.parquet'))
    d = d.join(hs, on=['pair_id', 'hand_id'], how='left')
    for c in [c for c in hs.columns if c.startswith('s_')]:
        d = d.with_columns(pl.col(c).fill_null(0.0))
    d = add_pair_pct(d)
    d = add_short_contrast(d)
    

    # "how early did this hand occur among the pair's most suspicious hands" - episode-onset ordering.
    # Expressed RELATIVE to the shortlist so it does not assume the episode starts at the window's beginning.
    if not no_time:
        d = d.with_columns(
            (pl.col('t_rank').rank().over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias('sh_time_rank'),
            ((pl.col('t_rank') - pl.col('t_rank').min().over('pair_id')) /
             (pl.col('t_rank').max().over('pair_id') - pl.col('t_rank').min().over('pair_id') + 1e-6)).cast(pl.Float32).alias('sh_time_frac'),
        )
        for sub in multi_k:
            # same "how early" signal but measured only among the pair's top-`sub` most suspicious hands:
            # a tighter candidate set is a purer read on episode onset, a looser one is less noisy.
            inb = pl.col('s1_pos') < sub
            d = d.with_columns(
                pl.when(inb)
                  .then(pl.col('t_rank').rank().over('pair_id') / inb.sum().over('pair_id'))
                  .otherwise(None).cast(pl.Float32).alias(f'sh_time_rank{sub}'))
    if onset:
        d = d.join(onset_cached(out, phase, norm_only=(onset == 3), pair_filter=pair_filter), on=['pair_id', 'hand_id'], how='left')
    if onset > 1:
        d = d.join(onset_features(out, phase, pair_filter, src='max', pfx='mx'), on=['pair_id', 'hand_id'], how='left')
    if extra_time:
        # gap to the neighbouring shortlist hands, and how far into the pair's whole shared history it sits
        d = d.sort(['pair_id', 't_rank']).with_columns(
            (pl.col('t_rank').diff().over('pair_id')).cast(pl.Float32).alias('sh_gap_prev'),
            (pl.col('t_rank').diff(-1).over('pair_id') * -1).cast(pl.Float32).alias('sh_gap_next'),
        ).with_columns(
            (pl.col('sh_gap_prev') / (pl.col('sh_gap_prev').median().over('pair_id') + 1e-6)).cast(pl.Float32).alias('sh_gap_prev_rel'),
        )
        if full_time is not None:
            d = d.join(full_time, on=['pair_id', 'hand_id'], how='left')
    return d


def onset_features(out, phase, pair_filter, qs=(0.80, 0.90, 0.95), src='gen', pfx='', norm_only=0):
    """Episode-onset features over the pair's FULL shared history, not just the shortlist.

    The planted evidence is the EARLIEST manipulated hands of an episode, so what identifies it is
    how much suspicious activity *precedes* a hand - a causal quantity `sh_time_rank` cannot see,
    because that one only orders the 20 hands we already picked. Here the whole history votes:
    a hand with little suspicious activity before it sits near the onset even if it is late in
    absolute time, which is exactly why the absolute-position feature (`full_tq`) found nothing.
    """
    hs = pl.read_parquet(out / ('hand_scores_dev.parquet' if phase == 'development' else 'hand_scores_eval.parquet'))
    if pair_filter is not None:
        hs = hs.join(pair_filter, on='pair_id', how='inner')
    tr = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 't_rank'])
    d = hs.join(tr, on='hand_id', how='left').sort(['pair_id', 't_rank'])
    if src == 'max':
        # strongest family signal for this hand - the generic score dilutes a family-specific manipulation
        d = d.with_columns(pl.max_horizontal('s_directed_transfer', 's_soft_play', 's_coordinated_isolation').alias('s_gen'))
    out_cols = ['pair_id', 'hand_id']
    for q in qs:
        t = int(q * 100)
        above = (pl.col('s_gen') > pl.col('s_gen').quantile(q).over('pair_id')).cast(pl.Int32)
        d = d.with_columns(above.alias(f'_a{t}'))
        d = d.with_columns(
            # strictly-earlier hands above the pair's q-quantile of suspicion
            (pl.col(f'_a{t}').cum_sum().over('pair_id') - pl.col(f'_a{t}')).cast(pl.Float32).alias(f'ons_nb{t}'),
            # time of the episode's apparent onset at this threshold
            pl.when(pl.col(f'_a{t}') == 1).then(pl.col('t_rank')).otherwise(None).min().over('pair_id').alias(f'_t0{t}'),
        )
        d = d.with_columns(
            # length-invariant twin of ons_nb: DENSITY of suspicious activity before this hand rather
            # than its raw count, so the feature does not shift when eval pairs are shorter.
            (pl.col(f'ons_nb{t}') / (pl.int_range(pl.len()).over('pair_id') + 1.0)).cast(pl.Float32).alias(f'ons_rb{t}'),
            (pl.col(f'ons_nb{t}') / (pl.col(f'_a{t}').sum().over('pair_id') + 1e-6)).cast(pl.Float32).alias(f'ons_fb{t}'),
            ((pl.col('t_rank') - pl.col(f'_t0{t}')) /
             (pl.col('t_rank').max().over('pair_id') - pl.col(f'_t0{t}') + 1e-6)).cast(pl.Float32).alias(f'ons_dl{t}'),
        )
        # ons_nb is an ABSOLUTE count: eval pairs are shorter (shared median 76 vs 112), so its
        # distribution shifts between the two phases. ons_fb / ons_dl are normalised and should not.
        out_cols += ([] if norm_only else [f'ons_nb{t}']) + [f'ons_rb{t}', f'ons_fb{t}', f'ons_dl{t}']
    d = d.with_columns(
        ((pl.col('s_gen').cum_sum().over('pair_id') - pl.col('s_gen')) /
         (pl.col('s_gen').sum().over('pair_id') + 1e-6)).cast(pl.Float32).alias('ons_cums'),
        (pl.int_range(pl.len()).over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias('ons_tq'),
    )
    d = d.select(out_cols + ['ons_cums', 'ons_tq'])
    if pfx:
        d = d.rename({c: pfx + c for c in d.columns if c.startswith('ons_')})
    return d


def onset_cached(out, phase, norm_only=0, pair_filter=None):
    """Compute the phase's onset features ONCE and reuse them.

    Three consumers need the identical columns (stage-1 ranker, per-family ranker, re-ranker). On the
    evaluation phase each call is a 5.5M-row join + sort + three quantile passes over 112k groups - the
    largest memory spike in the pipeline - so computing it three times is both slow and the most likely
    place to be OOM-killed on a 16 GB machine.
    The dev path filters to 372 positive pairs first, so it stays small and is not cached.
    """
    if pair_filter is not None:
        return onset_features(out, phase, pair_filter, norm_only=norm_only)
    f = out / f"onset_{'dev' if phase == 'development' else 'eval'}{'_n' if norm_only else ''}.parquet"
    if not f.exists():
        log(f'computing onset features for {phase} (cached to {f.name})')
        onset_features(out, phase, None, norm_only=norm_only).write_parquet(f)
    return pl.read_parquet(f)


def full_time_quantile(phase, pair_filter=None):
    parts = []
    for f in l2_files(phase):
        d = pl.read_parquet(f, columns=['pair_id', 'hand_id', 't_rank'])
        if pair_filter is not None:
            d = d.join(pair_filter, on='pair_id', how='inner')
        if d.height:
            parts.append(d)
    d = pl.concat(parts)
    return d.with_columns((pl.col('t_rank').rank().over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias('full_tq')).select(['pair_id', 'hand_id', 'full_tq'])


def main(run, K=20, rounds=400, seeds=3, lr=0.05, leaves=31, ev_tag='pf', ev2_tag='base', apply=0, extra_time=0, use_full_tq=0, no_time=0, tag='', multi_k='', onset=0):
    multi_k = tuple(int(x) for x in str(multi_k).split('-') if x) if multi_k else ()
    out = ROOT / '5_outputs/models' / run
    folds = json.load(open(out / 'folds_by_table.json'))
    lab = pl.read_csv(RAW / 'development_labels.csv')
    pos = lab.filter(pl.col('label') == 1).select(['pair_id', 'behavior_family'])
    ev = pl.read_csv(RAW / 'development_evidence.csv').select(['pair_id', 'hand_id']).with_columns(pl.lit(1).cast(pl.Int8).alias('is_ev'))
    log(f'building dev shortlist K={K}')
    ftq = full_time_quantile('development', pos.select('pair_id')) if use_full_tq else None
    d = build_shortlist(out, 'development', pos.select('pair_id'), K, ev_tag, ev2_tag, extra_time, ftq, no_time, multi_k, onset)
    log(f'no_time={no_time} multi_k={multi_k} onset={onset} tag={tag!r}')
    d = d.join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('is_ev').fill_null(0)).join(pos, on='pair_id')
    d = d.sort(['pair_id', 's1_pos'])
    log(f"shortlist rows {d.height:,}, evidence retained {int(d['is_ev'].sum())}/1817 ({d['is_ev'].sum()/1817:.3f}), pairs {d['pair_id'].n_unique()}")
    feats = [c for c in feature_cols(d.drop(['is_ev'])) if c != 'is_ev']
    fam_true = d['behavior_family'].to_numpy()
    onehot = np.zeros((d.height, 3), np.float32)
    for i, f in enumerate(FAMS):
        onehot[:, i] = (fam_true == f).astype(np.float32)
    feats_all = feats + ['fam_0', 'fam_1', 'fam_2']
    X = np.hstack([d.select(feats).to_numpy().astype(np.float32), onehot])
    Xs = np.hstack([swap_ab(d).select(feats).to_numpy().astype(np.float32), onehot])
    y = d['is_ev'].to_numpy().astype(int)
    pairs = d['pair_id'].to_numpy(); tables = d['table_id'].to_numpy()
    fold_id = np.array([folds[t] for t in tables])
    oof = np.zeros(d.height, np.float32); models = []
    for fo in range(5):
        tr = np.flatnonzero(fold_id != fo); tr = tr[np.argsort(pairs[tr], kind='mergesort')]
        va = np.flatnonzero(fold_id == fo); va = va[np.argsort(pairs[va], kind='mergesort')]
        gtr = pd.Series(pairs[tr]).groupby(pd.Series(pairs[tr]), sort=False).size().to_numpy()
        Xtr = np.vstack([X[tr], Xs[tr]]); ytr = np.concatenate([y[tr], y[tr]]); g2 = np.concatenate([gtr, gtr])
        preds = []
        for s in range(seeds):
            m = lgb.train(dict(PARAMS, seed=42 + s, learning_rate=lr, num_leaves=leaves),
                          lgb.Dataset(Xtr, ytr, group=g2, feature_name=feats_all), num_boost_round=rounds)
            preds.append(0.5 * (m.predict(X[va]) + m.predict(Xs[va]))); models.append(m)
        oof[va] = np.mean(preds, axis=0)
    # stage-1 baseline restricted to the same shortlist (upper bound of what re-ranking can fix)
    s1 = d['s1'].to_numpy()
    log(f"stage-1 within shortlist MAP@5: {map5(pairs, y, s1):.4f}")
    log(f"stage-2 re-ranked   MAP@5: {map5(pairs, y, oof):.4f}")
    for f in FAMS:
        m_ = fam_true == f
        log(f"   {f:24s} stage1 {map5(pairs[m_], y[m_], s1[m_]):.4f} -> stage2 {map5(pairs[m_], y[m_], oof[m_]):.4f}")
    for w in [0.0, 0.2, 0.35, 0.5]:
        blend = (1 - w) * pd.Series(oof).rank().to_numpy() / len(oof) + w * s1
        log(f"   blend stage2 {1-w:.2f} + stage1 {w:.2f}: {map5(pairs, y, blend):.4f}")
    sfx = f'_{tag}' if tag else ''
    d.select(['pair_id', 'hand_id']).with_columns(pl.Series('rr_score', oof), pl.Series('s1', s1)).write_parquet(out / f'rerank_dev{sfx}.parquet')
    for i, m in enumerate(models):
        m.save_model(str(out / f'rerank{sfx}_{i}.txt'))
    json.dump({'feats': feats, 'K': K, 'ev_tag': ev_tag, 'ev2_tag': ev2_tag, 'no_time': no_time}, open(out / f'rerank_cfg{sfx}.json', 'w'))
    if apply:
        log('scoring evaluation shortlist')
        pfe = pl.read_parquet(out / 'pair_features_eval.parquet')
        fam_ev, _ = family_from_pf(pfe)
        fmap = pl.DataFrame({'pair_id': pfe['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_ev], np.int32)})
        ftqe = full_time_quantile('evaluation') if use_full_tq else None
        de = build_shortlist(out, 'evaluation', None, K, ev_tag, ev2_tag, extra_time, ftqe, no_time, multi_k, onset).join(fmap, on='pair_id')
        oh = np.zeros((de.height, 3), np.float32); oh[np.arange(de.height), de['fam_idx'].to_numpy()] = 1.0
        Xe = np.hstack([de.select(feats).to_numpy().astype(np.float32), oh])
        Xes = np.hstack([swap_ab(de).select(feats).to_numpy().astype(np.float32), oh])
        sc = np.mean([0.5 * (m.predict(Xe) + m.predict(Xes)) for m in models], axis=0)
        de.select(['pair_id', 'hand_id', 's1']).with_columns(pl.Series('rr_score', sc.astype(np.float32))).write_parquet(out / f'rerank_eval{sfx}.parquet')
        log(f'written rerank_eval{sfx}.parquet')


if __name__ == '__main__':
    run = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        kw[k] = v if k in ('ev_tag', 'ev2_tag', 'tag', 'multi_k') else (float(v) if '.' in v else int(v))
    main(run, **kw)
