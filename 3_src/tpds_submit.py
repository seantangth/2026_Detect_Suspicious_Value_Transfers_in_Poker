"""Assemble a submission from: pair-stage risk (tpds_pair_stage) + evidence ranker (tpds_evidence_apply) + family.
Also reports the dev-solution score of the same assembly. Usage:
  tpds_submit.py <run> <tag> [risk_tag=ps1] [ev=1] [active_frac=0.01]
"""
import sys, json
from pathlib import Path
import numpy as np, polars as pl, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import FAMS, family_from_pf, make_submission_frame, dev_solution, rank_to_unit, log, PROC
from metric import score as kaggle_score
from tpds_paths import vpath


OC_TIERS = [(5, 400), (4, 700), (3, 1200)]
MIN_OBS = 12   # (z_n5 threshold, rank to insert at)


def boost_other_coordination(pair_ids, risk, phase, min_rank=5000, tiers=None, min_obs=None):
    """Promote pairs that are extreme within their own table but that the family models rank low.
    Rationale: `other_coordination` never appears in the public positive labels, so no supervised model can
    learn it; a within-table outlier test is the only detector that does not depend on the three signatures.
    21.8% of labelled positives carry z_n5>=3 versus 0.13% of confirmed non-targets (likelihood ratio 162).
    Insertion ranks are chosen so that the cost on dev is 0.0000 (essentially every labelled positive already
    ranks above them), which makes this a near-free option on the upside.
    Returns the boosted risk plus the boolean mask of promoted pairs."""
    an = pl.read_parquet(vpath(f'anomaly_{phase}.parquet')).select(['pair_id', 'z_n5'])
    ps = pl.read_parquet(vpath(f'pairstats_{phase}.parquet'))
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select(['pair_id', 'a', 'b'])
    nobs = cand.join(ps.select(['a', 'b', 'ab_n_ev', 'ba_n_ev']), on=['a', 'b'], how='left').select(
        'pair_id', (pl.col('ab_n_ev').fill_null(0) + pl.col('ba_n_ev').fill_null(0)).alias('nobs'))
    j = pl.DataFrame({'pair_id': pair_ids}).join(an, on='pair_id', how='left').join(nobs, on='pair_id', how='left')
    z = j['z_n5'].fill_null(0).to_numpy(); nob = j['nobs'].fill_null(0).to_numpy()
    n = len(risk)
    order = pd.Series(risk).rank(ascending=False).to_numpy()
    out = risk.copy(); promoted = np.zeros(n, bool)
    # MIN_OBS guards against short evaluation pairs producing extreme z by small-sample noise:
    # eval pairs carry a median of 18 response observations versus 37 in development, and all 81 labelled
    # positives with z_n5>=3 have >=12, so the filter costs nothing on the positive side.
    for thr, target in (tiers or OC_TIERS):
        m = (z >= thr) & (nob >= (MIN_OBS if min_obs is None else min_obs)) & (order > min_rank) & ~promoted
        if m.sum():
            # keep the model's relative order inside a tier so no ties are introduced
            jitter = (1.0 - pd.Series(np.where(m, order, np.nan)).rank(pct=True).fillna(0).to_numpy()) * (200.0 / n)
            out[m] = np.maximum(out[m], 1.0 - target / n + jitter[m])
            promoted |= m
    return out, promoted


def apply_rerank(rows, rr, w_rr=1.0):
    """Replace the shortlist's ordering with the stage-2 re-ranker score, keeping non-shortlist hands below it."""
    rr = rr.with_columns((pl.col('rr_score').rank().over('pair_id') / pl.len().over('pair_id')).alias('rr_rank'))
    d = rows.join(rr.select(['pair_id', 'hand_id', 'rr_rank']), on=['pair_id', 'hand_id'], how='left')
    # shortlist rows get 1 + blended rank (always above any non-shortlist row, whose score stays in [0,1])
    return d.with_columns(
        pl.when(pl.col('rr_rank').is_not_null())
          .then(1.0 + w_rr * pl.col('rr_rank') + (1.0 - w_rr) * pl.col('ev_score'))
          .otherwise(pl.col('ev_score')).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score'])


def top5_map(rows, score_col='ev_score'):
    t = (rows.sort(['pair_id', score_col], descending=[False, True]).group_by('pair_id', maintain_order=True)
             .agg(pl.col('hand_id').head(5).alias('ev')))
    return dict(zip(t['pair_id'].to_list(), t['ev'].to_list()))


def blend_rows(rank_rows, hand_rows, fam_map, w_rank=0.6, w_fam=0.4, w_gen=0.0, rank2_rows=None, w_rank2=0.0, rank3_rows=None, w_rank3=0.0):
    """Per-pair rank-normalised blend of the LambdaRank score and the family/generic hand-model scores."""
    h = hand_rows.join(fam_map, on='pair_id')
    h = h.with_columns(pl.when(pl.col('fam_idx') == 0).then(pl.col('s_directed_transfer'))
                         .when(pl.col('fam_idx') == 1).then(pl.col('s_soft_play'))
                         .otherwise(pl.col('s_coordinated_isolation')).alias('fam_score'))
    d = h.join(rank_rows, on=['pair_id', 'hand_id'], how='left')
    d = d.with_columns(pl.col('ev_score').fill_null(pl.col('s_gen')))
    if rank2_rows is not None:
        d = d.join(rank2_rows.rename({'ev_score': 'ev_score2'}), on=['pair_id', 'hand_id'], how='left')
        d = d.with_columns(pl.col('ev_score2').fill_null(pl.col('s_gen')))
    else:
        d = d.with_columns(pl.col('ev_score').alias('ev_score2'))
    if rank3_rows is not None:
        d = d.join(rank3_rows.rename({'ev_score': 'ev_score3'}), on=['pair_id', 'hand_id'], how='left')
        d = d.with_columns(pl.col('ev_score3').fill_null(pl.col('s_gen')))
    else:
        d = d.with_columns(pl.col('ev_score').alias('ev_score3'))
    d = d.with_columns(
        (pl.col('ev_score').rank().over('pair_id') / pl.len().over('pair_id')).alias('r_rank'),
        (pl.col('ev_score2').rank().over('pair_id') / pl.len().over('pair_id')).alias('r_rank2'),
        (pl.col('ev_score3').rank().over('pair_id') / pl.len().over('pair_id')).alias('r_rank3'),
        (pl.col('fam_score').rank().over('pair_id') / pl.len().over('pair_id')).alias('r_fam'),
        (pl.col('s_gen').rank().over('pair_id') / pl.len().over('pair_id')).alias('r_gen'))
    return d.with_columns((w_rank * pl.col('r_rank') + w_rank2 * pl.col('r_rank2') + w_rank3 * pl.col('r_rank3') + w_fam * pl.col('r_fam') + w_gen * pl.col('r_gen')).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score'])


def main(run, tag, risk_tag='', ev=1, active_frac=0.01, w_rank=0.6, w_fam=0.4, w_gen=0.0, ev_tag='', ev2_tag='', w_rank2=0.0, ev3_tag='', w_rank3=0.0, rerank=0, w_rr=1.0, dir_k=0, oc_boost=0, oc_min_rank=5000, oc_tiers='', oc_min_obs=0):
    # oc_tiers: 'z:rank-z:rank' e.g. '5:2000' -> [(5, 2000)]; empty keeps the module default
    tiers = [tuple(int(x) for x in t.split(':')) for t in str(oc_tiers).split('-') if t] or None
    min_obs = int(oc_min_obs) or None
    out = ROOT / '5_outputs/models' / run
    pf = pl.read_parquet(out / 'pair_features_dev.parquet'); pfe = pl.read_parquet(out / 'pair_features_eval.parquet')
    fam_dev, _ = family_from_pf(pf); fam_ev, _ = family_from_pf(pfe)
    oof = pd.read_parquet(out / (f'pair_oof_{risk_tag}.parquet' if risk_tag else 'pair_oof_dev.parquet'))
    oof = oof.set_index('pair_id').loc[pf['pair_id'].to_list()]
    risk_dev = rank_to_unit(oof.oof.to_numpy())
    # --- dev evidence: ranker for positive pairs (OOF), pooled hand score elsewhere
    hs = pl.read_parquet(out / 'hand_scores_dev.parquet')
    fam_idx = {p: FAMS.index(f) for p, f in zip(pf['pair_id'].to_list(), fam_dev)}
    hs = hs.with_columns(pl.col('pair_id').replace_strict(fam_idx, default=0).alias('fi'))
    hs = hs.with_columns(pl.when(pl.col('fi') == 0).then(pl.col('s_directed_transfer'))
                           .when(pl.col('fi') == 1).then(pl.col('s_soft_play'))
                           .otherwise(pl.col('s_coordinated_isolation')).alias('fam_score'))
    hs = hs.with_columns((pl.col('s_gen') + pl.col('fam_score')).alias('ev_score'))
    fmap_dev = pl.DataFrame({'pair_id': pf['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_dev], np.int32)})
    # the presence test must look at the ranker actually requested (ev_tag), not at the untagged file:
    # a run that only trained tagged rankers (v5nb) used to fall back to pooled hand scores on dev and then
    # crash on eval with an undefined `sfx` (2026-09-10 14:01).
    sfx = f'_{ev_tag}' if ev_tag and ev_tag != 'base' else ''
    sfx2 = f'_{ev2_tag}' if ev2_tag and ev2_tag != 'base' else ''
    if ev and (out / f'evidence_dev{sfx}.parquet').exists():
        edev = pl.read_parquet(out / f'evidence_dev{sfx}.parquet')
        edev2 = pl.read_parquet(out / f'evidence_dev{sfx2}.parquet') if ev2_tag else None
        sfx3 = f'_{ev3_tag}' if ev3_tag and ev3_tag != 'base' else ''
        edev3 = pl.read_parquet(out / f'evidence_dev{sfx3}.parquet') if ev3_tag else None
        log(f'evidence sources: primary=evidence_dev{sfx}.parquet w={w_rank}, secondary=' + (f'evidence_dev{sfx2}.parquet w={w_rank2}' if ev2_tag else 'none')
            + (f', tertiary=evidence_dev{sfx3}.parquet w={w_rank3}' if ev3_tag else ''))
        dev_rows = blend_rows(edev, pl.read_parquet(out / 'hand_scores_dev.parquet'), fmap_dev, w_rank, w_fam, w_gen, edev2, w_rank2, edev3, w_rank3)
        if rerank and (out / 'rerank_dev.parquet').exists():
            dev_rows = apply_rerank(dev_rows, pl.read_parquet(out / 'rerank_dev.parquet'), w_rr)
            log('dev evidence: stage-2 re-ranker applied to the shortlist')
        if dir_k:
            # directed_transfer direction invariant (tpds_direction.py): applied to pairs PREDICTED
            # as DT, direction inferred from the model's own top-k, so it is label-free at eval time.
            from tpds_direction import apply_constraint
            dev_rows = apply_constraint(dev_rows, 'development',
                                        [p for p, f in zip(pf['pair_id'].to_list(), fam_dev) if f == 'directed_transfer'], K=int(dir_k))
    else:
        dev_rows = hs.select(['pair_id', 'hand_id', 'ev_score'])
    sol = dev_solution(pf.select(['pair_id', 'label', 'behavior_family']))
    if oc_boost:
        risk_dev, prom_dev = boost_other_coordination(pf['pair_id'].to_list(), risk_dev, 'development', oc_min_rank, tiers, min_obs)
        fam_dev = np.where(prom_dev, 'other_coordination', fam_dev)
        log(f'oc_boost: promoted {prom_dev.sum()} development pairs (cost measured below)')
    best = None
    for af in [0.003, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0]:
        sub_dev = make_submission_frame(pf['pair_id'].to_list(), risk_dev, fam_dev, top5_map(dev_rows), af)
        sc, comp = kaggle_score(sol, sub_dev, 'pair_id', return_components=True)
        log(f"  active_frac {af}: score {sc:.4f} (Pair {comp['pair_ap']:.4f} Evid {comp['evidence_map']:.4f} Behav {comp['behavior_map']:.4f})")
        if best is None or sc > best[0]:
            best = (sc, af, comp)
    log(f"DEV best: score {best[0]:.4f} at active_frac {best[1]} | {best[2]['pair_ap']:.4f}/{best[2]['evidence_map']:.4f}/{best[2]['behavior_map']:.4f}")
    af = active_frac if active_frac > 0 else best[1]
    # --- eval side
    pred_f = out / (f'pair_pred_{risk_tag}.parquet' if risk_tag else 'pair_pred_eval.parquet')
    if pred_f.exists():
        pr = pd.read_parquet(pred_f).set_index('pair_id').loc[pfe['pair_id'].to_list()]
        risk_ev = rank_to_unit(pr.pred.to_numpy())
    else:
        base = pd.read_csv(out / 'submission.csv').set_index('pair_id').loc[pfe['pair_id'].to_list()]
        risk_ev = base.risk_score.to_numpy()
    fmap_ev = pl.DataFrame({'pair_id': pfe['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_ev], np.int32)})
    if ev and (out / f'evidence_eval{sfx}.parquet').exists():
        eev2 = pl.read_parquet(out / f'evidence_eval{sfx2}.parquet') if ev2_tag else None
        eev3 = pl.read_parquet(out / f'evidence_eval{sfx3}.parquet') if ev3_tag else None
        ev_rows = blend_rows(pl.read_parquet(out / f'evidence_eval{sfx}.parquet'), pl.read_parquet(out / 'hand_scores_eval.parquet'), fmap_ev, w_rank, w_fam, w_gen, eev2, w_rank2, eev3, w_rank3)
        if dir_k:
            from tpds_direction import apply_constraint
            ev_rows = apply_constraint(ev_rows, 'evaluation',
                                       [p for p, f in zip(pfe['pair_id'].to_list(), fam_ev) if f == 'directed_transfer'], K=int(dir_k))
        if rerank and (out / 'rerank_eval.parquet').exists():
            ev_rows = apply_rerank(ev_rows, pl.read_parquet(out / 'rerank_eval.parquet'), w_rr)
            log('eval evidence: stage-2 re-ranker applied to the shortlist')
    else:
        h2 = pl.read_parquet(out / 'hand_scores_eval.parquet')
        fam_idx2 = {p: FAMS.index(f) for p, f in zip(pfe['pair_id'].to_list(), fam_ev)}
        h2 = h2.with_columns(pl.col('pair_id').replace_strict(fam_idx2, default=0).alias('fi'))
        h2 = h2.with_columns(pl.when(pl.col('fi') == 0).then(pl.col('s_directed_transfer'))
                               .when(pl.col('fi') == 1).then(pl.col('s_soft_play'))
                               .otherwise(pl.col('s_coordinated_isolation')).alias('fam_score'))
        ev_rows = h2.with_columns((pl.col('s_gen') + pl.col('fam_score')).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score'])
    if oc_boost:
        risk_ev, prom_ev = boost_other_coordination(pfe['pair_id'].to_list(), risk_ev, 'evaluation', oc_min_rank, tiers, min_obs)
        fam_ev = np.where(prom_ev, 'other_coordination', fam_ev)
        log(f'oc_boost: promoted {prom_ev.sum()} evaluation pairs to other_coordination')
    sub = make_submission_frame(pfe['pair_id'].to_list(), risk_ev, fam_ev, top5_map(ev_rows), af)
    sample = pd.read_csv(RAW / 'sample_submission.csv')
    sub = sample[['pair_id']].merge(sub, on='pair_id', how='left')
    assert sub.isna().sum().sum() == 0
    path = out / f'submission_{tag}.csv'; sub.to_csv(path, index=False)
    log(f"written {path} (active_frac={af})")
    return best


if __name__ == '__main__':
    run, tag = sys.argv[1], sys.argv[2]
    kw = {}
    for a in sys.argv[3:]:
        k, v = a.split('=')
        kw[k] = v if k in ('risk_tag', 'ev_tag', 'ev2_tag', 'ev3_tag', 'oc_tiers') else (float(v) if '.' in v else int(v))
    main(run, tag, **kw)
