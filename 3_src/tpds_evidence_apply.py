"""Score evaluation-phase rows with the trained evidence ranker and rebuild a submission.
Usage: tpds_evidence_apply.py <run> <tag> [use_equity=2] [use_surprise=2] [active_frac=0.01] [risk_from=<tag or ''>]
"""
import sys, json, glob, time
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import read_l2, l2_files, swap_ab, FAMS, family_from_pf, make_submission_frame, dev_solution, rank_to_unit, log
from tpds_evidence import add_pair_pct, add_hand_scores
from metric import score as kaggle_score


def score_phase(phase, models, feats, fam_map, use_equity, use_surprise, out=None, stack=1, per_family=0, onset=0):
    """fam_map: DataFrame(pair_id, fam_idx). Returns per-row ev_score.

    `onset` MUST match the flag the ranker was trained with: the feature list is taken from the saved
    model, so a mismatch is not a silent degradation but a hard ColumnNotFoundError at predict time.
    """
    parts = []
    ons = None
    if onset:
        from tpds_rerank import onset_cached
        ons = onset_cached(out, phase, norm_only=(onset == 3))
    t0 = time.time(); files = l2_files(phase)
    for i, f in enumerate(files):
        d = read_l2(f, phase, use_equity, use_surprise)
        if d.height == 0:
            continue
        d = add_pair_pct(d).join(fam_map, on='pair_id', how='inner')
        if ons is not None:
            d = d.join(ons, on=['pair_id', 'hand_id'], how='left')
        if stack:
            d = add_hand_scores(d, out, phase)
        if d.height == 0:
            continue
        X = d.select(feats).to_numpy().astype(np.float32); Xs = swap_ab(d).select(feats).to_numpy().astype(np.float32)
        oh = np.zeros((d.height, 3), np.float32)
        fi = d['fam_idx'].to_numpy()
        oh[np.arange(d.height), fi] = 1.0
        X = np.hstack([X, oh]); Xs = np.hstack([Xs, oh])
        if per_family:
            s = np.zeros(d.height, np.float64)
            per = len(models) // 3
            for fi in range(3):
                sel = np.flatnonzero(d['fam_idx'].to_numpy() == fi)
                if len(sel) == 0:
                    continue
                mods = models[fi * per:(fi + 1) * per]
                s[sel] = np.mean([0.5 * (m.predict(X[sel]) + m.predict(Xs[sel])) for m in mods], axis=0)
        else:
            s = np.mean([0.5 * (m.predict(X) + m.predict(Xs)) for m in models], axis=0)
        parts.append(d.select(['pair_id', 'hand_id']).with_columns(pl.Series('ev_score', s.astype(np.float32))))
        if i % 100 == 0:
            log(f"  ev-score {phase} {i+1}/{len(files)} {time.time()-t0:.0f}s")
    return pl.concat(parts)


def top5(rows):
    t = (rows.sort(['pair_id', 'ev_score'], descending=[False, True]).group_by('pair_id', maintain_order=True)
             .agg(pl.col('hand_id').head(5).alias('ev')))
    return dict(zip(t['pair_id'].to_list(), t['ev'].to_list()))


def main(run, tag, use_equity=2, use_surprise=2, active_frac=0.01, oof_tag='', model_tag='', onset=0):
    out = ROOT / '5_outputs/models' / run
    suffix = f'_{model_tag}' if model_tag else ''
    import re as _re
    _rx = _re.compile(r'^evrank' + _re.escape(suffix) + r'_(\d+)\.txt$')   # exact tag only (2026-09-13 fix: prefix glob mixed in evrank_<tag>_<other>_k.txt)
    paths = sorted((p for p in out.glob(f'evrank{suffix}_*.txt') if _rx.match(p.name)), key=lambda p: int(_rx.match(p.name).group(1)))
    models = [lgb.Booster(model_file=str(p)) for p in paths]
    feats = json.load(open(out / f'evidence_features{suffix}.json'))
    cfg = json.load(open(out / f'evidence_cfg{suffix}.json')) if (out / f'evidence_cfg{suffix}.json').exists() else {}
    per_family = int(cfg.get('per_family', 0))
    log(f'per_family={per_family}, {len(models)} boosters')
    log(f"{len(models)} rankers, {len(feats)} base features")
    pf = pl.read_parquet(out / 'pair_features_dev.parquet'); pfe = pl.read_parquet(out / 'pair_features_eval.parquet')
    fam_dev, _ = family_from_pf(pf); fam_ev, _ = family_from_pf(pfe)
    fmap_dev = pl.DataFrame({'pair_id': pf['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_dev], np.int32)})
    fmap_ev = pl.DataFrame({'pair_id': pfe['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_ev], np.int32)})
    # dev check on positive pairs using the OOF evidence scores already saved
    edev = pl.read_parquet(out / f'evidence_dev{suffix}.parquet')
    oof = pd.read_parquet(out / (f'pair_oof_{oof_tag}.parquet' if oof_tag else 'pair_oof_dev.parquet'))
    risk_dev = rank_to_unit(oof.oof.to_numpy())
    sol = dev_solution(pf.select(['pair_id', 'label', 'behavior_family']))
    # combine: evidence from ranker where available (positive pairs), else fall back to the pooled hand score
    hs = pl.read_parquet(out / 'hand_scores_dev.parquet').select(['pair_id', 'hand_id', 's_gen']).rename({'s_gen': 'ev_score'})
    ev_all = pl.concat([edev, hs.join(edev.select('pair_id').unique(), on='pair_id', how='anti')])
    m_dev = top5(ev_all)
    sub_dev = make_submission_frame(pf['pair_id'].to_list(), risk_dev, fam_dev, m_dev, active_frac)
    sc, comp = kaggle_score(sol, sub_dev, 'pair_id', return_components=True)
    log(f"DEV with ranker evidence: score {sc:.4f} | PairAP {comp['pair_ap']:.4f} Evid {comp['evidence_map']:.4f} Behav {comp['behavior_map']:.4f}")
    log('scoring evaluation rows')
    stack = int(cfg.get('stack', 0))
    rows_ev = score_phase('evaluation', models, feats, fmap_ev, use_equity, use_surprise, out, stack, per_family, onset)
    rows_ev.write_parquet(out / f'evidence_eval{suffix}.parquet')
    risk_ev = rank_to_unit(pd.read_parquet(out / (f'pair_pred_{oof_tag}.parquet' if oof_tag else 'pair_pred_eval.parquet')).pred.to_numpy()) if (out / 'pair_pred_eval.parquet').exists() or oof_tag else None
    if risk_ev is None:
        base = pd.read_csv(out / 'submission.csv')
        base = base.set_index('pair_id').loc[pfe['pair_id'].to_list()]
        risk_ev = base.risk_score.to_numpy()
    sub = make_submission_frame(pfe['pair_id'].to_list(), risk_ev, fam_ev, top5(rows_ev), active_frac)
    sample = pd.read_csv(RAW / 'sample_submission.csv')
    sub = sample[['pair_id']].merge(sub, on='pair_id', how='left')
    path = out / f'submission_{tag}.csv'; sub.to_csv(path, index=False)
    log(f"written {path}")


if __name__ == '__main__':
    run, tag = sys.argv[1], sys.argv[2]
    kw = {}
    for a in sys.argv[3:]:
        k, v = a.split('=')
        kw[k] = v if k in ('oof_tag', 'model_tag') else (float(v) if '.' in v else int(v))
    main(run, tag, **kw)
