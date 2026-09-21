"""Fast pair-stage iteration: load a run's pooled pair features + hand scores, train pair model variants, score dev-solution, write submission.
Usage: tpds_pair_stage.py <run> <out_tag> [w_unknown=0.5] [pu_stage2=1] [rn_drop=0.02] [seeds=3] [rounds=600] [active_frac=0.5] [leaves=31]
"""
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import (PROC, RAW, FAMS, family_from_pf, pick_evidence, make_submission_frame, dev_solution, rank_to_unit, log, PAIR_PARAMS)
from metric import score as kaggle_score
from tpds_paths import vpath, eqpath
from sklearn.metrics import average_precision_score, roc_auc_score


def train_eval(pf, folds_by_table, w_unknown, rounds, seeds, leaves, drop_mask=None, extra_pos_mask=None, suspect_w=0.0, len_w=None):
    feats = [c for c in pf.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')]
    X = pf.select(feats).to_numpy().astype(np.float32)
    lab = pf['label'].to_numpy()
    y = (lab == 1).astype(int)
    w = np.where(lab == 1, 1.0, np.where(lab == 0, 1.0, w_unknown))
    if drop_mask is not None:
        w = np.where(drop_mask, suspect_w, w)
    if extra_pos_mask is not None:
        y = np.where(extra_pos_mask, 1, y); w = np.where(extra_pos_mask, 0.5, w)
    if len_w is not None:
        w = w * len_w
    fold_id = np.array([folds_by_table[t] for t in pf['table_id'].to_list()])
    oof = np.zeros(len(y)); models = []
    for fo in range(5):
        tr = (fold_id != fo) & (w > 0); va = fold_id == fo
        preds = []; fm = []
        for s in range(seeds):
            params = dict(PAIR_PARAMS, seed=42 + s, num_leaves=leaves)
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats), num_boost_round=rounds)
            preds.append(m.predict(X[va])); fm.append(m)
        oof[va] = np.mean(preds, axis=0); models.append(fm)
    return models, oof, feats


def add_anomaly(pf, phase):
    """Within-table robust z-scores of the Marginal-Impact statistics (tpds_anomaly.py)."""
    path = vpath(f'anomaly_{phase}.parquet')
    if not path.exists():
        return pf
    an = pl.read_parquet(path).drop(['a', 'b'])
    d = pf.join(an, on='pair_id', how='left')
    return d.with_columns([pl.col(c).fill_null(0.0) for c in an.columns if c != 'pair_id'])


def add_betsize(pf, phase, keep_n=0):
    """Opponent-conditional bet-size residuals (tpds_betsize.py), keyed (a, b).
    `bs_n_min` is deliberately excluded by default: it counts how often the two were aggressive in the
    same pot, correlates 0.78 with the existing `sum_hu`, and is a co-occurrence proxy rather than a
    behavioural signal - exactly the accidental-colluder confound task B7 warns about."""
    path = PROC / f'betsize_{phase}.parquet'   # deliberately NOT variant-suffixed (see tpds_paths)
    if not path.exists():
        return pf
    bs = pl.read_parquet(path)
    if not keep_n:
        bs = bs.drop('bs_n_min')
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select(['pair_id', 'a', 'b'])
    bs = cand.join(bs, on=['a', 'b'], how='left').drop(['a', 'b'])
    d = pf.join(bs, on='pair_id', how='left')
    return d.with_columns([pl.col(c).fill_null(0.0) for c in bs.columns if c != 'pair_id'])


def add_callvalue(pf, phase):
    """Pair-level call regret vs the partner (tpds_callvalue.py): chips a partner knowingly gave up by calling
    against the partner's ACTUAL hand, split by direction, plus bad-call rates. Keyed (a, b), fixed path
    (derived from equity tables only, so world-independent)."""
    path = eqpath(f'callvalue_pair_{phase}.parquet')
    if not path.exists():
        raise FileNotFoundError(path)
    cv = pl.read_parquet(path)
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select(['pair_id', 'a', 'b'])
    cv = cand.join(cv, on=['a', 'b'], how='left').drop(['a', 'b'])
    d = pf.join(cv, on='pair_id', how='left')
    return d.with_columns([pl.col(c).fill_null(0.0) for c in cv.columns if c != 'pair_id'])


def add_graph(pf, phase, tag=''):
    """Within-table graph features (tpds_graph.py). These are derived from the pair model's OWN score,
    so their single-feature AUC is self-referential and meaningless; only the incremental effect on the
    held-out rulers (known-positive ranks) counts."""
    path = PROC / f'graph_{phase}{tag}.parquet'
    if not path.exists():
        return pf
    g = pl.read_parquet(path)
    d = pf.join(g, on='pair_id', how='left')
    return d.with_columns([pl.col(c).fill_null(0.0) for c in g.columns if c != 'pair_id'])


def add_pairstats(pf, phase):
    """Join the opponent-conditional Marginal Impact profile (tpds_pairstats.py) onto pooled pair features."""
    path = vpath(f'pairstats_{phase}.parquet')
    if not path.exists():
        return pf
    ps = pl.read_parquet(path).drop('shared')
    cand = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select(['pair_id', 'a', 'b'])
    ps = cand.join(ps, on=['a', 'b'], how='left').drop(['a', 'b'])
    d = pf.join(ps, on='pair_id', how='left')
    return d.with_columns([pl.col(c).fill_null(0.0) for c in ps.columns if c != 'pair_id'])


def main(run, tag, w_unknown=0.5, pu_stage2=1, rn_drop=0.02, seeds=3, rounds=600, active_frac=0.5, leaves=31, pseudo_top=0, use_pairstats=1, drop_suspects=0, suspect_w=0.0, match_len=0, use_anomaly=0, use_graph=0, graph_tag='', use_betsize=0, use_callvalue=0):
    out = ROOT / '5_outputs/models' / run
    pf = pl.read_parquet(out / 'pair_features_dev.parquet'); pfe = pl.read_parquet(out / 'pair_features_eval.parquet')
    if use_pairstats:
        pf = add_pairstats(pf, 'development'); pfe = add_pairstats(pfe, 'evaluation')
        log(f"pairstats joined: {pf.width} pair features")
    if use_betsize:
        pf = add_betsize(pf, 'development', keep_n=(use_betsize > 1)); pfe = add_betsize(pfe, 'evaluation', keep_n=(use_betsize > 1))
        log(f"bet-size residuals joined: {pf.width} pair features")
    if use_callvalue:
        pf = add_callvalue(pf, 'development'); pfe = add_callvalue(pfe, 'evaluation')
        log(f"pair-level call regret joined: {pf.width} pair features")
    if use_graph:
        pf = add_graph(pf, 'development', graph_tag); pfe = add_graph(pfe, 'evaluation', graph_tag)
        log(f"within-table graph joined: {pf.width} pair features")
    if use_anomaly:
        pf = add_anomaly(pf, 'development'); pfe = add_anomaly(pfe, 'evaluation')
        log(f"within-table anomaly joined: {pf.width} pair features")
    rows_dev = pl.read_parquet(out / 'hand_scores_dev.parquet'); rows_ev = pl.read_parquet(out / 'hand_scores_eval.parquet')
    folds = json.load(open(out / 'folds_by_table.json'))
    lab = pf['label'].to_numpy()
    presuspect = None
    if drop_suspects:
        sp = PROC / 'suspect_hidden_positives.parquet'
        if sp.exists():
            sus = set(pl.read_parquet(sp)['pair_id'].to_list())
            presuspect = np.array([p in sus for p in pf['pair_id'].to_list()])
            log(f"drop_suspects: zero-weighting {presuspect.sum()} suspected hidden-positive pairs")
    len_w = None
    if match_len:
        ev_shared = pl.read_parquet(PROC / 'cand_pairs_evaluation.parquet')['shared'].to_numpy()
        dv_shared = pf['shared'].to_numpy()
        edges = np.array([0, 50, 60, 70, 80, 95, 110, 130, 160, 200, 260, 10**9])
        pe, _ = np.histogram(ev_shared, bins=edges); pe = pe / pe.sum()
        pd_, _ = np.histogram(dv_shared, bins=edges); pd_ = pd_ / max(pd_.sum(), 1)
        ratio = np.where(pd_ > 0, pe / np.maximum(pd_, 1e-9), 1.0)
        ratio = np.clip(ratio, 0.25, 4.0)
        idx = np.clip(np.digitize(dv_shared, edges) - 1, 0, len(ratio) - 1)
        len_w = ratio[idx]
        log(f"match_len: weight range {len_w.min():.2f}-{len_w.max():.2f}, mean {len_w.mean():.2f}")
    log(f"stage1 pair model (w_unknown={w_unknown})")
    models, oof, feats = train_eval(pf, folds, w_unknown, rounds, seeds, leaves, drop_mask=presuspect, suspect_w=suspect_w, len_w=len_w)
    drop = presuspect; pseudo = None
    if pu_stage2:
        unk = lab < 0
        thr = np.quantile(oof[unk], 1 - rn_drop)
        drop = unk & (oof >= thr)
        if presuspect is not None:
            drop = drop | presuspect
        log(f"stage2: dropping {drop.sum()} top-{rn_drop:.1%} unknown pairs from negatives (score >= {thr:.4f})")
        if pseudo_top:
            order = np.argsort(-np.where(unk, oof, -1))[:int(pseudo_top)]
            pseudo = np.zeros(len(oof), bool); pseudo[order] = True
            log(f"  pseudo-labelling top {pseudo.sum()} unknown pairs as positives (w=0.5)")
        models, oof, feats = train_eval(pf, folds, w_unknown, rounds, seeds, leaves, drop_mask=drop, extra_pos_mask=pseudo, suspect_w=suspect_w, len_w=len_w)
    y = (lab == 1).astype(int)
    known = lab >= 0
    log(f"labelled AP {average_precision_score(y[known], oof[known]):.4f} | AUC pos-vs-unknown {roc_auc_score(np.r_[np.ones(y.sum()), np.zeros((lab<0).sum())], np.r_[oof[lab==1], oof[lab<0]]):.4f}")
    fam_dev, _ = family_from_pf(pf)
    ev_map_dev = pick_evidence(rows_dev, pl.DataFrame({'pair_id': pf['pair_id'], 'fam': fam_dev}))
    risk_dev = rank_to_unit(oof)
    sol = dev_solution(pf.select(['pair_id', 'label', 'behavior_family']))
    sub_dev = make_submission_frame(pf['pair_id'].to_list(), risk_dev, fam_dev, ev_map_dev, active_frac)
    sc, comp = kaggle_score(sol, sub_dev, 'pair_id', return_components=True)
    ranks = pd.Series(oof).rank(ascending=False).to_numpy()
    sh = pf['shared'].to_numpy()
    for lo, hi in [(0, 90), (90, 130), (130, 10**9)]:
        m = (sh >= lo) & (sh < hi)
        if (y[m] == 1).sum() > 10:
            log(f"  AP for shared in [{lo},{hi}): n={m.sum()}, pos={(y[m]==1).sum()}, AP={average_precision_score(y[m], oof[m]):.4f}")
    oc = PROC.parent.parent / '5_outputs/oc_candidates.csv'
    if oc.exists():
        import csv
        occ = {r['pair_id'] for r in csv.DictReader(open(oc))}
        mm = np.array([p in occ for p in pf['pair_id'].to_list()])
        if mm.sum():
            log(f"  OC candidates (n={mm.sum()}): median rank {np.median(ranks[mm]):.0f}, in top 2000: {(ranks[mm]<=2000).sum()}")
    log(f"DEV-SOLUTION: score {sc:.4f} | PairAP {comp['pair_ap']:.4f} Evid {comp['evidence_map']:.4f} Behav {comp['behavior_map']:.4f} | pos median rank {np.median(ranks[lab==1]):.0f}, pos rank>1000: {(ranks[lab==1]>1000).sum()} | top-1000 unknowns {(ranks[lab<0]<=1000).sum()}")
    # eval
    Xe = pfe.select(feats).to_numpy().astype(np.float32)
    risk_ev = rank_to_unit(np.mean([m.predict(Xe) for fm in models for m in fm], axis=0))
    fam_ev, _ = family_from_pf(pfe)
    ev_map = pick_evidence(rows_ev, pl.DataFrame({'pair_id': pfe['pair_id'], 'fam': fam_ev}))
    sub = make_submission_frame(pfe['pair_id'].to_list(), risk_ev, fam_ev, ev_map, active_frac)
    sample = pd.read_csv(RAW / 'sample_submission.csv')
    sub = sample[['pair_id']].merge(sub, on='pair_id', how='left')
    path = out / f'submission_{tag}.csv'; sub.to_csv(path, index=False)
    pd.DataFrame({'pair_id': pf['pair_id'], 'oof': oof, 'label': lab}).to_parquet(out / f'pair_oof_{tag}.parquet')
    pd.DataFrame({'pair_id': pfe['pair_id'], 'pred': risk_ev}).to_parquet(out / f'pair_pred_{tag}.parquet')
    log(f"written {path}")
    return sc, comp


if __name__ == '__main__':
    run, tag = sys.argv[1], sys.argv[2]
    kw = {}
    for a in sys.argv[3:]:
        k, v = a.split('=')
        kw[k] = v if k == 'graph_tag' else (float(v) if '.' in v else int(v))
    main(run, tag, **kw)
