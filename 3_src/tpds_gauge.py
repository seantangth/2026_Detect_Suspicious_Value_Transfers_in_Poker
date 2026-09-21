"""Deployment-consistent dev gauges for one assembled recipe (run + risk tag + evidence blend).

Prints for the given run:
  * AP_mirror           dev PairAP on the eval-like population: drops the unknown pairs that share a player with a
                        public positive (eval's construction rule excludes those; they were 43% of top-1000 errors)
  * rulers              known-positive median rank, #>500, #>1000, weak-episode (n_ev<=4) median rank
  * dev Evid            EvidenceMAP@5 over the 372 positives, TRUE-routed (historical CV number) and ROUTED, i.e.
                        with mis-routed positives re-scored by the PREDICTED family's ranker + one-hot, which is what
                        tpds_evidence_apply.py does at eval (Codex 2026-09-10 finding: -0.0074 on v012)
  * composite           official metric on the dev-solution population at the production active_frac
Usage: [TPDS_VARIANT=nb] tpds_gauge.py <run> [risk_tag=bs] [ev_tag=pfons] [ev2_tag=ons] [w_rank=0.55] [w_rank2=0.30]
                                             [w_fam=0.15] [dir_k=5] [active_frac=0.2] [onset=3] [json=path]
"""
import sys, json, re
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_model import read_l2, swap_ab, FAMS, family_from_pf, make_submission_frame, dev_solution, rank_to_unit, log
from tpds_evidence import add_pair_pct
from tpds_submit import blend_rows, top5_map
from tpds_direction import apply_constraint
from tpds_rerank import onset_cached
from metric import score as kaggle_score, _average_precision


def ap5_per_pair(rows, truth, ids):
    pred = top5_map(rows); out = []
    for p in ids:
        y = np.array([h in truth[p] for h in pred.get(p, [])], float)
        out.append(float(np.sum(y * np.cumsum(y) / np.arange(1, len(y) + 1)) / min(len(truth[p]), 5)) if len(y) else 0.0)
    return np.array(out)


def rescore_misrouted(out, tag, mis_ids, fmap_pred, folds, onset):
    """Re-score the rows of `mis_ids` with the ranker/one-hot of their PREDICTED family, held-out fold models."""
    cfg = json.load(open(out / f'evidence_cfg_{tag}.json'))
    assert int(cfg.get('stack', 0)) == 0, 'gauge supports stack=0 rankers only'
    per = int(cfg.get('per_family', 0))
    feats = json.load(open(out / f'evidence_features_{tag}.json'))
    _rx = re.compile(r'^evrank_' + re.escape(tag) + r'_(\d+)\.txt$')   # exact tag: evrank_<tag>_<k>.txt only (not <tag>_<other>_k)
    paths = sorted((p for p in out.glob(f'evrank_{tag}_*.txt') if _rx.match(p.name)), key=lambda p: int(_rx.match(p.name).group(1)))
    seeds = len(paths) // (15 if per else 5)
    mis = pl.DataFrame({'pair_id': mis_ids})
    tables = (pl.read_parquet(PROC / 'cand_pairs_development.parquet', columns=['pair_id', 'table_id'])
                .join(mis, on='pair_id')['table_id'].unique().to_list())
    parts = []
    for t in tables:
        d = read_l2(PROC / 'l2' / 'development' / f'{t}.parquet', 'development', 2, 2)
        d = d.join(mis, on='pair_id', how='inner')
        if d.height:
            parts.append(add_pair_pct(d))
    x = pl.concat(parts)
    if onset:
        x = x.join(onset_cached(out, 'development', norm_only=(onset == 3), pair_filter=mis), on=['pair_id', 'hand_id'], how='left')
    x = x.join(fmap_pred, on='pair_id')
    preds = []
    for t in x['table_id'].unique().to_list():
        z = x.filter(pl.col('table_id') == t); fo = folds[t]
        for fi in z['fam_idx'].unique().to_list():
            q = z.filter(pl.col('fam_idx') == fi)
            oh = np.zeros((q.height, 3), np.float32); oh[:, fi] = 1
            a = np.hstack([q.select(feats).to_numpy().astype(np.float32), oh])
            b = np.hstack([swap_ab(q).select(feats).to_numpy().astype(np.float32), oh])
            start = ((fi * 5 + fo) if per else fo) * seeds
            s = [(lgb.Booster(model_file=str(paths[k])).predict(a) + lgb.Booster(model_file=str(paths[k])).predict(b)) / 2 for k in range(start, start + seeds)]
            preds.append(q.select(['pair_id', 'hand_id']).with_columns(pl.Series('ev_score', np.mean(s, axis=0).astype(np.float32))))
    return pl.concat(preds)


def main(run, risk_tag='bs', ev_tag='pfons', ev2_tag='ons', w_rank=0.55, w_rank2=0.30, w_fam=0.15, dir_k=5, active_frac=0.2, onset=3, json_out='', pair_only=0):
    out = ROOT / '5_outputs/models' / run
    pf = pl.read_parquet(out / 'pair_features_dev.parquet')
    fam_pred, _ = family_from_pf(pf)
    lab = pf['label'].to_numpy(); true_fam = pf['behavior_family'].to_numpy(); ids_all = pf['pair_id'].to_list()
    oof = pd.read_parquet(out / f'pair_oof_{risk_tag}.parquet').set_index('pair_id').loc[ids_all].oof.to_numpy()
    risk = rank_to_unit(oof)
    res = {'run': run, 'risk_tag': risk_tag, 'ev': f'{ev_tag}/{ev2_tag}', 'dir_k': dir_k}
    # ---- AP_mirror + rulers
    labs = pl.read_csv(RAW / 'development_labels.csv'); pos = labs.filter(pl.col('label') == 1)
    pos_players = set(pos['player_1'].to_list()) | set(pos['player_2'].to_list())
    cand = pl.read_parquet(PROC / 'cand_pairs_development.parquet', columns=['pair_id', 'a', 'b']).to_pandas().set_index('pair_id').loc[ids_all]
    touch = cand.a.isin(pos_players).to_numpy() | cand.b.isin(pos_players).to_numpy()
    y = (lab == 1).astype(int); keep = (lab == 1) | ~touch
    order = np.argsort(ids_all)   # official metric sorts by pair_id first
    res['AP_dev'] = float(_average_precision(y[order], risk[order]))
    res['AP_mirror'] = float(_average_precision(y[order][keep[order]], risk[order][keep[order]]))
    res['n_excluded'] = int((~keep).sum())
    ranks = pd.Series(oof).rank(ascending=False).to_numpy()
    n_ev = pl.read_csv(RAW / 'development_evidence.csv').group_by('pair_id').len().to_pandas().set_index('pair_id')['len']
    weak = np.array([n_ev.get(p, 0) <= 4 and l == 1 for p, l in zip(ids_all, lab)])
    res['pos_median_rank'] = float(np.median(ranks[lab == 1])); res['pos_gt500'] = int((ranks[lab == 1] > 500).sum())
    res['pos_gt1000'] = int((ranks[lab == 1] > 1000).sum()); res['weak_median_rank'] = float(np.median(ranks[weak]))
    res['top1000_unknown'] = int((ranks[lab < 0] <= 1000).sum())
    rk_m = pd.Series(np.where(keep, oof, -np.inf)).rank(ascending=False).to_numpy()   # ranks within the mirror population
    res['mirror_pos_median_rank'] = float(np.median(rk_m[lab == 1])); res['mirror_pos_gt500'] = int((rk_m[lab == 1] > 500).sum())
    res['mirror_weak_median_rank'] = float(np.median(rk_m[weak]))
    if pair_only:   # pair-axis controls have no rankers of their own
        print(json.dumps(res, indent=1))
        if json_out:
            Path(json_out).write_text(json.dumps(res, indent=1))
        return res
    # ---- evidence: true-routed (historical) and routed (deployment-consistent)
    pos_ids = pos['pair_id'].to_list()
    hs = pl.read_parquet(out / 'hand_scores_dev.parquet').join(pl.DataFrame({'pair_id': pos_ids}), on='pair_id')
    fmap = pl.DataFrame({'pair_id': pf['pair_id'], 'fam_idx': np.array([FAMS.index(f) for f in fam_pred], np.int32)})
    ev = pl.read_csv(RAW / 'development_evidence.csv')
    truth = {r['pair_id']: set(r['hand_id']) for r in ev.group_by('pair_id').agg(pl.col('hand_id')).to_dicts()}
    ids = sorted(truth)
    dt_pred = [p for p, f in zip(ids_all, fam_pred) if f == 'directed_transfer']
    def assemble(e1, e2):
        rows = blend_rows(e1, hs, fmap, w_rank, w_fam, 0.0, e2, w_rank2)
        return apply_constraint(rows, 'development', dt_pred, K=int(dir_k)) if dir_k else rows
    e1 = pl.read_parquet(out / f'evidence_dev_{ev_tag}.parquet'); e2 = pl.read_parquet(out / f'evidence_dev_{ev2_tag}.parquet')
    rows_true = assemble(e1, e2)
    ap_true = ap5_per_pair(rows_true, truth, ids); res['Evid_true_routed'] = float(ap_true.mean())
    mis = [p for p, l, tf, pfm in zip(ids_all, lab, true_fam, fam_pred) if l == 1 and tf != pfm]
    res['n_misrouted'] = len(mis)
    if mis:
        folds = json.load(open(out / 'folds_by_table.json'))
        c1 = rescore_misrouted(out, ev_tag, mis, fmap, folds, onset); c2 = rescore_misrouted(out, ev2_tag, mis, fmap, folds, onset)
        e1r = pl.concat([e1.filter(~pl.col('pair_id').is_in(mis)), c1.select(e1.columns)])
        e2r = pl.concat([e2.filter(~pl.col('pair_id').is_in(mis)), c2.select(e2.columns)])
        rows_routed = assemble(e1r, e2r)
    else:
        rows_routed = rows_true
    ap_routed = ap5_per_pair(rows_routed, truth, ids); res['Evid_routed'] = float(ap_routed.mean())
    fam_of = pos.to_pandas().set_index('pair_id').behavior_family
    for f in FAMS:
        m = np.array([fam_of[p] == f for p in ids]); res[f'Evid_routed_{f[:2]}'] = float(ap_routed[m].mean())
    # ---- composite on the dev-solution population (official metric), both routings
    sol = dev_solution(pf.select(['pair_id', 'label', 'behavior_family']))
    for name, rows in (('true', rows_true), ('routed', rows_routed)):
        sub = make_submission_frame(ids_all, risk, fam_pred, top5_map(rows), active_frac)
        sc, comp = kaggle_score(sol, sub, 'pair_id', return_components=True)
        res[f'composite_{name}'] = float(sc); res[f'Behav_{name}'] = float(comp['behavior_map'])
    res['family_acc_pos'] = float((fam_pred[lab == 1] == true_fam[lab == 1]).mean())
    print(json.dumps(res, indent=1))
    if json_out:
        Path(json_out).write_text(json.dumps(res, indent=1))
        # per-pair AP@5 next to the json (for paired table-bootstrap comparisons; aggregate numbers unchanged)
        pl.DataFrame({'pair_id': ids, 'ap_true': ap_true, 'ap_routed': ap_routed}).write_parquet(str(json_out).removesuffix('.json') + '_pairs.parquet')
    return res


if __name__ == '__main__':
    run = sys.argv[1]; kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        kw['json_out' if k == 'json' else k] = v if k in ('risk_tag', 'ev_tag', 'ev2_tag', 'json') else (float(v) if '.' in v else int(v))
    main(run, **kw)
