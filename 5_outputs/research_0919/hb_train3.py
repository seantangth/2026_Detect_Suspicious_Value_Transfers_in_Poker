"""H_B evidence decoder v2 — dev training + gauge (2026-09-19). CI uses the exact strength level (folds before the pair's first preflop raise) + cif features.
Rows come from the PRODUCTION loader (same env as v040) so the eval apply sees identical columns.
Usage: hb_train.py tag=<t> [seed0=42] [seeds=3] [rounds=500] [lr=0.03] [leaves=31] [ff=0.5] [fit=1] [w=0.5]
  - nested typing (no fold-fo label ever reaches the detector that predicts fold fo)
  - per-family planted-C / planted-S binary detectors, swap_ab augmentation, table-fold OOF
  - slot decode, DT direction re-decode, blend with the v040 production OOF (0.55/0.30/0.15), apply_constraint
  - fit=1: also fit all-dev models and save boosters for hb_apply.py
"""
import os, sys, json, time
os.environ.update({'TPDS_VARIANT': 'nb', 'TPDS_EQTAG': 'x', 'TPDS_CALLVAL': '1', 'TPDS_PB': '1', 'TPDS_PB_TAG': 'ni'})
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '3_src')); sys.path.insert(0, str(HERE))
import tpds_model as TM
import tpds_evidence as TE
from tpds_submit import blend_rows
from tpds_direction import apply_constraint, build as build_dir
from tpds_gauge import ap5_per_pair
from hb_common import FAMS, runs_from_ranked_times, decode_pair, build_labels, decode_levels
import ci_feats

RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'; OUT = ROOT / '5_outputs/models/v5nb'
kw = dict(x.split('=', 1) for x in sys.argv[1:])
TAG = kw.get('tag', 'hb2'); SEED0 = int(kw.get('seed0', 42)); SEEDS = int(kw.get('seeds', 3)); ROUNDS = int(kw.get('rounds', 500))
LR = float(kw.get('lr', 0.03)); LEAVES = int(kw.get('leaves', 31)); FF = float(kw.get('ff', 0.5)); FIT = int(kw.get('fit', 1)); W = float(kw.get('w', 0.5))
NJ = int(kw.get('njobs', 8)); NATNEG = int(kw.get('natneg', 0)); WN = float(kw.get('wn', 0.5)); SPLIT = int(kw.get('split', 0)); SELF = int(kw.get('selftrain', 0)); F2P = int(kw.get('f2p', 0)); ADT = float(kw.get('adt', 1.0)); KDT = float(kw.get('kdt', 1.0)); JOINTC = int(kw.get('jointc', 0)); HI = float(kw.get('hi', 0.9)); LO = float(kw.get('lo', 0.1))


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


P = dict(objective='binary', learning_rate=LR, num_leaves=LEAVES, min_data_in_leaf=20, feature_fraction=FF, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=NJ)
PT = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.5, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=2.0, verbose=-1, n_jobs=NJ)

# ------------------------------------------------------------------ rows
lab = pl.read_csv(RAW / 'development_labels.csv'); pos = lab.filter(pl.col('label') == 1).select(['pair_id', 'behavior_family'])
d = TE.load_rows('development', pos.select('pair_id'), 2, 2, onset=0, out=OUT).join(pos, on='pair_id').sort(['pair_id', 't_rank', 'hand_id'])
cif = ci_feats.build(d.select(['pair_id', 'hand_id', 'a', 'b'])); d = d.join(cif, on=['pair_id', 'hand_id'], how='left')
LEVEL_COLS = ['cif_first_mem_idx', 'cif_active_at_first', 'cif_nfold_before', 'cif_noutvol_before', 'cif_active_at_first_raise', 'cif_nfold_before_raise']
CIF = [c for c in cif.columns if c.startswith('cif_') and c not in LEVEL_COLS]
prod_feats = json.load(open(OUT / 'evidence_features_onscvxpbni4.json'))
feats = [c for c in prod_feats if not c.startswith('ons_')]
miss = [c for c in feats if c not in d.columns]; assert not miss, miss
log(f'rows {d.height:,} pairs {d["pair_id"].n_unique()} feats {len(feats)}')
evd = pl.read_csv(RAW / 'development_evidence.csv')
tr_of = dict(zip(d['hand_id'].to_list(), d['t_rank'].to_list()))
rank_of = {(p, h): r for p, h, r in evd.select(['pair_id', 'hand_id', 'evidence_rank']).iter_rows()}
# run index per (pair, hand)
run_of, nruns_of = {}, {}
for (p,), g in evd.sort(['pair_id', 'evidence_rank']).group_by(['pair_id'], maintain_order=True):
    hs = g['hand_id'].to_list(); r = runs_from_ranked_times([tr_of[h] for h in hs])
    for h, ri in zip(hs, r):
        run_of[(p, h)] = ri
    nruns_of[p] = r[-1] + 1
pid = d['pair_id'].to_numpy(); hid = d['hand_id'].to_numpy(); t = d['t_rank'].to_numpy(); fam = d['behavior_family'].to_numpy(); N = d.height
ev = np.array([(p, h) in run_of for p, h in zip(pid, hid)]); run = np.array([run_of.get((p, h), -1) for p, h in zip(pid, hid)])
nr = np.array([nruns_of[p] for p in pid]); n_ev = pd.Series(ev).groupby(pid).transform('sum').to_numpy()
assert ev.sum() == evd.height, (ev.sum(), evd.height)
folds = json.load(open(OUT / 'folds_by_table.json')); fold = np.array([folds[x] for x in d['table_id'].to_list()])
X = d.select(feats).to_numpy().astype(np.float32); Xs = TM.swap_ab(d).select(feats).to_numpy().astype(np.float32)
Xc = d.select(CIF).to_numpy().astype(np.float32)          # symmetric in A/B -> identical under swap
lvl = d['cif_nfold_before_raise'].fill_null(-1).fill_nan(-1).to_numpy().astype(int)
XCI = np.hstack([X, Xc]); XCIs = np.hstack([Xs, Xc])
if NATNEG:
    neg = lab.filter(pl.col('label') == 0).select('pair_id')
    dn = TE.load_rows('development', neg, 2, 2, onset=0, out=OUT)
    dn = dn.join(ci_feats.build(dn.select(['pair_id', 'hand_id', 'a', 'b'])), on=['pair_id', 'hand_id'], how='left')
    Xn = dn.select(feats).to_numpy().astype(np.float32); XnCI = np.hstack([Xn, dn.select(CIF).to_numpy().astype(np.float32)])
    fold_n = np.array([folds[x] for x in dn['table_id'].to_list()]); lvl_n = dn['cif_nfold_before_raise'].fill_null(-1).fill_nan(-1).to_numpy().astype(int)
    log(f'natural-negative rows {dn.height:,} from {dn["pair_id"].n_unique()} confirmed non-target pairs; level-0 rows {(lvl_n == 0).sum():,}')
CI = fam == 'coordinated_isolation'
f2p = ((d['A_fold_to_B'].fill_null(0) + d['B_fold_to_A'].fill_null(0)) > 0).to_numpy()


def typed_evidence(train_mask):
    """isC for evidence rows inside train_mask. 2+ runs: run 0 = C. 1 run: typing classifier fitted on train_mask's multi-run evidence.
    CI: single channel (all C)."""
    isC = np.zeros(N, bool)
    m_multi = ev & (nr >= 2); isC[m_multi] = run[m_multi] == 0
    isC[ev & CI] = True
    for F in FAMS[:2]:
        trn = train_mask & ev & (nr >= 2) & (fam == F); tgt = train_mask & ev & (nr == 1) & (fam == F)
        if tgt.sum() == 0:
            continue
        y = (run[trn] == 0).astype(int)
        pr = np.mean([lgb.train(dict(PT, seed=SEED0 + s), lgb.Dataset(np.vstack([X[trn], Xs[trn]]), np.concatenate([y, y])), num_boost_round=200).predict(X[tgt]) for s in range(2)], axis=0)
        isC[tgt] = pr > 0.5
        if SPLIT:      # levels are non-decreasing along a list: choose the split point maximising the typing likelihood
            ti = np.flatnonzero(tgt); prm = dict(zip(ti, np.clip(pr, 1e-4, 1 - 1e-4)))
            for p_ in np.unique(pid[ti]):
                ii = [i for i in ti if pid[i] == p_]; ii = sorted(ii, key=lambda i: rank_of[(pid[i], hid[i])])
                q = np.array([prm[i] for i in ii]); best = max(range(len(ii) + 1), key=lambda k: np.log(q[:k]).sum() + np.log(1 - q[k:]).sum())
                for j, i in enumerate(ii): isC[i] = j < best
    if F2P:
        isC[ev & ~CI & ~f2p] = False
    return isC


def ci_labels():
    """CI exact-level labels on level-0 rows: listed level-0 = 1; unlisted level-0 = 0 if before the last listed level-0 hand or if fewer
    than 5 level-0 hands are listed (then every planted level-0 hand is listed); otherwise censored (-1). Other levels: -1."""
    lb = np.full(N, -1)
    for idx in groups_all:
        if fam[idx[0]] != 'coordinated_isolation':
            continue
        l0 = idx[lvl[idx] == 0]
        if len(l0) == 0:
            continue
        e0 = ev[l0]; lab0 = np.zeros(len(l0), int); lab0[e0] = 1
        if e0.sum() >= 5:
            lab0[(~e0) & (t[l0] > t[l0][e0].max())] = -1
        lb[l0] = lab0
    return lb


def fit_models(train_mask, isC, pseudo=None):
    """per family -> {'C': [boosters], 'S': [boosters]}; CI -> {'C': [boosters on level-0 rows, feats+cif]}."""
    labC, labS = build_labels(pid, t, ev, n_ev, isC & ev, ev & ~isC)
    labCI = ci_labels()
    if pseudo is not None:      # self-training: censored rows (never seen in round 1) get confident pseudo-labels
        pC_, pS_, pCI_ = pseudo
        for lb, pp in ((labC, pC_), (labS, pS_)):
            cz = (lb < 0) & train_mask & ~CI; lb[cz & (pp >= HI)] = 1; lb[cz & (pp <= LO)] = 0
        cz = (labCI < 0) & train_mask & CI & (lvl >= 0); labCI[cz & (pCI_ >= HI)] = 1; labCI[cz & (pCI_ <= LO)] = 0
    out = {}
    for F in FAMS:
        out[F] = {}
        nat = (np.isin(fold_n, np.unique(fold[train_mask])) if NATNEG else None)
        if F == 'coordinated_isolation':
            m = train_mask & (labCI >= 0)      # level-0 rows of CI pairs only
            Xtr = np.vstack([XCI[m], XCIs[m]]); ytr = np.concatenate([labCI[m], labCI[m]]); wtr = np.ones(len(ytr))
            if NATNEG:
                mn = nat & (lvl_n == 0); Xtr = np.vstack([Xtr, XnCI[mn]]); ytr = np.concatenate([ytr, np.zeros(mn.sum(), int)]); wtr = np.concatenate([wtr, np.full(mn.sum(), WN)])
            out[F]['C'] = [lgb.train(dict(P, seed=SEED0 + s), lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=ROUNDS) for s in range(SEEDS)]
            continue
        for ch, lb in (('C', labC), ('S', labS)):
            if JOINTC and ch == 'C':       # DT-C and SP-C are the same concept (fold the better hand to the partner): one detector + family flag
                if F == 'soft_play':
                    out[F][ch] = out['directed_transfer'][ch]; continue
                m = train_mask & ~CI & (lb >= 0); ind = (fam[m] == 'directed_transfer').astype(np.float32)[:, None]
                Xtr = np.vstack([np.hstack([X[m], ind]), np.hstack([Xs[m], ind])]); ytr = np.concatenate([lb[m], lb[m]]); wtr = np.ones(len(ytr))
                out[F][ch] = [lgb.train(dict(P, seed=SEED0 + s_), lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=ROUNDS) for s_ in range(SEEDS)]; continue
            m = train_mask & (fam == F) & (lb >= 0)
            Xtr = np.vstack([X[m], Xs[m]]); ytr = np.concatenate([lb[m], lb[m]]); wtr = np.ones(len(ytr))
            if NATNEG:
                Xtr = np.vstack([Xtr, Xn[nat]]); ytr = np.concatenate([ytr, np.zeros(nat.sum(), int)]); wtr = np.concatenate([wtr, np.full(nat.sum(), WN)])
            out[F][ch] = [lgb.train(dict(P, seed=SEED0 + s), lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=ROUNDS) for s in range(SEEDS)]
    return out


def predict(models, mask):
    """-> dict F -> (pC, pS) on rows in mask (every family's detectors on every row: needed for predicted-family routing)."""
    res = {}
    for F in FAMS:
        A_, B_ = (XCI, XCIs) if F == 'coordinated_isolation' else (X, Xs)
        if JOINTC and F != 'coordinated_isolation':
            ind = np.full((N, 1), 1.0 if F == 'directed_transfer' else 0.0, np.float32); AC, BC = np.hstack([X, ind]), np.hstack([Xs, ind])
        else:
            AC, BC = A_, B_
        pc = np.mean([0.5 * (m.predict(AC[mask]) + m.predict(BC[mask])) for m in models[F]['C']], axis=0)
        ps = np.mean([0.5 * (m.predict(A_[mask]) + m.predict(B_[mask])) for m in models[F]['S']], axis=0) if 'S' in models[F] else np.zeros(mask.sum())
        res[F] = (pc, ps)
    return res


order_all = np.argsort(pid, kind='stable'); bnd_all = np.flatnonzero(np.r_[True, pid[order_all][1:] != pid[order_all][:-1], True])
groups_all = [order_all[bnd_all[i]:bnd_all[i + 1]] for i in range(len(bnd_all) - 1)]
t0 = time.time()
PC = {F: np.zeros(N) for F in FAMS}; PS = {F: np.zeros(N) for F in FAMS}
for fo in range(5):
    trm = fold != fo; tem = fold == fo
    isC = typed_evidence(trm)
    models = fit_models(trm, isC)
    if SELF:
        rr = predict(models, trm); pc1 = np.zeros(N); ps1 = np.zeros(N); pci1 = np.zeros(N)
        for F in FAMS[:2]:
            mf = trm & (fam == F); sub = (fam[trm] == F); pc1[mf] = rr[F][0][sub]; ps1[mf] = rr[F][1][sub]
        mf = trm & CI; pci1[mf] = rr['coordinated_isolation'][0][CI[trm]]
        models = fit_models(trm, isC, pseudo=(pc1, ps1, pci1))
    r = predict(models, tem)
    for F in FAMS:
        PC[F][tem], PS[F][tem] = r[F]
    log(f'fold {fo} done ({time.time()-t0:.0f}s)')

# ------------------------------------------------------------------ decode + gauge
dirtab = build_dir('development'); dmap = dict(zip(zip(dirtab['pair_id'].to_list(), dirtab['hand_id'].to_list()), dirtab['dir'].to_list())) if False else None
dirv = np.sign(d['A_net_bb'].to_numpy() - d['B_net_bb'].to_numpy())
order = np.argsort(pid, kind='stable'); bnd = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1], True])
groups = [order[bnd[i]:bnd[i + 1]] for i in range(len(bnd) - 1)]          # rows already time-sorted within pair


def hb_scores(route):
    """route: dict pair -> family used for detectors. Returns decode score with DT direction re-decode."""
    sc = np.zeros(N)
    for idx in groups:
        F = route[pid[idx[0]]]; c = PC[F][idx].copy(); s = PS[F][idx].copy()
        if F == 'coordinated_isolation':
            sc[idx] = decode_levels(np.where(lvl[idx] >= 0, c, 0.0), lvl[idx]); continue
        if F2P:
            c[~f2p[idx]] = 0.0
        aa = ADT if F == 'directed_transfer' else 1.0
        h = decode_pair(c, s, aa)
        if F == 'directed_transfer':
            top = np.argsort(-h)[:5]; d0 = np.sign((dirv[idx][top] * h[top]).sum())
            if d0 != 0:
                bad = dirv[idx] == -d0; c[bad] *= 0.02; s[bad] *= 0.02
                if KDT != 1.0:      # planted hands all run in the pair's direction, natural ones only ~40% -> posterior odds of same-direction hands x KDT
                    good = dirv[idx] == d0; c[good] = KDT * c[good] / (1 - c[good] + KDT * c[good]); s[good] = KDT * s[good] / (1 - s[good] + KDT * s[good])
                h = decode_pair(c, s, aa)
        sc[idx] = h
    return sc


truth = {r['pair_id']: set(r['hand_id']) for r in evd.group_by('pair_id').agg(pl.col('hand_id')).to_dicts()}
ids = sorted(truth); fam_true = dict(zip(pos['pair_id'].to_list(), pos['behavior_family'].to_list()))
_fo = ROOT / '5_outputs/revise_0917/family_clf_dev_oof.parquet'   # (release) optional: routes the printed development gauges only
fo_ = pl.read_parquet(_fo) if _fo.exists() else pl.DataFrame({'pair_id': [], 'fam_new': []}, schema={'pair_id': pl.Utf8, 'fam_new': pl.Utf8})
fam_pred = dict(zip(fo_['pair_id'].to_list(), fo_['fam_new'].to_list()))   # labelled pairs carry their P... id directly
n_mis = sum(fam_pred.get(p) not in (None, fam_true[p]) for p in ids); n_null = sum(fam_pred.get(p) is None for p in ids)
log(f'predicted-family routing: {n_mis} misrouted, {n_null} without prediction (fall back to truth)')
fam_pred = {p: (fam_pred.get(p) or fam_true[p]) for p in ids}
tab_of = dict(zip(pid, d['table_id'].to_list())); tabs = np.array([tab_of[p] for p in ids]); fam_ids = np.array([fam_true[p] for p in ids])
pf_rows = pl.read_parquet(OUT / 'evidence_dev_pfonscvxpbni4.parquet'); gen_rows = pl.read_parquet(OUT / 'evidence_dev_onscvxpbni4.parquet')
hs = pl.read_parquet(OUT / 'hand_scores_dev.parquet').join(pos.select('pair_id'), on='pair_id', how='inner')
key = d.select(['pair_id', 'hand_id'])


def prank(rows):
    return rows.with_columns((pl.col('ev_score').rank() .over('pair_id') / pl.len().over('pair_id')).alias('ev_score'))


def gauge(name, rows, base=None, route=None):
    rows = apply_constraint(rows, 'development', [p for p in ids if (route or fam_true)[p] == 'directed_transfer'], K=5)
    ap = ap5_per_pair(rows, truth, ids); s = f'{name:44s} AP@5 {ap.mean():.4f} | ' + ' '.join(f'{F[:2]} {ap[fam_ids == F].mean():.4f}' for F in FAMS)
    if base is not None:
        diff = ap - base; ut = np.unique(tabs); rng = np.random.default_rng(0); by = {x: diff[tabs == x] for x in ut}
        bs = [np.concatenate([by[x] for x in rng.choice(ut, len(ut))]).mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [5, 95])
        s += f' | delta {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] ' + ' '.join(f'{F[:2]} {diff[fam_ids == F].mean():+.4f}' for F in FAMS)
    log(s); return ap


# evidence-driven routing: family whose detectors give the largest expected number of listed hits in the top-5
def top5sum(route_const):
    sc = hb_scores({p: route_const for p in ids}); out = {}
    for idx in groups:
        out[pid[idx[0]]] = np.sort(sc[idx])[-5:].sum()
    return out
t5 = {F: top5sum(F) for F in FAMS}
fam_evd = {p: max(FAMS, key=lambda F: t5[F][p]) for p in ids}
fam_hyb = {p: (fam_evd[p] if t5[fam_evd[p]][p] > 1.5 * t5[fam_pred[p]][p] else fam_pred[p]) for p in ids}
log(f'routing accuracy: famclf {np.mean([fam_pred[p] == fam_true[p] for p in ids]):.4f} | evidence-driven {np.mean([fam_evd[p] == fam_true[p] for p in ids]):.4f} | hybrid {np.mean([fam_hyb[p] == fam_true[p] for p in ids]):.4f}')
res = {}
for rname, route in (('true', fam_true), ('pred', fam_pred), ('evd', fam_evd), ('hyb', fam_hyb)):
    fmap = pl.DataFrame({'pair_id': ids, 'fam_idx': np.array([FAMS.index(route[p]) for p in ids], np.int32)})
    prod = blend_rows(pf_rows, hs, fmap, 0.55, 0.15, 0.0, gen_rows, 0.30).join(key, on=['pair_id', 'hand_id'], how='inner')
    base = gauge(f'[{rname}] production v040 recipe', prod, route=route)
    hb = hb_scores(route); hb_rows = key.with_columns(pl.Series('ev_score', hb))
    a_hb = gauge(f'[{rname}] H_B decode alone', hb_rows, base, route)
    out_w = {}
    for w in sorted({0.3, 0.5, 0.7, W}):
        mix = prank(prod).join(prank(hb_rows).rename({'ev_score': 'hb'}), on=['pair_id', 'hand_id']).with_columns(((1 - w) * pl.col('ev_score') + w * pl.col('hb')).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score'])
        out_w[w] = gauge(f'[{rname}] blend prod*{1-w:.1f} + HB*{w:.1f}', mix, base, route)
    res[rname] = dict(base=float(base.mean()), hb=float(a_hb.mean()), blend={str(w): float(v.mean()) for w, v in out_w.items()})
    if rname == 'pred':
        pl.DataFrame({'pair_id': ids, 'ap_base': base, 'ap_hb': a_hb, f'ap_blend{W}': out_w[W]}).write_parquet(HERE / f'hb_{TAG}_pairs.parquet')
        key.with_columns(pl.Series('hb', hb), pl.Series('pC_true', np.array([PC[fam_true[p]][i] for i, p in enumerate(pid)])),
                         pl.Series('pS_true', np.array([PS[fam_true[p]][i] for i, p in enumerate(pid)]))).write_parquet(HERE / f'hb_{TAG}_oof.parquet')
json.dump(dict(tag=TAG, seed0=SEED0, seeds=SEEDS, rounds=ROUNDS, lr=LR, leaves=LEAVES, ff=FF, n_feats=len(feats), res=res), open(HERE / f'hb_{TAG}.json', 'w'), indent=1)

# ------------------------------------------------------------------ all-dev fit for the eval apply
if FIT:
    allm = np.ones(N, bool); isC = typed_evidence(allm); models = fit_models(allm, isC)
    if SELF:
        rr = predict(models, allm); pc1 = np.zeros(N); ps1 = np.zeros(N); pci1 = np.zeros(N)
        for F in FAMS[:2]:
            mf = fam == F; pc1[mf] = rr[F][0][mf]; ps1[mf] = rr[F][1][mf]
        pci1[CI] = rr['coordinated_isolation'][0][CI]
        models = fit_models(allm, isC, pseudo=(pc1, ps1, pci1))
    md = HERE / f'models_{TAG}'; md.mkdir(exist_ok=True)
    for F in FAMS:
        for ch, ms in models[F].items():
            for i, m in enumerate(ms):
                m.save_model(str(md / f'hb_{F}_{ch}_{i}.txt'))
    json.dump(dict(feats=feats, cif=CIF, f2p=F2P, split=SPLIT, jointc=JOINTC, env={k: os.environ[k] for k in ('TPDS_VARIANT', 'TPDS_EQTAG', 'TPDS_CALLVAL', 'TPDS_PB', 'TPDS_PB_TAG')}, seeds=SEEDS, seed0=SEED0), open(md / 'cfg.json', 'w'), indent=1)
    log(f'saved all-dev models to {md}')
log(f'done ({time.time()-t0:.0f}s)')
