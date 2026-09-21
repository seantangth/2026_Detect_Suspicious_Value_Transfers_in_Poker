"""H_B evidence decoder v3 — evaluation apply (f2p mask optional via model cfg) (2026-09-19). CI: exact strength levels + cif features.
Usage: hb_apply.py tag=<model tag> base=<submission csv in 5_outputs/submissions> out=<new csv name> [topn=3000] [w=0.5]
Only the five evidence columns of the base's top-<topn> pairs (by its own risk_score) are replaced; cols 1-3 stay byte-identical.
Production component = v040 evidence scores (pfonscvxpbni4 0.55 / onscvxpbni4 0.30 / family hand score 0.15, argmax routing, as shipped).
H_B component = per-family planted-C / planted-S detectors (family = trained family classifier) -> slot decode -> DT direction re-decode.
Reproduction gate: with w=0 the top-5 of the target pairs must equal the base's evidence (the base must carry v040 evidence)."""
import os, sys, json, time, hashlib, subprocess
os.environ.update({'TPDS_VARIANT': 'nb', 'TPDS_EQTAG': 'x', 'TPDS_CALLVAL': '1', 'TPDS_PB': '1', 'TPDS_PB_TAG': 'ni'})
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '3_src')); sys.path.insert(0, str(HERE))
import tpds_model as TM
from tpds_model import read_l2, l2_files, swap_ab, family_from_pf
from tpds_evidence import add_pair_pct
from tpds_submit import blend_rows, top5_map
from tpds_direction import apply_constraint
from hb_common import FAMS, decode_pair, decode_levels
import ci_feats

kw = dict(x.split('=', 1) for x in sys.argv[1:])
TAG = kw['tag']; BASE = kw['base']; OUTN = kw['out']; TOPN = int(kw.get('topn', 3000)); W = float(kw.get('w', 0.5)); ADT = float(kw.get('adt', 1.0)); KDT = float(kw.get('kdt', 1.0))
OUT = ROOT / '5_outputs/models/v5nb'; SUB = ROOT / '5_outputs/submissions'; MD = HERE / f'models_{TAG}'
EV = [f'evidence_hand_{i}' for i in range(1, 6)]


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


cfg = json.load(open(MD / 'cfg.json')); feats = cfg['feats']; CIF = cfg['cif']; F2P = int(cfg.get('f2p', 0))
models = {F: {ch: [lgb.Booster(model_file=str(p)) for p in sorted(MD.glob(f'hb_{F}_{ch}_*.txt'))] for ch in ('C', 'S')} for F in FAMS}
log({F: {ch: len(v) for ch, v in m.items()} for F, m in models.items()})
base = pd.read_csv(SUB / BASE, dtype=str, keep_default_na=False)
base['rk'] = base.risk_score.astype(float).rank(ascending=False, method='first').astype(int)
pbcov = set(pl.read_parquet(ROOT / '1_data/processed/pbni_evaluation.parquet', columns=['pair_id'])['pair_id'].unique().to_list())
target = [p for p, r in zip(base.pair_id, base.rk) if r <= TOPN and p in pbcov]
log(f'target pairs: {len(target)} of top {TOPN} (pb/ni coverage); top-600 covered {sum(1 for p, r in zip(base.pair_id, base.rk) if r <= 600 and p in pbcov)}/600')
fc = pl.read_parquet(ROOT / '5_outputs/revise_0917/family_clf_eval.parquet')
fam_new = dict(zip(fc['pair_id'].to_list(), fc['fam_new'].to_list()))
pfe = pl.read_parquet(OUT / 'pair_features_eval.parquet'); fam_arg_l, _ = family_from_pf(pfe); fam_arg = dict(zip(pfe['pair_id'].to_list(), fam_arg_l))
tset = set(target)
fmap_new = pl.DataFrame({'pair_id': target, 'fam_idx': np.array([FAMS.index(fam_new[p]) for p in target], np.int32)})
fmap_arg = pl.DataFrame({'pair_id': target, 'fam_idx': np.array([FAMS.index(fam_arg[p]) for p in target], np.int32)})
log(f'family (classifier) of targets: { {F: int((fmap_new["fam_idx"] == i).sum()) for i, F in enumerate(FAMS)} } | differs from argmax: {sum(fam_new[p] != fam_arg[p] for p in target)}')

# ------------------------------------------------------------------ score eval rows with the detectors
cache = HERE / f'hb_{TAG}_eval_rows_top{TOPN}.parquet'
RAWD = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
if cache.exists():
    rows = pl.read_parquet(cache); log('eval detector rows: cached')
else:
    parts = []; t0 = time.time(); files = l2_files('evaluation')
    keys = pl.concat([pl.read_parquet(f, columns=['pair_id', 'hand_id', 'a', 'b']).join(fmap_new.select('pair_id'), on='pair_id', how='inner') for f in files])
    cif_all = ci_feats.build(keys); log(f'cif features for {cif_all.height:,} eval rows ({time.time()-t0:.0f}s)')
    for i, f in enumerate(files):
        d = read_l2(f, 'evaluation', 2, 2)
        if d.height == 0:
            continue
        d = add_pair_pct(d).join(fmap_new, on='pair_id', how='inner')
        if d.height == 0:
            continue
        d = d.join(cif_all, on=['pair_id', 'hand_id'], how='left')
        X = d.select(feats).to_numpy().astype(np.float32); Xs = swap_ab(d).select(feats).to_numpy().astype(np.float32)
        Xc = d.select(CIF).to_numpy().astype(np.float32); XCI = np.hstack([X, Xc]); XCIs = np.hstack([Xs, Xc])
        fi = d['fam_idx'].to_numpy(); pc = np.zeros(d.height); ps = np.zeros(d.height)
        for k, F in enumerate(FAMS):
            sel = np.flatnonzero(fi == k)
            if len(sel) == 0:
                continue
            A_, B_ = (XCI, XCIs) if F == 'coordinated_isolation' else (X, Xs)
            pc[sel] = np.mean([0.5 * (m.predict(A_[sel]) + m.predict(B_[sel])) for m in models[F]['C']], axis=0)
            if models[F]['S']:
                ps[sel] = np.mean([0.5 * (m.predict(A_[sel]) + m.predict(B_[sel])) for m in models[F]['S']], axis=0)
        d = d.with_columns(pl.col('cif_nfold_before_raise').fill_null(-1).fill_nan(-1).cast(pl.Int32).alias('lvl'), ((pl.col('A_fold_to_B').fill_null(0) + pl.col('B_fold_to_A').fill_null(0)) > 0).alias('f2p'))
        parts.append(d.select(['pair_id', 'hand_id', 't_rank', 'fam_idx', 'A_net_bb', 'B_net_bb', 'lvl', 'f2p']).with_columns(pl.Series('pC', pc), pl.Series('pS', ps)))
        if i % 50 == 0:
            log(f'  eval files {i+1}/{len(files)} ({time.time()-t0:.0f}s)')
    rows = pl.concat(parts).sort(['pair_id', 't_rank', 'hand_id']); rows.write_parquet(cache)
log(f'eval detector rows {rows.height:,} pairs {rows["pair_id"].n_unique()}')

# ------------------------------------------------------------------ decode
pid = rows['pair_id'].to_numpy(); pc = rows['pC'].to_numpy(); ps = rows['pS'].to_numpy(); fi = rows['fam_idx'].to_numpy()
dirv = np.sign(rows['A_net_bb'].to_numpy() - rows['B_net_bb'].to_numpy()); lv = rows['lvl'].to_numpy(); f2p = rows['f2p'].to_numpy()
bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True]); hb = np.zeros(len(pid)); nflip = 0
for i in range(len(bnd) - 1):
    sl = slice(bnd[i], bnd[i + 1]); c = pc[sl].copy(); s = ps[sl].copy()
    if fi[bnd[i]] == 2:
        hb[sl] = decode_levels(np.where(lv[sl] >= 0, c, 0.0), lv[sl]); continue
    if F2P:
        c[~f2p[sl]] = 0.0
    aa = ADT if fi[bnd[i]] == 0 else 1.0
    h = decode_pair(c, s, aa)
    if fi[bnd[i]] == 0:
        top = np.argsort(-h)[:5]; d0 = np.sign((dirv[sl][top] * h[top]).sum())
        if d0 != 0:
            bad = dirv[sl] == -d0; c[bad] *= 0.02; s[bad] *= 0.02
            if KDT != 1.0:
                good = dirv[sl] == d0; c[good] = KDT * c[good] / (1 - c[good] + KDT * c[good]); s[good] = KDT * s[good] / (1 - s[good] + KDT * s[good])
            h = decode_pair(c, s, aa); nflip += int(bad.sum())
    hb[sl] = h
hb_rows = rows.select(['pair_id', 'hand_id']).with_columns(pl.Series('ev_score', hb))
log(f'decode done; DT opposite-direction rows damped: {nflip:,}; mean expected planted C per pair {rows.group_by("pair_id").agg(pl.col("pC").sum())["pC"].mean():.2f}, S {rows.group_by("pair_id").agg(pl.col("pS").sum())["pS"].mean():.2f}')

# ------------------------------------------------------------------ production component + reproduction gate
tp = pl.DataFrame({'pair_id': target})
pf_rows = pl.read_parquet(OUT / 'evidence_eval_pfonscvxpbni4.parquet').join(tp, on='pair_id', how='inner')
gen_rows = pl.read_parquet(OUT / 'evidence_eval_onscvxpbni4.parquet').join(tp, on='pair_id', how='inner')
hs = pl.read_parquet(OUT / 'hand_scores_eval.parquet').join(tp, on='pair_id', how='inner')
prod = blend_rows(pf_rows, hs, fmap_arg, 0.55, 0.15, 0.0, gen_rows, 0.30)
prod_c = apply_constraint(prod, 'evaluation', [p for p in target if fam_arg[p] == 'directed_transfer'], K=5)
top_prod = top5_map(prod_c); bi = base.set_index('pair_id')
same = sum(top_prod.get(p, []) == [h for h in bi.loc[p, EV].tolist() if h != 'NO_EVIDENCE'] for p in target)
log(f'reproduction gate (w=0 vs base evidence): {same}/{len(target)} identical top-5 lists')
if same < 0.98 * len(target):
    log('GATE FAILED: production component does not reproduce the base evidence; not writing a candidate'); sys.exit(2)


def prank(r):
    return r.with_columns((pl.col('ev_score').rank().over('pair_id') / pl.len().over('pair_id')).alias('ev_score'))


mix = prank(prod).join(prank(hb_rows).rename({'ev_score': 'hb'}), on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('hb').fill_null(0.0))
mix = mix.with_columns(((1 - W) * pl.col('ev_score') + W * pl.col('hb')).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score'])
mix = apply_constraint(mix, 'evaluation', [p for p in target if fam_new[p] == 'directed_transfer'], K=5)
top = top5_map(mix)
ch_set = {lo: 0 for lo in (150, 300, 600, 1000, TOPN)}; ov = []
for p in target:
    cur = [h for h in bi.loc[p, EV].tolist() if h != 'NO_EVIDENCE']; new = top.get(p, [])
    ov.append(len(set(cur) & set(new)))
    for lo in ch_set:
        if bi.loc[p, 'rk'] <= lo and set(cur) != set(new):
            ch_set[lo] += 1
log(f'pairs whose top-5 SET changed, by base rank cut: {ch_set}; mean overlap with base top-5: {np.mean(ov):.2f}/5')
o = base.drop(columns=['rk']).set_index('pair_id')
for p in target:
    o.loc[p, EV] = (top.get(p, []) + ['NO_EVIDENCE'] * 5)[:5]
o = o.reset_index()[['pair_id', 'risk_score', 'predicted_behavior'] + EV]
path = SUB / OUTN; o.to_csv(path, index=False)
b0 = pd.read_csv(SUB / BASE, dtype=str, keep_default_na=False)
assert (o[['pair_id', 'risk_score', 'predicted_behavior']].to_numpy() == b0[['pair_id', 'risk_score', 'predicted_behavior']].to_numpy()).all(), 'cols 1-3 changed'
r = subprocess.run([sys.executable, str(ROOT / '3_src/validate_submission.py'), str(path)], capture_output=True, text=True)
log(r.stdout.strip()[-300:], r.stderr.strip()[-200:])
log(f'written {path} md5 {hashlib.md5(open(path, "rb").read()).hexdigest()} | rows with changed evidence: {int((o[EV].to_numpy() != b0[EV].to_numpy()).any(1).sum())}')
