"""Merge two evidence-ranker seed sets (same recipe, different seed0) into one tag, interleaving boosters per (family, fold)
in the order tpds_gauge / tpds_evidence_apply expect: index = ((fam*5+fold) if per_family else fold) * n_seeds + seed.
Dev OOF = mean of the two OOF tables. Usage: merge_seed_sets.py <run> <tag1> <tag2> <new_tag>   e.g. v5nb onscvx onscvx_s2 onscvx4
(2026-09-13: v023 = onscvx4 / pfonscvx4 built this way from seed0=42 and seed0=44 rankers)."""
import sys, json, shutil, re, polars as pl
from pathlib import Path
run, t1, t2, tnew = sys.argv[1:5]
out = Path(__file__).resolve().parents[1] / '5_outputs/models' / run
def files(tag):
    rx = re.compile(r'^evrank_' + re.escape(tag) + r'_(\d+)\.txt$')
    return sorted((p for p in out.glob(f'evrank_{tag}_*.txt') if rx.match(p.name)), key=lambda p: int(rx.match(p.name).group(1)))
cfg = json.load(open(out / f'evidence_cfg_{t1}.json')); per = int(cfg.get('per_family', 0)); groups = 15 if per else 5
p1, p2 = files(t1), files(t2); s1 = len(p1) // groups; s2 = len(p2) // groups
assert len(p1) == groups * s1 and len(p2) == groups * s2 and s1 and s2, (t1, len(p1), t2, len(p2), groups)
k = 0
for g in range(groups):
    for p in list(p1[g * s1:(g + 1) * s1]) + list(p2[g * s2:(g + 1) * s2]):
        shutil.copyfile(p, out / f'evrank_{tnew}_{k}.txt'); k += 1
f1 = json.load(open(out / f'evidence_features_{t1}.json')); f2 = json.load(open(out / f'evidence_features_{t2}.json')); assert f1 == f2, 'feature lists differ'
shutil.copyfile(out / f'evidence_features_{t1}.json', out / f'evidence_features_{tnew}.json')
cfg2 = dict(cfg); cfg2.update({'seeds': s1 + s2, 'merged_from': [t1, t2]}); json.dump(cfg2, open(out / f'evidence_cfg_{tnew}.json', 'w'))
a = pl.read_parquet(out / f'evidence_dev_{t1}.parquet'); b = pl.read_parquet(out / f'evidence_dev_{t2}.parquet')
m = a.join(b, on=['pair_id', 'hand_id'], suffix='_2'); assert m.height == a.height == b.height
m.with_columns(((pl.col('ev_score') * s1 + pl.col('ev_score_2') * s2) / (s1 + s2)).cast(pl.Float32).alias('ev_score')).select(['pair_id', 'hand_id', 'ev_score']).write_parquet(out / f'evidence_dev_{tnew}.parquet')
print(f'{tnew}: {k} boosters ({groups} groups x {s1 + s2} seeds); OOF averaged')
