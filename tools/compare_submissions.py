"""Agreement between two submission files (e.g. a full rebuild vs the submitted file).
Usage: compare_submissions.py <reference.csv> <other.csv>
Reports: risk rank correlation and top-k set overlap; behaviour label agreement; evidence agreement on the reference's top pairs."""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

EV = [f'evidence_hand_{i}' for i in range(1, 6)]
a = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).set_index('pair_id')
b = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False).set_index('pair_id').loc[a.index]
ra, rb = a.risk_score.astype(float), b.risk_score.astype(float)
rank_a = ra.rank(ascending=False, method='first'); rank_b = rb.rank(ascending=False, method='first')
print(f'pairs {len(a):,} | identical rows {int((a.to_numpy() == b.to_numpy()).all(1).sum()):,} | identical risk strings {int((a.risk_score == b.risk_score).sum()):,}')
print(f'risk Spearman {spearmanr(ra, rb)[0]:.5f}')
for k in (150, 300, 600, 1000, 3000):
    top_a = set(rank_a[rank_a <= k].index); top_b = set(rank_b[rank_b <= k].index)
    print(f'  top-{k:<5d} set overlap {len(top_a & top_b)}/{k}')
act = a.predicted_behavior != 'none'
print(f'behaviour: same label {float((a.predicted_behavior == b.predicted_behavior).mean()):.4f} | among the reference\'s labelled pairs {float((a.predicted_behavior[act] == b.predicted_behavior[act]).mean()):.4f}')
for k in (600, 3000):
    idx = rank_a[rank_a <= k].index
    same = np.mean([list(a.loc[p, EV]) == list(b.loc[p, EV]) for p in idx])
    inter = np.mean([len(set(a.loc[p, EV]) & set(b.loc[p, EV]) - {'NO_EVIDENCE'}) for p in idx])
    print(f'evidence on the reference\'s top-{k}: identical ranked lists {same:.4f} | mean shared hands {inter:.2f}/5')
