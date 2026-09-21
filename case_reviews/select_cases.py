"""Case-review candidates from the SELECTED submission A: for the highest-risk pairs of each predicted family, render the five
submitted evidence hands (all hole cards, board, per-street equity of the players still in, every action). Usage: select_cases.py [n_per_family=6] [submission.csv]. Output: case_reviews/candidates.md (+ candidates.parquet)."""
import sys
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SUB = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'submission.csv'
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT)); import render_hands as rh
N_PER_FAM = int(sys.argv[1]) if len(sys.argv) > 1 else 6

sub = pl.read_csv(SUB, infer_schema_length=0).with_columns(pl.col('risk_score').cast(pl.Float64))
sub = sub.with_columns(pl.col('risk_score').rank('ordinal', descending=True).alias('rank'))
ep = pl.read_csv(RAW / 'evaluation_pairs.csv')
pc = [c for c in ep.columns if c != 'pair_id'][:2]
sub = sub.join(ep.select('pair_id', pl.col(pc[0]).alias('A'), pl.col(pc[1]).alias('B')), on='pair_id')
items = []
for fam in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
    top = sub.filter(pl.col('predicted_behavior') == fam).sort('rank').head(N_PER_FAM)
    for r in top.iter_rows(named=True):
        hands = [(r[f'evidence_hand_{i}'], f'SUBMITTED EVIDENCE #{i}') for i in range(1, 6) if r[f'evidence_hand_{i}'] != 'NO_EVIDENCE']
        items.append(dict(pair_id=r['pair_id'], A=r['A'], B=r['B'], header=f"predicted {fam} | risk rank {r['rank']} (risk_score {r['risk_score']:.6f})", hands=hands))
rh.write_pack('candidates.md', 'Case-review candidates: top-ranked pairs of the selected submission, with the five submitted evidence hands',
              'A and B are the two players of the pair (A = first player column of evaluation_pairs.csv); o<seat> = other players. '
              'Equity = probability of winning at showdown among the players still in, at the start of each street (exact post-flop, 400-sample Monte Carlo pre-flop).',
              items, 'evaluation')
pl.DataFrame([{k: v for k, v in it.items() if k != 'hands'} | {'hands': ' '.join(h for h, _ in it['hands'])} for it in items]).write_parquet(OUT / 'candidates.parquet')
