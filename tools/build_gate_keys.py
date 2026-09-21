"""Development (pair, hand) key frame of the public positive pairs, written where the private-information scripts expect it
(5_outputs/revise_0915/gate_frame.parquet).

Rows = every row of the 372 labelled positive pairs in the per-(pair, hand) tables (1_data/processed/l2/development), with the pair's
family (development_labels.csv) and whether the hand is a listed evidence hand (development_evidence.csv). This equals the key set of the
research frame used during the competition (28,268 rows). Its research-only diagnostic columns (is_miss, is_fp, brk: misses / false
positives of an earlier evidence ranker) only feed printed diagnostics and are filled with neutral values here."""
import glob
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
pos = pl.read_csv(RAW / 'development_labels.csv').filter(pl.col('label') == 1).select('pair_id', 'behavior_family')
rows = pl.concat([pl.read_parquet(f, columns=['pair_id', 'hand_id', 'a', 'b', 'table_id', 't_rank']).join(pos.select('pair_id'), on='pair_id', how='semi')
                  for f in sorted(glob.glob(str(ROOT / '1_data/processed/l2/development/*.parquet')))])
ev = pl.read_csv(RAW / 'development_evidence.csv').select('pair_id', 'hand_id').with_columns(pl.lit(True).alias('is_ev'))
out = (rows.join(pos, on='pair_id').join(ev, on=['pair_id', 'hand_id'], how='left')
       .with_columns(pl.col('is_ev').fill_null(False), pl.lit(False).alias('is_miss'), pl.lit(False).alias('is_fp'), pl.lit(0, pl.Int64).alias('brk'))
       .select('pair_id', 'hand_id', 'a', 'b', 'table_id', 't_rank', 'behavior_family', 'is_ev', 'is_miss', 'is_fp', 'brk')
       .sort(['pair_id', 't_rank', 'hand_id']))
assert out.height == 28268 and out['pair_id'].n_unique() == 372 and int(out['is_ev'].sum()) == pl.read_csv(RAW / 'development_evidence.csv').height
dst = ROOT / '5_outputs/revise_0915/gate_frame.parquet'; dst.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(dst); print('gate_frame keys', out.shape, '->', dst)
