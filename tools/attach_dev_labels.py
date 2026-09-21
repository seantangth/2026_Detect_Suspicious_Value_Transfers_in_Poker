"""Re-attach the development labels to the checkpoint tables that carry them (label / behavior_family columns).
The published checkpoint ships these two columns empty; they are filled here from your own copy of development_labels.csv:
label = 1 / 0 for the listed pairs, -1 for unlisted pairs; behavior_family = the listed family ('none' for confirmed non-targets), 'unknown' otherwise.
Column order and dtypes are preserved, so the restored tables are identical to the ones used for the submission (replay.sh checks the md5)."""
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FILES = ['1_data/processed/cand_pairs_development.parquet', '5_outputs/models/v5x/pair_features_dev.parquet',
         '5_outputs/research_0919/band/tw/W1/pair_features.parquet', '5_outputs/research_0919/band/tw/W2/pair_features.parquet']
lab = pl.read_csv(ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker/development_labels.csv').select(
    'pair_id', pl.col('label').cast(pl.Int8).alias('_label'), pl.col('behavior_family').alias('_family'))
for rel in FILES:
    p = ROOT / rel
    d = pl.read_parquet(p)
    if d['label'].null_count() == 0:
        print(f'{rel}: labels already present'); continue
    d = (d.join(lab, on='pair_id', how='left', maintain_order='left')
          .with_columns(pl.col('_label').fill_null(-1).cast(pl.Int8).alias('label'), pl.col('_family').fill_null('unknown').alias('behavior_family'))
          .drop('_label', '_family'))
    d.write_parquet(p)
    print(f'{rel}: labels attached ({int((d["label"] == 1).sum())} positive, {int((d["label"] == 0).sum())} confirmed non-target)')
