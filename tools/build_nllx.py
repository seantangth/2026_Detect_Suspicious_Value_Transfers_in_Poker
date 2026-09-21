"""nllx_hand_player.parquet: entropy-adjusted normal-behaviour surprisal per (hand, player) = sum over the player's actions in the
hand of (nll_type_N - entropy_N), from the cross-fitted gbnll normal-behaviour model output (5_outputs/seqnll_0912/gbnll_action.parquet).
Under the normal policy E[nll_type - entropy] = 0 in every decision state, which removes the state-mix confound of raw NLL."""
import sys
from pathlib import Path
import polars as pl
R = str(Path(__file__).resolve().parents[1]) + '/'
out = sys.argv[1] if len(sys.argv) > 1 else R + '5_outputs/eqx_0913/nllx_hand_player.parquet'
x = (pl.scan_parquet(R + '5_outputs/seqnll_0912/gbnll_action.parquet').select('hand_id', 'player_id', 'nll_type_N', 'entropy_N')
     .with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8), (pl.col('nll_type_N') - pl.col('entropy_N')).alias('x'))
     .group_by('hand_id', 'player_id').agg(pl.col('x').sum().alias('nllx_sum')).collect(engine='streaming'))
x = x.with_columns(pl.col('nllx_sum').cast(pl.Float32))
x.write_parquet(out); print('nllx_hand_player', x.shape, '->', out)
