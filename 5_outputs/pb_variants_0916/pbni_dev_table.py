"""pbni_development.parquet = pb_development (16 pb_*) + ni_corrected2.parquet renamed ni_* (11, table-fold OOF, exactly the columns feat_test pbni used).
Checks equality with gate_frame's ni_* columns."""
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[2]; PROC = ROOT / '1_data/processed'; REV = ROOT / '5_outputs/revise_0915'
pb = pl.read_parquet(PROC / 'pb_development.parquet')
ni = pl.read_parquet(REV / 'ni_corrected2.parquet'); ni = ni.rename({c: 'ni_' + c for c in ni.columns if c not in ('pair_id', 'hand_id')})
NI = [c for c in ni.columns if c.startswith('ni_')]
ni = ni.with_columns([pl.col(c).cast(pl.Float32) for c in NI])
out = pb.join(ni, on=['pair_id', 'hand_id'], how='full', coalesce=True)
if all(c in pl.read_parquet_schema(REV / 'gate_frame.parquet') for c in NI):   # (release) self-check only when the frame carries ni_* values
  g = pl.read_parquet(REV / 'gate_frame.parquet', columns=['pair_id', 'hand_id'] + NI)
  chk = out.join(g, on=['pair_id', 'hand_id'], suffix='_g'); assert chk.height == g.height == out.height, (chk.height, g.height, out.height)
  bad = 0
  for c in NI:
    dmax = chk.select((pl.col(c).cast(pl.Float64) - pl.col(c + '_g').cast(pl.Float64)).abs().max()).item(); nn = int((chk[c].is_null() != chk[c + '_g'].is_null()).sum())
    bad += (dmax or 0) > 1e-4 or nn > 0; print(f'  {c:26s} max|diff| {dmax}  null mismatch {nn}  nonnull {float(chk[c].is_not_null().mean()):.3f}')
  assert bad == 0
out.write_parquet(PROC / 'pbni_development.parquet'); print('pbni_development.parquet', out.shape)
