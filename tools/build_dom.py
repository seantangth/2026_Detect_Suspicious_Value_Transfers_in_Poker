"""dom_{phase}_v5x.parquet: same-player 'dominance' of a pair's top-5 hand scores (struct_hand recipe; the build_dom function of
5_outputs/eqx_0913/pair_chain_v5xcp2.py, verbatim, with run v5x hand scores). For every candidate pair, its top-5 hands by the
generic hand score are compared with the scores the SAME hands receive in other candidate pairs that share a player."""
import sys
from pathlib import Path
import polars as pl
R = str(Path(__file__).resolve().parents[1]) + '/'
PROC = Path(R) / '1_data/processed'; M = R + '5_outputs/models/v5x'; OUT = R + '5_outputs/eqx_0913/'


def build_dom(phase):
    cp = pl.read_parquet(PROC / f'cand_pairs_{phase}.parquet').select('pair_id', 'a', 'b').with_row_index('pi')
    hs = pl.scan_parquet(M + f'/hand_scores_{"dev" if phase == "development" else "eval"}.parquet').select('pair_id', 'hand_id', 's_gen')
    F = hs.join(cp.lazy().select('pair_id', 'pi'), on='pair_id').select('pi', pl.col('hand_id').cast(pl.Categorical).to_physical().alias('h'), 's_gen').collect()
    T = F.sort('s_gen', descending=True).group_by('pi', maintain_order=True).head(5)
    J = T.join(F.rename({'pi': 'qi', 's_gen': 's_q'}), on='h').filter(pl.col('qi') != pl.col('pi'))
    ab = cp.select('pi', 'a', 'b')
    J = J.join(ab, on='pi').join(ab.rename({'pi': 'qi', 'a': 'qa', 'b': 'qb'}), on='qi')
    J = J.filter((pl.col('a') == pl.col('qa')) | (pl.col('a') == pl.col('qb')) | (pl.col('b') == pl.col('qa')) | (pl.col('b') == pl.col('qb')))
    Hm = J.group_by('pi', 'h').agg(nmax=pl.col('s_q').max())
    T = T.join(Hm, on=['pi', 'h'], how='left').with_columns(pl.col('nmax').fill_null(0.0))
    agg = T.group_by('pi').agg(dom5=(pl.col('nmax') > pl.col('s_gen')).mean(),
        dommass=((pl.col('nmax') > pl.col('s_gen')).cast(pl.Float32) * pl.col('s_gen')).sum() / pl.col('s_gen').sum(),
        ratio5=(pl.col('nmax') / pl.col('s_gen').clip(1e-6)).clip(0, 5).mean(), ntop=pl.len())
    return cp.join(agg, on='pi', how='left').select('pair_id', 'dom5', 'dommass', 'ratio5', 'ntop')


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT
    for ph in ('development', 'evaluation'):
        d = build_dom(ph); d.write_parquet(out_dir.rstrip('/') + f'/dom_{ph}_v5x.parquet'); print(ph, d.shape)
