"""Materialise L2 v2: join equity + surprise into each table's L2 file once (avoids repeated 12M-row hash joins).
Writes 1_data/processed/l2v2/{phase}/{table}.parquet
"""
import sys, time
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from tpds_paths import vpath, eqpath, L2V2
SURP_P = ['surp_sum', 'surp_max', 'surpw_sum', 'surpw_max', 'surp_fold', 'surp_call', 'surp_agg', 'surp_post', 'p_min']


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(phase):
    outdir = PROC / L2V2 / phase; outdir.mkdir(parents=True, exist_ok=True)
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id', 'phase']).filter(pl.col('phase') == phase)
    eq = pl.read_parquet(eqpath(f'equity_{phase}.parquet')).join(l1.select(['hand_id', 'table_id']), on='hand_id')
    pfeq_path = eqpath(f'pfeq_{phase}.parquet')
    pfeq_d = None
    if pfeq_path.exists():
        pfeq_d = pl.read_parquet(pfeq_path).join(l1.select(['hand_id', 'table_id']), on='hand_id').partition_by('table_id', as_dict=True)
    sp = pl.read_parquet(vpath('surprise_player.parquet')).join(l1.select(['hand_id', 'table_id']), on='hand_id')
    sr = pl.read_parquet(vpath('surprise_resp.parquet')).join(l1.select(['hand_id', 'table_id']), on='hand_id')
    eq_d = eq.partition_by('table_id', as_dict=True); sp_d = sp.partition_by('table_id', as_dict=True); sr_d = sr.partition_by('table_id', as_dict=True)
    files = sorted((PROC / 'l2' / phase).glob('*.parquet'))
    t0 = time.time()
    for i, f in enumerate(files):
        t = f.stem
        out = outdir / f'{t}.parquet'
        if out.exists():
            continue
        d = pl.read_parquet(f)
        e = eq_d.get((t,), None); p = sp_d.get((t,), None); r = sr_d.get((t,), None)
        if e is not None:
            d = d.join(e.drop('table_id'), on=['hand_id', 'a', 'b'], how='left')
        if pfeq_d is not None:
            pe = pfeq_d.get((t,), None)
            if pe is not None:
                d = d.join(pe.drop('table_id'), on=['hand_id', 'a', 'b'], how='left')
        if p is not None:
            p = p.drop('table_id')
            d = d.join(p.rename({'player_id': 'a', **{c: f'A_{c}' for c in SURP_P}}), on=['hand_id', 'a'], how='left')
            d = d.join(p.rename({'player_id': 'b', **{c: f'B_{c}' for c in SURP_P}}), on=['hand_id', 'b'], how='left')
        if r is not None:
            r = r.drop('table_id')
            d = d.join(r.rename({'responder': 'a', 'aggressor': 'b', 'rsurp_max': 'rAB_surp_max', 'rsurp_sum': 'rAB_surp_sum', 'rsurpw_max': 'rAB_surpw_max'}), on=['hand_id', 'a', 'b'], how='left')
            d = d.join(r.rename({'responder': 'b', 'aggressor': 'a', 'rsurp_max': 'rBA_surp_max', 'rsurp_sum': 'rBA_surp_sum', 'rsurpw_max': 'rBA_surpw_max'}), on=['hand_id', 'a', 'b'], how='left')
        d = d.with_columns(
            ((pl.col('A_fold_to_B') == 1) & (pl.col('A_eq_at_fold') > 0.5)).cast(pl.Int8).alias('A_fold_eqplus_to_B'),
            ((pl.col('B_fold_to_A') == 1) & (pl.col('B_eq_at_fold') > 0.5)).cast(pl.Int8).alias('B_fold_eqplus_to_A'),
            (pl.col('eq_flop_A') - 0.5).abs().alias('eq_flop_abs'),
            pl.when(pl.col('A_fold_to_B') == 1).then(pl.col('A_eq_at_fold')).when(pl.col('B_fold_to_A') == 1).then(pl.col('B_eq_at_fold')).otherwise(None).cast(pl.Float32).alias('folder_eq_vs_partner'),
            pl.max_horizontal('A_surp_max', 'B_surp_max').alias('surp_max_pair'),
            (pl.col('A_surp_sum').fill_null(0) + pl.col('B_surp_sum').fill_null(0)).alias('surp_sum_pair'),
            pl.max_horizontal('rAB_surp_max', 'rBA_surp_max').alias('rsurp_max_pair'),
            pl.max_horizontal('A_surpw_max', 'B_surpw_max').alias('surpw_max_pair'),
        )
        if 'pf_eq_A' in d.columns:
            d = d.with_columns(
                (pl.col('pf_eq_A') - 0.5).abs().alias('pf_eq_abs'),
                # folded preflop while ahead of the partner
                ((pl.col('A_fold_street') == 0) & (pl.col('pf_eq_A') > 0.5)).cast(pl.Int8).alias('A_pf_fold_ahead'),
                ((pl.col('B_fold_street') == 0) & (pl.col('pf_eq_A') < 0.5)).cast(pl.Int8).alias('B_pf_fold_ahead'),
                # equity the folder gave up preflop, weighted by the pot
                pl.when(pl.col('A_fold_street') == 0).then((pl.col('pf_eq_A') - 0.5) * pl.col('A_fold_pot_bb'))
                  .when(pl.col('B_fold_street') == 0).then((0.5 - pl.col('pf_eq_A')) * pl.col('B_fold_pot_bb'))
                  .otherwise(None).cast(pl.Float32).alias('pf_fold_ev_loss'),
                # equity at the moment of folding, whichever street (preflop now covered)
                pl.when(pl.col('A_fold_street') == 0).then(pl.col('pf_eq_A')).otherwise(pl.col('A_eq_at_fold')).cast(pl.Float32).alias('A_eq_fold_any'),
                pl.when(pl.col('B_fold_street') == 0).then(1.0 - pl.col('pf_eq_A')).otherwise(pl.col('B_eq_at_fold')).cast(pl.Float32).alias('B_eq_fold_any'),
            )
        d.write_parquet(out)
        if i % 100 == 0:
            log(f"{phase}: {i+1}/{len(files)} {time.time()-t0:.0f}s cols={len(d.columns)}")
    log(f"{phase} done {time.time()-t0:.0f}s -> {outdir}")


if __name__ == '__main__':
    for ph in (sys.argv[1:] or ['development', 'evaluation']):
        run(ph)
