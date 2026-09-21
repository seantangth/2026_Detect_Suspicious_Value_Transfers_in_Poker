"""Rebuild 1_data/processed/pb_development.parquet from the cloud PokerBench scores (scores.parquet) and the dev decision
prompts (prompts.parquet, keys only): the per-(pair, hand) pb_* aggregation of pokerbench_features.py lines 7-38, without the research gate."""
import sys; from pathlib import Path
import numpy as np, polars as pl
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / '5_outputs/pokerbench_0915'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '1_data/processed/pb_development.parquet'
sc = pl.read_parquet(HERE / 'scores.parquet'); pr = pl.read_parquet(HERE / 'prompts.parquet')
if 'pair_idx' in pr.columns:   # the shipped key file stores each positive pair as its index in the sorted list of positive pair IDs
    pos_ids = sorted(pl.read_csv(ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker/development_labels.csv').filter(pl.col('label') == 1)['pair_id'].to_list())
    pr = pr.with_columns(pl.col('pair_idx').replace_strict(dict(enumerate(pos_ids)), return_dtype=pl.Utf8).alias('pair_idx')).rename({'pair_idx': 'pair_id'})
m = pr.join(sc, on=['hand_id', 'action_no'], how='left')
m = m.with_columns(pl.max_horizontal('lp_allin', 'lp_all').alias('lp_allin2'))
facing = pl.col('legal') == 'fold,call,raise'
def lse(cols): return pl.Series(np.logaddexp.reduce(np.column_stack([m[c].to_numpy() for c in cols]), axis=1))
m = m.with_columns(lse(['lp_fold', 'lp_call', 'lp_raise', 'lp_allin2']).alias('Z_face'), lse(['lp_check', 'lp_bet', 'lp_raise', 'lp_allin2']).alias('Z_nf'),
                   lse(['lp_raise', 'lp_allin2']).alias('A_face'), lse(['lp_bet', 'lp_raise', 'lp_allin2']).alias('A_nf'))
m = m.with_columns(
    pl.when(facing).then(pl.col('lp_fold') - pl.col('Z_face')).otherwise(None).alias('p_fold_l'),
    pl.when(facing).then(pl.col('lp_call') - pl.col('Z_face')).otherwise(None).alias('p_call_l'),
    pl.when(facing).then(pl.col('A_face') - pl.col('Z_face')).otherwise(pl.col('A_nf') - pl.col('Z_nf')).alias('p_agg_l'),
    pl.when(~facing).then(pl.col('lp_check') - pl.col('Z_nf')).otherwise(None).alias('p_check_l'))
taken_lp = (pl.when(pl.col('taken') == 'fold').then(pl.col('p_fold_l')).when(pl.col('taken') == 'call').then(pl.col('p_call_l'))
            .when(pl.col('taken') == 'check').then(pl.col('p_check_l')).otherwise(pl.col('p_agg_l')))
m = m.with_columns(taken_lp.alias('lp_taken')).with_columns((-pl.col('lp_taken')).alias('surp'))
best = pl.when(facing).then(pl.when((pl.col('p_fold_l') >= pl.col('p_call_l')) & (pl.col('p_fold_l') >= pl.col('p_agg_l'))).then(pl.lit('fold')).when(pl.col('p_call_l') >= pl.col('p_agg_l')).then(pl.lit('call')).otherwise(pl.lit('agg'))) \
         .otherwise(pl.when(pl.col('p_check_l') >= pl.col('p_agg_l')).then(pl.lit('check')).otherwise(pl.lit('agg')))
tk = pl.when(pl.col('taken').is_in(['bet', 'raise', 'allin'])).then(pl.lit('agg')).otherwise(pl.col('taken'))
m = m.with_columns(best.alias('best'), tk.alias('tk')).with_columns((pl.col('best') != pl.col('tk')).cast(pl.Float32).alias('disagree'))
hf = m.group_by(['pair_id', 'hand_id']).agg(
    pl.col('surp').max().alias('pb_surp_max'), pl.col('surp').sum().alias('pb_surp_sum'), pl.col('surp').mean().alias('pb_surp_mean'),
    pl.col('surp').filter(pl.col('role') == 'A').max().alias('pb_surp_A_max'), pl.col('surp').filter(pl.col('role') == 'B').max().alias('pb_surp_B_max'),
    pl.col('surp').filter(pl.col('taken') == 'fold').max().alias('pb_surp_fold_max'), pl.col('surp').filter(pl.col('taken') == 'call').max().alias('pb_surp_call_max'),
    pl.col('surp').filter(pl.col('tk') == 'agg').max().alias('pb_surp_agg_max'), pl.col('surp').filter(pl.col('taken') == 'check').max().alias('pb_surp_check_max'),
    pl.col('disagree').sum().alias('pb_n_disagree'), pl.col('disagree').mean().alias('pb_disagree_rate'),
    pl.col('p_fold_l').filter(pl.col('taken') == 'fold').min().alias('pb_fold_lp_min'), pl.col('p_call_l').filter(pl.col('taken') == 'call').min().alias('pb_call_lp_min'),
    pl.col('p_agg_l').filter(pl.col('tk') == 'agg').min().alias('pb_agg_lp_min'),
    pl.col('p_fold_l').filter(pl.col('legal') == 'fold,call,raise').max().alias('pb_pfold_max'), pl.len().cast(pl.Float32).alias('pb_n_dec'))
hf = hf.with_columns([pl.col(c).cast(pl.Float32) for c in hf.columns if c.startswith('pb_')])
hf.write_parquet(OUT); print('pb_development', hf.shape, '->', OUT)
