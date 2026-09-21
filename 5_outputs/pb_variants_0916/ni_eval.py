"""Eval-side corrected N/I hand features for v040 (= v036 pb recipe + 11 ni_* columns).
Dev features are the table-fold OOF of ni_corrected2.py (models trained on the other folds' positive-pair member decisions).
Eval: C0 / C1 / C multiclass policy models with the SAME features and params, trained on ALL dev positive-pair member decisions (300 rounds),
applied to the pair-member decisions of every candidate hand of the 8,000 pairs that have pb features (top-8,000 by v032 risk);
logpN from the population model mN[1-g] with g = table-hash parity (dev: v5 fold parity). Hand aggregation identical to ni_corrected2.py.
Writes 1_data/processed/pbni_evaluation.parquet = pb_evaluation (16 pb_*) full-joined with ni_* (11). Hook: TPDS_PB=1 TPDS_PB_TAG=ni."""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; REV = ROOT / '5_outputs/revise_0915'; PROC = ROOT / '1_data/processed'
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'; OUTM = ROOT / '5_outputs/models/v5nb'
def log(*x): print(time.strftime('[%H:%M:%S]'), *x, f'| maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB', flush=True)
def coll(lf):
    try:
        return lf.collect(engine='streaming')
    except Exception:
        return lf.collect()
folds_v5 = json.load(open(ROOT / '5_outputs/models/v5/folds_by_table.json'))
mN = {g: lgb.Booster(model_file=str(PROC / f'policy_model_g{g}_nbcf.txt')) for g in (0, 1)}; featsN = mN[0].feature_name()
base = pl.read_parquet(PROC / 'player_baselines.parquet').filter(pl.col('bphase') == 'all').drop('bphase')
PUB = ['p_rel_pos', 'p_last_y', 'p_folded_before', 'p_invested_bb', 'p_n_act_before']
PRIV = ['p_chen', 'p_pf_pair', 'p_pf_suited', 'p_pf_hi', 'p_pf_lo', 'p_str_now', 'p_chen_gap', 'p_str_gap']
SETS = {'C0': featsN, 'C1': featsN + PUB, 'C': featsN + PUB + PRIV}
P = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=6, seed=42)
L0COLS = ['hand_id', 'player_id', 'rel_pos', 'stack_bb', 'chen', 'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo', 'str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3']

def load_tables(hids):
    h = hids.lazy()
    a = coll(pl.scan_parquet(PROC / 'action_ctx.parquet').join(h, on='hand_id', how='semi'))
    raw = coll(pl.scan_parquet(RAW / 'actions.parquet').select(['hand_id', 'action_no', 'player_id', 'stack_before', 'action', 'amount']).join(h, on='hand_id', how='semi'))
    l0 = coll(pl.scan_parquet(PROC / 'seat_l0.parquet').select(L0COLS).join(h, on='hand_id', how='semi'))
    l1 = coll(pl.scan_parquet(PROC / 'hands_l1.parquet').select(['hand_id', 'big_blind', 'n_board', 'phase', 'table_id']).join(h, on='hand_id', how='semi'))
    return a, raw, l0, l1

def member_decisions(pr, tabs, phase):
    """ni_corrected2.member_decisions with pre-loaded tables; eval N-model group = table-hash parity; logpN computed on member rows (row-wise identical)."""
    hids = pr.select('hand_id').unique()
    a, raw, l0, l1 = [t.join(hids, on='hand_id', how='semi') for t in tabs]
    a = a.join(raw.drop('player_id'), on=['hand_id', 'action_no'])
    a = a.join(l0, on=['hand_id', 'player_id']).join(l1, on='hand_id').join(base, on='player_id', how='left')
    a = a.sort(['hand_id', 'action_no']).with_columns(
        (pl.col('stack_before') / pl.col('big_blind')).cast(pl.Float32).alias('stack_before_bb'),
        pl.col('is_agg').cast(pl.Int32).cum_sum().over('hand_id').alias('_cum_agg')).with_columns(
        (pl.col('_cum_agg') - pl.col('is_agg').cast(pl.Int32)).alias('n_agg_before'),
        (pl.col('stack_bb') - pl.col('stack_before') / pl.col('big_blind')).cast(pl.Float32).alias('invested_bb'),
        (pl.col('to_call_bb') / (pl.col('pot_before_bb') + pl.col('to_call_bb') + 1e-3)).cast(pl.Float32).alias('pot_odds'),
        (pl.col('stack_before') / pl.col('big_blind') / (pl.col('pot_before_bb') + 1e-3)).clip(0, 200).cast(pl.Float32).alias('spr'),
        pl.when(pl.col('sidx') == 1).then(pl.col('str_pct_1')).when(pl.col('sidx') == 2).then(pl.col('str_pct_2')).when(pl.col('sidx') == 3).then(pl.col('str_pct_3')).otherwise(None).cast(pl.Float32).alias('str_now'),
        pl.when(pl.col('sidx') == 1).then(pl.col('str_cat_1')).when(pl.col('sidx') == 2).then(pl.col('str_cat_2')).when(pl.col('sidx') == 3).then(pl.col('str_cat_3')).otherwise(None).cast(pl.Int8).alias('cat_now'),
        pl.when(pl.col('is_fold')).then(0).when(pl.col('action') == 'check').then(1).when(pl.col('is_call')).then(2).otherwise(3).cast(pl.Int8).alias('y'),
        pl.col('facing').cast(pl.Int8))
    if phase == 'development':
        a = a.with_columns(pl.col('table_id').replace_strict(folds_v5, return_dtype=pl.Int64).alias('fold_v5')).with_columns((pl.col('fold_v5') % 2).alias('grp'))
    else:
        a = a.with_columns((pl.col('table_id').hash(seed=0) % 2).cast(pl.Int64).alias('grp'))
    mem = pl.concat([pr.select(['hand_id', pl.col('a').alias('player_id')]), pr.select(['hand_id', pl.col('b').alias('player_id')])]).unique()
    a = a.join(mem, on=['hand_id', 'player_id'], how='semi')
    XN = a.select(featsN).to_numpy().astype(np.float32); y = a['y'].to_numpy(); grp = a['grp'].to_numpy(); pN = np.empty(a.height, np.float32)
    for g in (0, 1):
        ii = np.flatnonzero(grp == g)
        if len(ii):
            p = mN[1 - g].predict(XN[ii]); pN[ii] = p[np.arange(len(ii)), y[ii]]
    a = a.with_columns(pl.Series('logpN', np.log(np.clip(pN, 1e-6, 1))))
    m1 = a.drop('table_id').join(pr, on='hand_id', how='inner').filter((pl.col('player_id') == pl.col('a')) | (pl.col('player_id') == pl.col('b')))
    m1 = m1.with_columns(pl.when(pl.col('player_id') == pl.col('a')).then(pl.col('b')).otherwise(pl.col('a')).alias('partner'))
    pl0 = l0.select(['hand_id', 'player_id', 'rel_pos', 'chen', 'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo', 'str_pct_1', 'str_pct_2', 'str_pct_3']).rename(
        {'player_id': 'partner', 'rel_pos': 'p_rel_pos', 'chen': 'p_chen', 'pf_pair': 'p_pf_pair', 'pf_suited': 'p_pf_suited', 'pf_hi': 'p_pf_hi', 'pf_lo': 'p_pf_lo', 'str_pct_1': 'p_s1', 'str_pct_2': 'p_s2', 'str_pct_3': 'p_s3'})
    m1 = m1.join(pl0, on=['hand_id', 'partner'], how='inner').with_columns(
        pl.when(pl.col('sidx') == 1).then(pl.col('p_s1')).when(pl.col('sidx') == 2).then(pl.col('p_s2')).when(pl.col('sidx') == 3).then(pl.col('p_s3')).otherwise(None).cast(pl.Float32).alias('p_str_now'),
        (pl.col('p_chen') - pl.col('chen')).cast(pl.Float32).alias('p_chen_gap'))
    m1 = m1.with_columns((pl.col('p_str_now') - pl.col('str_now')).cast(pl.Float32).alias('p_str_gap'))
    pa = raw.select(['hand_id', 'action_no', 'player_id', 'action', 'amount']).rename({'player_id': 'partner', 'action_no': 'p_action_no', 'action': 'p_action', 'amount': 'p_amount'})
    pa = pa.with_columns(pl.when(pl.col('p_action') == 'fold').then(0).when(pl.col('p_action') == 'check').then(1).when(pl.col('p_action') == 'call').then(2).otherwise(3).cast(pl.Int8).alias('p_y'))
    pa = pa.sort(['hand_id', 'partner', 'p_action_no']).with_columns(pl.col('p_amount').cum_sum().over(['hand_id', 'partner']).alias('p_cum_amt'))
    m1 = m1.with_row_index('ri')
    tmp = m1.select(['ri', 'hand_id', 'partner', 'action_no']).join(pa, on=['hand_id', 'partner'], how='left').filter(pl.col('p_action_no') < pl.col('action_no'))
    tmp = tmp.sort(['ri', 'p_action_no']).group_by('ri', maintain_order=True).agg(pl.col('p_y').last().alias('p_last_y'), (pl.col('p_action') == 'fold').any().cast(pl.Int8).alias('p_folded_before'), pl.col('p_cum_amt').last().alias('p_invested'), pl.len().cast(pl.Int8).alias('p_n_act_before'))
    m1 = m1.join(tmp, on='ri', how='left').with_columns(pl.col('p_last_y').fill_null(-1).cast(pl.Int8), pl.col('p_folded_before').fill_null(0), pl.col('p_invested').fill_null(0).cast(pl.Float32), pl.col('p_n_act_before').fill_null(0))
    m1 = m1.with_columns((pl.col('p_invested') / pl.col('big_blind')).cast(pl.Float32).alias('p_invested_bb'))
    return m1

def aggregate(pos):
    return pos.group_by(['pair_id', 'hand_id']).agg(
        *[pl.col(c).max().alias(f'{c}_max') for c in ['lr_private', 'lr_public', 'lr_total']], *[pl.col(c).sum().alias(f'{c}_sum') for c in ['lr_private', 'lr_public', 'lr_total']],
        pl.col('lr_private').filter(pl.col('is_fold')).max().alias('lr_private_fold_max'), pl.col('lr_private').filter(pl.col('is_call')).max().alias('lr_private_call_max'),
        pl.col('lr_private').filter(pl.col('is_agg')).max().alias('lr_private_agg_max'), pl.col('lr_private').filter(pl.col('p_folded_before') == 0).max().alias('lr_private_pin_max'),
        pl.col('lr_private').filter(pl.col('p_folded_before') == 0).sum().alias('lr_private_pin_sum'))

def add_lr(m1, models):
    y = m1['y'].to_numpy()
    for k, cols in SETS.items():
        p = models[k].predict(m1.select(cols).to_numpy().astype(np.float32))
        m1 = m1.with_columns(pl.Series(f'logp{k}', np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).astype(np.float32)))
    return m1.with_columns((pl.col('logpC') - pl.col('logpC1')).alias('lr_private'), (pl.col('logpC1') - pl.col('logpC0')).alias('lr_public'), (pl.col('logpC') - pl.col('logpN')).alias('lr_total'))

# ---- 1) all-dev policy models ----
try:
    print('ni_corrected2.json (dev OOF reference):', str(json.load(open(REV / 'ni_corrected2.json')))[:400], flush=True)
except Exception as e:
    print('no ni_corrected2.json', e)
d = pl.read_parquet(REV / 'gate_frame.parquet', columns=['pair_id', 'hand_id', 'a', 'b', 'table_id']).unique()
tabs = load_tables(d.select('hand_id').unique()); pos = member_decisions(d, tabs, 'development'); del tabs; gc.collect()
log('dev positive-pair member decisions', pos.height, '| mean logpN', float(pos['logpN'].mean()))
y = pos['y'].to_numpy(); models = {}
for k, cols in SETS.items():
    X = pos.select(cols).to_numpy().astype(np.float32); m = lgb.train(P, lgb.Dataset(X, y), 300); models[k] = m
    p = m.predict(X); log(f'  model {k} ({len(cols)} feats): in-sample NLL {-np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean():.4f}')
    m.save_model(str(HERE / f'ni_policy_{k}_alldev.txt'))
del pos, X; gc.collect()
# ---- 2) eval candidate pair-hands of the pb pairs ----
top = pl.read_parquet(PROC / 'pb_evaluation.parquet', columns=['pair_id']).unique()
cand = coll(pl.scan_parquet(OUTM / 'evidence_eval_pfonscvxpb4.parquet').select(['pair_id', 'hand_id']).join(top.lazy(), on='pair_id', how='semi'))
ep = pl.read_csv(RAW / 'evaluation_pairs.csv').select(['pair_id', pl.col('player_1').alias('a'), pl.col('player_2').alias('b')])
h1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id'])
pre = cand.join(ep, on='pair_id', how='left').join(h1, on='hand_id', how='left'); assert pre['a'].null_count() == 0 and pre['table_id'].null_count() == 0
log('eval candidate pair-hands', pre.height, '| pairs', pre['pair_id'].n_unique(), '| hands', pre['hand_id'].n_unique())
tabs = load_tables(pre.select('hand_id').unique()); log('eval tables loaded', [t.height for t in tabs])
pairs = sorted(pre['pair_id'].unique().to_list()); CH = 1000; outs = []; n_dec = 0
for ci in range(0, len(pairs), CH):
    prc = pre.join(pl.DataFrame({'pair_id': pairs[ci:ci + CH]}), on='pair_id', how='semi')
    m1 = add_lr(member_decisions(prc, tabs, 'evaluation'), models); n_dec += m1.height
    outs.append(aggregate(m1)); log(f'  chunk {ci // CH + 1}/{(len(pairs) + CH - 1) // CH}: decisions {m1.height:,} | lr_private mean {float(m1["lr_private"].mean()):+.4f} | lr_total mean {float(m1["lr_total"].mean()):+.4f}')
    del m1; gc.collect()
hf = pl.concat(outs); hf = hf.rename({c: 'ni_' + c for c in hf.columns if c not in ('pair_id', 'hand_id')})
NI = [c for c in hf.columns if c.startswith('ni_')]; hf = hf.with_columns([pl.col(c).cast(pl.Float32) for c in NI])
pbe = pl.read_parquet(PROC / 'pb_evaluation.parquet'); out = pbe.join(hf, on=['pair_id', 'hand_id'], how='full', coalesce=True)
dev = pl.read_parquet(PROC / 'pbni_development.parquet'); assert out.columns == dev.columns, (out.columns, dev.columns)
out.write_parquet(PROC / 'pbni_evaluation.parquet'); log('pbni_evaluation.parquet', out.shape, '| decisions', n_dec, '| rows with ni', int(out['ni_lr_total_max'].is_not_null().sum()), '| rows with pb', int(out['pb_n_dec'].is_not_null().sum()))
_v032 = ROOT / '5_outputs/submissions/submission_v032cand_v5xcp2_eqx4.csv'   # (release) research-era submission, feeds the printed column means only
if not _v032.exists(): log('NI_EVAL_DONE'); raise SystemExit(0)
sub = pl.read_csv(_v032, columns=['pair_id', 'risk_score']).with_columns(pl.col('risk_score').rank(descending=True).alias('rk'))
t500 = out.join(sub.filter(pl.col('rk') <= 500).select('pair_id'), on='pair_id', how='semi')
print('column means  dev(OOF, positive pairs) | eval top-500 pairs | eval all 8000:')
for c in NI: print(f'  {c:26s} {float(dev[c].mean()):+.4f} | {float(t500[c].mean()):+.4f} | {float(out[c].mean()):+.4f}   nonnull {float(dev[c].is_not_null().mean()):.3f}/{float(t500[c].is_not_null().mean()):.3f}')
log('NI_EVAL_DONE')
