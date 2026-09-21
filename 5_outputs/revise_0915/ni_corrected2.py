"""Corrected N/I test, v2: separates the PRIVATE partner-card channel from PUBLIC partner-state, and adds a confirmed
non-target control.  Models on pair-member decisions (cross-fitted by table fold):
  pC0 = population feats (29)                      pC1 = pC0 + public partner state (rel_pos, last action, folded, invested, n_act)
  pC  = pC1 + PRIVATE partner cards (chen, pf_*, str_now, gaps)
  lr_private = log pC - log pC1 ; lr_public = log pC1 - log pC0 ; nllN = population crossfit policy
Control: same three models on decisions of 300 sampled confirmed_non_target pairs (hands where both are dealt)."""
import sys, json; from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; PROC = ROOT / '1_data/processed'; RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
def log(*x): print(*x, flush=True)
folds_v5 = json.load(open(ROOT / '5_outputs/models/v5/folds_by_table.json')); folds_nb = json.load(open(ROOT / '5_outputs/models/v5nb/folds_by_table.json'))
mN = {g: lgb.Booster(model_file=str(PROC / f'policy_model_g{g}_nbcf.txt')) for g in (0, 1)}; featsN = mN[0].feature_name()
base = pl.read_parquet(PROC / 'player_baselines.parquet').filter(pl.col('bphase') == 'all').drop('bphase')
PUB = ['p_rel_pos', 'p_last_y', 'p_folded_before', 'p_invested_bb', 'p_n_act_before']
PRIV = ['p_chen', 'p_pf_pair', 'p_pf_suited', 'p_pf_hi', 'p_pf_lo', 'p_str_now', 'p_chen_gap', 'p_str_gap']

def member_decisions(pr):
    """pr: (pair_id, hand_id, a, b, table_id). Returns pair-member decision rows with population feats, public+private partner feats, logpN."""
    hids = pr.select('hand_id').unique()
    a = pl.read_parquet(PROC / 'action_ctx.parquet').join(hids, on='hand_id', how='semi')
    raw = pl.read_parquet(RAW / 'actions.parquet', columns=['hand_id', 'action_no', 'player_id', 'stack_before', 'action', 'amount']).join(hids, on='hand_id', how='semi')
    a = a.join(raw.drop('player_id'), on=['hand_id', 'action_no'])
    l0 = pl.read_parquet(PROC / 'seat_l0.parquet', columns=['hand_id', 'player_id', 'rel_pos', 'stack_bb', 'chen', 'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo', 'str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3']).join(hids, on='hand_id', how='semi')
    l1 = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'big_blind', 'n_board', 'phase', 'table_id']).join(hids, on='hand_id', how='semi')
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
    a = a.with_columns(pl.col('table_id').replace_strict(folds_v5, return_dtype=pl.Int64).alias('fold_v5')).with_columns((pl.col('fold_v5') % 2).alias('grp'))
    XN = a.select(featsN).to_numpy().astype(np.float32); y = a['y'].to_numpy(); grp = a['grp'].to_numpy(); pN = np.empty(a.height, np.float32)
    for g in (0, 1):
        ii = np.flatnonzero(grp == g); p = mN[1 - g].predict(XN[ii]); pN[ii] = p[np.arange(len(ii)), y[ii]]
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
    m1 = m1.with_columns((pl.col('p_invested') / pl.col('big_blind')).cast(pl.Float32).alias('p_invested_bb'), pl.col('table_id').replace_strict(folds_nb, return_dtype=pl.Int64).alias('fold'))
    return m1

P = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=6, seed=42)
def crossfit_nll(m1, tag):
    sets = {'C0': featsN, 'C1': featsN + PUB, 'C': featsN + PUB + PRIV}
    X = {k: m1.select(v).to_numpy().astype(np.float32) for k, v in sets.items()}; y = m1['y'].to_numpy(); fold = m1['fold'].to_numpy()
    out = {k: np.zeros(m1.height, np.float32) for k in sets}
    for fo in range(5):
        tr = np.flatnonzero(fold != fo); va = np.flatnonzero(fold == fo)
        if len(va) == 0: continue
        for k in sets:
            m = lgb.train(P, lgb.Dataset(X[k][tr], y[tr]), 300); p = m.predict(X[k][va]); out[k][va] = np.log(np.clip(p[np.arange(len(va)), y[va]], 1e-6, 1))
    nll = {k: float(-out[k].mean()) for k in sets}; nll['N'] = float(-m1['logpN'].mean())
    log(f'[{tag}] decisions {m1.height:,}  NLL  N {nll["N"]:.4f}  C0 {nll["C0"]:.4f}  C1(+public partner) {nll["C1"]:.4f}  C(+private partner cards) {nll["C"]:.4f}  '
        f'-> public gain {nll["C0"]-nll["C1"]:+.4f}  PRIVATE gain {nll["C1"]-nll["C"]:+.4f} nats/decision')
    return m1.with_columns(pl.Series('logpC0', out['C0']), pl.Series('logpC1', out['C1']), pl.Series('logpC', out['C'])), nll

# ---- positive pairs (candidate hands) ----
d = pl.read_parquet(HERE / 'gate_frame.parquet', columns=['pair_id', 'hand_id', 'a', 'b', 'table_id', 'behavior_family', 'is_ev', 'is_miss', 'is_fp', 'brk'])
_sc = ROOT / '5_outputs/evsel_0915/dev_candidates_scripts.parquet'   # (release) research-only script labels, printed diagnostics only
d = d.join(pl.read_parquet(_sc, columns=['pair_id', 'hand_id', 'script']), on=['pair_id', 'hand_id'], how='left') if _sc.exists() else d.with_columns(pl.lit(None, pl.Utf8).alias('script'))
pos = member_decisions(d.select(['pair_id', 'hand_id', 'a', 'b', 'table_id']).unique())
pos, nll_pos = crossfit_nll(pos, 'positive pairs')
# ---- control: 300 confirmed non-target pairs, hands where both are dealt ----
lab = pl.read_csv(RAW / 'development_labels.csv').filter(pl.col('label') == 0).sample(300, seed=0).select(['pair_id', pl.col('player_1').alias('a'), pl.col('player_2').alias('b')])
seat = pl.scan_parquet(PROC / 'seat_l0.parquet').select(['hand_id', 'player_id']).filter(pl.col('player_id').is_in(lab['a'].to_list() + lab['b'].to_list())).collect()
ph = lab.join(seat.rename({'player_id': 'a'}), on='a').join(seat.rename({'player_id': 'b'}), on=['b', 'hand_id'])
ph = ph.join(pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id', 'phase']).filter(pl.col('phase') == 'development'), on='hand_id').select(['pair_id', 'hand_id', 'a', 'b', 'table_id']).unique()
log(f'control: {lab.height} non-target pairs, {ph.height:,} pair-hands')
neg = member_decisions(ph); neg, nll_neg = crossfit_nll(neg, 'non-target control')
# ---- per-family private gain on positive pairs (evidence-hand decisions vs other decisions) ----
pos = pos.with_columns((pl.col('logpC') - pl.col('logpC1')).alias('lr_private'), (pl.col('logpC1') - pl.col('logpC0')).alias('lr_public'), (pl.col('logpC') - pl.col('logpN')).alias('lr_total'))
pos = pos.join(d.select(['pair_id', 'hand_id', 'is_ev', 'behavior_family']), on=['pair_id', 'hand_id'], how='left')
print(pos.group_by(['behavior_family', 'is_ev']).agg(pl.len(), pl.col('lr_private').mean().round(4).alias('lr_private_mean'), pl.col('lr_public').mean().round(4).alias('lr_public_mean')).sort(['behavior_family', 'is_ev']))
hf = pos.group_by(['pair_id', 'hand_id']).agg(
    *[pl.col(c).max().alias(f'{c}_max') for c in ['lr_private', 'lr_public', 'lr_total']], *[pl.col(c).sum().alias(f'{c}_sum') for c in ['lr_private', 'lr_public', 'lr_total']],
    pl.col('lr_private').filter(pl.col('is_fold')).max().alias('lr_private_fold_max'), pl.col('lr_private').filter(pl.col('is_call')).max().alias('lr_private_call_max'),
    pl.col('lr_private').filter(pl.col('is_agg')).max().alias('lr_private_agg_max'), pl.col('lr_private').filter(pl.col('p_folded_before') == 0).max().alias('lr_private_pin_max'),
    pl.col('lr_private').filter(pl.col('p_folded_before') == 0).sum().alias('lr_private_pin_sum'))
g = d.join(hf, on=['pair_id', 'hand_id'], how='left'); newc = [c for c in hf.columns if c not in ('pair_id', 'hand_id')]
g.select(['pair_id', 'hand_id'] + newc).write_parquet(HERE / 'ni_corrected2.parquet')
def wp_auc(df, col, p_, n_):
    xx = df.filter((pl.col(p_) | pl.col(n_)) & pl.col(col).is_not_null()).select(['pair_id', pl.col(col).cast(pl.Float64), p_]).to_pandas()
    num = den = 0.0
    for _, gg in xx.groupby('pair_id'):
        p = gg.loc[gg[p_], col].to_numpy(); n = gg.loc[~gg[p_], col].to_numpy()
        if len(p) == 0 or len(n) == 0: continue
        dd = p[:, None] - n[None, :]; num += (dd > 0).sum() + 0.5 * (dd == 0).sum(); den += dd.size
    return num / den if den else float('nan')
g = g.with_columns((~pl.col('is_ev')).alias('not_ev'), (pl.col('script') != 'E').alias('sig'))
fams = ['directed_transfer', 'soft_play', 'coordinated_isolation']
print(f'\n{"col":22s} ' + ' '.join(f'{f[:2]:>28s}' for f in fams)); print(f'{"":22s} ' + ' '.join(f'{"miss/fp  ev/rest  ev/rest@sig":>28s}' for _ in fams))
res = {'nll_pos': nll_pos, 'nll_neg': nll_neg}
for c in newc:
    row = []
    for fam in fams:
        gf = g.filter(pl.col('behavior_family') == fam); gs = gf.filter(pl.col('sig'))
        v = (wp_auc(gf, c, 'is_miss', 'is_fp'), wp_auc(gf, c, 'is_ev', 'not_ev'), wp_auc(gs, c, 'is_ev', 'not_ev')); res[f'{c}|{fam}'] = v
        row.append(f'{v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f}'.rjust(28))
    print(f'{c:22s} ' + ' '.join(row))
json.dump(res, open(HERE / 'ni_corrected2.json', 'w'), indent=1)
