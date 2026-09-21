"""TPDS modelling v1: hand-level LightGBM (generic + 3 families) -> per-pair pooling -> pair-level LightGBM (PU).
Folds: GroupKFold by table_id shared by both levels. Time (t_rank/session_id) used only for windowed pooling.
Outputs: 5_outputs/models/<run>/ {hand_oof, pair_features, pair_oof, submission}.
"""
import os, sys, time, json, glob, gc
from pathlib import Path
import numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '1_data/raw/detect-suspicious-value-transfers-in-poker'
PROC = ROOT / '1_data/processed'
sys.path.insert(0, str(ROOT / '3_src'))
from metric import score as kaggle_score
from tpds_paths import L2V2, eqpath

FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
KEY_COLS = {'hand_id', 'table_id', 'a', 'b', 'pair_id', 't_rank', 'session_id'}
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- A/B symmetry
def swap_name(c):
    for pre in ('pct_', 'q_', 'm_', 'mx_', 't3_'):
        if c.startswith(pre):
            return pre + swap_name(c[len(pre):])
    special = {'A_fold_to_B': 'B_fold_to_A', 'B_fold_to_A': 'A_fold_to_B',
               'A_fold_to_B_after_war': 'B_fold_to_A_after_war', 'B_fold_to_A_after_war': 'A_fold_to_B_after_war',
               'A_built_pot_for_B': 'B_built_pot_for_A', 'B_built_pot_for_A': 'A_built_pot_for_B',
               'A_sd_worse': 'B_sd_worse', 'B_sd_worse': 'A_sd_worse',
               'A_pf_fold_ahead': 'B_pf_fold_ahead', 'B_pf_fold_ahead': 'A_pf_fold_ahead',
               'A_eq_fold_any': 'B_eq_fold_any', 'B_eq_fold_any': 'A_eq_fold_any',
               'A_self_vs_pop_max': 'B_self_vs_pop_max', 'B_self_vs_pop_max': 'A_self_vs_pop_max',
               'A_fold_eqplus_to_B': 'B_fold_eqplus_to_A', 'B_fold_eqplus_to_A': 'A_fold_eqplus_to_B',
               'A_folded_better_to_B': 'B_folded_better_to_A', 'B_folded_better_to_A': 'A_folded_better_to_B',
               'tr_A_to_B': 'tr_B_to_A', 'tr_B_to_A': 'tr_A_to_B'}
    if c in special:
        return special[c]
    for p, q in (('A_', 'B_'), ('Ab_', 'Bb_'), ('rAB_', 'rBA_'), ('rtA_', 'rtB_')):
        if c.startswith(p):
            return q + c[len(p):]
        if c.startswith(q):
            return p + c[len(q):]
    return c


SIGN_FLIP = ['chen_diff', 'tr_dir', 'river_margin_AB']
ONE_MINUS = ['eq_flop_A', 'eq_turn_A', 'eq_river_A', 'pf_eq_A']


def swap_ab(df: pl.DataFrame) -> pl.DataFrame:
    d = df.rename({c: swap_name(c) for c in df.columns})
    d = d.with_columns([(-pl.col(c)).alias(c) for c in SIGN_FLIP if c in d.columns])
    d = d.with_columns([(1.0 - pl.col(c)).alias(c) for c in ONE_MINUS if c in d.columns])
    return d.select(df.columns)


# ----------------------------------------------------------------------------- data access
def l2_files(phase):
    return sorted(glob.glob(str(PROC / 'l2' / phase / '*.parquet')))


_EQ_CACHE = {}


_SURP_CACHE = {}
SURP_P = ['surp_sum', 'surp_max', 'surpw_sum', 'surpw_max', 'surp_fold', 'surp_call', 'surp_agg', 'surp_post', 'p_min']
PCT_BASE = ['pot_bb', 'pair_contrib', 'tr_any', 'net_gap', 'hu_streets', 'chen_min', 'chen_max', 'A_n_agg', 'B_n_agg',
            'surp_max_pair', 'surpw_max_pair', 'rsurp_max_pair', 'A_surp_max', 'B_surp_max', 'eq_flop_abs', 'folder_eq_vs_partner',
            'pf_eq_abs', 'pf_fold_ev_loss', 'fold_ev_loss_max', 'call_regret_max', 'call_regret_sum', 'call_edge_min', 'pass_val_max', 'pass_val_sum',
            'nll_max_pair', 'nll_sum_pair', 'gain_max_pair', 'gain_sum_pair', 'excess_max_pair', 'excess_sum_pair']


# --- pct_* length calibration (2026-09-18, audit_0917/stat-power tP/tQ2). Env TPDS_PCT; default '' = rank/n within the pair
# (byte-identical legacy). 'shrink<a>' (e.g. shrink30) = empirical-Bayes shrinkage of the within-pair percentile toward the
# table-phase percentile: (rank_in_pair + a * pct_in_table) / (n_nonnull_in_pair + a); nulls stay null. Short pairs no longer
# reach the extremes by chance (dev null pairs: >=3 extreme pct cols per hand short/long 1.33x -> 0.92x), and the denominator
# no longer counts null rows (pct_eq_flop_abs is ~90% null).
PCT_MODE = os.environ.get('TPDS_PCT', '')
if PCT_MODE:
    print(f'[tpds_model] TPDS_PCT={PCT_MODE!r}: length-calibrated pct_* features', flush=True)


_SELF_CACHE = {}
SELF_C = ['self_surp_sum', 'self_surp_max', 'self_vs_pop_max', 'self_vs_pop_sum', 'self_surp_fold', 'self_surp_call', 'self_surp_agg', 'self_surpw_max']


def read_l2(f, phase, use_equity=0, use_surprise=0, use_self=0):
    """use_equity/use_surprise >=1 -> read the augmented l2v2 file; >=2 -> also add per-pair percentile features."""
    aug = use_equity or use_surprise
    path = Path(str(f).replace('/l2/', f'/{L2V2}/')) if aug else Path(f)
    d = pl.read_parquet(path)
    if use_self:
        if 'sp' not in _SELF_CACHE:
            _SELF_CACHE['sp'] = pl.read_parquet(PROC / 'selfpolicy_player.parquet')
        sp = _SELF_CACHE['sp']
        d = d.join(sp.rename({'player_id': 'a', **{c: f'A_{c}' for c in SELF_C}}), on=['hand_id', 'a'], how='left')
        d = d.join(sp.rename({'player_id': 'b', **{c: f'B_{c}' for c in SELF_C}}), on=['hand_id', 'b'], how='left')
        d = d.with_columns(
            pl.max_horizontal('A_self_surp_max', 'B_self_surp_max').alias('self_max_pair'),
            (pl.col('A_self_surp_sum').fill_null(0) + pl.col('B_self_surp_sum').fill_null(0)).alias('self_sum_pair'),
            pl.max_horizontal('A_self_vs_pop_max', 'B_self_vs_pop_max').alias('svp_max_pair'))
    if USE_ONSET:
        on = _onset(phase)
        d = d.join(on, on=['pair_id', 'hand_id'], how='left')
        d = d.with_columns([pl.col(c).fill_null(0.0) for c in on.columns if c not in ('pair_id', 'hand_id')])
    if USE_CALLVAL:
        cv = _callval(phase)
        d = d.join(cv, on=['hand_id', 'a', 'b'], how='left')
        d = d.with_columns([pl.col(c).fill_null(0) for c in cv.columns if c.endswith(CALLVAL_FILL0)])   # no call -> 0; eq/edge stay null
    if USE_PASSVAL:
        pv = _passval(phase)
        d = d.join(pv, on=['hand_id', 'a', 'b'], how='left')
        d = d.with_columns([pl.col(c).fill_null(0) for c in pv.columns if c.endswith(PASSVAL_FILL0)])   # no passive action -> 0; edge stays null
    if USE_NBRDOM:
        nd = _nbrdom(phase)
        d = d.join(nd, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('nbr_dom').fill_null(0))
    if USE_PB:
        d = d.join(_pb(phase), on=['pair_id', 'hand_id'], how='left')
    if USE_AGGVAL:
        av = _aggval(phase)
        d = d.join(av, on=['hand_id', 'a', 'b'], how='left')
        d = d.with_columns([pl.col(c).fill_null(0) for c in av.columns if c.endswith(AGGVAL_FILL0)])   # no aggression -> 0; edge stays null
    if USE_NLL:
        nl = _nll_shard(f, phase)
        d = d.join(nl, on=['pair_id', 'hand_id'], how='left')
        d = d.with_columns([pl.col(c).fill_null(0.0) for c in nl.columns if c.endswith(NLL_FILL0)])   # no action -> 0; max/excess stay null
    if USE_SEQ:
        sq = _seq_shard(f, phase)
        if sq is not None:
            d = d.join(sq, on=['hand_id', 'a', 'b'], how='left')
            d = d.with_columns([pl.col(c).fill_null(0.0) for c in sq.columns if c not in ('hand_id', 'a', 'b')])
    if max(use_equity, use_surprise) >= 2:
        cols = [c for c in PCT_BASE if c in d.columns]
        if PCT_MODE.startswith('shrink'):
            a = float(PCT_MODE[len('shrink'):] or 30)
            d = d.with_columns([((pl.col(c).rank(method='average').over('pair_id') + a * pl.col(c).rank(method='average') / pl.col(c).count())
                                 / (pl.col(c).count().over('pair_id') + a)).cast(pl.Float32).alias(f'pct_{c}') for c in cols])
        else:
            d = d.with_columns([(pl.col(c).rank(method='average').over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias(f'pct_{c}') for c in cols])
    return d


# --- normal-play NLL features from the cloud N/I models (SPEC_seqnll.md §7 / SPEC_gbnll.md §8). Env TPDS_NLL=<prefix>
# (e.g. gbnll or seqnll); default '' = byte-identical behaviour. Shards live in PROC/nll_<prefix>/<phase>/<table>.parquet keyed
# (pair_id, hand_id); every consumer of read_l2 (hand model, rankers, gauge) sees the identical columns.
USE_NLL = os.environ.get('TPDS_NLL', '')
NLL_FILL0 = ('_nll_sum', '_nll_size_sum', '_n_act', 'nll_sum_pair', 'gain_sum_pair')


def _nll_shard(f, phase):
    q = PROC / f'nll_{USE_NLL}' / phase / Path(f).name
    if not q.exists():
        raise FileNotFoundError(f'TPDS_NLL={USE_NLL}: missing shard {q}')
    d = pl.read_parquet(q).drop([c for c in ('a', 'b', 'table_id') if c in pl.read_parquet_schema(q)])
    return d


# --- action-sequence features (tpds_seq.py). Enabled per-run by env TPDS_SEQ=1 so that the hand model,
# the evidence ranker and the re-ranker all see the SAME feature set without threading a flag through
# five argument parsers. Screened by tpds_lift.py: kept only columns with best-family AUC >= 0.75 AND
# max |corr| < 0.88 against the existing pooled pair features, because v6 showed that adding collinear
# columns costs more than the marginal signal is worth.
USE_SEQ = int(os.environ.get('TPDS_SEQ', '0'))
# --- commit-side call regret vs the partner (tpds_callvalue.py). Env TPDS_CALLVAL=1, same rationale as TPDS_SEQ:
# every consumer of read_l2 (hand model, rankers, gauge) sees the identical columns. Default off.
USE_CALLVAL = int(os.environ.get('TPDS_CALLVAL', '0'))
_CALLVAL_CACHE = {}
CALLVAL_FILL0 = ('_n_call_vs_p', '_call_regret_sum', '_call_regret_max', '_n_bad_call', 'call_regret_max', 'call_regret_sum')


def _callval(phase):
    if phase not in _CALLVAL_CACHE:
        _CALLVAL_CACHE[phase] = pl.read_parquet(eqpath(f'callvalue_{phase}.parquet'))
    return _CALLVAL_CACHE[phase]


# --- passivity value vs the partner (tpds_passvalue.py). Env TPDS_PASSVAL=1; default off (byte-identical behaviour).
USE_PASSVAL = int(os.environ.get('TPDS_PASSVAL', '0'))
_PASSVAL_CACHE = {}
PASSVAL_FILL0 = ('_pass_val_max', '_pass_val_sum', '_chk_hu_val_max', '_call_ahead_val_max', '_n_pass_ahead', 'pass_val_max', 'pass_val_sum', 'n_pass_ahead')


def _passval(phase):
    if phase not in _PASSVAL_CACHE:
        _PASSVAL_CACHE[phase] = pl.read_parquet(eqpath(f'passvalue_{phase}.parquet'))
    return _PASSVAL_CACHE[phase]


# --- same-hand neighbour dominance (2026-09-10 evidence-axis screen). Env TPDS_NBRDOM=1; default off.
# nbr_max = highest s_gen among OTHER candidate pairs in the hand that share a player with this pair; nbr_dom = nbr_max > own
# s_gen; nbr_gap = nbr_max - own. Built from the v5nb hand scores. The dev table covers the 372 positive pairs only.
USE_NBRDOM = int(os.environ.get('TPDS_NBRDOM', '0'))
_NBRDOM_CACHE = {}


def _nbrdom(phase):
    if phase not in _NBRDOM_CACHE:
        _NBRDOM_CACHE[phase] = pl.read_parquet(PROC / f'nbrdom_{phase}.parquet')
    return _NBRDOM_CACHE[phase]



# --- PokerBench-SFT solver-policy surprisal features at the HAND level (5_outputs/pokerbench_0915/, 2026-09-15). Env TPDS_PB=1; default off.
# 16 pb_* columns per (pair_id, hand_id) aggregated from the pair members' decisions; rows without scores stay null.
USE_PB = int(os.environ.get('TPDS_PB', '0'))
PB_TAG = os.environ.get('TPDS_PB_TAG', '')   # 2026-09-16: '' = pb_{phase}.parquet (v036); 'both' = pbboth_{phase}.parquet (pb_* + pbid_* OOD-split columns)
_PB_CACHE = {}


def _pb(phase):
    if phase not in _PB_CACHE:
        _PB_CACHE[phase] = pl.read_parquet(PROC / f'pb{PB_TAG}_{phase}.parquet')
        print(f'[tpds_model] pb table: pb{PB_TAG}_{phase}.parquet {_PB_CACHE[phase].shape}', flush=True)
    return _PB_CACHE[phase]

# --- aggression regret vs the partner (tpds_aggvalue.py). Env TPDS_AGGVAL=1; default off.
USE_AGGVAL = int(os.environ.get('TPDS_AGGVAL', '0'))
_AGGVAL_CACHE = {}
AGGVAL_FILL0 = ('_agg_regret_max', '_agg_regret_sum', '_agg_regret_post', '_agg_behind_bb', '_n_bad_agg', 'agg_regret_max', 'agg_regret_sum', 'n_bad_agg')


def _aggval(phase):
    if phase not in _AGGVAL_CACHE:
        _AGGVAL_CACHE[phase] = pl.read_parquet(eqpath(f'aggvalue_{phase}.parquet'))
    return _AGGVAL_CACHE[phase]
# --- episode-onset features at the HAND level (env TPDS_ONSET=<run whose hand_scores seed the graph>).
# The `is_ev` label marks the EARLIEST manipulated hands, so it is time-contaminated: without a
# "how much suspicious activity came before" feature the hand model is asked to separate early
# manipulated hands from LATE manipulated ones on features that are identical between them.
# Fixing this at the ranker stage was worth +0.104 MAP@5; the hand model has the same defect.
# Bootstrap, not circularity: the onset features come from a PREVIOUS run's hand scores, which are
# out-of-fold by table on dev and the fold ensemble on eval.
USE_ONSET = os.environ.get('TPDS_ONSET', '')
_ONSET_CACHE = {}


def _onset(phase):
    if phase not in _ONSET_CACHE:
        from tpds_rerank import onset_cached          # lazy: tpds_rerank imports this module
        _ONSET_CACHE[phase] = onset_cached(ROOT / '5_outputs/models' / USE_ONSET, phase, norm_only=1)
    return _ONSET_CACHE[phase]
# Symmetric under swap_ab by construction: every rAB_x has its rBA_x twin, and the two non-directional
# columns are already A/B invariant. seq_n_A was dropped for exactly this reason - swap_name has no rule
# for it, so the symmetry augmentation would have silently fed the model an unswapped column.
SEQ_COLS = ['rAB_bg22', 'rBA_bg22', 'rAB_chk_chk', 'rBA_chk_chk', 'rAB_bg44', 'rBA_bg44',
            'rAB_bg21', 'rBA_bg21', 'rAB_bg40', 'rBA_bg40', 'rAB_bg20', 'rBA_bg20',
            'rAB_bg32', 'rBA_bg32', 'rAB_bg10', 'rBA_bg10', 'rAB_bg12', 'rBA_bg12',
            'rAB_chk_agg', 'rBA_chk_agg', 'rAB_agg_agg_outf', 'rBA_agg_agg_outf',
            'rAB_agg_agg_adj', 'rBA_agg_agg_adj', 'seq_n_pair', 'seq_out_agg_within']
_SEQ_CACHE = {}


def _seq_shard(f, phase):
    q = PROC / 'seq' / phase / Path(f).name
    if not q.exists():
        return None
    d = pl.read_parquet(q)
    cols = [c for c in SEQ_COLS if c in d.columns]
    return d.select(['hand_id', 'a', 'b'] + cols)


def feature_cols(df):
    return [c for c in df.columns if c not in KEY_COLS and df.schema[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt32, pl.Boolean)]


def load_training_rows(dev_cand, unknown_per_table, rng, use_equity=0, use_surprise=0, clean_pu=0, use_self=0):
    """L2 rows for labelled pairs + sampled unknown pairs (dev phase)."""
    lab = dev_cand.filter(pl.col('label') >= 0)
    unk_pool = dev_cand.filter(pl.col('label') < 0)
    if clean_pu:
        sp = PROC / 'suspect_hidden_positives.parquet'
        if sp.exists():
            sus = pl.read_parquet(sp)
            before = unk_pool.height
            unk_pool = unk_pool.join(sus, on='pair_id', how='anti')
            log(f"clean_pu: removed {before - unk_pool.height} suspected hidden-positive pairs from the unlabelled pool")
    unk = (unk_pool.sample(fraction=1.0, shuffle=True, seed=SEED)
           .group_by('table_id', maintain_order=True).head(unknown_per_table))
    keep = pl.concat([lab.select(['pair_id', 'table_id', 'label', 'behavior_family']), unk.select(['pair_id', 'table_id', 'label', 'behavior_family'])])
    parts = []
    for f in l2_files('development'):
        d = read_l2(f, 'development', use_equity, use_surprise, use_self)
        d = d.join(keep.select(['pair_id']), on='pair_id', how='inner')
        if d.height:
            parts.append(d)
    rows = pl.concat(parts)
    ev = pl.read_csv(RAW / 'development_evidence.csv').select(['pair_id', 'hand_id']).with_columns(pl.lit(1).cast(pl.Int8).alias('is_ev'))
    rows = rows.join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('is_ev').fill_null(0))
    rows = rows.join(keep.select(['pair_id', 'label', 'behavior_family']), on='pair_id')
    return rows


# ----------------------------------------------------------------------------- hand models
HAND_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=8, seed=SEED)
if os.environ.get('TPDS_HAND_SEED', '').strip():   # default off: LightGBM seed of the HAND models only (folds and unknown-pair sampling keep SEED)
    HAND_PARAMS = dict(HAND_PARAMS, seed=int(os.environ['TPDS_HAND_SEED']))


def train_hand_models(rows, feats, folds_by_table, n_rounds=500, w_unknown=0.5, w_posnonev=0.7):
    """Returns dict target -> list of fold boosters, plus OOF predictions frame."""
    X = rows.select(feats)
    Xs = swap_ab(rows).select(feats)
    gc.collect()
    label = rows['label'].to_numpy(); fam = rows['behavior_family'].to_numpy(); is_ev = rows['is_ev'].to_numpy()
    w = np.where(label == 0, 1.0, np.where(label < 0, w_unknown, np.where(is_ev == 1, 1.0, w_posnonev)))
    targets = {'gen': is_ev.astype(int)}
    for f in FAMS:
        targets[f] = ((is_ev == 1) & (fam == f)).astype(int)
    tables = rows['table_id'].to_numpy()
    fold_id = np.array([folds_by_table[t] for t in tables])
    n = rows.height
    Xnp = np.empty((2 * n, len(feats)), dtype=np.float32)
    Xnp[:n] = X.to_numpy().astype(np.float32, copy=False)
    del X; gc.collect()
    Xnp[n:] = Xs.to_numpy().astype(np.float32, copy=False)
    del Xs; gc.collect()
    models = {k: [] for k in targets}
    oof = {k: np.zeros(rows.height, dtype=np.float32) for k in targets}
    for k, y in targets.items():
        yy = np.concatenate([y, y]); ww = np.concatenate([w, w]); ff = np.concatenate([fold_id, fold_id])
        full = lgb.Dataset(Xnp, yy, weight=ww, feature_name=feats, free_raw_data=False, params={'max_bin': 63}).construct()
        for fo in range(5):
            tr = np.flatnonzero(ff != fo); va = np.arange(rows.height)[fold_id == fo]
            ds = full.subset(tr)
            m = lgb.train(HAND_PARAMS, ds, num_boost_round=n_rounds)
            models[k].append(m)
            del ds; gc.collect()
            p = 0.5 * (m.predict(Xnp[va]) + m.predict(Xnp[rows.height + va]))
            oof[k][va] = p
        from sklearn.metrics import roc_auc_score, average_precision_score
        log(f"hand model {k}: OOF AUC {roc_auc_score(y, oof[k]):.4f} AP {average_precision_score(y, oof[k]):.4f} (pos {y.sum()})")
    return models, oof


def predict_hand(models_k, X: np.ndarray, Xs: np.ndarray, fold=None):
    if fold is None:
        return np.mean([0.5 * (m.predict(X) + m.predict(Xs)) for m in models_k], axis=0)
    m = models_k[fold]
    return 0.5 * (m.predict(X) + m.predict(Xs))


# ----------------------------------------------------------------------------- pooling
POOL_SRC = ['gen'] + FAMS
AGG_MEAN_COLS = ['both_vpip', 'both_flop', 'both_sd', 'hu_streets', 'both_check_hu', 'war_streets', 'A_fold_to_B', 'B_fold_to_A',
                 'A_fold_to_other', 'B_fold_to_other', 'A_folded_better_to_B', 'B_folded_better_to_A', 'tr_any', 'pair_contrib', 'net_gap',
                 'pot_bb', 'A_vpip', 'B_vpip', 'A_n_agg', 'B_n_agg', 'A_sd', 'B_sd', 'rAB_fold', 'rBA_fold', 'rAB_call', 'rBA_call', 'rAB_raise', 'rBA_raise',
                 'rAB_n', 'rBA_n', 'rtA_fold', 'rtB_fold', 'A_dev_vpip', 'B_dev_vpip', 'A_dev_agg', 'B_dev_agg', 'A_dev_post_check', 'B_dev_post_check',
                 'A_n_allin', 'B_n_allin', 'chen_max', 'chen_min']


def pool_table(d: pl.DataFrame, scores: dict) -> pl.DataFrame:
    """d: L2 rows of one table (with t_rank); scores: name -> np.array aligned with d. Returns per-pair features."""
    d = d.with_columns([pl.Series(f's_{k}', v.astype(np.float32)) for k, v in scores.items()])
    d = d.sort(['pair_id', 't_rank'])
    aggs = [pl.len().cast(pl.Int32).alias('n_rows')]
    for k in scores:
        c = pl.col(f's_{k}')
        aggs += [c.max().alias(f'{k}_max'), c.mean().alias(f'{k}_mean'), c.top_k(3).mean().alias(f'{k}_top3'), c.top_k(5).mean().alias(f'{k}_top5'),
                 (c > 0.5).sum().cast(pl.Int16).alias(f'{k}_n50'), (c > 0.2).sum().cast(pl.Int16).alias(f'{k}_n20'), c.sum().alias(f'{k}_sum'),
                 c.rolling_mean(window_size=20, min_samples=5).max().alias(f'{k}_win20'),
                 c.rolling_mean(window_size=10, min_samples=3).max().alias(f'{k}_win10')]
    aggs += [pl.col(c).mean().alias(f'm_{c}') for c in AGG_MEAN_COLS if c in d.columns]
    for c in ('A_fold_eqplus_to_B', 'B_fold_eqplus_to_A', 'folder_eq_vs_partner', 'eq_flop_abs', 'surp_max_pair', 'surp_sum_pair', 'rsurp_max_pair', 'surpw_max_pair', 'rAB_surp_max', 'rBA_surp_max'):
        if c in d.columns:
            aggs += [pl.col(c).mean().alias(f'm_{c}'), pl.col(c).max().alias(f'mx_{c}'), pl.col(c).top_k(3).mean().alias(f't3_{c}')]
    aggs += [pl.col('tr_A_to_B').sum().alias('sum_tr_AB'), pl.col('tr_B_to_A').sum().alias('sum_tr_BA'),
             pl.col('A_net_bb').sum().alias('sum_A_net'), pl.col('B_net_bb').sum().alias('sum_B_net'),
             pl.col('hu_streets').sum().alias('sum_hu'), pl.col('war_streets').sum().alias('sum_war'),
             pl.col('A_fold_to_B').sum().alias('sum_A_fold_to_B'), pl.col('B_fold_to_A').sum().alias('sum_B_fold_to_A'),
             pl.col('Ab_vpip').first().alias('Ab_vpip'), pl.col('Bb_vpip').first().alias('Bb_vpip'),
             pl.col('Ab_fold_facing').first().alias('Ab_fold_facing'), pl.col('Bb_fold_facing').first().alias('Bb_fold_facing'),
             pl.col('Ab_agg_rate').first().alias('Ab_agg_rate'), pl.col('Bb_agg_rate').first().alias('Bb_agg_rate'),
             pl.col('session_id').n_unique().cast(pl.Int16).alias('n_sessions')]
    # direction consistency among the top-5 hands by generic score (DT: same partner wins every planted hand)
    topk = (d.sort(['pair_id', 's_gen'], descending=[False, True]).group_by('pair_id', maintain_order=True).head(5)
              .with_columns(pl.when(pl.col('A_net_bb') > pl.col('B_net_bb')).then(1.0).when(pl.col('A_net_bb') < pl.col('B_net_bb')).then(-1.0).otherwise(0.0).alias('sgn'))
              .group_by('pair_id').agg(pl.col('sgn').mean().abs().alias('top5_dir_consistency'),
                                       pl.col('tr_any').mean().alias('top5_tr_any'), pl.col('pair_contrib').mean().alias('top5_pair_contrib'),
                                       pl.col('hu_streets').mean().alias('top5_hu'), pl.col('both_check_hu').mean().alias('top5_both_check'),
                                       pl.col('war_streets').mean().alias('top5_war'), (pl.col('A_fold_to_B') + pl.col('B_fold_to_A')).mean().alias('top5_fold_to_partner'),
                                       (pl.col('A_folded_better_to_B') + pl.col('B_folded_better_to_A')).mean().alias('top5_folded_better'),
                                       pl.col('chen_min').mean().alias('top5_chen_min'), pl.col('pot_bb').mean().alias('top5_pot'),
                                       pl.col('t_rank').std().alias('top5_t_spread')))
    g = d.group_by('pair_id').agg(aggs).join(topk, on='pair_id', how='left')
    g = g.with_columns(
        ((pl.col('sum_tr_AB') - pl.col('sum_tr_BA')).abs() / (pl.col('sum_tr_AB') + pl.col('sum_tr_BA') + 1.0)).alias('tr_consistency'),
        pl.max_horizontal('sum_tr_AB', 'sum_tr_BA').alias('tr_max_dir'),
        (pl.col('sum_A_net') - pl.col('sum_B_net')).abs().alias('net_flow_abs'),
        # directional differentials: fold-to-partner rate minus own baseline fold-facing rate
        (pl.col('m_rAB_fold') / (pl.col('m_rAB_n') + 1e-3) - pl.col('Ab_fold_facing')).alias('A_fold_vs_B_diff'),
        (pl.col('m_rBA_fold') / (pl.col('m_rBA_n') + 1e-3) - pl.col('Bb_fold_facing')).alias('B_fold_vs_A_diff'),
        (pl.col('m_hu_streets') / (pl.col('Ab_vpip') * pl.col('Bb_vpip') + 1e-3)).alias('hu_norm'),
    )
    return g


def build_pair_features(phase, cand, hand_models, feats, folds_by_table=None, use_equity=0, use_surprise=0, use_self=0):
    """Score all L2 rows of a phase per table and pool. For dev, use the held-out fold model of each table (OOF)."""
    parts = []
    t0 = time.time()
    files = l2_files(phase)
    for i, f in enumerate(files):
        d = read_l2(f, phase, use_equity, use_surprise, use_self)
        if d.height == 0:
            continue
        t = d['table_id'][0]
        X = d.select(feats).to_numpy().astype(np.float32); Xs = swap_ab(d).select(feats).to_numpy().astype(np.float32)
        fold = folds_by_table[t] if folds_by_table is not None else None
        scores = {k: predict_hand(hand_models[k], X, Xs, fold) for k in POOL_SRC}
        parts.append(pool_table(d, scores))
        # keep per-row scores for evidence selection (top rows only to save space)
        d_small = d.select(['pair_id', 'hand_id']).with_columns([pl.Series(f's_{k}', scores[k].astype(np.float32)) for k in POOL_SRC])
        parts[-1] = (parts[-1], d_small)
        if i % 100 == 0:
            log(f"pool {phase}: {i+1}/{len(files)} {time.time()-t0:.0f}s")
    pf = pl.concat([p[0] for p in parts]); rows = pl.concat([p[1] for p in parts])
    pf = cand.select(['pair_id', 'table_id', 'shared', 'label', 'behavior_family']).join(pf, on='pair_id', how='left')
    return pf, rows


# ----------------------------------------------------------------------------- pair model
PAIR_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=40, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, n_jobs=8, seed=SEED)


def train_pair_model(pf, folds_by_table, w_unknown=0.5, n_rounds=600, seeds=(42, 7, 2024)):
    feats = [c for c in pf.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')]
    X = pf.select(feats).to_numpy().astype(np.float32)
    y = (pf['label'] == 1).to_numpy().astype(int)
    lab = pf['label'].to_numpy()
    w = np.where(lab == 1, 1.0, np.where(lab == 0, 1.0, w_unknown))
    fold_id = np.array([folds_by_table[t] for t in pf['table_id'].to_list()])
    oof = np.zeros(len(y)); models = []
    for fo in range(5):
        tr = fold_id != fo; va = fold_id == fo
        preds = []
        fold_models = []
        for s in seeds:
            params = dict(PAIR_PARAMS, seed=s)
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats), num_boost_round=n_rounds)
            preds.append(m.predict(X[va])); fold_models.append(m)
        oof[va] = np.mean(preds, axis=0); models.append(fold_models)
    return models, oof, feats


def predict_pair(models, pf, feats):
    X = pf.select(feats).to_numpy().astype(np.float32)
    return np.mean([m.predict(X) for fm in models for m in fm], axis=0)


# ----------------------------------------------------------------------------- assembling predictions
def family_from_pf(pf):
    s = np.stack([pf[f'{f}_top3'].to_numpy() for f in FAMS], axis=1)
    return np.array(FAMS)[np.nan_to_num(s, nan=-1).argmax(1)], s


def pick_evidence(rows, pf_fam, k=5):
    """rows: per-row scores (pair_id, hand_id, s_*). pf_fam: DataFrame pair_id -> family. Returns pair_id -> list of hand_ids."""
    r = rows.join(pf_fam, on='pair_id')
    r = r.with_columns(
        pl.when(pl.col('fam') == 'directed_transfer').then(pl.col('s_directed_transfer'))
          .when(pl.col('fam') == 'soft_play').then(pl.col('s_soft_play'))
          .when(pl.col('fam') == 'coordinated_isolation').then(pl.col('s_coordinated_isolation'))
          .otherwise(pl.col('s_gen')).alias('sel'))
    top = (r.sort(['pair_id', 'sel'], descending=[False, True]).group_by('pair_id', maintain_order=True)
             .agg(pl.col('hand_id').head(k).alias('ev')))
    return dict(zip(top['pair_id'].to_list(), top['ev'].to_list()))


def make_submission_frame(pair_ids, risk, fam, ev_map, active_frac=0.5):
    n = len(pair_ids)
    order = np.argsort(-risk, kind='mergesort'); active = np.zeros(n, bool); active[order[:int(round(n * active_frac))]] = True
    beh = np.where(active, fam, 'none')
    rows = []
    for i, pid in enumerate(pair_ids):
        ev = list(ev_map.get(pid, []))[:5]
        ev = ev + ['NO_EVIDENCE'] * (5 - len(ev))
        rows.append([pid, float(risk[i]), beh[i]] + ev)
    return pd.DataFrame(rows, columns=['pair_id', 'risk_score', 'predicted_behavior'] + [f'evidence_hand_{i}' for i in range(1, 6)])


def dev_solution(dev_cand):
    ev = pd.read_csv(RAW / 'development_evidence.csv')
    evm = ev.sort_values(['pair_id', 'evidence_rank']).groupby('pair_id').hand_id.apply(list).to_dict()
    sol = []
    for pid, lab, fam in zip(dev_cand['pair_id'].to_list(), dev_cand['label'].to_list(), dev_cand['behavior_family'].to_list()):
        e = evm.get(pid, [])[:5] if lab == 1 else []
        e = e + ['NO_EVIDENCE'] * (5 - len(e))
        sol.append([pid, 1 if lab == 1 else 0, fam if lab == 1 else 'none'] + e)
    return pd.DataFrame(sol, columns=['pair_id', 'risk_score', 'predicted_behavior'] + [f'evidence_hand_{i}' for i in range(1, 6)])


def rank_to_unit(x):
    r = pd.Series(x).rank(method='first').to_numpy()
    return (r - 0.5) / len(r)


# ----------------------------------------------------------------------------- main
def main(run='v1', unknown_per_table=40, w_unknown_hand=0.5, w_unknown_pair=0.5, active_frac=0.5, hand_rounds=500, pair_rounds=600, use_equity=0, use_surprise=0, clean_pu=0, use_self=0, w_posnonev=0.7):
    # w_posnonev: training weight of NON-evidence hands inside positive pairs (default 0.7 = every run up to v014).
    # Those hands include LATER manipulated hands (is_ev marks only the earliest <=5), so a lower weight asks the hand
    # models for 'manipulated' rather than 'earliest manipulated' (2026-09-10 direction A).
    out = ROOT / '5_outputs/models' / run; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dev_cand = pl.read_parquet(PROC / 'cand_pairs_development.parquet'); ev_cand = pl.read_parquet(PROC / 'cand_pairs_evaluation.parquet')
    tables = sorted(dev_cand['table_id'].unique().to_list())
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_by_table = {}
    for fo, (_, va) in enumerate(gkf.split(tables, groups=tables)):
        for i in va:
            folds_by_table[tables[i]] = fo
    json.dump(folds_by_table, open(out / 'folds_by_table.json', 'w'))
    log('loading training rows'); rows = load_training_rows(dev_cand, unknown_per_table, rng, use_equity, use_surprise, clean_pu, use_self)
    feats = feature_cols(rows.drop(['is_ev', 'label']))
    feats = [c for c in feats if c not in ('is_ev', 'label')]
    _fl = os.environ.get('TPDS_HAND_FEATS', '').strip()   # default off: train on a saved feature list verbatim (e.g. production's 280)
    if _fl:
        _want = json.load(open(_fl)); _miss = [c for c in _want if c not in feats]
        assert not _miss, f'TPDS_HAND_FEATS: {len(_miss)} listed columns are not available, e.g. {_miss[:5]}'
        log(f'TPDS_HAND_FEATS={_fl}: {len(_want)} of {len(feats)} available features; dropped {sorted(set(feats) - set(_want))}')
        feats = list(_want)
    log(f"training rows {rows.height:,} (evidence {int(rows['is_ev'].sum())}), features {len(feats)}")
    json.dump(feats, open(out / 'hand_features.json', 'w'))
    log(f'training hand models (w_posnonev={w_posnonev})'); hand_models, hand_oof = train_hand_models(rows, feats, folds_by_table, n_rounds=hand_rounds, w_unknown=w_unknown_hand, w_posnonev=w_posnonev)
    for k, ms in hand_models.items():
        for i, m in enumerate(ms):
            m.save_model(str(out / f'hand_{k}_f{i}.txt'))
    # MAP@5 within positive pairs from OOF (rows restricted to positive pairs)
    pos_rows = rows.filter(pl.col('label') == 1)
    mask = (rows['label'] == 1).to_numpy()
    for k in POOL_SRC:
        pr = pos_rows.select(['pair_id', 'hand_id', 'is_ev', 'behavior_family']).with_columns(pl.Series('s', hand_oof[k][mask]))
        top = pr.sort(['pair_id', 's'], descending=[False, True]).group_by('pair_id', maintain_order=True).agg(pl.col('is_ev').head(5).alias('hits'), pl.col('is_ev').sum().alias('nrel'), pl.col('behavior_family').first().alias('fam'))
        def ap5(h, nrel):
            hits = 0; ps = 0.0
            for r_, v in enumerate(h, 1):
                if v: hits += 1; ps += hits / r_
            return ps / min(nrel, 5)
        top = top.with_columns(pl.struct(['hits', 'nrel']).map_elements(lambda s: ap5(s['hits'], s['nrel']), return_dtype=pl.Float64).alias('ap5'))
        log(f"  MAP@5 within positive pairs using score '{k}': all {top['ap5'].mean():.4f} | " + ' '.join(f"{f[:4]} {top.filter(pl.col('fam')==f)['ap5'].mean():.4f}" for f in FAMS))
    log('pooling dev'); pf_dev, rows_dev = build_pair_features('development', dev_cand, hand_models, feats, folds_by_table, use_equity, use_surprise, use_self)
    log('pooling eval'); pf_ev, rows_ev = build_pair_features('evaluation', ev_cand, hand_models, feats, None, use_equity, use_surprise, use_self)
    pf_dev.write_parquet(out / 'pair_features_dev.parquet'); pf_ev.write_parquet(out / 'pair_features_eval.parquet')
    rows_dev.write_parquet(out / 'hand_scores_dev.parquet'); rows_ev.write_parquet(out / 'hand_scores_eval.parquet')
    log('training pair model'); pair_models, pair_oof, pfeats = train_pair_model(pf_dev, folds_by_table, w_unknown=w_unknown_pair, n_rounds=pair_rounds)
    for fo, fm in enumerate(pair_models):
        for i, m in enumerate(fm):
            m.save_model(str(out / f'pair_f{fo}_s{i}.txt'))
    # ---- dev-solution metric
    fam_dev, _ = family_from_pf(pf_dev)
    ev_map_dev = pick_evidence(rows_dev, pl.DataFrame({'pair_id': pf_dev['pair_id'], 'fam': fam_dev}))
    risk_dev = rank_to_unit(pair_oof)
    sub_dev = make_submission_frame(pf_dev['pair_id'].to_list(), risk_dev, fam_dev, ev_map_dev, active_frac)
    sol = dev_solution(pf_dev.select(['pair_id', 'label', 'behavior_family']))
    sc, comp = kaggle_score(sol, sub_dev, 'pair_id', return_components=True)
    log(f"DEV-SOLUTION OOF: score {sc:.4f} | PairAP {comp['pair_ap']:.4f} EvidenceMAP {comp['evidence_map']:.4f} BehaviorMAP {comp['behavior_map']:.4f} | per-family AP {comp['behavior_ap_per_family']}")
    # family accuracy on positives
    pos = (pf_dev['label'] == 1).to_numpy()
    acc = (fam_dev[pos] == pf_dev['behavior_family'].to_numpy()[pos]).mean()
    log(f"family accuracy on positives: {acc:.3f}")
    # active fraction grid
    for af in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0]:
        s_ = make_submission_frame(pf_dev['pair_id'].to_list(), risk_dev, fam_dev, ev_map_dev, af)
        _, c_ = kaggle_score(sol, s_, 'pair_id', return_components=True)
        log(f"  active_frac {af}: BehaviorMAP {c_['behavior_map']:.4f}")
    pd.DataFrame({'pair_id': pf_dev['pair_id'], 'oof': pair_oof, 'label': pf_dev['label'], 'fam_pred': fam_dev}).to_parquet(out / 'pair_oof_dev.parquet')
    # ---- eval submission
    risk_ev = rank_to_unit(predict_pair(pair_models, pf_ev, pfeats))
    fam_ev, _ = family_from_pf(pf_ev)
    ev_map = pick_evidence(rows_ev, pl.DataFrame({'pair_id': pf_ev['pair_id'], 'fam': fam_ev}))
    sub = make_submission_frame(pf_ev['pair_id'].to_list(), risk_ev, fam_ev, ev_map, active_frac)
    sample = pd.read_csv(RAW / 'sample_submission.csv')
    sub = sample[['pair_id']].merge(sub, on='pair_id', how='left')
    assert sub.isna().sum().sum() == 0, 'missing rows in submission'
    sub.to_csv(out / 'submission.csv', index=False)
    log(f"submission written: {out/'submission.csv'} rows {len(sub)}")
    json.dump({'dev_score': sc, **{k: (v if not isinstance(v, dict) else v) for k, v in comp.items()}, 'family_acc': float(acc)}, open(out / 'dev_metrics.json', 'w'), indent=1)


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('='); kw[k] = float(v) if '.' in v else int(v)
    main(run, **kw)
