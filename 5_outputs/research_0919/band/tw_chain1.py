"""截短世界第 4 部分（第一階段）：v5xcp2 配對鏈在「eval 長度窗口」的 dev 世界重訓。
世界：F＝生產全長 dev 表；W1＝t_rank[0,2000)、W2＝[1000,3000)（tw_base／tw_pc／tw_pairpol 建）。W 世界只留窗口內同桌手數 ≥38 的配對（＝eval 候選規則 1.9%）。
臂：F（對照＝生產配方）、W（W1+W2）、FW（F+W1+W2）。全部依桌五折；量尺在 W 世界的留出桌上讀（＝像 eval 的母體）。
用法: python tw_chain1.py <lgb|full> ；lgb＝只跑 LGB 三臂量尺；full＝LGB+CatBoost、寫 OOF 與 eval 基底分數。"""
import sys, os, json, time, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
R = (str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/')
sys.path.insert(0, R + '3_src'); import tpds_pair_stage as tps
from tpds_model import PROC, PAIR_PARAMS
M, EQ, TW = R + '5_outputs/models/v5x', R + '5_outputs/eqx_0913/', R + '5_outputs/research_0919/band/tw/'
OUT = R + '5_outputs/research_0919/band/'; mode = sys.argv[1] if len(sys.argv) > 1 else 'lgb'
ARMS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['F', 'W', 'FW']
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
rk = lambda v: rankdata(v) / len(v)
def world(tag):
    if tag == 'F':
        pf = pl.read_parquet(M + '/pair_features_dev.parquet'); pc = EQ + 'presence_contrast_x_development.parquet'; pp = EQ + 'pairpol_feats_development.parquet'; p2 = EQ + 'presence_contrast_v2_development.parquet'
    elif tag == 'E':
        pf = pl.read_parquet(M + '/pair_features_eval.parquet'); pc = EQ + 'presence_contrast_x_evaluation.parquet'; pp = EQ + 'pairpol_feats_evaluation.parquet'; p2 = EQ + 'presence_contrast_v2_evaluation.parquet'
    else:
        pf = pl.read_parquet(TW + tag + '/pair_features.parquet').filter(pl.col('shared') >= 38); pc = TW + tag + '/presence_contrast_x.parquet'; pp = TW + tag + '/pairpol_feats.parquet'; p2 = TW + tag + '/presence_contrast_v2.parquet'
    pf = tps.add_betsize(pf, 'evaluation' if tag == 'E' else 'development')
    for f in (pc, pp, p2): pf = pf.join(pl.read_parquet(f), on='pair_id', how='left').fill_null(0.0)
    return pf
pfF = world('F'); fe = [c for c in pfF.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')]
W = {t: world(t) for t in ('W1', 'W2')}; pfE = world('E')
for t, d in W.items(): assert [c for c in d.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')] == fe, f'{t} 欄位順序不同'
assert [c for c in pfE.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')] == fe
log(f'feats {len(fe)} | F {pfF.height} W1 {W["W1"].height} W2 {W["W2"].height} E {pfE.height}')
folds = json.load(open(M + '/folds_by_table.json')); sus = set(pl.read_parquet(PROC / 'suspect_hidden_positives.parquet')['pair_id'].to_list())
def pack(d):
    X = d.select(fe).to_numpy().astype(np.float32); lab = d['label'].to_numpy(); y = (lab == 1).astype(int)
    w = np.where(lab == 1, 1.0, np.where(lab == 0, 1.0, 0.5)); pre = np.array([p in sus for p in d['pair_id'].to_list()]); w = np.where(pre, 0.0, w)
    return dict(X=X, y=y, w=w, pre=pre, fold=np.array([folds[t] for t in d['table_id'].to_list()]), pid=d['pair_id'].to_list(), shared=d['shared'].to_numpy())
D = {'F': pack(pfF), 'W1': pack(W['W1']), 'W2': pack(W['W2'])}; Xe = pfE.select(fe).to_numpy().astype(np.float32); pids_e = pfE['pair_id'].to_list()
# 分佈相近度（對抗驗證的替代讀數）：幾個長度敏感欄在各世界正例以外的中位數
for c in ('shared', 'n_rows', 'gen_top3', 'gen_n20', 'pc_nll_sum_tg_mean', 'pp_llr_z_max'):
    if c in fe: log(f'  中位數 {c:22s} F {np.median(D["F"]["X"][:, fe.index(c)]):.4f} W1 {np.median(D["W1"]["X"][:, fe.index(c)]):.4f} W2 {np.median(D["W2"]["X"][:, fe.index(c)]):.4f} eval {np.median(Xe[:, fe.index(c)]):.4f}')
TRAIN = {'F': ['F'], 'W': ['W1', 'W2'], 'FW': ['F', 'W1', 'W2']}
def fit_lgb(Xt, yt, wt, seed): return lgb.train(dict(PAIR_PARAMS, seed=seed, num_leaves=31), lgb.Dataset(Xt, yt, weight=wt), num_boost_round=600)
def fit_cb(Xt, yt, wt, seed):
    import catboost as cb
    m = cb.CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=10, rsm=0.7, bootstrap_type='Bernoulli', subsample=0.8, random_seed=seed, verbose=0, thread_count=8, allow_writing_files=False)
    m.fit(Xt, yt, sample_weight=wt); return m
p_lgb = lambda m, Z: m.predict(Z); p_cb = lambda m, Z: m.predict_proba(Z)[:, 1]
def stack(tags, fo=None):
    Xs, ys, ws = [], [], []
    for t in tags:
        d = D[t]; m = (d['w'] > 0) if fo is None else ((d['fold'] != fo) & (d['w'] > 0)); Xs.append(d['X'][m]); ys.append(d['y'][m]); ws.append(d['w'][m])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
def gauge(name, t, o):
    d = D[t]; keep = ~d['pre']; y = d['y'][keep]; s = o[keep]; r = rankdata(-s, method='ordinal'); pr = np.sort(r[y == 1])
    bands = [(1, 150), (151, 300), (301, 450), (451, 600), (601, 1000)]
    bp = ' '.join(f'{lo}-{hi}:{y[(r >= lo) & (r <= hi)].mean():.2f}' for lo, hi in bands)
    log(f'  [{name:>3s}→{t}] AP(排除疑似) {average_precision_score(y, s):.4f} | 正例名次 中位 {np.median(pr):.0f} q90 {np.quantile(pr, .9):.0f} >600:{(pr > 600).sum()} >1000:{(pr > 1000).sum()} | 名次帶正例率 {bp}')
    return average_precision_score(y, s)
res = {}
for arm in ARMS:
    t0 = time.time(); oof = {t: {'lgb': np.zeros(len(D[t]['y'])), 'cb': np.zeros(len(D[t]['y']))} for t in D}
    for fo in range(5):
        Xt, yt, wt = stack(TRAIN[arm], fo)
        ms = [fit_lgb(Xt, yt, wt, s) for s in (42, 43, 44)]
        for t in D:
            va = D[t]['fold'] == fo; oof[t]['lgb'][va] = np.mean([p_lgb(m, D[t]['X'][va]) for m in ms], axis=0)
        if mode == 'full':
            mc = [fit_cb(Xt, yt, wt, s) for s in (42, 43, 44)]
            for t in D:
                va = D[t]['fold'] == fo; oof[t]['cb'][va] = np.mean([p_cb(m, D[t]['X'][va]) for m in mc], axis=0)
        log(f'arm {arm} fold {fo} done ({time.time() - t0:.0f}s, train rows {len(yt):,})')
    log(f'=== arm {arm}（訓練世界 {TRAIN[arm]}）')
    for t in D:
        o = oof[t]['lgb'] if mode == 'lgb' else (rk(oof[t]['lgb']) + rk(oof[t]['cb'])) / 2
        res[(arm, t)] = gauge(arm, t, o)
        if mode == 'full':
            pd.DataFrame({'pair_id': D[t]['pid'], 'oof': o, 'lgb': oof[t]['lgb'], 'cb': oof[t]['cb']}).to_parquet(OUT + f'tw_oof_{arm}_{t}.parquet')
    if mode == 'full':
        Xt, yt, wt = stack(TRAIN[arm])
        le = np.mean([p_lgb(fit_lgb(Xt, yt, wt, s), Xe) for s in (42, 43, 44)], axis=0); ce = np.mean([p_cb(fit_cb(Xt, yt, wt, s), Xe) for s in (42, 43, 44)], axis=0)
        pd.DataFrame({'pair_id': pids_e, 'raw': (rk(le) + rk(ce)) / 2, 'lgb_raw': le, 'cb_raw': ce}).to_parquet(OUT + f'tw_eval_raw_{arm}.parquet'); log(f'arm {arm} eval base written')
log('SUMMARY ' + json.dumps({f'{a}->{t}': round(v, 4) for (a, t), v in res.items()}))
