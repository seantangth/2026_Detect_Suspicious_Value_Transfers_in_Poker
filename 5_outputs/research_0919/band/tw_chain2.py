"""截短世界第 5 部分（第二階段＝頭部重排，決定帶區排序的那一層）＋eval 套用＋拼接。
讀 tw_chain1.py full 的產物（tw_oof_<arm>_<world>.parquet、tw_eval_raw_<arm>.parquet）。
每個臂：第二階段訓練列＝該臂各訓練世界的頭部（依世界大小等比例取前 2.178%），情境特徵 context() 在各世界內各自計算；
量尺一律在 W1／W2 的留出桌讀（像 eval 的母體）。對照臂 F＝生產配方原樣（第一、二階段都只看全長 dev），再拿去打 W 世界＝生產在 eval 的處境。
用法: python tw_chain2.py <arm,arm,...> [splice_base_csv]"""
import sys, os, json, time, hashlib, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
R = (str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/')
sys.path.insert(0, R + '3_src'); import tpds_pair_stage as tps
from tpds_model import PROC, rank_to_unit
M, EQ, SQ = R + '5_outputs/models/v5x', R + '5_outputs/eqx_0913/', R + '5_outputs/seqnll_0912/'
B = R + '5_outputs/research_0919/band/'; TW = B + 'tw/'; raw = R + '1_data/raw/detect-suspicious-value-transfers-in-poker/'
ARMS = sys.argv[1].split(','); SPLICE = sys.argv[2] if len(sys.argv) > 2 else R + '5_outputs/submissions/submission_HB2_FINALD2_hbdecode_cilvl_w70.csv'
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
sys.path.insert(0, R + '5_outputs/eqx_0913'); from head_rerank_helpers import context, EXTRA, P2, lg, neutralise, place   # verbatim slice of seqnll_0912/ens_head_rerank.py
HEAD_FRAC = 3000 / 137739
def world(tag):
    if tag == 'F': pf = pl.read_parquet(M + '/pair_features_dev.parquet'); fs = [EQ + 'presence_contrast_x_development.parquet', EQ + 'pairpol_feats_development.parquet', EQ + 'presence_contrast_v2_development.parquet']
    elif tag == 'E': pf = pl.read_parquet(M + '/pair_features_eval.parquet'); fs = [EQ + 'presence_contrast_x_evaluation.parquet', EQ + 'pairpol_feats_evaluation.parquet', EQ + 'presence_contrast_v2_evaluation.parquet']
    else: pf = pl.read_parquet(TW + tag + '/pair_features.parquet').filter(pl.col('shared') >= 38); fs = [TW + tag + '/presence_contrast_x.parquet', TW + tag + '/pairpol_feats.parquet', TW + tag + '/presence_contrast_v2.parquet']
    pf = tps.add_betsize(pf, 'evaluation' if tag == 'E' else 'development')
    for f in fs: pf = pf.join(pl.read_parquet(f), on='pair_id', how='left').fill_null(0.0)
    return pf
def build_dom(tag, pids):
    """pair_chain_v5xcp2.build_dom 的配方，手分數換成該世界的檔、配對限該世界的候選。"""
    f = TW + tag + '/dom.parquet'
    if os.path.exists(f): return f
    cp = pl.read_parquet(PROC / 'cand_pairs_development.parquet').select('pair_id', 'a', 'b').filter(pl.col('pair_id').is_in(pids)).with_row_index('pi')
    hs = pl.scan_parquet(TW + tag + '/hand_scores.parquet').select('pair_id', 'hand_id', 's_gen')
    F = hs.join(cp.lazy().select('pair_id', 'pi'), on='pair_id').select('pi', pl.col('hand_id').cast(pl.Categorical).to_physical().alias('h'), 's_gen').collect()
    T = F.sort('s_gen', descending=True).group_by('pi', maintain_order=True).head(5)
    J = T.join(F.rename({'pi': 'qi', 's_gen': 's_q'}), on='h').filter(pl.col('qi') != pl.col('pi'))
    ab = cp.select('pi', 'a', 'b'); J = J.join(ab, on='pi').join(ab.rename({'pi': 'qi', 'a': 'qa', 'b': 'qb'}), on='qi')
    J = J.filter((pl.col('a') == pl.col('qa')) | (pl.col('a') == pl.col('qb')) | (pl.col('b') == pl.col('qa')) | (pl.col('b') == pl.col('qb')))
    Hm = J.group_by('pi', 'h').agg(nmax=pl.col('s_q').max()); T = T.join(Hm, on=['pi', 'h'], how='left').with_columns(pl.col('nmax').fill_null(0.0))
    agg = T.group_by('pi').agg(dom5=(pl.col('nmax') > pl.col('s_gen')).mean(), dommass=((pl.col('nmax') > pl.col('s_gen')).cast(pl.Float32) * pl.col('s_gen')).sum() / pl.col('s_gen').sum(),
                               ratio5=(pl.col('nmax') / pl.col('s_gen').clip(1e-6)).clip(0, 5).mean(), ntop=pl.len())
    cp.join(agg, on='pi', how='left').select('pair_id', 'dom5', 'dommass', 'ratio5', 'ntop').write_parquet(f); return f
folds = json.load(open(M + '/folds_by_table.json')); sus = set(pl.read_parquet(PROC / 'suspect_hidden_positives.parquet')['pair_id'].to_list())
pos = pl.read_csv(raw + 'development_labels.csv').filter(pl.col('label') == 1); pp = set(pos['player_1'].to_list()) | set(pos['player_2'].to_list())
candD = pl.read_parquet(PROC / 'cand_pairs_development.parquet', columns=['pair_id', 'a', 'b']).to_pandas().set_index('pair_id')
PF = {t: world(t) for t in ('F', 'W1', 'W2')}; pfE = world('E'); fe = [c for c in PF['F'].columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')]
Xe = pfE.select(fe).to_numpy().astype(np.float32); pids_e = pfE['pair_id'].to_list()
def prep(arm, t):
    """世界 t 在臂 arm 的第一階段 OOF 下的第二階段輸入。"""
    d = PF[t]; pids = d['pair_id'].to_list(); X = d.select(fe).to_numpy().astype(np.float32); y = (d['label'].to_numpy() == 1).astype(int)
    base = pd.read_parquet(B + f'tw_oof_{arm}_{t}.parquet').set_index('pair_id').loc[pids].oof.to_numpy()
    hs = M + '/hand_scores_dev.parquet' if t == 'F' else TW + t + '/hand_scores.parquet'; dom = EQ + 'dom_development_v5x.parquet' if t == 'F' else build_dom(t, pids)
    cd = context('development', pids, base, hs, dom); Xh = np.hstack([X, cd[EXTRA].to_numpy(dtype=np.float32)])
    c = candD.loc[pids]; pre = np.array([p in sus for p in pids]); hid = pre & ~(c.a.isin(pp) | c.b.isin(pp)).to_numpy()
    head = pd.Series(-base).rank(method='first').to_numpy() <= round(HEAD_FRAC * len(pids)); contam = (cd.ovl.to_numpy() > 0) | (cd.dom5.to_numpy() >= 0.2)
    return dict(pids=pids, y=y, base=base, Xh=Xh, Xn=neutralise(Xh, fe + EXTRA, base), head=head, contam=contam, hid=hid, pre=pre, fold=np.array([folds[x] for x in d['table_id'].to_list()]))
def gauge(name, S, final):
    keep = ~S['pre']; y = S['y'][keep]; s = final[keep]; r = rankdata(-s, method='ordinal'); pr = np.sort(r[y == 1])
    bp = ' '.join(f'{lo}-{hi}:{y[(r >= lo) & (r <= hi)].mean():.2f}' for lo, hi in [(1, 150), (151, 300), (301, 450), (451, 600), (601, 1000)])
    ap = average_precision_score(y, s); log(f'  [{name}] AP(排除疑似) {ap:.4f} | 正例名次 中位 {np.median(pr):.0f} q90 {np.quantile(pr, .9):.0f} >600:{(pr > 600).sum()} >1000:{(pr > 1000).sum()} | 名次帶正例率 {bp}'); return ap
TRAIN = {'F': ['F'], 'W': ['W1', 'W2'], 'FW': ['F', 'W1', 'W2']}; f2 = fe + EXTRA; summary = {}
for arm in ARMS:
    t0 = time.time(); S = {t: prep(arm, t) for t in ('F', 'W1', 'W2')}; log(f'arm {arm}: contexts built ({time.time() - t0:.0f}s)')
    fin = {}
    for t in S: S[t]['p2'] = np.full(len(S[t]['y']), 0.5); S[t]['p2n'] = np.full(len(S[t]['y']), 0.5)
    for fo in range(5):
        Xt = np.vstack([S[t]['Xh'][S[t]['head'] & ~S[t]['hid'] & (S[t]['fold'] != fo)] for t in TRAIN[arm]]); yt = np.concatenate([S[t]['y'][S[t]['head'] & ~S[t]['hid'] & (S[t]['fold'] != fo)] for t in TRAIN[arm]])
        ms = [lgb.train(dict(P2, seed=s), lgb.Dataset(Xt, yt, feature_name=f2), num_boost_round=400) for s in (1, 2, 3)]
        for t in S:
            va = S[t]['head'] & (S[t]['fold'] == fo)
            S[t]['p2'][va] = np.mean([m.predict(S[t]['Xh'][va]) for m in ms], axis=0); S[t]['p2n'][va] = np.mean([m.predict(S[t]['Xn'][va]) for m in ms], axis=0)
    log(f'=== arm {arm} 第二階段後（訓練世界 {TRAIN[arm]}；{time.time() - t0:.0f}s）')
    for t in S:
        adj = lg(np.where(~S[t]['contam'], np.maximum(S[t]['p2'], S[t]['p2n']), S[t]['p2'])); final = place(S[t]['base'], S[t]['head'], adj)
        summary[f'{arm}->{t}:stage1'] = round(gauge(f'{arm}→{t} 只第一階段', S[t], S[t]['base']), 4); summary[f'{arm}->{t}:stage2'] = round(gauge(f'{arm}→{t} 含頭部重排 ', S[t], final), 4)
        pd.DataFrame({'pair_id': S[t]['pids'], 'oof': final, 'base': S[t]['base'], 'p2': S[t]['p2'], 'head': S[t]['head'], 'contam': S[t]['contam']}).to_parquet(B + f'tw_final_oof_{arm}_{t}.parquet')
    # ---- eval
    base_e = pd.read_parquet(B + f'tw_eval_raw_{arm}.parquet').set_index('pair_id').loc[pids_e].raw.to_numpy()
    ce = context('evaluation', pids_e, base_e, M + '/hand_scores_eval.parquet', EQ + 'dom_evaluation_v5x.parquet')
    Xeh = np.hstack([Xe, ce[EXTRA].to_numpy(dtype=np.float32)]); Xen = neutralise(Xeh, f2, base_e)
    head_e = pd.Series(-base_e).rank(method='first').to_numpy() <= 2451; contam_e = (ce.ovl.to_numpy() > 0) | (ce.dom5.to_numpy() >= 0.2)
    Xt = np.vstack([S[t]['Xh'][S[t]['head'] & ~S[t]['hid']] for t in TRAIN[arm]]); yt = np.concatenate([S[t]['y'][S[t]['head'] & ~S[t]['hid']] for t in TRAIN[arm]])
    models = [lgb.train(dict(P2, seed=s), lgb.Dataset(Xt, yt, feature_name=f2), num_boost_round=400) for s in (1, 2, 3)]
    p2e = np.full(len(base_e), 0.5); p2ne = np.full(len(base_e), 0.5)
    p2e[head_e] = np.mean([m.predict(Xeh[head_e]) for m in models], axis=0); p2ne[head_e] = np.mean([m.predict(Xen[head_e]) for m in models], axis=0)
    s_e = place(base_e, head_e, lg(np.where(~contam_e, np.maximum(p2e, p2ne), p2e))); risk = pd.Series(rank_to_unit(s_e), index=pd.Series(pids_e))
    pd.DataFrame({'pair_id': pids_e, 'final': s_e, 'base': base_e, 'p2': p2e, 'head': head_e}).to_parquet(B + f'tw_eval_final_{arm}.parquet')
    ref = pd.read_csv(SPLICE, dtype=str, keep_default_na=False); out = ref.copy(); out['risk_score'] = [repr(float(v)) for v in risk.loc[out.pair_id].to_numpy()]
    fn = R + f'5_outputs/submissions/submission_TW{arm}_{os.path.basename(SPLICE).replace("submission_", "")}'; out.to_csv(fn, index=False)
    q = pd.read_csv(fn, dtype=str, keep_default_na=False); assert (q.drop(columns='risk_score').to_numpy() == ref.drop(columns='risk_score').to_numpy()).all() and q.risk_score.astype(float).between(0, 1).all() and q.risk_score.astype(float).duplicated().sum() == 0
    a = ref.risk_score.astype(float).to_numpy(); b = out.risk_score.astype(float).to_numpy(); oa, ob = np.argsort(-a), np.argsort(-b)
    from scipy.stats import spearmanr
    log(f'arm {arm} eval：與拼接基底相比 前150 重疊 {len(set(oa[:150]) & set(ob[:150]))}｜前300 {len(set(oa[:300]) & set(ob[:300]))}｜前600 {len(set(oa[:600]) & set(ob[:600]))}｜帶區301-600 留下 {len(set(oa[300:600]) & set(ob[300:600]))}｜Spearman {spearmanr(a, b)[0]:.4f}')
    log(f'wrote {fn} md5 {hashlib.md5(open(fn, "rb").read()).hexdigest()}')
    # ---- D′ 同規則變體（FINAL-D2／HB2w70 的 risk 含 D′：最終排序的帶區 301–600 內 dom5≥0.6 或 ovl≥0.4 → 放到第 1001／1002 名之間）
    rv = risk.loc[pids_e].to_numpy(); rnk = pd.Series(-rv).rank(method='first').to_numpy().astype(int)
    cf = context('evaluation', pids_e, s_e, M + '/hand_scores_eval.parquet', EQ + 'dom_evaluation_v5x.parquet')
    flag = (rnk >= 301) & (rnk <= 600) & ((cf.dom5.to_numpy() >= 0.6) | (cf.ovl.to_numpy() >= 0.4))
    r_hi = rv[np.flatnonzero(rnk == 1001)[0]]; r_lo = rv[np.flatnonzero(rnk == 1002)[0]]; order = pd.Series(-rv[flag]).rank(method='first').to_numpy()
    rv2 = rv.copy(); rv2[flag] = r_hi - (r_hi - r_lo) * order / (flag.sum() + 1); assert pd.Series(rv2).duplicated().sum() == 0
    risk2 = pd.Series(rv2, index=pd.Series(pids_e)); out2 = ref.copy(); out2['risk_score'] = [repr(float(v)) for v in risk2.loc[out2.pair_id].to_numpy()]
    fn2 = fn.replace(f'submission_TW{arm}_', f'submission_TW{arm}Dp_'); out2.to_csv(fn2, index=False)
    _dp = R + '5_outputs/research_0916/submission_D2_v038_dom06_demote_pairs.parquet'; Dold = set(pd.read_parquet(_dp).pair_id) if os.path.exists(_dp) else set()   # log-only
    log(f"arm {arm} D′ 變體：降權 {int(flag.sum())} 對（與 09-16 的 29 對重疊 {len(Dold & set(np.array(pids_e)[flag]))}）→ {fn2} md5 {hashlib.md5(open(fn2, 'rb').read()).hexdigest()}")
log('SUMMARY ' + json.dumps(summary, ensure_ascii=False))
