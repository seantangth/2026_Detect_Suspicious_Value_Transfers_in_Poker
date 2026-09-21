"""0917-F eval 側規模估計＋候選檔：全部 372 個 dev 正例訓練（5 種子），對 eval 配對預測家族；
組合＝0.5*LGB 機率＋0.5*top3 正規化分數（與 family_clf2.py 相同、事先固定）。
輸出：與現行 argmax 家族不同的配對數（依 K1 名次帶），以及只改 predicted_behavior 的候選檔 K3b。"""
import hashlib, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
R = (str(__import__('pathlib').Path(__file__).resolve().parents[2]) + '/')
M, OUT = R + '5_outputs/models/v5x', R + '5_outputs/eqx_0913/'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
def load(ph, short):
    pf = pl.read_parquet(M + f'/pair_features_{short}.parquet')
    for f in ['presence_contrast_x_', 'pairpol_feats_', 'presence_contrast_v2_']:
        pf = pf.join(pl.read_parquet(OUT + f + ph + '.parquet'), on='pair_id', how='left')
    return pf.fill_null(0.0)
pd_, pe = load('development', 'dev').filter(pl.col('label') == 1), load('evaluation', 'eval')
fe = [c for c in pd_.columns if c not in ('pair_id', 'table_id', 'label', 'behavior_family')]
assert all(c in pe.columns for c in fe), [c for c in fe if c not in pe.columns][:5]
y = np.array([FAMS.index(f) for f in pd_['behavior_family'].to_list()])
X, Xe = np.nan_to_num(pd_.select(fe).to_numpy().astype(np.float64)), np.nan_to_num(pe.select(fe).to_numpy().astype(np.float64))
proba = np.zeros((pe.height, 3))
for sd in range(5):
    m = lgb.LGBMClassifier(objective='multiclass', n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=8, colsample_bytree=0.3,
                           subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=sd, verbose=-1).fit(X, y)
    proba += m.predict_proba(Xe) / 5
s = np.clip(np.stack([np.nan_to_num(pe[f'{f}_top3'].to_numpy(), nan=0) for f in FAMS], 1), 1e-6, None)
p_arg = s / s.sum(1, keepdims=True)
new = np.array(FAMS)[(0.5 * proba + 0.5 * p_arg).argmax(1)]; arg = np.array(FAMS)[s.argmax(1)]
fam = pd.DataFrame({'pair_id': pe['pair_id'].to_list(), 'fam_new': new, 'fam_argmax': arg, 'p_new_max': (0.5 * proba + 0.5 * p_arg).max(1)})
fam.to_parquet(R + '5_outputs/revise_0917/family_clf_eval.parquet')   # (release) moved up; the K3b block below is a research side output
if not __import__('os').path.exists(R + '5_outputs/submissions/submission_K1_D3risk_v040evid.csv'): raise SystemExit(0)
sub = pd.read_csv(R + '5_outputs/submissions/submission_K1_D3risk_v040evid.csv', dtype=str, keep_default_na=False)  # 字串層級：risk／證據逐位元保留
sub['rk'] = sub['risk_score'].astype(float).rank(ascending=False, method='first').astype(int)
d = sub.merge(fam, on='pair_id', how='left')
act = d['predicted_behavior'] != 'none'
print('現行提交的家族 == 本腳本重算的 argmax？（檢查特徵表與生產一致）:', float((d.loc[act, 'predicted_behavior'] == d.loc[act, 'fam_argmax']).mean()))
for lo, hi in [(1, 150), (151, 300), (301, 400), (401, 600), (601, 1000)]:
    q = d[(d.rk >= lo) & (d.rk <= hi)]; ch = q[q.predicted_behavior != q.fam_new]
    print(f'名次 {lo}-{hi}: 改判 {len(ch)}/{len(q)} 對', dict(pd.crosstab(ch.predicted_behavior, ch.fam_new).stack().loc[lambda v: v > 0]))
out = d.copy(); out.loc[act, 'predicted_behavior'] = out.loc[act, 'fam_new']
out = out[sub.columns.drop('rk')]
p = R + '5_outputs/submissions/submission_K3b_K1_famclf.csv'; out.to_csv(p, index=False)
print(p, hashlib.md5(open(p, 'rb').read()).hexdigest(), '| 與 K1 不同的列數', int((out.predicted_behavior.values != sub.predicted_behavior.values).sum()))
fam.to_parquet(R + '5_outputs/revise_0917/family_clf_eval.parquet')
