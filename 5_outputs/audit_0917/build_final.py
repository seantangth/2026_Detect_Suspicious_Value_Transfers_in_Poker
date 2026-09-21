"""把「已分別確認為正」的改動組裝成最終候選（字串層級，risk／證據欄不做浮點往返）。
    risk        ← <risk_src>（新配對鏈的 *_head.csv，欄 1–3 來自 v022 模板）
    evidence    ← <evidence_src>（預設 v040：dev 量尺 0.7014，全場最高）
    behavior    ← 訓練式家族分類器（5_outputs/revise_0917/family_clf_eval.parquet，dev 正例準確率 0.9927 vs argmax 0.9758）
                  none 集合改用「本候選自己的 risk 前 active_frac」（生產 v011 起凍結的 none 集合與 K1 自己的前 20% 只重疊 62%）
用法：build_final.py <risk_src.csv> <out.csv> [evidence_src.csv] [active_frac=0.2]
輸出：驗證結果、與各來源的差異列數、md5。
"""
import sys, hashlib, subprocess
import numpy as np, pandas as pd

R = (str(__import__('pathlib').Path(__file__).resolve().parents[2]) + '/')
EV = [f'evidence_hand_{i}' for i in range(1, 6)]
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']

risk_src, out = sys.argv[1], sys.argv[2]
ev_src = sys.argv[3] if len(sys.argv) > 3 else R + '5_outputs/submissions/submission_v040cand_v5xcp2_pbni4.csv'
af = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

a = pd.read_csv(risk_src, dtype=str, keep_default_na=False)      # risk 來源（字串原樣）
b = pd.read_csv(ev_src, dtype=str, keep_default_na=False)        # 證據來源
assert list(a.columns) == list(b.columns) and (a.pair_id == b.pair_id).all()
fam = pd.read_parquet(R + '5_outputs/revise_0917/family_clf_eval.parquet').set_index('pair_id')

o = a.copy()
o[EV] = b[EV].to_numpy()
rk = a.risk_score.astype(float).rank(ascending=False, method='first').to_numpy()
active = rk <= round(len(o) * af)
o['predicted_behavior'] = np.where(active, fam.loc[o.pair_id, 'fam_new'].to_numpy(), 'none')
o.to_csv(out, index=False)

print('risk 欄與 risk 來源相同列數 :', int((o.risk_score == a.risk_score).sum()), '/', len(o))
print('證據五欄與證據來源相同列數  :', int((o[EV].to_numpy() == b[EV].to_numpy()).all(1).sum()), '/', len(o))
print('behavior 與 risk 來源不同列數:', int((o.predicted_behavior != a.predicted_behavior).sum()),
      '| 與證據來源不同列數:', int((o.predicted_behavior != b.predicted_behavior).sum()))
print('behavior 分佈:', o.predicted_behavior.value_counts().to_dict())
for lo, hi in [(1, 150), (151, 300), (301, 600), (601, 1000)]:
    m = (rk >= lo) & (rk <= hi)
    print(f'  名次 {lo}-{hi}: 家族與 argmax 版不同 {int((o.predicted_behavior.to_numpy()[m] != a.predicted_behavior.to_numpy()[m]).sum())}/{int(m.sum())}')
r = subprocess.run([sys.executable, R + '3_src/validate_submission.py', out], capture_output=True, text=True)
print(r.stdout.strip()[-260:], r.stderr.strip()[-200:])
print('md5', hashlib.md5(open(out, 'rb').read()).hexdigest(), out)
