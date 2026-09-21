"""截短世界（truncated world）第 1 部分：基底池化特徵（146 欄）。
用法: python tw_base.py <wtag> <lo> <hi> [ntables]
對 dev 每桌：L2 列依 t_rank∈[lo,hi) 過濾 → 在窗口內重算 pct_*（手模型的 16 個配對內分位特徵）→ 以該桌留出折的
v5x 手模型打分（OOF，與生產同）→ tpds_model.pool_table。shared 改為窗口內兩人同桌手數。
輸出 band/tw/<wtag>/{pair_features,hand_scores}.parquet。wtag=FULLCHK（lo=0 hi=3000）時只跑前 ntables 桌並與生產表逐欄比對。"""
import os, sys, json, time
os.environ.setdefault('TPDS_EQTAG', 'x'); os.environ.setdefault('POLARS_MAX_THREADS', '6')
R = (str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/')
sys.path.insert(0, R + '3_src')
import numpy as np, polars as pl, lightgbm as lgb
from tpds_model import PROC, RAW, l2_files, read_l2, swap_ab, pool_table, predict_hand, POOL_SRC, PCT_BASE, log
wtag, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); ntab = int(sys.argv[4]) if len(sys.argv) > 4 else 0
M = R + '5_outputs/models/v5x/'; OUT = R + f'5_outputs/research_0919/band/tw/{wtag}/'; os.makedirs(OUT, exist_ok=True)
feats = json.load(open(M + 'hand_features.json')); folds = json.load(open(M + 'folds_by_table.json'))
models = {k: [lgb.Booster(model_file=M + f'hand_{k}_f{i}.txt') for i in range(5)] for k in POOL_SRC}
cand = pl.read_parquet(PROC / 'cand_pairs_development.parquet')
files = l2_files('development'); files = files[:ntab] if ntab else files
# 窗口內同桌手數（shared）
h = pl.read_parquet(PROC / 'hands_l1.parquet', columns=['hand_id', 'table_id', 'phase', 't_rank']).filter((pl.col('phase') == 'development') & (pl.col('t_rank') >= lo) & (pl.col('t_rank') < hi))
s = pl.read_parquet(RAW / 'seats.parquet', columns=['hand_id', 'player_id']).with_columns(pl.col('hand_id').cast(pl.Utf8), pl.col('player_id').cast(pl.Utf8)).join(h.select('hand_id', 'table_id'), on='hand_id')
log(f'window [{lo},{hi}) hands {h.height:,} seat rows {s.height:,}')
parts, rows, sh = [], [], []
t0 = time.time()
for i, f in enumerate(files):
    d = read_l2(f, 'development', 1, 1, 0)
    if d.height == 0: continue
    t = d['table_id'][0]
    assert d['t_rank'].min() >= 0 and d['t_rank'].max() < 3000, 't_rank 不是桌內 0..2999'
    d = d.filter((pl.col('t_rank') >= lo) & (pl.col('t_rank') < hi))
    cols = [c for c in PCT_BASE if c in d.columns]
    d = d.with_columns([(pl.col(c).rank(method='average').over('pair_id') / pl.len().over('pair_id')).cast(pl.Float32).alias(f'pct_{c}') for c in cols])
    X = d.select(feats).to_numpy().astype(np.float32); Xs = swap_ab(d).select(feats).to_numpy().astype(np.float32)
    sc = {k: predict_hand(models[k], X, Xs, folds[t]) for k in POOL_SRC}
    parts.append(pool_table(d, sc))
    rows.append(d.select(['pair_id', 'hand_id']).with_columns([pl.Series(f's_{k}', sc[k].astype(np.float32)) for k in POOL_SRC]))
    st = s.filter(pl.col('table_id') == t).select('hand_id', 'player_id')
    pr = st.join(st.rename({'player_id': 'b'}), on='hand_id').filter(pl.col('player_id') < pl.col('b')).group_by('player_id', 'b').len().rename({'player_id': 'a', 'len': 'shared_w'})
    sh.append(pr)
    if i % 50 == 0: log(f'{wtag}: {i + 1}/{len(files)} {time.time() - t0:.0f}s')
pf = pl.concat(parts); hs = pl.concat(rows); sh = pl.concat(sh)
c2 = cand.with_columns(pl.min_horizontal('a', 'b').alias('_a'), pl.max_horizontal('a', 'b').alias('_b')).join(sh.rename({'a': '_a', 'b': '_b'}), on=['_a', '_b'], how='left').with_columns(pl.col('shared_w').fill_null(0).cast(pl.Int32))
out = c2.select(['pair_id', 'table_id', pl.col('shared_w').alias('shared'), 'label', 'behavior_family']).join(pf, on='pair_id', how='left')
if wtag == 'FULLCHK':
    ref = pl.read_parquet(M + 'pair_features_dev.parquet'); tabs = set(pl.concat(parts)['pair_id'].to_list())
    a = out.filter(pl.col('pair_id').is_in(list(tabs))).sort('pair_id'); b = ref.filter(pl.col('pair_id').is_in(list(tabs))).sort('pair_id')
    assert a['pair_id'].to_list() == b['pair_id'].to_list()
    worst = []
    for c in b.columns:
        if c in ('pair_id', 'table_id', 'behavior_family'): continue
        x = a[c].cast(pl.Float64).fill_null(-9).to_numpy(); y = b[c].cast(pl.Float64).fill_null(-9).to_numpy()
        worst.append((float(np.abs(x - y).max()), c))
    worst.sort(reverse=True); log(f'FULLCHK pairs {a.height}: 最大絕對差前 6 欄 {worst[:6]}')
else:
    out.write_parquet(OUT + 'pair_features.parquet'); hs.write_parquet(OUT + 'hand_scores.parquet')
    log(f'written {OUT} pairs {out.height} (shared>=38: {int((out["shared"] >= 38).sum())}) hand rows {hs.height:,}')
