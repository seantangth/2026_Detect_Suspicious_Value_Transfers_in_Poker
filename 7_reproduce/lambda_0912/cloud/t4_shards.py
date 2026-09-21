#!/usr/bin/env python3
"""TPDS T4 — 候選配對 × 共享手的特徵分片（依 table_id 切檔）。

本機記憶體不足以在本地 join T3（60,000,000 列），所以在雲端算好再 rsync 回去。
輸入：
  --t2     seqnll_player.parquet   12,000,000 列（2,000,000 手 × 6 座位）
  --t3     seqnll_pair.parquet     60,000,000 列（每手 6×5 = 30 個有序 (X,Y)）
  --hands  raw hands.parquet       2,000,000 列（hand_id, table_id, phase）
  --seats  raw seats.parquet       12,000,000 列（判定「該手有沒有發牌給該玩家」）
  --cand-dev  cand_pairs_development.parquet（pair_id, a, b, table_id, shared；a < b 已成立）
  --cand-eval evaluation_pairs.csv（pair_id, player_1, player_2, shared_hands）
輸出：
  <out>/nll_<prefix>/{development,evaluation}/<table_id>.parquet（各 400 檔，zstd level 9）
  <out>/nll_<prefix>/rowcount_check.json

分塊策略（為什麼這樣切，見 README 等級的說明）：
  * T2（12M 列）與 hand→table 對應（2M 列）都夠小，一次讀進來，join 上 table_id 後
    用 partition_by 切成 400 份常駐記憶體（每桌 30,000 列），總量約 1–2 GB。
  * T3（60M 列）**只掃一次**：用 pyarrow 的 iter_batches 分批讀（每批數百萬列），
    每批立刻 (a) 只取需要的 5 個欄、(b) 把 id 轉成 String、(c) NaN 轉 null、
    (d) join 上 table_id、(e) partition_by 成 400 份丟進 buckets。
    掃完後每桌手上是一疊小 DataFrame，處理該桌時才 concat。
    這樣避免了「每桌 filter 一次全檔 → 掃檔 400 次」，也避免把 60M 列一次攤成
    一個巨大的 pandas DataFrame；常駐量約 60M × (3 個字串 + 2 個 float64) ≈ 7 GB，
    在 222 GB RAM 下非常寬裕。
  * 主迴圈一次處理一張桌、算完立刻寫檔並 pop 掉該桌的 bucket，記憶體不會累積。

數值細節：
  * infer.py 用 nanify() 把 -inf 寫成 **NaN**（不是 null），所以 T2 的 *_max 欄與
    T3 的 gain_max 在 n_act=0 的座位上是 NaN。polars 的 NaN ≠ null，會污染 sum 與
    max_horizontal，所以讀檔當下就 fill_nan(None)，之後全部用 null 語意運算。
  * excess 的「同手其他 Y 平均」用『總和 − 自己』與『計數 − 自己』的通式算，
    不寫死 5 與 4；分母為 0 時輸出 null（polars 的 x/0 會給 inf，必須用 when 擋掉）。
  * 中間運算一律 Float64，最後 select 時才 cast 成 Float32，避免 float32 累加漂移。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import polars as pl
import pyarrow.parquet as pq

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


# ---------------------------------------------------------------- 欄位對應
# T2 原欄名 -> 輸出時去掉 _N_ 的後綴（再加 A_/B_ 前綴）
T2_MAP = {
    'nll_N_sum': 'nll_sum',
    'nll_N_max': 'nll_max',
    'nll_N_pre': 'nll_pre',
    'nll_N_flop': 'nll_flop',
    'nll_N_turn': 'nll_turn',
    'nll_N_river': 'nll_river',
    'nll_N_fold_max': 'nll_fold_max',
    'nll_N_call_max': 'nll_call_max',
    'nll_N_agg_max': 'nll_agg_max',
    'nll_size_N_sum': 'nll_size_sum',
}
T2_FLOAT = list(T2_MAP.keys())
T2_COLS = ['hand_id', 'player_id', 'n_act'] + T2_FLOAT
T3_COLS = ['hand_id', 'player_id', 'other_id', 'gain_sum', 'gain_max']

# A_/B_ 兩側各 11 欄（10 個 float + n_act）
_SIDE = [T2_MAP[c] for c in T2_FLOAT] + ['n_act']

OUT_SCHEMA = {'pair_id': pl.String, 'hand_id': pl.String, 'a': pl.String, 'b': pl.String}
for _p in ('A', 'B'):
    for _s in _SIDE:
        OUT_SCHEMA[f'{_p}_{_s}'] = pl.Float32
for _c in ('gain_AB_sum', 'gain_AB_max', 'gain_BA_sum', 'gain_BA_max',
           'excess_AB_sum', 'excess_AB_max', 'excess_BA_sum', 'excess_BA_max',
           'nll_max_pair', 'nll_sum_pair', 'gain_max_pair', 'gain_sum_pair',
           'excess_max_pair', 'excess_sum_pair'):
    OUT_SCHEMA[_c] = pl.Float32

EMPTY_OUT = pl.DataFrame(schema=OUT_SCHEMA)
ZSTD = dict(compression='zstd', compression_level=9)

# 規格上記載的真實資料期望列數（只當「輸入檔是不是給錯了」的軟性提醒，不影響流程）
KNOWN_EXPECT = {'development': 16_648_186, 'evaluation': 9_651_820}


def s(col):
    """parquet 的 dictionary 編碼讀進 polars 可能是 Categorical/Enum，一律轉 String 再 join。"""
    return pl.col(col).cast(pl.String)


def f64(col):
    """轉 Float64 並把 NaN 當成缺值（infer.py 的 nanify 會寫出 NaN 而非 null）。"""
    return pl.col(col).cast(pl.Float64).fill_nan(None)


def excess_expr(gain, tot, cnt):
    """excess = gain − mean_{Y≠對手}；用『總和 − 自己』/『計數 − 自己』的通式。

    tot/cnt 是該 (hand, X) 全部 Y 列的非空總和與非空計數（已排除 NaN）。
    扣掉自己這一列之後若沒有其他 Y（分母 0）就回 null，不能讓 polars 算出 inf。
    """
    o_sum = pl.col(tot) - pl.col(gain).fill_null(0.0)
    o_cnt = pl.col(cnt) - pl.col(gain).is_not_null().cast(pl.Int64)
    return (pl.when(o_cnt > 0)
              .then(pl.col(gain) - o_sum / o_cnt)
              .otherwise(None))


def hmax(*cols):
    """跳過 null 取大；NaN 會傳染給 max_horizontal，所以先轉 null（此處為雙保險）。"""
    return pl.max_horizontal(*[pl.col(c).fill_nan(None) for c in cols])


# ---------------------------------------------------------------- 載入
def load_hands(path):
    hf = (pl.read_parquet(path, columns=['hand_id', 'table_id', 'phase'])
            .select(s('hand_id'), s('table_id'), s('phase')))
    tables = sorted(hf['table_id'].unique().to_list())
    log(f'hands: {hf.height:,} 列、{len(tables)} 張桌、'
        f'phase={hf["phase"].value_counts().to_dicts()}')
    return hf, tables


def load_dealt(seats_path, hf):
    """dealt = 每手每個「被發牌」的玩家，附上該手的 table_id 與 phase。

    T2 其實也是 (hand_id, player_id) 的粒度且同樣是 12M 列，但規格指定用 raw seats
    判定發牌與否，這裡照做（也讓 T2 缺列時能以 null 呈現而不是整列消失）。
    """
    st = (pl.read_parquet(seats_path, columns=['hand_id', 'player_id'])
            .select(s('hand_id'), s('player_id')))
    dealt = st.join(hf, on='hand_id', how='inner')
    log(f'seats: {st.height:,} 列 -> dealt {dealt.height:,} 列')
    del st
    parts = {k[0]: v for k, v in
             dealt.partition_by('table_id', as_dict=True, include_key=False).items()}
    del dealt
    return parts


def load_cand_dev(path):
    cd = (pl.read_parquet(path, columns=['pair_id', 'a', 'b', 'table_id', 'shared'])
            .select(s('pair_id'), s('a'), s('b'), s('table_id'),
                    pl.col('shared').cast(pl.Int64)))
    bad = cd.filter(pl.col('a') >= pl.col('b')).height
    if bad:
        log(f'警告：cand_dev 有 {bad} 列不滿足 a < b（規格說已成立），維持原樣不重排')
    log(f'cand_dev: {cd.height:,} 對、shared 總和 {cd["shared"].sum():,}')
    return cd


def load_cand_eval(path, dealt_parts, tables):
    """eval 候選對：自己算 a=min、b=max；table_id 依規格「由任一共享手取得」。

    做法：逐桌把該桌 evaluation 期的座位做 hand_id 自連接，取出「同一手同時被發牌」
    的無序玩家對 (a<b)，去重後每桌只剩 C(30,2) 量級（幾百列），全部 concat 起來也才
    十幾萬列，最後跟 evaluation_pairs 對一次 join 即可。
    比「先 join 全部共享手再取第一手」省掉數千萬列的中間結果，但語意完全相同：
    有被指派到的桌，一定存在一手把 a 與 b 同時發牌。
    """
    ce = pl.read_csv(path, columns=['pair_id', 'player_1', 'player_2', 'shared_hands'])
    ce = ce.select(
        s('pair_id'),
        pl.min_horizontal(s('player_1'), s('player_2')).alias('a'),
        pl.max_horizontal(s('player_1'), s('player_2')).alias('b'),
        pl.col('shared_hands').cast(pl.Int64).alias('shared'),
    )
    co_all = []
    for t in tables:
        d = dealt_parts.get(t)
        if d is None:
            continue
        d = d.filter(pl.col('phase') == 'evaluation').select('hand_id', 'player_id')
        if d.height == 0:
            continue
        co = (d.join(d.select('hand_id', pl.col('player_id').alias('p2')), on='hand_id')
                .filter(pl.col('player_id') < pl.col('p2'))
                .select(pl.col('player_id').alias('a'), pl.col('p2').alias('b'))
                .unique()
                .with_columns(pl.lit(t, dtype=pl.String).alias('table_id')))
        co_all.append(co)
    codealt = pl.concat(co_all) if co_all else pl.DataFrame(
        schema={'a': pl.String, 'b': pl.String, 'table_id': pl.String})
    log(f'eval 同手共現的無序玩家對：{codealt.height:,} 組（跨 {len(co_all)} 桌）')

    hit = (ce.select('pair_id', 'a', 'b').join(codealt, on=['a', 'b'], how='inner')
             .group_by('pair_id')
             .agg(pl.first('table_id').alias('table_id'), pl.len().alias('n_tbl')))
    multi = hit.filter(pl.col('n_tbl') > 1).height
    if multi:
        log(f'警告：{multi} 對 eval 配對在多張桌都有共現手，取第一張')
    ce = ce.join(hit.select('pair_id', 'table_id'), on='pair_id', how='left')
    missing = ce.filter(pl.col('table_id').is_null())
    if missing.height:
        log(f'警告：{missing.height} 對 eval 配對找不到任何共享手，無法指派 table_id，'
            f'將不會產生任何列（shared_hands 總和 {missing["shared"].sum():,}）')
    log(f'cand_eval: {ce.height:,} 對、shared_hands 總和 {ce["shared"].sum():,}')
    return ce, int(missing.height), int(missing['shared'].sum() or 0)


def load_t2_parts(path, h2t):
    """T2 一次讀完（12M 列），join 上 table_id 後切成 400 份。"""
    t2 = pl.read_parquet(path, columns=T2_COLS).select(
        s('hand_id'), s('player_id'), pl.col('n_act').cast(pl.Float64),
        *[f64(c) for c in T2_FLOAT])
    log(f'T2: {t2.height:,} 列')
    t2 = t2.join(h2t, on='hand_id', how='inner')
    parts = {k[0]: v for k, v in
             t2.partition_by('table_id', as_dict=True, include_key=False).items()}
    del t2
    log(f'T2 已切成 {len(parts)} 桌')
    return parts


def load_t3_buckets(path, h2t, batch_rows):
    """T3 只掃一次：分批讀 -> 轉型 -> 接 table_id -> partition_by 丟進每桌的 bucket。"""
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    log(f'T3: {n_rows:,} 列、{pf.metadata.num_row_groups} 個 row group，'
        f'分批讀（batch={batch_rows:,}）')
    buckets = defaultdict(list)
    done = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows, columns=T3_COLS)):
        d = pl.from_arrow(batch)
        if isinstance(d, pl.Series):          # 保險：單欄時 from_arrow 會回 Series
            d = d.to_frame()
        d = d.select(s('hand_id'), s('player_id'), s('other_id'),
                     f64('gain_sum'), f64('gain_max'))
        d = d.join(h2t, on='hand_id', how='inner')
        for k, part in d.partition_by('table_id', as_dict=True, include_key=False).items():
            buckets[k[0]].append(part)
        done += batch.num_rows
        del d, batch
        if bi % 5 == 0:
            log(f'  T3 {done:,}/{n_rows:,}')
    log(f'T3 掃描完成 {done:,} 列 -> {len(buckets)} 桌')
    return buckets, done


# ---------------------------------------------------------------- 單桌計算
def build_table_phase(cand_t, dealt_ph, t2_t, t3_t, grp_t):
    """算出「該桌 × 該 phase」的所有列。cand_t 已經是這張桌的候選配對。"""
    if cand_t.height == 0 or dealt_ph.height == 0:
        return EMPTY_OUT

    # 骨架：候選配對 × 「a 被發牌的手」，再用 semi-join 留下「b 也被發牌」的手
    skel = (cand_t.select('pair_id', 'a', 'b')
            .join(dealt_ph, left_on='a', right_on='player_id', how='inner')
            .join(dealt_ph, left_on=['hand_id', 'b'],
                  right_on=['hand_id', 'player_id'], how='semi'))
    if skel.height == 0:
        return EMPTY_OUT

    # --- T2：同一手中 player_id==a 的列給 A_*，==b 的列給 B_*
    # 一律用 left join，T2 萬一缺列也只是變 null，不能讓列數少掉（列數要等於 shared）
    for side, key in (('A', 'a'), ('B', 'b')):
        t2s = t2_t.select(
            pl.col('hand_id'), pl.col('player_id').alias(key),
            pl.col('n_act').alias(f'{side}_n_act'),
            *[pl.col(c).alias(f'{side}_{T2_MAP[c]}') for c in T2_FLOAT])
        skel = skel.join(t2s, on=['hand_id', key], how='left')

    # --- T3：(X=a, Y=b) 給 gain_AB_*、(X=b, Y=a) 給 gain_BA_*
    gab = t3_t.select(pl.col('hand_id'), pl.col('player_id').alias('a'),
                      pl.col('other_id').alias('b'),
                      pl.col('gain_sum').alias('gain_AB_sum'),
                      pl.col('gain_max').alias('gain_AB_max'))
    gba = t3_t.select(pl.col('hand_id'), pl.col('player_id').alias('b'),
                      pl.col('other_id').alias('a'),
                      pl.col('gain_sum').alias('gain_BA_sum'),
                      pl.col('gain_max').alias('gain_BA_max'))
    skel = skel.join(gab, on=['hand_id', 'a', 'b'], how='left')
    skel = skel.join(gba, on=['hand_id', 'a', 'b'], how='left')

    # --- (hand, X) 的 gain 總和／計數，用來扣掉自己算「其他 Y 的平均」
    for side, key in (('A', 'a'), ('B', 'b')):
        g = grp_t.select(
            pl.col('hand_id'), pl.col('player_id').alias(key),
            pl.col('gs_tot').alias(f'{side}_gs_tot'), pl.col('gs_cnt').alias(f'{side}_gs_cnt'),
            pl.col('gm_tot').alias(f'{side}_gm_tot'), pl.col('gm_cnt').alias(f'{side}_gm_cnt'))
        skel = skel.join(g, on=['hand_id', key], how='left')

    skel = skel.with_columns(
        excess_expr('gain_AB_sum', 'A_gs_tot', 'A_gs_cnt').alias('excess_AB_sum'),
        excess_expr('gain_AB_max', 'A_gm_tot', 'A_gm_cnt').alias('excess_AB_max'),
        excess_expr('gain_BA_sum', 'B_gs_tot', 'B_gs_cnt').alias('excess_BA_sum'),
        excess_expr('gain_BA_max', 'B_gm_tot', 'B_gm_cnt').alias('excess_BA_max'),
    ).with_columns(
        hmax('A_nll_max', 'B_nll_max').alias('nll_max_pair'),
        (pl.col('A_nll_sum') + pl.col('B_nll_sum')).alias('nll_sum_pair'),
        hmax('gain_AB_max', 'gain_BA_max').alias('gain_max_pair'),
        (pl.col('gain_AB_sum') + pl.col('gain_BA_sum')).alias('gain_sum_pair'),
        hmax('excess_AB_max', 'excess_BA_max').alias('excess_max_pair'),
        (pl.col('excess_AB_sum') + pl.col('excess_BA_sum')).alias('excess_sum_pair'),
    )
    return skel.select([pl.col(c).cast(dt) for c, dt in OUT_SCHEMA.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t2', required=True, help='seqnll_player.parquet')
    ap.add_argument('--t3', required=True, help='seqnll_pair.parquet')
    ap.add_argument('--hands', required=True, help='raw hands.parquet')
    ap.add_argument('--seats', required=True, help='raw seats.parquet')
    ap.add_argument('--cand-dev', required=True, help='cand_pairs_development.parquet')
    ap.add_argument('--cand-eval', required=True, help='evaluation_pairs.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', required=True, choices=['seqnll', 'gbnll'])
    ap.add_argument('--batch-rows', type=int, default=4_000_000,
                    help='T3 分批讀的每批列數')
    args = ap.parse_args()

    root = os.path.join(args.out, f'nll_{args.prefix}')
    dirs = {p: os.path.join(root, p) for p in ('development', 'evaluation')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    hf, tables = load_hands(args.hands)
    h2t = hf.select('hand_id', 'table_id')
    dealt_parts = load_dealt(args.seats, hf)
    del hf

    cd = load_cand_dev(args.cand_dev)
    ce, n_eval_no_table, shared_eval_no_table = load_cand_eval(args.cand_eval, dealt_parts, tables)
    expect = {'development': int(cd['shared'].sum()), 'evaluation': int(ce['shared'].sum())}
    for ph, n in KNOWN_EXPECT.items():
        if expect[ph] != n:
            log(f'提醒：{ph} 的候選 shared 總和 {expect[ph]:,} 與規格記載的 {n:,} 不同'
                f'（合成資料測試時屬正常；跑真實資料時請確認輸入檔沒給錯）')

    unknown_dev_tbl = cd.filter(~pl.col('table_id').is_in(tables)).height
    if unknown_dev_tbl:
        log(f'警告：cand_dev 有 {unknown_dev_tbl} 列的 table_id 不在 hands 裡')

    cd_parts = {k[0]: v for k, v in
                cd.partition_by('table_id', as_dict=True, include_key=False).items()}
    ce_parts = {k[0]: v for k, v in
                ce.drop_nulls('table_id')
                  .partition_by('table_id', as_dict=True, include_key=False).items()}

    t2_parts = load_t2_parts(args.t2, h2t)
    t3_buckets, _ = load_t3_buckets(args.t3, h2t, args.batch_rows)
    del h2t

    # ------------------------------------------------------------ 主迴圈：一桌一寫
    empty_t2 = pl.DataFrame(schema={'hand_id': pl.String, 'player_id': pl.String,
                                    'n_act': pl.Float64,
                                    **{c: pl.Float64 for c in T2_FLOAT}})
    empty_t3 = pl.DataFrame(schema={'hand_id': pl.String, 'player_id': pl.String,
                                    'other_id': pl.String,
                                    'gain_sum': pl.Float64, 'gain_max': pl.Float64})
    empty_dealt = pl.DataFrame(schema={'hand_id': pl.String, 'player_id': pl.String,
                                       'phase': pl.String})
    written = {'development': 0, 'evaluation': 0}
    nfiles = {'development': 0, 'evaluation': 0}

    for i, t in enumerate(tables):
        t2_t = t2_parts.pop(t, empty_t2)
        chunks = t3_buckets.pop(t, None)
        t3_t = pl.concat(chunks) if chunks else empty_t3
        del chunks
        dealt_t = dealt_parts.get(t, empty_dealt)

        # 每個 (hand, X) 的 gain 非空總和與非空計數（NaN 在讀檔時已轉 null，不會污染）
        grp_t = t3_t.group_by(['hand_id', 'player_id']).agg(
            pl.col('gain_sum').sum().alias('gs_tot'),
            pl.col('gain_sum').count().cast(pl.Int64).alias('gs_cnt'),
            pl.col('gain_max').sum().alias('gm_tot'),
            pl.col('gain_max').count().cast(pl.Int64).alias('gm_cnt'))

        for ph, cand_parts in (('development', cd_parts), ('evaluation', ce_parts)):
            cand_t = cand_parts.pop(t, None)
            dealt_ph = dealt_t.filter(pl.col('phase') == ph).select('hand_id', 'player_id')
            out = (build_table_phase(cand_t, dealt_ph, t2_t, t3_t, grp_t)
                   if cand_t is not None else EMPTY_OUT)
            out.write_parquet(os.path.join(dirs[ph], f'{t}.parquet'), **ZSTD)
            written[ph] += out.height
            nfiles[ph] += 1
            del out
        del t2_t, t3_t, grp_t
        if i % 25 == 0 or i == len(tables) - 1:
            log(f'  桌 {i + 1}/{len(tables)} ({t}) dev={written["development"]:,} '
                f'eval={written["evaluation"]:,}')

    # ------------------------------------------------------------ 列數核對
    left_dev = sum(v.height for v in cd_parts.values())
    left_eval = sum(v.height for v in ce_parts.values())
    if left_dev or left_eval:
        log(f'警告：有候選配對的 table_id 不在 hands 的桌清單裡（dev {left_dev}、eval {left_eval}），'
            f'這些配對沒有輸出')

    report = {'prefix': args.prefix, 'n_tables': len(tables)}
    ok = True
    for ph in ('development', 'evaluation'):
        d = written[ph] - expect[ph]
        ok = ok and d == 0
        report[ph] = {'rows_written': written[ph], 'rows_expected': expect[ph],
                      'diff': d, 'files': nfiles[ph], 'ok': d == 0}
    report['eval_pairs_without_table'] = n_eval_no_table
    report['eval_shared_without_table'] = shared_eval_no_table
    report['cand_pairs_dropped_unknown_table'] = {'development': left_dev, 'evaluation': left_eval}
    report['ok'] = ok
    with open(os.path.join(root, 'rowcount_check.json'), 'w') as f:
        json.dump(report, f, indent=2)

    log('列數核對：')
    for ph in ('development', 'evaluation'):
        r = report[ph]
        log(f'  {ph:12s} 寫出 {r["rows_written"]:,} / 期望 {r["rows_expected"]:,} '
            f'差 {r["diff"]:+,}  檔案 {r["files"]} 個  {"OK" if r["ok"] else "**不符**"}')
    log(f'輸出：{root}')
    if not ok:
        log('列數核對失敗，檔案已全部寫完，以非 0 退出')
        sys.exit(1)
    log('全部完成')


if __name__ == '__main__':
    main()
