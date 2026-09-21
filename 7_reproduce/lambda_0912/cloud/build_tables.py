#!/usr/bin/env python3
"""TPDS seqnll — 建表：把 raw/processed parquet 壓成訓練用的緊緻 numpy 陣列。

輸出到 --out（預設 /home/ubuntu/tpds/tables）：
  hand-level  (N_H)  : h_bb, h_grp, h_phase, h_nact, h_aoff, h_board_rank[5], h_board_suit[5], h_tbl
  action-level(N_A)  : a_hand, a_slot, a_sidx, a_type, a_amtb, a_potb, a_tcb, a_sprb, a_nactive,
                       a_foldbm, a_actedbm, a_lastaggr, a_isagg, a_iscall, a_isfold, a_actno
  seat-level  (N_S=6*N_H, 列序 = hand_idx*6 + seat_no)
                     : s_hole_rank[2], s_hole_suit[2], s_relpos, s_pcode,
                       s_chen, s_stackbb, s_pfhi, s_pflo, s_pfpair, s_pfsuited,
                       s_strpct[4], s_strcat[4]
  player-level       : p_base[n_player,11] (標準化後)
  ids                : hand_ids.npy(2M str), player_ids.npy, meta.json
硬規則：player_id / hand_id / table_id 只當 join key，不進特徵；不使用任何未來資訊。
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import polars as pl
from numba import njit

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
STREETS = ['preflop', 'flop', 'turn', 'river']
ACTIONS = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
BASE_COLS = ['b_vpip', 'b_pfr', 'b_agg_rate', 'b_fold_facing', 'b_call_facing',
             'b_raise_facing', 'b_sd_rate', 'b_post_check', 'b_contrib', 'b_allin', 'b_saw_flop']
T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


@njit(cache=True)
def _scan(aoff, nact, slot, sidx, isfold, isagg, y4, amount, sb_slot, bb_slot, sb_amt, bb_amt,
          foldbm, actedbm, lastaggr, nagg_before, nagg_street, nacted_street,
          actor_nprev, actor_prevtype, actor_prevagg, contrib6):
    """單趟掃描：算出每個動作『之前』可知的公開狀態（全部只看過去）。
    seqnll 用前三個遮罩；gbnll 另外用後六個因果計數欄。兩邊共用同一份定義。"""
    for h in range(aoff.shape[0]):
        s = aoff[h]
        e = s + nact[h]
        fbm = np.uint8(0)
        abm = np.uint8(0)
        last = np.int8(-1)
        cur = np.int8(-1)
        nagg_h = 0            # 本手在此動作之前的攻擊性動作數
        nagg_s = 0            # 本街之前的攻擊性動作數
        nact_s = 0            # 本街之前的動作數
        pn = np.zeros(6, np.int16)      # 每位座位本手先前的動作數
        pt = np.zeros(6, np.int8)       # 每位座位上一個動作的 4 類型別，4 = 尚未行動
        pa = np.zeros(6, np.int8)       # 每位座位本手先前是否攻擊過
        con = np.zeros(6, np.int64)     # 每位座位本手至此的累計投注（籌碼；不隨街重置）
        for k in range(6):
            pn[k] = 0
            pt[k] = 4
            pa[k] = 0
            con[k] = 0
        con[sb_slot[h]] = sb_amt[h]     # 小盲
        con[bb_slot[h]] = bb_amt[h]     # 大盲
        for i in range(s, e):
            if sidx[i] != cur:
                cur = sidx[i]
                abm = np.uint8(0)
                last = np.int8(-1)
                nagg_s = 0
                nact_s = 0
            sl = slot[i]
            foldbm[i] = fbm
            actedbm[i] = abm
            lastaggr[i] = last
            nagg_before[i] = nagg_h
            nagg_street[i] = nagg_s
            nacted_street[i] = nact_s
            actor_nprev[i] = pn[sl]
            actor_prevtype[i] = pt[sl]
            actor_prevagg[i] = pa[sl]
            for k in range(6):
                contrib6[i, k] = con[k]
            # ---- 更新（供後續動作使用）
            abm = np.uint8(abm | np.uint8(1 << sl))
            nact_s += 1
            pn[sl] += 1
            pt[sl] = y4[i]
            if isfold[i]:
                fbm = np.uint8(fbm | np.uint8(1 << sl))
            con[sl] += amount[i]
            if isagg[i]:
                last = np.int8(sl)
                nagg_h += 1
                nagg_s += 1
                pa[sl] = 1


@njit(inline='always')
def _pc(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True)
def _board_tex(brank, bsuit, win, sm, pr, hi, sw):
    """當街公牌質地。索引 [hand, sidx]，sidx=0（翻牌前）與資料不足者填 -1（下游轉成 null）。"""
    NV = (0, 3, 4, 5)
    for h in range(brank.shape[0]):
        for si in range(1, 4):
            n = NV[si]
            if brank[h, n - 1] < 0:
                sm[h, si] = -1; pr[h, si] = -1; hi[h, si] = -1; sw[h, si] = -1
                continue
            c0 = 0; c1 = 0; c2 = 0; c3 = 0
            rmask = np.int64(0); dup = 0; bhi = np.int64(-1)
            for j in range(n):
                r = np.int64(brank[h, j]); su = np.int64(bsuit[h, j])
                bit = np.int64(1) << r
                if rmask & bit:
                    dup = 1
                rmask |= bit
                if r > bhi:
                    bhi = r
                if su == 0: c0 += 1
                elif su == 1: c1 += 1
                elif su == 2: c2 += 1
                else: c3 += 1
            mx = c0
            if c1 > mx: mx = c1
            if c2 > mx: mx = c2
            if c3 > mx: mx = c3
            best = 0
            for w in range(10):
                c = _pc(rmask & win[w])
                if c > best: best = c
            sm[h, si] = mx; pr[h, si] = dup; hi[h, si] = bhi; sw[h, si] = best


@njit(cache=True)
def _hand_tex(hr, hs, brank, bsuit, win, sm, pb, oc, sw):
    """行動者底牌 ＋ 當街公牌的同一組質地量。索引 [seat_row, sidx]。"""
    NV = (0, 3, 4, 5)
    for i in range(hr.shape[0]):
        h = i // 6
        for si in range(1, 4):
            n = NV[si]
            if brank[h, n - 1] < 0:
                sm[i, si] = -1; pb[i, si] = -1; oc[i, si] = -1; sw[i, si] = -1
                continue
            c0 = 0; c1 = 0; c2 = 0; c3 = 0
            bmask = np.int64(0); bhi = np.int64(-1)
            for j in range(n):
                r = np.int64(brank[h, j]); su = np.int64(bsuit[h, j])
                bmask |= (np.int64(1) << r)
                if r > bhi: bhi = r
                if su == 0: c0 += 1
                elif su == 1: c1 += 1
                elif su == 2: c2 += 1
                else: c3 += 1
            rmask = bmask
            paired = 0; over = 0
            for k in range(2):
                r = np.int64(hr[i, k]); su = np.int64(hs[i, k])
                if (bmask >> r) & 1:
                    paired = 1
                if r > bhi:
                    over += 1
                rmask |= (np.int64(1) << r)
                if su == 0: c0 += 1
                elif su == 1: c1 += 1
                elif su == 2: c2 += 1
                else: c3 += 1
            mx = c0
            if c1 > mx: mx = c1
            if c2 > mx: mx = c2
            if c3 > mx: mx = c3
            best = 0
            for w in range(10):
                c = _pc(rmask & win[w])
                if c > best: best = c
            sm[i, si] = mx; pb[i, si] = paired; oc[i, si] = over; sw[i, si] = best


# 10 個 5 張順子窗（A 可當低）：A2345 再加 23456 .. TJQKA
STRAIGHT_WIN = np.array(
    [(1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)] + [((1 << 5) - 1) << w for w in range(9)],
    np.int64)


def log_edges(lo, hi, nbins):
    """nbins 個對數等距桶 -> 回傳 nbins-1 個內部邊界。"""
    e = np.exp(np.linspace(np.log(lo), np.log(hi), nbins + 1))
    return e[1:-1].astype(np.float64)


def lin_edges(lo, hi, nbins):
    e = np.linspace(lo, hi, nbins + 1)
    return e[1:-1].astype(np.float64)


RMAP = {c: i for i, c in enumerate(RANKS)}
SMAP = {c: i for i, c in enumerate(SUITS)}


def _card_cols(expr):
    """單張牌字串 expr -> (rank int8, suit int8)，空/None -> -1。"""
    r = expr.str.slice(0, 1).replace_strict(RMAP, default=-1, return_dtype=pl.Int8)
    s = expr.str.slice(1, 1).replace_strict(SMAP, default=-1, return_dtype=pl.Int8)
    return r, s


def parse_hole(df, col):
    r, s = _card_cols(pl.col(col))
    o = df.select(r.alias('r'), s.alias('s'))
    return o['r'].to_numpy().astype(np.int8).reshape(-1, 1), o['s'].to_numpy().astype(np.int8).reshape(-1, 1)


def parse_board(df, col, ncard=5):
    sp = pl.col(col).str.split(' ')
    exprs = []
    for j in range(ncard):
        r, s = _card_cols(sp.list.get(j, null_on_oob=True).fill_null(''))
        exprs += [r.alias(f'r{j}'), s.alias(f's{j}')]
    o = df.select(exprs)
    rk = np.stack([o[f'r{j}'].to_numpy().astype(np.int8) for j in range(ncard)], 1)
    st = np.stack([o[f's{j}'].to_numpy().astype(np.int8) for j in range(ncard)], 1)
    return rk, st


def collect_stream(lf):
    """polars 各版本的 streaming collect API 名稱不同，逐一退回。"""
    for kw in ({'engine': 'streaming'}, {'streaming': True}, {}):
        try:
            return lf.collect(**kw)
        except TypeError:
            continue
    return lf.collect()


def rd(path, columns, pick=None):
    """limit-hands 模式下用 streaming filter 只讀需要的手，記憶體才不會爆。"""
    if pick is None:
        return pl.read_parquet(path, columns=columns)
    return (pl.scan_parquet(path).select(columns)
              .filter(pl.col('hand_id').is_in(pick)).pipe(collect_stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/ubuntu/tpds/data')
    ap.add_argument('--out', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--limit-hands', type=int, default=0, help='>0 只取前 N 手（smoke test）')
    args = ap.parse_args()
    RAW = os.path.join(args.data, 'raw')
    PROC = os.path.join(args.data, 'processed')
    os.makedirs(args.out, exist_ok=True)
    meta = {'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'limit_hands': args.limit_hands}

    # ---------------------------------------------------------------- hands
    log('讀 hands.parquet')
    h = pl.read_parquet(os.path.join(RAW, 'hands.parquet'),
                        columns=['hand_id', 'table_id', 'phase', 'big_blind', 'small_blind',
                                 'button_seat', 'board_cards']).sort('hand_id')
    if args.limit_hands:
        h = h.head(args.limit_hands)
    N_H = h.height
    log(f'N_H = {N_H:,}')
    hand_ids = h['hand_id'].to_numpy().astype('U')
    # 交叉擬合半區：crc32 決定，跨進程穩定（不可用 python hash()）
    grp = np.array([zlib.crc32(x.encode()) & 1 for x in hand_ids], np.int8)
    bb = h['big_blind'].to_numpy()
    bb_levels = sorted(np.unique(bb).tolist())
    assert len(bb_levels) <= 4, bb_levels
    h_bb = np.searchsorted(np.array(bb_levels), bb).astype(np.int8)
    h_bb_raw = bb.astype(np.int64)
    btn = h['button_seat'].to_numpy().astype(np.int64)
    sb_slot = ((btn + 1) % 6).astype(np.int64)
    bb_slot = ((btn + 2) % 6).astype(np.int64)
    sb_amt = h['small_blind'].to_numpy().astype(np.int64)
    bb_amt = bb.astype(np.int64)
    h_phase = (h['phase'].to_numpy() == 'evaluation').astype(np.int8)
    tbl_uniq = np.sort(h['table_id'].unique().to_numpy().astype('U'))
    h_tbl = np.searchsorted(tbl_uniq, h['table_id'].to_numpy().astype('U')).astype(np.int32)
    board_rank, board_suit = parse_board(h, 'board_cards', 5)
    hidx = pl.DataFrame({'hand_id': hand_ids, 'hidx': np.arange(N_H, dtype=np.int32)})
    pick = pl.Series('hand_id', hand_ids) if args.limit_hands else None

    # ---------------------------------------------------------------- actions
    log('讀 actions.parquet + action_ctx.parquet')
    a = rd(os.path.join(RAW, 'actions.parquet'),
           ['hand_id', 'action_no', 'street', 'player_id', 'action',
            'amount', 'to_call', 'pot_before', 'stack_before', 'players_active'], pick)
    ac = rd(os.path.join(PROC, 'action_ctx.parquet'),
            ['hand_id', 'action_no', 'sidx', 'is_agg', 'is_call', 'is_fold',
             'amt_pot', 'to_call_bb', 'pot_before_bb'], pick)
    n_raw = a.height
    a = a.join(hidx, on='hand_id', how='inner').join(ac, on=['hand_id', 'action_no'], how='inner')
    assert a.height == n_raw, f'join 掉了列：{a.height} != {n_raw}（action_ctx 與 actions 對不齊）'
    a = a.sort(['hidx', 'action_no'])
    N_A = a.height
    log(f'N_A = {N_A:,}')
    if not args.limit_hands:
        assert N_H == 2_000_000 and N_A == 18_609_028, f'全量列數不符 N_H={N_H} N_A={N_A}'

    a_hand = a['hidx'].to_numpy().astype(np.int32)
    a_actno = a['action_no'].to_numpy().astype(np.int16)
    sidx = a['sidx'].to_numpy().astype(np.int8)
    # street 一致性
    st_map = {s: i for i, s in enumerate(STREETS)}
    assert (sidx == a['street'].replace_strict(st_map, return_dtype=pl.Int8).to_numpy()).all(), 'sidx 與 street 不一致'

    act_map = {s: i for i, s in enumerate(ACTIONS)}
    a_type = pl.Series(a['action']).replace_strict(act_map, return_dtype=pl.Int8).to_numpy().astype(np.int8)
    amount = a['amount'].to_numpy().astype(np.int64)
    to_call = a['to_call'].to_numpy().astype(np.int64)
    pot_before = a['pot_before'].to_numpy().astype(np.int64)
    stack_before = a['stack_before'].to_numpy().astype(np.int64)
    nactive = a['players_active'].to_numpy().astype(np.int64)
    is_agg = a['is_agg'].to_numpy()
    is_call = a['is_call'].to_numpy()
    is_fold = a['is_fold'].to_numpy()
    amt_pot = a['amt_pot'].to_numpy().astype(np.float64)
    # 核對 action_ctx 的 all_in 語義（bet/raise 或 all_in 且 amount>to_call 才算攻擊）
    ref_agg = np.isin(a_type, [3, 4]) | ((a_type == 5) & (amount > to_call))
    assert (ref_agg == is_agg).all(), 'is_agg 與重算不符'
    meta['allin_frac'] = float((a_type == 5).mean())
    meta['allin_is_agg_frac'] = float(is_agg[a_type == 5].mean()) if (a_type == 5).any() else 0.0

    # 每手動作數與位移（actions 已依 hidx, action_no 排序 -> 每手連續）
    cnt = np.bincount(a_hand, minlength=N_H).astype(np.int64)
    h_nact = cnt.astype(np.int16)
    h_aoff = np.zeros(N_H, np.int64)
    np.cumsum(cnt[:-1], out=h_aoff[1:])
    assert cnt.max() <= 64, f'每手動作數 {cnt.max()} 超過 model.MAXLEN=64'
    meta['max_actions_per_hand'] = int(cnt.max())
    meta['mean_actions_per_hand'] = float(cnt.mean())
    # action_no 必須是每手 0..n-1
    exp = np.arange(N_A, dtype=np.int64) - np.repeat(h_aoff, cnt)
    assert (a_actno.astype(np.int64) == exp).all(), 'action_no 不是每手 0..n-1'

    # ---------------------------------------------------------------- seats
    log('讀 seat_l0.parquet')
    SL_COLS = ['hand_id', 'player_id', 'seat_no', 'rel_pos', 'stack_bb', 'chen',
               'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo',
               'str_pct_1', 'str_cat_1', 'str_pct_2', 'str_cat_2', 'str_pct_3', 'str_cat_3',
               'hole_card_1', 'hole_card_2']
    sl = rd(os.path.join(PROC, 'seat_l0.parquet'), SL_COLS, pick)
    sl = sl.join(hidx, on='hand_id', how='inner').sort(['hidx', 'seat_no'])
    N_S = sl.height
    assert N_S == 6 * N_H, (N_S, N_H)
    seat_no = sl['seat_no'].to_numpy()
    assert (seat_no == np.tile(np.arange(6), N_H)).all(), 'seat_no 不是每手 0..5'
    dup = sl.group_by('hidx').agg(pl.col('player_id').n_unique().alias('u')).filter(pl.col('u') != 6).height
    assert dup == 0, f'{dup} 手的 6 個座位不是 6 位不同玩家'
    log(f'N_S = {N_S:,}')

    player_ids = np.sort(sl['player_id'].unique().to_numpy().astype('U'))
    pcode_map = pl.DataFrame({'player_id': player_ids, 'pcode': np.arange(len(player_ids), dtype=np.int32)})
    sl = sl.with_row_index('_rix').join(pcode_map, on='player_id', how='left').sort('_rix')
    assert sl['pcode'].null_count() == 0, '有座位的 player_id 不在 pcode 表'
    s_pcode = sl['pcode'].to_numpy().astype(np.int32)
    s_relpos = sl['rel_pos'].to_numpy().astype(np.int8)
    assert s_relpos.min() >= 0 and s_relpos.max() <= 5
    hr1, hs1 = parse_hole(sl, 'hole_card_1')
    hr2, hs2 = parse_hole(sl, 'hole_card_2')
    s_hole_rank = np.concatenate([hr1, hr2], 1)
    s_hole_suit = np.concatenate([hs1, hs2], 1)
    assert s_hole_rank.min() >= 0 and s_hole_suit.min() >= 0, '底牌解析失敗'
    s_chen = sl['chen'].to_numpy().astype(np.float32)
    s_stackbb = sl['stack_bb'].to_numpy().astype(np.float32)
    s_pfhi = sl['pf_hi'].to_numpy().astype(np.int8)
    s_pflo = sl['pf_lo'].to_numpy().astype(np.int8)
    s_pfpair = sl['pf_pair'].to_numpy().astype(np.int8)
    s_pfsuited = sl['pf_suited'].to_numpy().astype(np.int8)
    s_strpct = np.zeros((N_S, 4), np.float32)
    s_strcat = np.zeros((N_S, 4), np.int8)
    for k in (1, 2, 3):
        s_strpct[:, k] = np.nan_to_num(sl[f'str_pct_{k}'].to_numpy().astype(np.float32), nan=0.0)
        s_strcat[:, k] = np.nan_to_num(sl[f'str_cat_{k}'].to_numpy().astype(np.float32), nan=0.0).astype(np.int8)

    # 每個動作的行動者座位 slot（= seat_no）
    log('對應行動者 seat_no')
    seat_key = sl.select(['hidx', 'player_id', 'seat_no'])
    js = (a.select(['hidx', 'player_id']).with_row_index('_rix')
            .join(seat_key, on=['hidx', 'player_id'], how='left').sort('_rix'))
    assert js.height == N_A, f'行動者對座位的 join 產生了重複列 {js.height} != {N_A}'
    assert js['seat_no'].null_count() == 0, '有動作找不到對應座位'
    a_slot = js['seat_no'].to_numpy().astype(np.int8)

    # ---------------------------------------------------------------- 過去狀態遮罩
    # gbnll 的 4 類目標：fold 0 / check 1 / call 2 / 攻擊性(bet,raise,all_in) 3
    # 注意這是「依動作型別」分的（兩份規格的字面定義），與 action_ctx 修正過
    # all-in 語義的 is_agg/is_call 不同；T2 的 fold/call/agg 分組仍用 action_ctx。
    y4 = np.where(a_type == 0, 0, np.where(a_type == 1, 1, np.where(a_type == 2, 2, 3))).astype(np.int8)
    log('掃描每動作的過去狀態（遮罩＋因果計數）')
    a_foldbm = np.zeros(N_A, np.uint8)
    a_actedbm = np.zeros(N_A, np.uint8)
    a_lastaggr = np.zeros(N_A, np.int8)
    a_naggb = np.zeros(N_A, np.int16)
    a_naggs = np.zeros(N_A, np.int8)
    a_nacts = np.zeros(N_A, np.int8)
    a_apn = np.zeros(N_A, np.int8)
    a_apt = np.zeros(N_A, np.int8)
    a_apa = np.zeros(N_A, np.int8)
    a_contrib6 = np.zeros((N_A, 6), np.int32)
    _scan(h_aoff, cnt.astype(np.int64), a_slot, sidx, is_fold, is_agg, y4, amount,
          sb_slot, bb_slot, sb_amt, bb_amt,
          a_foldbm, a_actedbm, a_lastaggr, a_naggb, a_naggs, a_nacts, a_apn, a_apt, a_apa, a_contrib6)

    # ---------------------------------------------------------------- 公牌／底牌質地
    log('計算公牌與底牌質地（gbnll 的 8 個新欄）')
    bt_sm = np.zeros((N_H, 4), np.int8); bt_pr = np.zeros((N_H, 4), np.int8)
    bt_hi = np.zeros((N_H, 4), np.int8); bt_sw = np.zeros((N_H, 4), np.int8)
    _board_tex(board_rank, board_suit, STRAIGHT_WIN, bt_sm, bt_pr, bt_hi, bt_sw)
    ht_sm = np.zeros((N_S, 4), np.int8); ht_pb = np.zeros((N_S, 4), np.int8)
    ht_oc = np.zeros((N_S, 4), np.int8); ht_sw = np.zeros((N_S, 4), np.int8)
    _hand_tex(s_hole_rank, s_hole_suit, board_rank, board_suit, STRAIGHT_WIN,
              ht_sm, ht_pb, ht_oc, ht_sw)
    for arr in (bt_sm, bt_pr, bt_hi, bt_sw, ht_sm, ht_pb, ht_oc, ht_sw):
        arr[:, 0] = -1        # 翻牌前無公牌 -> 下游轉成 null

    # ---------------------------------------------------------------- gbnll 的原始連續欄
    bbv = h_bb_raw[a_hand].astype(np.float64)
    a_tocall_bb = (to_call / bbv).astype(np.float32)
    a_pot_bb = (pot_before / bbv).astype(np.float32)
    a_stackbf_bb = (stack_before / bbv).astype(np.float32)
    a_invested_bb = (s_stackbb[a_hand * 6 + a_slot] - a_stackbf_bb).astype(np.float32)
    # 交叉驗證：由動作序列累加出的行動者投注額，必須等於 SPEC 的 起始籌碼 − 當下籌碼
    _inv2 = (a_contrib6[np.arange(N_A), a_slot] / bbv).astype(np.float32)
    _d = np.abs(_inv2 - a_invested_bb)
    meta['invested_bb_max_abs_diff'] = float(_d.max())
    meta['invested_bb_mismatch_gt_1e3'] = int((_d > 1e-3).sum())
    a_amtpot = amt_pot.astype(np.float32)

    # 公牌可見性檢查：第 k 街的動作必須有至少 NVIS[k] 張公牌
    nvis_need = np.array([0, 3, 4, 5])[sidx]
    n_board_avail = (board_rank >= 0).sum(1)[a_hand]
    assert (n_board_avail >= nvis_need).all(), '有動作所處的街缺少應已公開的公牌'
    meta['board_len_counts'] = {int(k): int(v) for k, v in
                                zip(*np.unique((board_rank >= 0).sum(1), return_counts=True))}

    # ---------------------------------------------------------------- 分桶
    log('計算分桶邊界')
    pos = amt_pot > 0
    amt_lo = float(np.quantile(amt_pot[pos], 0.005)) if pos.any() else 0.01
    amt_hi = float(np.quantile(amt_pot[pos], 0.995)) if pos.any() else 3.0
    amt_e = log_edges(max(amt_lo, 1e-3), max(amt_hi, amt_lo * 2), 12)
    a_amtb = np.zeros(N_A, np.int8)
    a_amtb[pos] = (1 + np.digitize(amt_pot[pos], amt_e)).astype(np.int8)

    potv = np.log1p(np.maximum(a['pot_before_bb'].to_numpy().astype(np.float64), 0.0))
    pot_lo, pot_hi = float(np.quantile(potv, 0.001)), float(np.quantile(potv, 0.999))
    pot_e = lin_edges(pot_lo, max(pot_hi, pot_lo + 1e-3), 12)
    a_potb = np.digitize(potv, pot_e).astype(np.int8)

    tcbb = a['to_call_bb'].to_numpy().astype(np.float64)
    potbb = a['pot_before_bb'].to_numpy().astype(np.float64)
    ratio = np.where(tcbb > 0, tcbb / np.maximum(tcbb + potbb, 1e-9), 0.0)
    tc_e = lin_edges(0.0, 1.0, 9)
    a_tcb = np.zeros(N_A, np.int8)
    m = ratio > 0
    a_tcb[m] = (1 + np.digitize(ratio[m], tc_e)).astype(np.int8)

    spr = np.where(pot_before > 0, stack_before / np.maximum(pot_before, 1), 0.0)
    sm = spr > 0
    spr_lo = float(np.quantile(spr[sm], 0.005)) if sm.any() else 0.1
    spr_hi = float(np.quantile(spr[sm], 0.995)) if sm.any() else 100.0
    spr_e = log_edges(max(spr_lo, 1e-3), max(spr_hi, spr_lo * 2), 9)
    a_sprb = np.zeros(N_A, np.int8)
    a_sprb[sm] = (1 + np.digitize(spr[sm], spr_e)).astype(np.int8)

    a_nactive = np.clip(nactive, 0, 7).astype(np.int8)
    meta['buckets'] = {'amt_pot': amt_e.tolist(), 'log1p_pot_bb': pot_e.tolist(),
                       'to_call_ratio': tc_e.tolist(), 'spr': spr_e.tolist(),
                       'amt_range': [amt_lo, amt_hi], 'spr_range': [spr_lo, spr_hi]}
    meta['players_active_range'] = [int(nactive.min()), int(nactive.max())]

    # ---------------------------------------------------------------- baselines
    log('讀 player_baselines.parquet')
    pb = pl.read_parquet(os.path.join(PROC, 'player_baselines.parquet')).filter(pl.col('bphase') == 'all')
    pb = pcode_map.join(pb.select(['player_id'] + BASE_COLS), on='player_id', how='left').sort('pcode')
    assert pb.height == len(player_ids)
    p_base = pb.select(BASE_COLS).to_numpy().astype(np.float64)
    miss = np.isnan(p_base).any(1).sum()
    p_base = np.nan_to_num(p_base, nan=0.0)
    mu, sd = p_base.mean(0), p_base.std(0) + 1e-6
    p_base = ((p_base - mu) / sd).astype(np.float32)
    meta['baseline_missing_players'] = int(miss)
    meta['baseline_mu'] = mu.tolist()
    meta['baseline_sd'] = sd.tolist()

    # ---------------------------------------------------------------- 交叉擬合檢查
    tb = np.stack([np.bincount(h_tbl, weights=(grp == g).astype(np.float64), minlength=h_tbl.max() + 1) for g in (0, 1)])
    frac = tb[0] / np.maximum(tb.sum(0), 1)
    ok = tb.sum(0) >= 20
    meta['crossfit'] = {'n_tables': int(ok.sum()),
                        'table_g0_frac_min': float(frac[ok].min()) if ok.any() else None,
                        'table_g0_frac_max': float(frac[ok].max()) if ok.any() else None,
                        'hand_g0': int((grp == 0).sum()), 'hand_g1': int((grp == 1).sum())}
    pg = np.zeros((len(player_ids), 2), np.int64)
    np.add.at(pg, (s_pcode, np.repeat(grp, 6)), 1)
    meta['crossfit']['players_missing_a_half'] = int(((pg == 0).any(1)).sum())
    meta['crossfit']['n_players'] = int(len(player_ids))

    # ---------------------------------------------------------------- 存檔
    log('寫出陣列')
    O = args.out
    def sv(name, arr):
        np.save(os.path.join(O, name + '.npy'), arr)
    sv('h_bb', h_bb); sv('h_grp', grp); sv('h_phase', h_phase); sv('h_nact', h_nact)
    sv('h_aoff', h_aoff); sv('h_board_rank', board_rank); sv('h_board_suit', board_suit); sv('h_tbl', h_tbl)
    sv('a_hand', a_hand); sv('a_actno', a_actno); sv('a_slot', a_slot); sv('a_sidx', sidx)
    sv('a_type', a_type); sv('a_amtb', a_amtb); sv('a_potb', a_potb); sv('a_tcb', a_tcb)
    sv('a_sprb', a_sprb); sv('a_nactive', a_nactive); sv('a_foldbm', a_foldbm)
    sv('a_actedbm', a_actedbm); sv('a_lastaggr', a_lastaggr)
    sv('a_isagg', is_agg.astype(np.bool_)); sv('a_iscall', is_call.astype(np.bool_)); sv('a_isfold', is_fold.astype(np.bool_))
    sv('s_hole_rank', s_hole_rank); sv('s_hole_suit', s_hole_suit); sv('s_relpos', s_relpos)
    sv('s_pcode', s_pcode); sv('s_chen', s_chen); sv('s_stackbb', s_stackbb)
    sv('s_pfhi', s_pfhi); sv('s_pflo', s_pflo); sv('s_pfpair', s_pfpair); sv('s_pfsuited', s_pfsuited)
    sv('s_strpct', s_strpct); sv('s_strcat', s_strcat)
    sv('p_base', p_base)
    # ---- 共用決策狀態表（gbnll 用；seqnll 的桶欄亦由同一份原始值算出）
    sv('a_y4', y4); sv('a_naggb', a_naggb); sv('a_naggs', a_naggs); sv('a_nacts', a_nacts)
    sv('a_apn', a_apn); sv('a_apt', a_apt); sv('a_apa', a_apa)
    sv('a_tocall_bb', a_tocall_bb); sv('a_pot_bb', a_pot_bb)
    sv('a_stackbf_bb', a_stackbf_bb); sv('a_invested_bb', a_invested_bb); sv('a_amtpot', a_amtpot)
    sv('bt_sm', bt_sm); sv('bt_pr', bt_pr); sv('bt_hi', bt_hi); sv('bt_sw', bt_sw)
    sv('ht_sm', ht_sm); sv('ht_pb', ht_pb); sv('ht_oc', ht_oc); sv('ht_sw', ht_sw)
    sv('h_bb_raw', h_bb_raw); sv('a_contrib6', a_contrib6)
    np.save(os.path.join(O, 'hand_ids.npy'), hand_ids)
    np.save(os.path.join(O, 'player_ids.npy'), player_ids)
    meta['N_H'], meta['N_A'], meta['N_S'] = int(N_H), int(N_A), int(N_S)
    meta['action_type_counts'] = {ACTIONS[i]: int((a_type == i).sum()) for i in range(6)}
    p = np.array([meta['action_type_counts'][k] for k in ACTIONS], np.float64)
    p /= p.sum()
    meta['marginal_type_entropy_nats'] = float(-(p * np.log(p)).sum())
    with open(os.path.join(O, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('meta:', json.dumps({k: v for k, v in meta.items() if k not in ('buckets', 'baseline_mu', 'baseline_sd')}, ensure_ascii=False))
    log('完成')


if __name__ == '__main__':
    main()
