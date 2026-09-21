#!/usr/bin/env python3
"""純合成的小資料集，用來在沒有 GPU 的機器上把 pairpol 整條鏈跑通（程式自測，不是實驗）。

產生 <root>/{tables, data/raw, data/processed, data/extra}，結構與真實資料一致：
每桌固定一組玩家、每手發 6 個不同座位、決策狀態由 build_tables._scan 真的掃出來，
所以 pairpol_accept.py 的因果性檢查在這裡也是真的在檢查東西。
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tables as BT       # noqa: E402

NVIS = np.array([0, 3, 4, 5])
ACTIONS = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tables-n', type=int, default=4)
    ap.add_argument('--players-per-table', type=int, default=12)
    ap.add_argument('--hands-per-table', type=int, default=180)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    r = np.random.default_rng(a.seed)
    TB, PT, HT = a.tables_n, a.players_per_table, a.hands_per_table
    N_H = TB * HT
    npl = TB * PT
    T = os.path.join(a.root, 'tables')
    RAW = os.path.join(a.root, 'data', 'raw')
    PROC = os.path.join(a.root, 'data', 'processed')
    EX = os.path.join(a.root, 'data', 'extra')
    for d in (T, RAW, PROC, EX):
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------------- 手
    hand_ids = np.array([f'H{i:013X}' for i in range(N_H)], dtype='U')
    order = np.argsort(hand_ids, kind='stable')          # build_tables 會 sort('hand_id')
    hand_ids = hand_ids[order]
    h_tbl = np.repeat(np.arange(TB), HT)[order].astype(np.int32)
    h_phase = (r.random(N_H) < 0.4).astype(np.int8)
    tbl_ids = np.array([f'T{i:08X}' for i in range(TB)], dtype='U')
    started = np.arange(N_H) * 60
    nact = r.integers(4, 22, N_H).astype(np.int64)
    N_A = int(nact.sum())
    aoff = np.zeros(N_H, np.int64)
    np.cumsum(nact[:-1], out=aoff[1:])
    a_hand = np.repeat(np.arange(N_H, dtype=np.int64), nact)
    sidx = np.zeros(N_A, np.int8)
    for h in range(N_H):
        s, n = aoff[h], nact[h]
        v = np.zeros(n, np.int8)
        for c in np.sort(r.integers(0, n, r.integers(0, 4))):
            v[c:] += 1
        sidx[s:s + n] = np.minimum(v, 3)
    maxs = np.array([sidx[aoff[h]:aoff[h] + nact[h]].max() for h in range(N_H)])
    nb = NVIS[maxs]
    board_rank = np.full((N_H, 5), -1, np.int8)
    board_suit = np.full((N_H, 5), -1, np.int8)
    for h in range(N_H):
        k = int(nb[h])
        if k:
            board_rank[h, :k] = r.integers(0, 13, k)
            board_suit[h, :k] = r.integers(0, 4, k)

    # ---------------------------------------------------------------- 座位
    player_ids = np.array([f'U{i:012X}' for i in range(npl)], dtype='U')
    seat_pc = np.empty((N_H, 6), np.int32)
    for h in range(N_H):
        pool = np.arange(h_tbl[h] * PT, (h_tbl[h] + 1) * PT)
        seat_pc[h] = r.choice(pool, 6, replace=False)
    s_pcode = seat_pc.reshape(-1).astype(np.int32)

    # ---------------------------------------------------------------- 動作與因果掃描
    a_slot = np.empty(N_A, np.int8)
    for h in range(N_H):
        a_slot[aoff[h]:aoff[h] + nact[h]] = r.integers(0, 6, nact[h])
    a_type = r.integers(0, 6, N_A).astype(np.int8)
    y4 = np.where(a_type == 0, 0, np.where(a_type == 1, 1, np.where(a_type == 2, 2, 3))).astype(np.int8)
    isfold = (a_type == 0)
    isagg = (a_type >= 3)
    amount = np.where(isagg, r.integers(1, 400, N_A), 0).astype(np.int64)
    btn = r.integers(0, 6, N_H).astype(np.int64)
    sb_slot = ((btn + 1) % 6).astype(np.int64)
    bb_slot = ((btn + 2) % 6).astype(np.int64)
    sb_amt = np.full(N_H, 1, np.int64)
    bb_amt = np.full(N_H, 2, np.int64)
    o = [np.zeros(N_A, np.uint8), np.zeros(N_A, np.uint8), np.zeros(N_A, np.int8),
         np.zeros(N_A, np.int16), np.zeros(N_A, np.int8), np.zeros(N_A, np.int8),
         np.zeros(N_A, np.int16), np.zeros(N_A, np.int8), np.zeros(N_A, np.int8),
         np.zeros((N_A, 6), np.int64)]
    BT._scan(aoff, nact, a_slot, sidx, isfold, isagg, y4, amount,
             sb_slot, bb_slot, sb_amt, bb_amt, *o)
    (a_foldbm, a_actedbm, a_lastaggr, a_naggb, a_naggs, a_nacts,
     a_apn, a_apt, a_apa, a_contrib6) = o

    # ---------------------------------------------------------------- 存 tables/
    sv = lambda n, x: np.save(os.path.join(T, n + '.npy'), x)
    N_S = 6 * N_H
    sv('h_bb', r.integers(0, 3, N_H).astype(np.int8))
    sv('h_grp', r.integers(0, 2, N_H).astype(np.int8))
    sv('h_phase', h_phase); sv('h_nact', nact.astype(np.int16)); sv('h_aoff', aoff)
    sv('h_board_rank', board_rank); sv('h_board_suit', board_suit); sv('h_tbl', h_tbl)
    sv('h_bb_raw', np.full(N_H, 2, np.int64))
    sv('a_hand', a_hand.astype(np.int32))
    sv('a_actno', (np.arange(N_A) - np.repeat(aoff, nact)).astype(np.int16))
    sv('a_slot', a_slot); sv('a_sidx', sidx); sv('a_type', a_type); sv('a_y4', y4)
    sv('a_amtb', np.where(isagg, r.integers(1, 13, N_A), 0).astype(np.int8))
    sv('a_potb', r.integers(0, 12, N_A).astype(np.int8))
    sv('a_tcb', r.integers(0, 10, N_A).astype(np.int8))
    sv('a_sprb', r.integers(0, 10, N_A).astype(np.int8))
    sv('a_nactive', r.integers(2, 7, N_A).astype(np.int8))
    sv('a_foldbm', a_foldbm); sv('a_actedbm', a_actedbm); sv('a_lastaggr', a_lastaggr)
    sv('a_naggb', a_naggb); sv('a_naggs', a_naggs); sv('a_nacts', a_nacts)
    sv('a_apn', a_apn); sv('a_apt', a_apt); sv('a_apa', a_apa); sv('a_contrib6', a_contrib6)
    sv('a_isagg', isagg); sv('a_iscall', (a_type == 2)); sv('a_isfold', isfold)
    sv('a_tocall_bb', np.abs(r.normal(3, 3, N_A)).astype(np.float32))
    sv('a_pot_bb', np.abs(r.normal(20, 15, N_A)).astype(np.float32))
    sv('a_stackbf_bb', np.abs(r.normal(100, 40, N_A)).astype(np.float32))
    sv('a_invested_bb', np.abs(r.normal(5, 5, N_A)).astype(np.float32))
    sv('a_amtpot', np.abs(r.normal(0.6, 0.4, N_A)).astype(np.float32))
    sv('s_hole_rank', r.integers(0, 13, (N_S, 2)).astype(np.int8))
    sv('s_hole_suit', r.integers(0, 4, (N_S, 2)).astype(np.int8))
    sv('s_relpos', np.tile(np.arange(6), N_H).astype(np.int8))
    sv('s_pcode', s_pcode)
    sv('s_chen', r.normal(5, 3, N_S).astype(np.float32))
    sv('s_stackbb', np.abs(r.normal(100, 30, N_S)).astype(np.float32))
    sv('s_pfhi', r.integers(2, 15, N_S).astype(np.int8))
    sv('s_pflo', r.integers(2, 15, N_S).astype(np.int8))
    sv('s_pfpair', r.integers(0, 2, N_S).astype(np.int8))
    sv('s_pfsuited', r.integers(0, 2, N_S).astype(np.int8))
    sp = r.random((N_S, 4)).astype(np.float32); sp[:, 0] = 0
    sv('s_strpct', sp)
    sc = r.integers(1, 10, (N_S, 4)).astype(np.int8); sc[:, 0] = 0
    sv('s_strcat', sc)
    sv('p_base', r.normal(0, 1, (npl, 11)).astype(np.float32))
    for nm, sh in (('bt_sm', (N_H, 4)), ('bt_pr', (N_H, 4)), ('bt_hi', (N_H, 4)), ('bt_sw', (N_H, 4)),
                   ('ht_sm', (N_S, 4)), ('ht_pb', (N_S, 4)), ('ht_oc', (N_S, 4)), ('ht_sw', (N_S, 4))):
        v = r.integers(0, 5, sh).astype(np.float32)
        v[:, 0] = -1                       # 翻牌前無公牌 -> 下游轉 NaN
        sv(nm, v)
    np.save(os.path.join(T, 'hand_ids.npy'), hand_ids)
    np.save(os.path.join(T, 'player_ids.npy'), player_ids)
    json.dump({'N_H': int(N_H), 'N_A': int(N_A), 'N_S': int(N_S), 'synthetic': True},
              open(os.path.join(T, 'meta.json'), 'w'))

    # ---------------------------------------------------------------- raw / processed
    pl.DataFrame({
        'hand_id': hand_ids, 'table_id': tbl_ids[h_tbl],
        'phase': np.array(['development', 'evaluation'])[h_phase],
        'started_at': started, 'button_seat': btn,
        'small_blind': sb_amt, 'big_blind': bb_amt,
        'board_cards': [' '.join('23456789TJQKA'[board_rank[h, k]] + 'cdhs'[board_suit[h, k]]
                                 for k in range(int(nb[h]))) for h in range(N_H)],
    }).write_parquet(os.path.join(RAW, 'hands.parquet'))
    pl.DataFrame({
        'hand_id': hand_ids[a_hand],
        'action_no': (np.arange(N_A) - np.repeat(aoff, nact)).astype(np.int64),
        'player_id': player_ids[seat_pc[a_hand, a_slot.astype(np.int64)]],
        'action': np.array(ACTIONS)[a_type], 'amount': amount,
    }).write_parquet(os.path.join(RAW, 'actions.parquet'))

    # 候選對：每桌所有無序對（a<b 字串序），shared＝該期兩人都被發牌的手數
    rows = {'development': [], 'evaluation': []}
    for t in range(TB):
        pool = np.arange(t * PT, (t + 1) * PT)
        for ph_i, ph in enumerate(('development', 'evaluation')):
            hs = np.where((h_tbl == t) & (h_phase == ph_i))[0]
            if len(hs) == 0:
                continue
            present = np.zeros((npl, len(hs)), bool)
            for sl in range(6):
                present[seat_pc[hs, sl], np.arange(len(hs))] = True
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    x, y = pool[i], pool[j]
                    n = int((present[x] & present[y]).sum())
                    if n == 0:
                        continue
                    aa, bb = sorted([player_ids[x], player_ids[y]])
                    rows[ph].append((f'P{t:03d}{i:03d}{j:03d}', aa, bb, n, tbl_ids[t]))
    cd = pl.DataFrame(rows['development'], schema=['pair_id', 'a', 'b', 'shared', 'table_id'],
                      orient='row').with_columns(pl.col('shared').cast(pl.Int32))
    lab = np.where(np.arange(cd.height) % 7 == 0, 1, np.where(np.arange(cd.height) % 7 == 1, 0, -1))
    cd = cd.with_columns(pl.Series('label', lab).cast(pl.Int8),
                         pl.lit('unknown').alias('behavior_family'),
                         pl.lit('development').alias('phase'))
    cd.write_parquet(os.path.join(PROC, 'cand_pairs_development.parquet'))
    ce = pl.DataFrame(rows['evaluation'], schema=['pair_id', 'a', 'b', 'shared', 'table_id'],
                      orient='row')
    ce.select('pair_id', pl.col('a').alias('player_1'), pl.col('b').alias('player_2'),
              pl.col('shared').alias('shared_hands')).write_csv(
        os.path.join(RAW, 'evaluation_pairs.csv'))
    dl = cd.filter(pl.col('label') >= 0).select(
        'pair_id', pl.col('a').alias('player_1'), pl.col('b').alias('player_2'),
        pl.col('label').cast(pl.Int64),
        pl.when(pl.col('label') == 1).then(pl.lit('confirmed_target'))
          .otherwise(pl.lit('confirmed_non_target')).alias('label_status'),
        pl.lit('none').alias('behavior_family'))
    dl.write_csv(os.path.join(RAW, 'development_labels.csv'))
    n = cd.height
    pl.DataFrame({'pair_id': cd['pair_id'],
                  'pc_nll_sum_tg_mean': r.normal(0, 1, n) + (lab == 1) * 1.5,
                  'pc_nll_sum_tg_min': r.normal(0, 1, n),
                  'pc_nll_sum_dg_mean': r.normal(0, 1, n)}).write_parquet(
        os.path.join(EX, 'presence_contrast_development.parquet'))
    pl.DataFrame({'pair_id': cd['pair_id'],
                  'oof': r.random(n) + (lab == 1) * 0.5}).write_parquet(
        os.path.join(EX, 'pair_oof_v5x_head_s1.parquet'))
    print(f'合成完成 N_H={N_H} N_A={N_A} 玩家={npl} 桌={TB} '
          f'dev對={cd.height} eval對={ce.height} -> {a.root}')


if __name__ == '__main__':
    main()
