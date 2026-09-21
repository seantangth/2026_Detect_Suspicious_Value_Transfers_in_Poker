#!/usr/bin/env python3
"""T1／T2／T3 的共用寫檔器：seqnll 與 gbnll 走同一份實作，schema 保證一致。"""
import json, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

T2_COLS = ['n_act', 'nll_N_sum', 'nll_N_max', 'nll_N_pre', 'nll_N_flop', 'nll_N_turn', 'nll_N_river',
           'nll_N_fold_max', 'nll_N_call_max', 'nll_N_agg_max', 'nll_size_N_sum']


def nanify(a):
    return np.where(np.isneginf(a), np.nan, a)


def _dic(idx, d):
    """每個 row group 只帶這一塊用到的那段字典（hand_id 有 2,000,000 個相異值）。"""
    if len(idx) == 0:
        return pa.DictionaryArray.from_arrays(pa.array(idx.astype(np.int32)), d)
    lo, hi = int(idx.min()), int(idx.max()) + 1
    if hi - lo < len(d):
        return pa.DictionaryArray.from_arrays(pa.array((idx - lo).astype(np.int32)), d.slice(lo, hi - lo))
    return pa.DictionaryArray.from_arrays(pa.array(idx.astype(np.int32)), d)


def write_all(out, prefix, tables, T1, T2, T3sum, T3max, T3gmax, log=print, chunk_hands=200000):
    """T1/T2/T3 一次寫完，並回傳摘要（含品質檢查 5 需要的平均 gain）。"""
    hand_ids = np.load(os.path.join(tables, 'hand_ids.npy'), allow_pickle=True)
    player_ids = np.load(os.path.join(tables, 'player_ids.npy'), allow_pickle=True)
    a_hand = np.load(os.path.join(tables, 'a_hand.npy')).astype(np.int64)
    a_actno = np.load(os.path.join(tables, 'a_actno.npy'))
    a_slot = np.load(os.path.join(tables, 'a_slot.npy')).astype(np.int64)
    a_sidx = np.load(os.path.join(tables, 'a_sidx.npy'))
    s_pcode = np.load(os.path.join(tables, 's_pcode.npy'))
    N_H, N_A, N_S = len(hand_ids), len(a_hand), len(s_pcode)
    pa_hand = pa.array([str(x) for x in hand_ids])
    pa_play = pa.array([str(x) for x in player_ids])
    zst = dict(compression='zstd', compression_level=6)
    os.makedirs(out, exist_ok=True)

    log(f'寫 {prefix}_action')
    pid_act = s_pcode[a_hand * 6 + a_slot]
    sch = pa.schema([('hand_id', pa.dictionary(pa.int32(), pa.string())), ('action_no', pa.int32()),
                     ('player_id', pa.dictionary(pa.int32(), pa.string())), ('sidx', pa.int8()),
                     ('nll_type_N', pa.float32()), ('nll_size_N', pa.float32()),
                     ('p_taken_N', pa.float32()), ('entropy_N', pa.float32())])
    with pq.ParquetWriter(os.path.join(out, f'{prefix}_action.parquet'), sch, **zst) as w:
        for s in range(0, N_A, 4_000_000):
            e = min(s + 4_000_000, N_A)
            w.write_table(pa.Table.from_arrays(
                [_dic(a_hand[s:e], pa_hand), pa.array(a_actno[s:e].astype(np.int32)),
                 _dic(pid_act[s:e], pa_play), pa.array(a_sidx[s:e]),
                 pa.array(T1['nll_type_N'][s:e]), pa.array(T1['nll_size_N'][s:e]),
                 pa.array(T1['p_taken_N'][s:e]), pa.array(T1['entropy_N'][s:e])], schema=sch))
    del pid_act

    log(f'寫 {prefix}_player')
    s_hidx = np.repeat(np.arange(N_H, dtype=np.int64), 6)
    sch = pa.schema([('hand_id', pa.dictionary(pa.int32(), pa.string())),
                     ('player_id', pa.dictionary(pa.int32(), pa.string())), ('n_act', pa.int32())]
                    + [(c, pa.float32()) for c in T2_COLS[1:]])
    with pq.ParquetWriter(os.path.join(out, f'{prefix}_player.parquet'), sch, **zst) as w:
        for s in range(0, N_S, 4_000_000):
            e = min(s + 4_000_000, N_S)
            w.write_table(pa.Table.from_arrays(
                [_dic(s_hidx[s:e], pa_hand), _dic(s_pcode[s:e], pa_play), pa.array(T2['n_act'][s:e])]
                + [pa.array(T2[c][s:e]) for c in T2_COLS[1:]], schema=sch))

    log(f'寫 {prefix}_pair')
    XS = np.repeat(np.arange(6), 5).astype(np.int64)
    YS = np.array([y for x in range(6) for y in range(6) if y != x], np.int64)
    sch = pa.schema([('hand_id', pa.dictionary(pa.int32(), pa.string())),
                     ('player_id', pa.dictionary(pa.int32(), pa.string())),
                     ('other_id', pa.dictionary(pa.int32(), pa.string())),
                     ('nll_I_sum', pa.float32()), ('nll_I_max', pa.float32()),
                     ('gain_sum', pa.float32()), ('gain_max', pa.float32())])
    gs_acc, grow, gm_acc, gm_n = 0.0, 0, 0.0, 0
    with pq.ParquetWriter(os.path.join(out, f'{prefix}_pair.parquet'), sch, **zst) as w:
        for h0 in range(0, N_H, chunk_hands):
            h1 = min(h0 + chunk_hands, N_H)
            nh = h1 - h0
            hh = np.repeat(np.arange(h0, h1, dtype=np.int64), 30)
            xs = np.tile(XS, nh); ys = np.tile(YS, nh)
            srow = hh * 6 + xs
            idx = srow * 6 + ys
            g = (T2['nll_N_sum'][srow] - T3sum[idx]).astype(np.float32)
            w.write_table(pa.Table.from_arrays(
                [_dic(hh, pa_hand), _dic(s_pcode[srow], pa_play), _dic(s_pcode[hh * 6 + ys], pa_play),
                 pa.array(T3sum[idx]), pa.array(T3max[idx]), pa.array(g), pa.array(T3gmax[idx])],
                schema=sch))
            gs_acc += float(np.nansum(g)); grow += int(np.isfinite(g).sum())
            fm = np.isfinite(T3gmax[idx])
            gm_acc += float(T3gmax[idx][fm].sum()); gm_n += int(fm.sum())
            if (h0 // chunk_hands) % 2 == 0:
                log(f'  {prefix}_pair {h1:,}/{N_H:,}')
    summ = {'prefix': prefix, 'N_H': int(N_H), 'N_A': int(N_A), 'N_S': int(N_S),
            'T1_rows': int(N_A), 'T2_rows': int(N_S), 'T3_rows': int(N_H * 30),
            'mean_gain_sum': gs_acc / max(grow, 1), 'mean_gain_max': gm_acc / max(gm_n, 1),
            'seats_with_actions': int((T2['n_act'] > 0).sum()),
            'mean_nll_N_per_action': float(T2['nll_N_sum'].sum() / max(N_A, 1))}
    json.dump(summ, open(os.path.join(out, f'{prefix}_infer_summary.json'), 'w'), indent=2)
    return summ
