#!/usr/bin/env python3
"""產生一組合成的小表，用來在沒有 GPU 的機器上把整條鏈跑通（純程式自測，不是實驗）。"""
import argparse, json, os
import numpy as np

NVIS = np.array([0, 3, 4, 5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--hands', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = np.random.default_rng(a.seed)
    N_H = a.hands
    nact = r.integers(1, 21, N_H)
    N_A = int(nact.sum())
    aoff = np.zeros(N_H, np.int64); np.cumsum(nact[:-1], out=aoff[1:])
    a_hand = np.repeat(np.arange(N_H, dtype=np.int32), nact)
    # 每手的街：非遞減，從 0 開始
    sidx = np.zeros(N_A, np.int8)
    for h in range(N_H):
        s, n = aoff[h], nact[h]
        top = r.integers(0, 4)
        cuts = np.sort(r.integers(0, n, top))
        v = np.zeros(n, np.int8)
        for c in cuts:
            v[c:] += 1
        sidx[s:s + n] = np.minimum(v, 3)
    maxs = np.zeros(N_H, np.int64)
    for h in range(N_H):
        maxs[h] = sidx[aoff[h]:aoff[h] + nact[h]].max()
    nb = NVIS[maxs]                                    # 至少要有這麼多張公牌
    board_rank = np.full((N_H, 5), -1, np.int8)
    board_suit = np.full((N_H, 5), -1, np.int8)
    for h in range(N_H):
        k = int(nb[h])
        if k:
            board_rank[h, :k] = r.integers(0, 13, k)
            board_suit[h, :k] = r.integers(0, 4, k)
    sv = lambda n, x: np.save(os.path.join(a.out, n + '.npy'), x)
    sv('h_bb', r.integers(0, 3, N_H).astype(np.int8))
    sv('h_grp', r.integers(0, 2, N_H).astype(np.int8))
    sv('h_phase', r.integers(0, 2, N_H).astype(np.int8))
    sv('h_nact', nact.astype(np.int16)); sv('h_aoff', aoff)
    sv('h_board_rank', board_rank); sv('h_board_suit', board_suit)
    sv('h_tbl', (np.arange(N_H) // 20).astype(np.int32))
    sv('a_hand', a_hand); sv('a_actno', (np.arange(N_A) - np.repeat(aoff, nact)).astype(np.int16))
    sv('a_slot', r.integers(0, 6, N_A).astype(np.int8)); sv('a_sidx', sidx)
    atype = r.integers(0, 6, N_A).astype(np.int8)
    sv('a_type', atype)
    sv('a_amtb', np.where(atype >= 3, r.integers(1, 13, N_A), 0).astype(np.int8))
    sv('a_potb', r.integers(0, 12, N_A).astype(np.int8))
    sv('a_tcb', r.integers(0, 10, N_A).astype(np.int8))
    sv('a_sprb', r.integers(0, 10, N_A).astype(np.int8))
    sv('a_nactive', r.integers(2, 7, N_A).astype(np.int8))
    sv('a_foldbm', r.integers(0, 64, N_A).astype(np.uint8))
    sv('a_actedbm', r.integers(0, 64, N_A).astype(np.uint8))
    sv('a_lastaggr', r.integers(-1, 6, N_A).astype(np.int8))
    sv('a_isagg', (atype >= 3)); sv('a_iscall', (atype == 2)); sv('a_isfold', (atype == 0))
    N_S = 6 * N_H
    sv('s_hole_rank', r.integers(0, 13, (N_S, 2)).astype(np.int8))
    sv('s_hole_suit', r.integers(0, 4, (N_S, 2)).astype(np.int8))
    sv('s_relpos', np.tile(np.arange(6), N_H).astype(np.int8))
    npl = 60
    # 每手 6 個座位必須是 6 位不同玩家（真實資料就是如此，build_tables 也有斷言）
    pc = np.stack([r.choice(npl, 6, replace=False) for _ in range(N_H)]).reshape(-1)
    sv('s_pcode', pc.astype(np.int32))
    sv('s_chen', r.normal(5, 3, N_S).astype(np.float32))
    sv('s_stackbb', np.abs(r.normal(100, 30, N_S)).astype(np.float32))
    sv('s_pfhi', r.integers(2, 15, N_S).astype(np.int8)); sv('s_pflo', r.integers(2, 15, N_S).astype(np.int8))
    sv('s_pfpair', r.integers(0, 2, N_S).astype(np.int8)); sv('s_pfsuited', r.integers(0, 2, N_S).astype(np.int8))
    sp = r.random((N_S, 4)).astype(np.float32); sp[:, 0] = 0
    sv('s_strpct', sp)
    sc = r.integers(1, 10, (N_S, 4)).astype(np.int8); sc[:, 0] = 0
    sv('s_strcat', sc)
    sv('p_base', r.normal(0, 1, (npl, 11)).astype(np.float32))
    np.save(os.path.join(a.out, 'hand_ids.npy'), np.array([f'H{i:014X}' for i in range(N_H)]).astype('U'))
    np.save(os.path.join(a.out, 'player_ids.npy'), np.array([f'U{i:012X}' for i in range(npl)]).astype('U'))
    json.dump({'N_H': N_H, 'N_A': N_A, 'N_S': N_S, 'synthetic': True,
               'marginal_type_entropy_nats': 1.79}, open(os.path.join(a.out, 'meta.json'), 'w'))
    print(f'合成表完成 N_H={N_H} N_A={N_A} N_S={N_S} -> {a.out}')


if __name__ == '__main__':
    main()
