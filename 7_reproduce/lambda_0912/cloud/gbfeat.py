#!/usr/bin/env python3
"""gbnll 的扁平決策狀態特徵組裝。所有欄位都從 build_tables.py 產出的
共用決策狀態表取得，與 seqnll 用的是同一份定義（同一組陣列、同一份因果掃描）。"""
import os
import numpy as np

BASE29 = ['sidx', 'rel_pos', 'players_active', 'to_call_bb', 'pot_before_bb', 'pot_odds',
          'stack_before_bb', 'spr', 'invested_bb', 'n_agg_before', 'facing', 'chen',
          'pf_pair', 'pf_suited', 'pf_hi', 'pf_lo', 'str_now', 'cat_now',
          'b_vpip', 'b_pfr', 'b_agg_rate', 'b_fold_facing', 'b_call_facing', 'b_raise_facing',
          'b_sd_rate', 'b_post_check', 'b_contrib', 'b_allin', 'b_saw_flop']
NEW15 = ['n_agg_street', 'n_acted_street', 'actor_n_prev', 'actor_prev_type', 'actor_prev_agg',
         'facing_own', 'pos_vs_aggr', 'b_suit_max', 'b_paired', 'b_high', 'b_straight_win',
         'h_suit_max', 'h_pair_board', 'h_overcards', 'h_straight_win']
FEAT_BASE = BASE29 + NEW15
FEAT_Y = ['Y_chen', 'Y_pf_pair', 'Y_pf_suited', 'Y_pf_hi', 'Y_pf_lo', 'Y_str_now', 'Y_cat_now',
          'Y_rel_pos', 'Y_pos_diff', 'Y_folded', 'Y_last_aggr', 'Y_acted_street', 'Y_invested_bb']
FEAT_I = FEAT_BASE + FEAT_Y
CAT_BASE = ['sidx', 'rel_pos', 'actor_prev_type', 'pos_vs_aggr']
NAN = np.float32(np.nan)


def _nanify(v, sentinel=-1):
    v = v.astype(np.float32)
    v[v == sentinel] = NAN
    return v


def hand_actions(aoff, nact, hands):
    """把一批 hand 的動作索引攤平成一維（每手的動作在陣列裡本來就連續）。"""
    n = nact[hands]
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    off = np.zeros(len(hands), np.int64)
    np.cumsum(n[:-1], out=off[1:])
    return np.repeat(aoff[hands] - off, n) + np.arange(tot, dtype=np.int64)


class Feats:
    def __init__(self, tdir):
        L = lambda n: np.load(os.path.join(tdir, n + '.npy'))
        self.a_hand = L('a_hand').astype(np.int64)
        self.a_slot = L('a_slot').astype(np.int64)
        self.a_sidx = L('a_sidx').astype(np.int64)
        self.a_y4 = L('a_y4')
        self.a_amtb = L('a_amtb')
        self.a_nactive = L('a_nactive')
        self.a_naggb = L('a_naggb'); self.a_naggs = L('a_naggs'); self.a_nacts = L('a_nacts')
        self.a_apn = L('a_apn'); self.a_apt = L('a_apt'); self.a_apa = L('a_apa')
        self.a_foldbm = L('a_foldbm'); self.a_actedbm = L('a_actedbm'); self.a_lastaggr = L('a_lastaggr')
        self.a_tocall_bb = L('a_tocall_bb'); self.a_pot_bb = L('a_pot_bb')
        self.a_stackbf_bb = L('a_stackbf_bb'); self.a_invested_bb = L('a_invested_bb')
        self.a_contrib6 = L('a_contrib6')
        self.s_relpos = L('s_relpos').astype(np.int64)
        self.s_chen = L('s_chen'); self.s_pfpair = L('s_pfpair'); self.s_pfsuited = L('s_pfsuited')
        self.s_pfhi = L('s_pfhi'); self.s_pflo = L('s_pflo')
        self.s_strpct = L('s_strpct'); self.s_strcat = L('s_strcat')
        self.s_pcode = L('s_pcode').astype(np.int64)
        self.p_base = L('p_base')
        self.bt = np.stack([L('bt_sm'), L('bt_pr'), L('bt_hi'), L('bt_sw')], -1)   # [N_H,4,4]
        self.ht = np.stack([L('ht_sm'), L('ht_pb'), L('ht_oc'), L('ht_sw')], -1)   # [N_S,4,4]
        self.h_bb_raw = L('h_bb_raw').astype(np.float32)
        self.h_grp = L('h_grp')
        self.h_aoff = L('h_aoff').astype(np.int64)
        self.h_nact = L('h_nact').astype(np.int64)
        self.N_A = len(self.a_hand)
        self.N_H = len(self.h_grp)
        self.N_S = 6 * self.N_H

    # ---------------------------------------------------------------- 共用中介量
    def ctx(self, idx):
        h = self.a_hand[idx]
        sl = self.a_slot[idx]
        si = self.a_sidx[idx]
        return h, sl, si, h * 6 + sl, self.h_bb_raw[h]

    def base(self, idx):
        h, sl, si, srow, bb = self.ctx(idx)
        n = len(idx)
        X = np.empty((n, len(FEAT_BASE)), np.float32)
        tc = self.a_tocall_bb[idx]; pot = self.a_pot_bb[idx]; stk = self.a_stackbf_bb[idx]
        pre = si == 0
        c = 0
        def put(v):
            nonlocal c
            X[:, c] = v; c += 1
        put(si)
        put(self.s_relpos[srow])
        put(self.a_nactive[idx])
        put(tc)
        put(pot)
        put(tc / (pot + tc + 1e-3))
        put(stk)
        put(np.clip(stk / (pot + 1e-3), 0, 200))
        put(self.a_invested_bb[idx])
        put(self.a_naggb[idx])
        put((tc > 0).astype(np.float32))
        put(self.s_chen[srow])
        put(self.s_pfpair[srow]); put(self.s_pfsuited[srow])
        put(self.s_pfhi[srow]); put(self.s_pflo[srow])
        sp = self.s_strpct[srow, si].copy(); sp[pre] = NAN; put(sp)             # 翻牌前 null
        sc = self.s_strcat[srow, si].astype(np.float32); sc[pre] = NAN; put(sc)
        pb = self.p_base[self.s_pcode[srow]]
        for j in range(pb.shape[1]):
            put(pb[:, j])
        put(self.a_naggs[idx]); put(self.a_nacts[idx]); put(self.a_apn[idx])
        put(self.a_apt[idx]); put(self.a_apa[idx])
        la = self.a_lastaggr[idx].astype(np.int64)
        put((la == sl).astype(np.float32))
        rp_a = self.s_relpos[h * 6 + np.maximum(la, 0)]
        pv = np.where(la < 0, 6, (rp_a - self.s_relpos[srow]) % 6)
        put(pv.astype(np.float32))
        bt = self.bt[h, si]                    # [n,4]
        for j in range(4):
            put(_nanify(bt[:, j]))
        ht = self.ht[srow, si]
        for j in range(4):
            put(_nanify(ht[:, j]))
        assert c == len(FEAT_BASE), (c, len(FEAT_BASE))
        return X

    def with_y(self, idx, yslot):
        h, sl, si, srow, bb = self.ctx(idx)
        Xb = self.base(idx)
        n = len(idx)
        Y = np.empty((n, len(FEAT_Y)), np.float32)
        yrow = h * 6 + yslot
        pre = si == 0
        c = 0
        def put(v):
            nonlocal c
            Y[:, c] = v; c += 1
        put(self.s_chen[yrow])
        put(self.s_pfpair[yrow]); put(self.s_pfsuited[yrow])
        put(self.s_pfhi[yrow]); put(self.s_pflo[yrow])
        sp = self.s_strpct[yrow, si].copy(); sp[pre] = NAN; put(sp)
        sc = self.s_strcat[yrow, si].astype(np.float32); sc[pre] = NAN; put(sc)
        put(self.s_relpos[yrow])
        put(((self.s_relpos[yrow] - self.s_relpos[srow]) % 6).astype(np.float32))
        put(((self.a_foldbm[idx].astype(np.int64) >> yslot) & 1).astype(np.float32))
        put((self.a_lastaggr[idx].astype(np.int64) == yslot).astype(np.float32))
        put(((self.a_actedbm[idx].astype(np.int64) >> yslot) & 1).astype(np.float32))
        put(self.a_contrib6[idx, yslot] / bb)
        assert c == len(FEAT_Y), (c, len(FEAT_Y))
        return np.concatenate([Xb, Y], 1)
