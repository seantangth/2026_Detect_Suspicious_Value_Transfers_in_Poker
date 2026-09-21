#!/usr/bin/env python3
"""TPDS seqnll — 模型與批次組裝。

序列佈局（相對 SPEC §2.1 的唯一偏離，理由見 PROGRESS.md）：
  位置 p 攜帶 = (第 p-1 個動作的類型／尺寸桶／行動者相對位置)  ⊕
                (第 p 個動作的『前置公開狀態』：街、當街已公開公牌、pot 桶、
                 to_call 比例桶、SPR 桶、在局人數、行動者相對位置)
  因果注意力下，輸出位置 p 的資訊集 = {動作 0..p-1} ∪ {動作 p 的前置公開狀態}，
  正好等於 SPEC §0 硬規則第 2 條允許的集合。輸出位置 p 預測動作 p。
私有資訊（底牌）一律只在預測頭進入，編碼器完全看不到。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_LAYER, N_HEAD, D_FFN, DROPOUT = 192, 4, 4, 512, 0.1
N_TYPE, N_SIZE = 6, 13
MAXLEN = 64
NVIS = torch.tensor([0, 3, 4, 5])          # 各街已公開公牌張數
BSLOT = torch.tensor([0, 0, 0, 1, 2])      # 公牌槽：翻牌三張同組（順序無意義）、轉牌、河牌
NX_NUM, NY_NUM = 7 + 11, 4


class Block(nn.Module):
    def __init__(self, d=D_MODEL, nh=N_HEAD, ffn=D_FFN, p=DROPOUT):
        super().__init__()
        self.nh, self.hd, self.p = nh, d // nh, p
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, L, Dm = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, L, Dm)))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class Encoder(nn.Module):
    """只吃公開資訊。"""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.e_ptype = nn.Embedding(7, d)    # 0..5 動作型別，6 = [HAND]
        self.e_pamtb = nn.Embedding(14, d)   # 0..12 尺寸桶，13 = 無
        self.e_ppos = nn.Embedding(7, d)     # 0..5 相對位置，6 = 無
        self.e_bb = nn.Embedding(4, d)
        self.e_sidx = nn.Embedding(4, d)
        self.e_cpos = nn.Embedding(6, d)
        self.e_potb = nn.Embedding(12, d)
        self.e_tcb = nn.Embedding(10, d)
        self.e_sprb = nn.Embedding(10, d)
        self.e_nact = nn.Embedding(8, d)
        self.e_brank = nn.Embedding(13, d)
        self.e_bsuit = nn.Embedding(4, d)
        self.e_bslot = nn.Embedding(3, d)
        self.e_pos = nn.Embedding(MAXLEN, d)
        self.ln_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(N_LAYER)])
        self.ln_out = nn.LayerNorm(d)

    def forward(self, f):
        B, T = f['sidx'].shape
        ar = torch.arange(T, device=f['sidx'].device)
        # 公牌：先算每手的前綴和，再依當街可見張數取用（避免 [B,T,5,d] 的大張量）
        cards = (self.e_brank(f['brank'].clamp(min=0)) + self.e_bsuit(f['bsuit'].clamp(min=0))
                 + self.e_bslot(BSLOT.to(f['brank'].device))[None])
        cards = cards * (f['brank'] >= 0).unsqueeze(-1)
        bcum = torch.cat([torch.zeros_like(cards[:, :1]), cards.cumsum(1)], 1)      # [B,6,d]
        bemb = bcum.gather(1, f['nvis'].unsqueeze(-1).expand(B, T, cards.shape[-1]))
        x = (self.e_ptype(f['p_type']) + self.e_pamtb(f['p_amtb']) + self.e_ppos(f['p_pos'])
             + self.e_sidx(f['sidx']) + self.e_cpos(f['c_pos']) + self.e_potb(f['potb'])
             + self.e_tcb(f['tcb']) + self.e_sprb(f['sprb']) + self.e_nact(f['nactive'])
             + bemb + self.e_bb(f['bb'])[:, None, :] + self.e_pos(ar)[None, :, :])
        x = self.ln_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)


class Head(nn.Module):
    """私有資訊只在這裡進來：X 自己的底牌 ＋ 可選的揭露槽（另一位玩家 Y）。"""

    def __init__(self, d=D_MODEL, hid=384):
        super().__init__()
        self.e_xr = nn.Embedding(13, d)
        self.e_xs = nn.Embedding(4, d)
        self.e_xcat = nn.Embedding(10, d)
        self.e_xpos = nn.Embedding(6, d)
        self.x_num = nn.Linear(NX_NUM, d)
        self.ln_x = nn.LayerNorm(d)
        self.e_yr = nn.Embedding(13, d)
        self.e_ys = nn.Embedding(4, d)
        self.e_ycat = nn.Embedding(10, d)
        self.e_ypos = nn.Embedding(6, d)
        self.y_num = nn.Linear(NY_NUM, d)
        self.ln_y = nn.LayerNorm(d)
        self.w_none = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(nn.Linear(3 * d, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU())
        self.out_type = nn.Linear(hid, N_TYPE)
        self.out_size = nn.Linear(hid, N_SIZE)

    @staticmethod
    def _pair(er, es, r, s):
        """每張牌先各自成一個向量再相加：c0 + c1 與 c1 + c0 逐位元相同。
        （若寫成 er0+er1+es0+es1 依序相加，交換順序會改變中間值的捨入，
          品質檢查 4 會以 ~5e-7 的差距失敗——自測時實際踩到過。）"""
        return (er(r[:, 0]) + es(s[:, 0])) + (er(r[:, 1]) + es(s[:, 1]))

    def forward(self, h, x, y=None, use_y=None):
        # 兩張底牌以「每張各自成向量再相加」進來 -> 交換順序輸出逐位元不變
        xv = (self._pair(self.e_xr, self.e_xs, x['r'], x['s'])
              + self.e_xcat(x['cat']) + self.e_xpos(x['pos']) + self.x_num(x['num']))
        xv = F.gelu(self.ln_x(xv))
        if y is None:
            yv = self.w_none.to(xv.dtype).expand_as(xv)
        else:
            yv = (self._pair(self.e_yr, self.e_ys, y['r'], y['s'])
                  + self.e_ycat(y['cat']) + self.e_ypos(y['pos']) + self.y_num(y['num']))
            yv = F.gelu(self.ln_y(yv))
            if use_y is not None:
                yv = torch.where(use_y[:, None], yv, self.w_none.to(yv.dtype).expand_as(yv))
        z = self.mlp(torch.cat([h, xv, yv], -1))
        return self.out_type(z), self.out_size(z)


class SeqNLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.head = Head()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- 資料
class Tables:
    """所有特徵陣列常駐 GPU（約 0.7 GB），訓練完全沒有 dataloader。"""

    KEYS_A = ['a_hand', 'a_slot', 'a_sidx', 'a_type', 'a_amtb', 'a_potb', 'a_tcb', 'a_sprb',
              'a_nactive', 'a_foldbm', 'a_actedbm', 'a_lastaggr', 'a_isagg', 'a_iscall', 'a_isfold']
    KEYS_H = ['h_bb', 'h_grp', 'h_phase', 'h_nact', 'h_aoff', 'h_board_rank', 'h_board_suit', 'h_tbl']
    KEYS_S = ['s_hole_rank', 's_hole_suit', 's_relpos', 's_pcode', 's_chen', 's_stackbb',
              's_pfhi', 's_pflo', 's_pfpair', 's_pfsuited', 's_strpct', 's_strcat']

    def __init__(self, path, device='cuda'):
        import numpy as np, os, json
        self.device = device
        for k in self.KEYS_A + self.KEYS_H + self.KEYS_S + ['p_base']:
            a = np.load(os.path.join(path, k + '.npy'))
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            setattr(self, k, torch.from_numpy(a).to(device))
        self.h_aoff = self.h_aoff.long()
        self.h_nact = self.h_nact.long()
        self.s_strpct = self.s_strpct.float()
        self.s_chen = self.s_chen.float()
        self.s_stackbb = self.s_stackbb.float()
        self.meta = json.load(open(os.path.join(path, 'meta.json')))
        self.N_H = int(self.h_nact.shape[0])
        self.N_A = int(self.a_type.shape[0])
        self.nvis_tab = NVIS.to(device)

    # ---- 編碼器輸入 -------------------------------------------------------
    def enc_feats(self, hb):
        dev = self.device
        nact = self.h_nact[hb]
        T = int(nact.max().item())
        ar = torch.arange(T, device=dev)
        valid = ar[None, :] < nact[:, None]
        aidx = (self.h_aoff[hb][:, None] + ar[None, :]).clamp(min=0)
        aidx = torch.where(valid, aidx, torch.zeros_like(aidx))
        prev = (aidx - 1).clamp(min=0)
        has_prev = (ar > 0)[None, :] & valid
        slot = self.a_slot[aidx].long()
        srow = hb[:, None] * 6 + slot
        sidx = self.a_sidx[aidx].long()
        pslot = self.a_slot[prev].long()
        psrow = hb[:, None] * 6 + pslot
        f = {
            'sidx': sidx,
            'c_pos': self.s_relpos[srow].long(),
            'potb': self.a_potb[aidx].long(),
            'tcb': self.a_tcb[aidx].long(),
            'sprb': self.a_sprb[aidx].long(),
            'nactive': self.a_nactive[aidx].long(),
            'nvis': self.nvis_tab[sidx],
            'brank': self.h_board_rank[hb].long(),
            'bsuit': self.h_board_suit[hb].long(),
            'bb': self.h_bb[hb].long(),
            'p_type': torch.where(has_prev, self.a_type[prev].long(), torch.full_like(sidx, 6)),
            'p_amtb': torch.where(has_prev, self.a_amtb[prev].long(), torch.full_like(sidx, 13)),
            'p_pos': torch.where(has_prev, self.s_relpos[psrow].long(), torch.full_like(sidx, 6)),
        }
        return f, aidx, valid, T

    # ---- 預測頭輸入 -------------------------------------------------------
    def x_feats(self, srow, sidx):
        num = torch.stack([
            self.s_chen[srow] / 10.0,
            self.s_pfhi[srow].float() / 14.0,
            self.s_pflo[srow].float() / 14.0,
            self.s_pfpair[srow].float(),
            self.s_pfsuited[srow].float(),
            self.s_strpct[srow, sidx],
            torch.log1p(self.s_stackbb[srow].clamp(min=0)) / 5.0,
        ], -1)
        num = torch.cat([num, self.p_base[self.s_pcode[srow].long()]], -1)
        return {'r': self.s_hole_rank[srow].long(), 's': self.s_hole_suit[srow].long(),
                'cat': self.s_strcat[srow, sidx].long(), 'pos': self.s_relpos[srow].long(), 'num': num}

    def y_feats(self, hb_flat, yslot, aflat, sidx):
        srow = hb_flat * 6 + yslot
        folded = ((self.a_foldbm[aflat].long() >> yslot) & 1).float()
        acted = ((self.a_actedbm[aflat].long() >> yslot) & 1).float()
        lastag = (self.a_lastaggr[aflat].long() == yslot).float()
        num = torch.stack([folded, acted, lastag, self.s_strpct[srow, sidx]], -1)
        return {'r': self.s_hole_rank[srow].long(), 's': self.s_hole_suit[srow].long(),
                'cat': self.s_strcat[srow, sidx].long(), 'pos': self.s_relpos[srow].long(), 'num': num}
