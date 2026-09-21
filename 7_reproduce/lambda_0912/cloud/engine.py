#!/usr/bin/env python3
"""共用的批次前向：訓練、評估、推論、品質檢查都走這裡，保證資訊集一致。"""
import torch
import torch.nn.functional as F


def flat_ctx(tb, model, hb):
    f, aidx, valid, T = tb.enc_feats(hb)
    h = model.enc(f)
    sel = valid.reshape(-1).nonzero(as_tuple=True)[0]
    hflat = h.reshape(-1, h.shape[-1])[sel]
    aflat = aidx.reshape(-1)[sel]
    hb_flat = hb[:, None].expand(-1, T).reshape(-1)[sel]
    sidx = tb.a_sidx[aflat].long()
    slot = tb.a_slot[aflat].long()
    srow = hb_flat * 6 + slot
    return dict(h=hflat, aflat=aflat, hb=hb_flat, sidx=sidx, slot=slot, srow=srow,
                sel=sel, valid=valid, T=T, n=int(sel.numel()))


def head_nll(model, tb, c, yslot=None, use_y=None):
    """回傳 (nll_type, nll_size, p_taken, entropy)。yslot=None -> N 模式。"""
    x = tb.x_feats(c['srow'], c['sidx'])
    if yslot is None:
        lt, ls = model.head(c['h'], x, None, None)
    else:
        y = tb.y_feats(c['hb'], yslot, c['aflat'], c['sidx'])
        lt, ls = model.head(c['h'], x, y, use_y)
    lt, ls = lt.float(), ls.float()
    tgt_t = tb.a_type[c['aflat']].long()
    tgt_s = tb.a_amtb[c['aflat']].long()
    agg = tgt_t >= 3                       # bet / raise / all_in（SPEC §2.2）
    logp_t = F.log_softmax(lt, -1)
    nll_t = -logp_t.gather(1, tgt_t[:, None]).squeeze(1)
    nll_s = -F.log_softmax(ls, -1).gather(1, tgt_s[:, None]).squeeze(1) * agg.float()
    p_taken = logp_t.gather(1, tgt_t[:, None]).squeeze(1).exp()
    ent = -(logp_t.exp() * logp_t).sum(-1)
    return nll_t, nll_s, p_taken, ent, agg
