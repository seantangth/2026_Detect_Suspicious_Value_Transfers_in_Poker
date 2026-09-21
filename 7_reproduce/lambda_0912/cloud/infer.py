#!/usr/bin/env python3
"""TPDS seqnll — 全量推論，輸出 T1/T2/T3。所有分數都是 out-of-sample
（半區 1 由 g0 打分、半區 0 由 g1 打分）。"""
import argparse, json, os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from model import SeqNLL, Tables
from engine import flat_ctx, head_nll
from outputs import write_all, nanify, T2_COLS

T0 = time.time()
NEG = float('-inf')


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--ckpt', default='/home/ubuntu/tpds/ckpt')
    ap.add_argument('--out', default='/home/ubuntu/tpds/out')
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--chunk-hands', type=int, default=200000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--prefix', default='seqnll')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    tb = Tables(args.tables, dev)
    N_H, N_A, N_S = tb.N_H, tb.N_A, 6 * tb.N_H
    log(f'N_H={N_H:,} N_A={N_A:,} N_S={N_S:,}')

    # ---- 輸出累加器（常駐 GPU）
    f32 = lambda n, v=0.0: torch.full((n,), v, dtype=torch.float32, device=dev)
    t1_nt, t1_ns, t1_pt, t1_en = f32(N_A), f32(N_A), f32(N_A), f32(N_A)
    t2_n = torch.zeros(N_S, dtype=torch.int32, device=dev)
    t2_sum, t2_size = f32(N_S), f32(N_S)
    t2_max, t2_fmax, t2_cmax, t2_amax = f32(N_S, NEG), f32(N_S, NEG), f32(N_S, NEG), f32(N_S, NEG)
    t2_str = f32(N_S * 4)
    t3_sum = f32(N_S * 6)
    t3_max, t3_gmax = f32(N_S * 6, NEG), f32(N_S * 6, NEG)

    by_n = np.zeros(6, np.float64); by_i = np.zeros(6, np.float64); by_c = np.zeros(6, np.float64)
    n_done = 0
    for g in (0, 1):
        sd = torch.load(os.path.join(args.ckpt, f'g{g}_best.pt'), map_location=dev)
        model = SeqNLL().to(dev)
        model.load_state_dict(sd)
        model.eval()
        target = (tb.h_grp == (1 - g)).nonzero(as_tuple=True)[0]   # g0 打分半區 1
        log(f'g{g} 打分半區 {1-g}：{target.numel():,} 手')
        t_g = time.time()
        with torch.no_grad():
            for i in range(0, target.numel(), args.bs):
                hb = target[i:i + args.bs]
                c = flat_ctx(tb, model, hb)
                aflat, srow = c['aflat'], c['srow']
                nt, ns, pt, en, agg = head_nll(model, tb, c, None, None)
                tot_N = nt + ns
                t1_nt[aflat] = nt; t1_ns[aflat] = torch.where(agg, ns, torch.full_like(ns, float('nan')))
                t1_pt[aflat] = pt; t1_en[aflat] = en
                one = torch.ones_like(srow, dtype=torch.int32)
                t2_n.scatter_add_(0, srow, one)
                t2_sum.scatter_add_(0, srow, tot_N)
                t2_size.scatter_add_(0, srow, ns)
                t2_str.scatter_add_(0, srow * 4 + c['sidx'], tot_N)
                t2_max.scatter_reduce_(0, srow, tot_N, reduce='amax', include_self=True)
                for arr, m in ((t2_fmax, tb.a_isfold[aflat]), (t2_cmax, tb.a_iscall[aflat]),
                               (t2_amax, tb.a_isagg[aflat])):
                    arr.scatter_reduce_(0, srow, torch.where(m, tot_N, torch.full_like(tot_N, NEG)),
                                        reduce='amax', include_self=True)
                ty = tb.a_type[aflat].long()
                by_n += torch.bincount(ty, weights=tot_N.double(), minlength=6).cpu().numpy()
                by_c += torch.bincount(ty, minlength=6).cpu().numpy()
                for k in range(1, 6):
                    ys = (c['slot'] + k) % 6
                    it, is_, _, _, _ = head_nll(model, tb, c, ys,
                                                torch.ones(c['n'], dtype=torch.bool, device=dev))
                    tot_I = it + is_
                    by_i += torch.bincount(ty, weights=tot_I.double(), minlength=6).cpu().numpy() / 5.0
                    idx = srow * 6 + ys
                    t3_sum.scatter_add_(0, idx, tot_I)
                    t3_max.scatter_reduce_(0, idx, tot_I, reduce='amax', include_self=True)
                    t3_gmax.scatter_reduce_(0, idx, tot_N - tot_I, reduce='amax', include_self=True)
                n_done += hb.numel()
                if (i // args.bs) % 200 == 0:
                    el = time.time() - t_g
                    log(f'  g{g} {i:,}/{target.numel():,} '
                        f'{(i+args.bs)/max(el,1e-9):.0f} hands/s '
                        f'eta={(target.numel()-i)/max((i+args.bs)/max(el,1e-9),1e-9)/60:.1f}min')
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()
    assert n_done == N_H, (n_done, N_H)
    log('推論完成，搬回 CPU')

    def cpu(t):
        return t.cpu().numpy()

    T1 = dict(nll_type_N=cpu(t1_nt), nll_size_N=cpu(t1_ns), p_taken_N=cpu(t1_pt), entropy_N=cpu(t1_en))
    T2 = dict(n_act=cpu(t2_n), nll_N_sum=cpu(t2_sum), nll_N_max=nanify(cpu(t2_max)),
              nll_N_fold_max=nanify(cpu(t2_fmax)), nll_N_call_max=nanify(cpu(t2_cmax)),
              nll_N_agg_max=nanify(cpu(t2_amax)), nll_size_N_sum=cpu(t2_size))
    st = cpu(t2_str).reshape(-1, 4)
    for k, nm in enumerate(['pre', 'flop', 'turn', 'river']):
        T2[f'nll_N_{nm}'] = st[:, k].copy()
    del st
    T3sum, T3max, T3gmax = cpu(t3_sum), nanify(cpu(t3_max)), nanify(cpu(t3_gmax))
    del t1_nt, t1_ns, t1_pt, t1_en, t2_sum, t2_size, t2_max, t2_fmax, t2_cmax, t2_amax, t2_str
    del t3_sum, t3_max, t3_gmax
    if dev == 'cuda':
        torch.cuda.empty_cache()
    summ = write_all(args.out, args.prefix, args.tables, T1, T2, T3sum, T3max, T3gmax,
                     log=log, chunk_hands=args.chunk_hands)
    ACT = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    summ['by_action_type'] = {ACT[i]: {'n': int(by_c[i]),
                                       'nll_N': by_n[i] / max(by_c[i], 1),
                                       'nll_I_mean_over_Y': by_i[i] / max(by_c[i], 1),
                                       'mean_gain': (by_n[i] - by_i[i]) / max(by_c[i], 1)}
                              for i in range(6)}
    json.dump(summ, open(os.path.join(args.out, f'{args.prefix}_infer_summary.json'), 'w'), indent=2)
    log('完成', json.dumps(summ))


if __name__ == '__main__':
    main()
