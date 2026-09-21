#!/usr/bin/env python3
"""TPDS seqnll — 交叉擬合訓練。g0 訓練在半區 0、替半區 1 打分；g1 反之。
訓練不使用任何勾結標籤；選 epoch 只看另一半的 held-out NLL。"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from model import SeqNLL, Tables
from engine import flat_ctx, head_nll

T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)


def loss_batch(model, tb, hb, gen):
    c = flat_ctx(tb, model, hb)
    n = c['n']
    dev = hb.device
    off = torch.randint(1, 6, (n,), device=dev, generator=gen)
    yslot = (c['slot'] + off) % 6
    use_y = torch.rand(n, device=dev, generator=gen) < 0.5      # SPEC §2.2：各半機率
    nll_t, nll_s, _, _, agg = head_nll(model, tb, c, yslot, use_y)
    return (nll_t.sum() + nll_s.sum()) / n, nll_t, nll_s, agg, n


@torch.no_grad()
def evaluate(model, tb, hands, bs, seed=1234, amp=True):
    model.eval()
    dev = hands.device
    acc = {k: 0.0 for k in ['n', 'nagg', 'N_t', 'N_s', 'I_t', 'I_s', 'M_t', 'M_s']}
    for i in range(0, hands.numel(), bs):
        hb = hands[i:i + bs]
        g = torch.Generator(device=dev); g.manual_seed(seed + i)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            c = flat_ctx(tb, model, hb)
            n = c['n']
            nt, ns, _, _, agg = head_nll(model, tb, c, None, None)
            off = torch.randint(1, 6, (n,), device=dev, generator=g)
            yslot = (c['slot'] + off) % 6
            it, is_, _, _, _ = head_nll(model, tb, c, yslot, torch.ones(n, dtype=torch.bool, device=dev))
            um = torch.rand(n, device=dev, generator=g) < 0.5
        acc['n'] += n; acc['nagg'] += float(agg.sum())
        acc['N_t'] += float(nt.sum()); acc['N_s'] += float(ns.sum())
        acc['I_t'] += float(it.sum()); acc['I_s'] += float(is_.sum())
        acc['M_t'] += float(torch.where(um, it, nt).sum())
        acc['M_s'] += float(torch.where(um, is_, ns).sum())
    n, na = acc['n'], max(acc['nagg'], 1.0)
    model.train()
    return {'n_actions': int(n), 'n_agg': int(acc['nagg']),
            'N_type': acc['N_t'] / n, 'N_size_per_action': acc['N_s'] / n,
            'N_size_per_agg': acc['N_s'] / na, 'N_total': (acc['N_t'] + acc['N_s']) / n,
            'I_type': acc['I_t'] / n, 'I_size_per_action': acc['I_s'] / n,
            'I_total': (acc['I_t'] + acc['I_s']) / n,
            'MIX_total': (acc['M_t'] + acc['M_s']) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='/home/ubuntu/tpds/tables')
    ap.add_argument('--out', default='/home/ubuntu/tpds/ckpt')
    ap.add_argument('--group', type=int, required=True, choices=[0, 1])
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--eval-hands', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--throughput-only', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed + args.group)
    dev = args.device
    AMP = dev == 'cuda'
    if AMP:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tb = Tables(args.tables, dev)
    log(f"表載入完成 N_H={tb.N_H:,} N_A={tb.N_A:,} "
        f"GPU={torch.cuda.memory_allocated()/1e9:.2f} GB" if AMP else f"表載入完成 N_H={tb.N_H:,} N_A={tb.N_A:,} (cpu)")
    grp = tb.h_grp
    tr = (grp == args.group).nonzero(as_tuple=True)[0]
    ho = (grp != args.group).nonzero(as_tuple=True)[0]
    g_ev = torch.Generator(device=dev); g_ev.manual_seed(999)
    ev = ho[torch.randperm(ho.numel(), device=dev, generator=g_ev)[:min(args.eval_hands, ho.numel())]]
    log(f'g{args.group}: 訓練 {tr.numel():,} 手、held-out 母體 {ho.numel():,} 手、評估抽樣 {ev.numel():,} 手')

    model = SeqNLL().to(dev)
    log(f'參數量 {model.n_params():,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    spe = max(1, tr.numel() // args.bs)
    total = spe * args.epochs
    warm = min(args.warmup, max(1, total // 10))

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    gen = torch.Generator(device=dev); gen.manual_seed(args.seed * 7 + args.group)
    hist, step = [], 0
    best = (1e9, -1)
    for ep in range(args.epochs):
        model.train()
        perm = tr[torch.randperm(tr.numel(), device=dev, generator=gen)]
        t_ep, run, seen = time.time(), 0.0, 0
        for bi in range(spe):
            hb = perm[bi * args.bs:(bi + 1) * args.bs]
            for pg in opt.param_groups:
                pg['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=AMP):
                loss, _, _, _, n = loss_batch(model, tb, hb, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * n; seen += n; step += 1
            if bi % 200 == 0:
                el = time.time() - t_ep
                log(f'  ep{ep} step {bi}/{spe} loss={run/max(seen,1):.4f} '
                    f'{(bi+1)/max(el,1e-9):.2f} it/s eta_ep={(spe-bi-1)/max((bi+1)/max(el,1e-9),1e-9)/60:.1f}min')
            if args.throughput_only and bi == 60:
                el = time.time() - t_ep
                print(json.dumps({'it_per_s': (bi + 1) / el, 'hands_per_s': (bi + 1) * args.bs / el,
                                  'steps_per_epoch': spe}))
                return
        ep_s = time.time() - t_ep
        m = evaluate(model, tb, ev, args.bs, amp=AMP)
        m.update({'epoch': ep, 'train_loss': run / max(seen, 1), 'epoch_sec': ep_s, 'lr_end': lr_at(step - 1)})
        hist.append(m)
        log(f"  ep{ep} 完成 {ep_s/60:.1f}min train={m['train_loss']:.4f} "
            f"heldout MIX={m['MIX_total']:.4f} N_total={m['N_total']:.4f} N_type={m['N_type']:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out, f'g{args.group}_ep{ep}.pt'))
        if m['MIX_total'] < best[0]:
            best = (m['MIX_total'], ep)
            torch.save(model.state_dict(), os.path.join(args.out, f'g{args.group}_best.pt'))
        json.dump({'group': args.group, 'best_epoch': best[1], 'best_mix': best[0],
                   'history': hist, 'args': vars(args), 'steps_per_epoch': spe, 'warmup': warm},
                  open(os.path.join(args.out, f'g{args.group}_hist.json'), 'w'), indent=2)
    log(f'g{args.group} 訓練結束，最佳 epoch = {best[1]}（held-out MIX NLL {best[0]:.4f}）')


if __name__ == '__main__':
    main()
