"""H_B evidence decoder — shared pieces (2026-09-19).

Listing rule recovered from development_evidence.csv (label-only test, see REPORT_0919.md):
    evidence list = planted hands sorted by (script priority, time), THEN truncated at 5.
  * priority script ("C": fold/soft response to the partner) is listed wherever it occurs in time (uniform over the phase);
  * the lower-priority script ("S": check-down / pay-off) only fills the remaining slots with its EARLIEST hands.
Decoder: per-family planted-C / planted-S detectors (binary, censoring-aware labels) + Poisson-binomial slot probabilities.
"""
import numpy as np

FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']


def runs_from_ranked_times(t):
    """t: times in evidence_rank order -> run index per element (new run when time decreases)."""
    r = [0]
    for i in range(1, len(t)):
        r.append(r[-1] + (1 if t[i] < t[i - 1] else 0))
    return r


def tail_le4(ps):
    """P(sum of independent Bernoulli(ps) <= 4)."""
    d = np.zeros(6); d[0] = 1.0
    for p in ps:
        nd = d * (1 - p); nd[1:] += d[:-1] * p; nd[5] += d[5] * p; d = nd
    return float(d[:5].sum())


def decode_pair(c, s, a=1.0):
    """c, s: planted-C / planted-S probabilities of ONE pair's hands in TIME order -> P(listed) per hand.
    C listed iff <=4 planted C earlier; S listed iff (all planted C, any time) + (planted S earlier) <= 4."""
    n = len(c); out = np.zeros(n)
    # prefix DP for C-before and (C-before + S-before); the 'all other C' term needs C after j too
    for j in range(n):
        pc = tail_le4(c[:j])
        ps = tail_le4(np.concatenate([c[:j], c[j + 1:], s[:j]]))
        out[j] = (c[j] ** a) * pc + (s[j] ** a) * ps      # a < 1 softens the detector's own probability relative to the slot term
    return out


def build_labels(pid, t, ev, n_ev, isC_ev, isS_ev):
    """Censoring-aware training labels. -1 = excluded from training.
    C: positives = listed C. If the list is 5 hands of C only, C-like hands AFTER the last listed C are censored.
       Otherwise (list contains S, or fewer than 5 listed) every planted C is listed -> all other hands are true negatives.
    S: positives = listed S. If 5 listed and no S among them, S status is unknown everywhere (slots exhausted by C).
       If 5 listed incl. S: hands after the last listed S are censored. Fewer than 5 listed: everything else is negative."""
    N = len(pid); labC = np.full(N, -1); labS = np.full(N, -1)
    order = np.argsort(pid, kind='stable'); b = np.flatnonzero(np.r_[True, pid[order][1:] != pid[order][:-1], True])
    for i in range(len(b) - 1):
        idx = order[b[i]:b[i + 1]]; tt = t[idx]; e = ev[idx]; c = isC_ev[idx]; s = isS_ev[idx]
        n5 = n_ev[idx][0] >= 5; hasS = s.any()
        lc = np.zeros(len(idx), int); lc[c] = 1
        if n5 and not hasS and c.any():
            lc[(~e) & (tt > tt[c].max())] = -1
        labC[idx] = lc
        ls = np.zeros(len(idx), int); ls[s] = 1
        if n5 and not hasS:
            ls[:] = -1
        elif n5 and hasS:
            ls[(~e) & (tt > tt[s].max())] = -1
        labS[idx] = ls
    return labC, labS


def decode_levels(p, lv):
    """Exact-level families (CI): p = planted probability, lv = integer strength level (0 = strongest, <0 = not a candidate), TIME order.
    A planted hand of level k is listed iff (#planted of level < k, any time) + (#planted of level k earlier) <= 4."""
    n = len(p); out = np.zeros(n)
    for j in range(n):
        if lv[j] < 0 or p[j] <= 0:
            continue
        m = ((lv >= 0) & (lv < lv[j])) | ((lv == lv[j]) & (np.arange(n) < j))
        out[j] = p[j] * tail_le4(p[m])
    return out
