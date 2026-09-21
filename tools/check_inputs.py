"""Verify that input files are in place and byte-identical to the ones used for the submission.
Usage: check_inputs.py [manifest ...]   (default: raw_inputs.md5 frozen_inputs.md5; replay.sh uses raw_inputs.md5 checkpoint.md5)"""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


bad = 0
for manifest in (sys.argv[1:] or ['raw_inputs.md5', 'frozen_inputs.md5']):
    rows = [l.split('  ', 1) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    miss = [r for _, r in rows if not (ROOT / r).exists()]
    if miss:
        print(f'{manifest}: {len(miss)} of {len(rows)} files missing, e.g. {miss[:3]}'); bad += len(miss); continue
    diff = [r for h, r in rows if md5(ROOT / r) != h]
    print(f'{manifest}: {len(rows)} files present, {len(diff)} with a different md5' + (f', e.g. {diff[:3]}' if diff else ''))
    bad += len(diff)
sys.exit(1 if bad else 0)
