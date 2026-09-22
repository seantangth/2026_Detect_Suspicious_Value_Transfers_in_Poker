"""Verify that input files are in place and byte-identical to the ones used for the submission.
Usage: check_inputs.py [manifest ...]   (default: raw_inputs.md5 frozen_inputs.md5; replay.sh uses raw_inputs.md5 checkpoint.md5)
With TPDS_REGENERATED_CLOUD=1 (cloud-stage outputs regenerated with the code in 7_reproduce/ and 5_outputs/pokerbench_0915/, see
CLOUD_STAGES.md), those outputs are only checked for presence, since a re-run does not reproduce them bit for bit, and the optional
development family OOF may be absent; the configuration and key files are still checked by md5."""
import hashlib, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGEN = os.environ.get('TPDS_REGENERATED_CLOUD') == '1'
CLOUD = ('5_outputs/seqnll_0912/', '5_outputs/pairpol_0913/', '1_data/processed/pairpol/', '5_outputs/pokerbench_0915/')
OPTIONAL = ('5_outputs/revise_0917/family_clf_dev_oof.parquet',)


def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


bad = 0
for manifest in (sys.argv[1:] or ['raw_inputs.md5', 'frozen_inputs.md5']):
    rows = [l.split('  ', 1) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    if REGEN:
        rows = [(h, r) for h, r in rows if not (r in OPTIONAL and not (ROOT / r).exists())]
    miss = [r for _, r in rows if not (ROOT / r).exists()]
    if miss:
        print(f'{manifest}: {len(miss)} of {len(rows)} files missing, e.g. {miss[:3]}'); bad += len(miss); continue
    pres = [r for _, r in rows if REGEN and r.startswith(CLOUD)]
    diff = [r for h, r in rows if not (REGEN and r.startswith(CLOUD)) and md5(ROOT / r) != h]
    print(f'{manifest}: {len(rows)} files present, {len(diff)} with a different md5' + (f', e.g. {diff[:3]}' if diff else '')
          + (f' ({len(pres)} regenerated cloud-stage outputs checked for presence only)' if pres else ''))
    bad += len(diff)
sys.exit(1 if bad else 0)
