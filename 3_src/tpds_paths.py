"""Variant-aware paths for the processed artifacts that DEPEND ON THE POLICY (surprise) MODEL.

Set TPDS_VARIANT=<tag> to build and read a parallel world of derived artifacts
(policy_model, surprise_player/resp, l2v2, pairstats, anomaly) without touching the default ones.
An already-submitted chain (v012 = A slot) therefore stays reproducible from the untouched default
artifacts while a corrected policy model is evaluated side by side.  Unset -> exactly the historical
paths, byte-for-byte the same behaviour as before this module existed.

Artifacts that do NOT depend on the policy model keep their fixed paths on purpose:
l2, equity, pfeq, direction, betsize, cand_pairs, seat_l0, hands_l1, responses, player_baselines,
suspect_hidden_positives.  (betsize is trained from the policy state table but is not rebuilt per
variant: keeping it fixed is what makes a variant a single-variable change at the pair layer.)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / '1_data/processed'
VARIANT = os.environ.get('TPDS_VARIANT', '').strip()
SFX = f'_{VARIANT}' if VARIANT else ''
# TPDS_EQTAG=<tag> (2026-09-13): parallel world of the EQUITY-dependent artifacts (equity, pfeq, callvalue, passvalue,
# aggvalue and the L2V2 directory that materialises the equity columns). Unset -> byte-identical historical paths.
EQTAG = os.environ.get('TPDS_EQTAG', '').strip()
EQSFX = f'_{EQTAG}' if EQTAG else ''
L2V2 = f'l2v2{SFX}{EQSFX}'          # directory name under PROC


def vpath(name: str) -> Path:
    """'surprise_player.parquet' -> PROC/'surprise_player_<variant>.parquet' (unchanged when no variant)."""
    p = Path(name)
    return PROC / f'{p.stem}{SFX}{p.suffix}'


def eqpath(name: str) -> Path:
    """'equity_development.parquet' -> PROC/'equity_development_<eqtag>.parquet' (unchanged when TPDS_EQTAG unset)."""
    p = Path(name)
    return PROC / f'{p.stem}{EQSFX}{p.suffix}'


if VARIANT:
    print(f"[tpds_paths] TPDS_VARIANT={VARIANT!r}: policy-derived artifacts use suffix {SFX!r}, L2 dir {L2V2!r}", flush=True)
if EQTAG:
    print(f"[tpds_paths] TPDS_EQTAG={EQTAG!r}: equity-derived artifacts use suffix {EQSFX!r}, L2 dir {L2V2!r}", flush=True)
