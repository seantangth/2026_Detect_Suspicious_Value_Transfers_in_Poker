# Detect Suspicious Value Transfers in Poker — 9th place solution (reproduction package)

Kaggle competition: [Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker) (Slash).
Team `seantangth`. Solution write-up: _link_. Case reviews: [`CASE_REVIEWS.md`](CASE_REVIEWS.md).
Artifacts: Kaggle dataset [`seantangth/tpds-9th-place-artifacts`](https://www.kaggle.com/datasets/seantangth/tpds-9th-place-artifacts).

| selected submission | file | md5 | public | private |
|---|---|---|---|---|
| **A** (final ranking) | `submission.csv` = `5_outputs/submissions/submission_TWFWDp_HB3k12_FINALD2_split_f2p_kdt3_w90.csv` | `c303970ed74faa833cf671ebdfd4efb6` | 0.92668 | **0.92677** |
| B (hedge: same behaviour and evidence columns, production risk) | `5_outputs/submissions/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv` | `f17b3b4de833994b8ec994546f3f802b` | 0.92559 | 0.92414 |

There are two ways to check the submission, both CPU only:

| | what is re-run | inputs | result | time |
|---|---|---|---|---|
| **1. Replay** — `replay.sh` | the final model layer: stage-1 pair ensemble (LightGBM + CatBoost, both training worlds), stage-2 head re-ranker and rank rule, family classifier, evidence-listing decoder (evaluation inference + decoding), assembly | raw data + **checkpoint** (3.5 GB) | **byte-identical** A, B and base file (md5 checked), on macOS arm64 and on Linux x86-64 | ~40 min (Apple M4, 10 cores); ~2 h 10 min (Kaggle CPU notebook, 4 cores) |
| **2. Full rebuild** — `run_all.sh` | everything from the raw tables | raw data + **frozen inputs** (1.4 GB) | an equivalent file, not byte-identical (see §5) | ~4 h (Apple M4, 10 cores) |

The Kaggle notebook [TPDS 9th place - replay of the selected submission](https://www.kaggle.com/code/seantangth/tpds-9th-place-replay-of-the-selected-submission)
runs the replay in a fresh Kaggle CPU session (Python 3.13 and the pinned packages are installed by the notebook); its run reproduced all
three files byte for byte.

## 1. Setup

Python 3.13; the exact package versions are in `requirements.txt`.

```bash
git clone https://github.com/seantangth/2026_Detect_Suspicious_Value_Transfers_in_Poker.git tpds && cd tpds
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# competition data (accept the competition rules first)
kaggle competitions download -c detect-suspicious-value-transfers-in-poker -p 1_data/raw/detect-suspicious-value-transfers-in-poker
(cd 1_data/raw/detect-suspicious-value-transfers-in-poker && unzip -q detect-suspicious-value-transfers-in-poker.zip)
```

## 2. Replay (byte-identical)

```bash
# artifacts (4.7 GB): Kaggle serves the uploaded archives as the folders checkpoint/, frozen_inputs/ and code/
kaggle datasets download seantangth/tpds-9th-place-artifacts -p artifacts --unzip
cp -R artifacts/checkpoint/. .            # competition-time intermediate files, at the repository root
PY=.venv/bin/python ./replay.sh
```

`replay.sh` checks every input against `raw_inputs.md5` and `checkpoint.md5`, writes the development labels back into the four checkpoint
tables that use them (`tools/attach_dev_labels.py`; the published checkpoint does not contain labels), re-runs the final model layer and
prints three md5 checks, which should all read "identical to the submitted file".

## 3. Full rebuild (from the raw tables)

In a fresh clone (set up as in §1, without the checkpoint):

```bash
# frozen inputs (downloaded in §2): outputs of three cloud stages + small configuration files, at the repository root
cp -R artifacts/frozen_inputs/. .
PY=.venv/bin/python ./run_all.sh          # one log per step in logs/; peak RSS ~6 GB
.venv/bin/python tools/compare_submissions.py artifacts/submission_TWFWDp_HB3k12_FINALD2_split_f2p_kdt3_w90.csv submission.csv
```

| steps | what | main code |
|---|---|---|
| s01 | per-hand / per-seat tables, candidate pairs, one row per (pair, shared hand) | `3_src/tpds_features.py` |
| s02–s04 | hand equity: Monte-Carlo world and exact world (post-flop enumeration, pre-flop 10k MC) | `3_src/tpds_equity*.py`, `tpds_preflop_eq.py` |
| s05–s07 | population action (policy) models → per-action surprise; with / without look-ahead; table-group cross-fit | `3_src/tpds_policy.py` |
| s08a–s13 | feature rows with equity + surprise (three feature worlds), bet-size residuals, chip-flow direction, call regret | `3_src/tpds_augment.py`, `tpds_betsize.py`, `tpds_direction.py`, `tpds_callvalue.py` |
| s14–s16 | hand models (per family + generic) and pooled pair features, two feature worlds | `3_src/tpds_model.py`, `tpds_pair_stage.py` |
| s17–s21 | pair families: dominance context, partner-presence contrast (entropy-adjusted), pair-conditional policy | `tools/build_dom.py`, `tools/build_nllx.py`, `5_outputs/eqx_0913/build_presence_contrast*.py`, `tools/build_pairpol_feats.py` |
| s22–s28 | eval-length-window training world (W1 = hands [0,2000), W2 = [1000,3000)) and the two-stage pair model; arm F = full-length dev only (production risk), arm FW = full + both windows (selected risk) | `5_outputs/research_0919/band/tw_*.py` |
| s29 | behaviour: family classifier | `5_outputs/revise_0917/family_clf_eval.py` |
| s30–s34 | evidence hand features: PokerBench-SFT surprisal, private-information likelihood ratios | `tools/build_pb_dev.py`, `tools/build_gate_keys.py`, `5_outputs/pokerbench_0915/`, `5_outputs/revise_0915/ni_corrected2.py`, `5_outputs/pb_variants_0916/` |
| s35–s43 | evidence rankers (LambdaRank, per family + generic, 2 seed sets each), evaluation scoring, blend | `3_src/tpds_evidence*.py`, `7_reproduce/merge_seed_sets.py`, `3_src/tpds_submit.py` |
| s44 | base file: production risk + ranker evidence + family labels | `5_outputs/audit_0917/build_final.py` |
| s45–s46 | evidence-listing decoder (12 seeds) → selected B | `5_outputs/research_0919/hb_train3.py`, `hb_apply3.py` |
| s47 | eval-length-window risk (arm FW) + D′ rank rule spliced onto B → selected A | `5_outputs/research_0919/band/tw_chain2.py` |

Directory names under `5_outputs/` are those of the research project, so every script can be traced to its experiment log; outputs are
written next to the code, as during the competition. `release_changes.json` lists every edit made to the research code for this release:
path handling, one `exec` of a code slice replaced by an import of the identical frozen module, and research-only side outputs and
diagnostics made optional. Small builders that were run ad hoc during the competition were re-written in `tools/` and checked against the
original tables: identical content, except `build_nllx.py`, which matches to within 2e-6 (float32 summation order).

## 4. What is shipped, and what is not

**Frozen inputs** (`frozen_inputs.md5`, needed by the full rebuild):

| file(s) | produced by | why frozen |
|---|---|---|
| `5_outputs/seqnll_0912/gbnll_{player,action}.parquet` | `7_reproduce/lambda_0912/cloud/` (LightGBM normal-behaviour model, rented CPU) | cloud stage ([CLOUD_STAGES.md](CLOUD_STAGES.md)) |
| `5_outputs/pairpol_0913/*`, `1_data/processed/pairpol/*` | `7_reproduce/lambda_0913/cloud/pairpol.py` (1× A10) | cloud stage |
| `5_outputs/pokerbench_0915/{prompts,prompts_eval_top8000,scores,scores_eval_top8000}.parquet` | `score_vllm.py` with `YiPz/llama3-8b-pokerbench-sft` (8× A100) | cloud stage; prompt tables are keys only (no hand text) |
| `5_outputs/models/v5/{hand_features.json,folds_by_table.json}`, `1_data/processed/suspect_hidden_positives.parquet` | first baseline run | configuration: hand-model feature list, table folds, 171 unlabelled development pairs given weight 0 |
| `5_outputs/models/v5nb/evidence_eval_pfonscvxpb4.parquet` | an earlier evidence run | (pair, hand) keys of the 8,000 evaluation pairs with PokerBench features |
| `5_outputs/revise_0917/family_clf_dev_oof.parquet` | development family-classifier OOF | only feeds a development score printed by `hb_train3.py` |

**Checkpoint** (`checkpoint.md5`, needed by the replay): the 897 competition-time files that the replayed steps read (per-pair feature tables
of the three training worlds and of evaluation, hand scores, evaluation per-(pair, hand) rows, the ranker evidence scores, the 60 decoder
detectors, the base evidence file). They were listed by tracing the file reads of each replayed script.

**Not shipped:** no raw competition table, no hand text, no label or evidence list. The four development tables of the checkpoint that carry
labels have their label columns emptied, and the development prompt keys store each positive pair as an index into the sorted list of
positive pair IDs; both are resolved at run time from your own copy of the competition files.

## 5. Determinism

The replay is byte-identical: we ran it on macOS 15.7 (Apple M4) and in a Kaggle CPU notebook (Linux x86-64), and both reproduced the
md5 of A, B and the base file. The full rebuild is not, for two reasons we measured: polars' multi-threaded `group_by` returns rows in a
run-dependent order (e.g. the candidate-pair and per-(pair, hand) tables of step s01 have identical content but a different row order in
every run), and it sums floats in a run-dependent order (two runs of `tw_pc.py` on identical inputs differ at ~1e-15). Several learners
depend on row order (the seeded sampling of unlabelled pairs, LightGBM bagging and bin sampling, CatBoost Bernoulli subsampling), so the
retrained models differ slightly from the competition-time ones. `tools/compare_submissions.py` quantifies the difference. Our clean-room
run (fresh virtual environment from `requirements.txt`, only the raw tables and the frozen inputs) against the submitted A: risk Spearman
0.981; top-300 / top-600 / top-1000 overlap 288 / 573 / 928; same behaviour label for 94.9% of the pairs; on the submitted top-600 pairs,
95.5% of the evidence lists are identical (4.94 of 5 hands shared on average).
All seeds are fixed; given identical input files every script we re-ran reproduced its original output bit for bit. This includes the
12-seed training of the evidence-decoder detectors (`hb_train3.py`, 60 boosters identical); it is not part of the replay only because it
reads development-side tables that we do not publish (the replay uses the trained detectors from the checkpoint).

## 6. Rules compliance

Coordination is inferred from gameplay only. IDs are join keys (and hash seeds for two cross-fitting splits); chronology comes from
`hands.started_at` (no ties within a table), never from row or file order; `players.parquet` is not read; no evaluation labels and no manual
labelling are used. The external model `YiPz/llama3-8b-pokerbench-sft` (Llama 3 Community License) is used only to score decisions.

## License

MIT (see `LICENSE`). The competition data and the PokerBench-SFT model are subject to their own terms and are not redistributed here.
