# Cloud stages (their outputs are shipped as frozen inputs)

Three feature sources were computed on rented machines during the competition. Their outputs are in the artifacts dataset (md5 in
`frozen_inputs.md5`); `run_all.sh` reads them and does not re-run the stages, and they were not re-run for this release. The code and the
commands that were used are below; after regenerating the outputs, run `run_all.sh` with `TPDS_REGENERATED_CLOUD=1` (README §3). gbnll and pairpol use no labels; the development PokerBench prompts cover the candidate hands of the
372 positive development pairs, on which the evidence rankers are trained.

## Layout and environment

The scripts ran with the machine root `/home/ubuntu/tpds`: `<root>/cloud` held the files of both `7_reproduce/lambda_0912/cloud` and
`7_reproduce/lambda_0913/cloud`, `<root>/data/raw` the competition files and `<root>/data/processed` four tables written by step s01 of
`run_all.sh`. `bash 7_reproduce/cloud_layout.sh <root>` assembles this layout (run step s01 first); `export TPDS_CLOUD_ROOT=<root>` points
`run_stage.sh` and `run_pairpol.sh` to it (default `/home/ubuntu/tpds`). `pairpol.py --help` and the other scripts also run in place.

Environment: Lambda Cloud Ubuntu image with PyTorch and CUDA, plus `polars pyarrow numba lightgbm` (`setup.sh`), NumPy 1.26.4 (the image's
PyTorch is built against NumPy 1.x). A CPU smoke test of the layout, `build_tables.py`, gbnll and pairpol on 20,000 hands (`--device cpu`)
ran on 2026-09-22 with numba 0.67, NumPy 2.5.3, polars 1.44.1, PyTorch 2.14 and LightGBM 4.7; PokerBench scoring needs vLLM and GPUs.

## 1. Normal-behaviour model (`gbnll`): per-action surprisal and entropy

- Code: `7_reproduce/lambda_0912/cloud/`: `build_tables.py` (decision-state tables from the raw hands, seats and actions and the step-s01
  tables), `gbfeat.py` (features), `gbnll.py` (LightGBM multiclass models of action type and size), `outputs.py`, `accept.py` (checks).
  `train.py` / `infer.py` / `model.py` / `engine.py` belong to a Transformer variant that was tested and not used.
- Cross-fitting: two halves by `crc32(hand_id) & 1`; each half is scored by the model trained on the other half.
- Commands (in `<root>/cloud`):
  ```bash
  python3 build_tables.py --data $TPDS_CLOUD_ROOT/data --out $TPDS_CLOUD_ROOT/tables
  python3 gbnll.py --tables $TPDS_CLOUD_ROOT/tables --out $TPDS_CLOUD_ROOT/out --jobs 22 --train-rows 6000000 --rounds 1000
  ```
  (stage `gbnll_full` of `run_stage.sh`, which also runs `accept.py` and `t4_shards.py`; learning rate 0.05, 127 leaves, seed 0, early
  stopping on a 3% validation split; recorded in `gbnll_train_info.json`).
- Outputs used: `<root>/out/gbnll_{player,action}.parquet` → `5_outputs/seqnll_0912/` (per hand × player; per action: `nll_type_N`,
  `nll_size_N`, `p_taken_N`, `entropy_N`).
- Used by: `tools/build_nllx.py`, `5_outputs/eqx_0913/build_presence_contrast*.py`, `5_outputs/research_0919/band/tw_pc.py`.

## 2. Pair-conditional policy (`pairpol`): does knowing *who else is at the hand* improve the prediction of a player's actions?

- Code: `7_reproduce/lambda_0913/cloud/pairpol.py` (+ `run_pairpol.sh`, `pairpol_accept.py`), reusing `build_tables.py` and `gbfeat.py`
  from `lambda_0912/cloud`. An MLP trunk (256-256-128, GELU) on the decision state, an 8-d drift embedding per (player, phase, time
  quintile) and an 8-d embedding per ordered (player, other player, phase), summed over the other players dealt in; additive heads for
  action type and size. Trained on the two `crc32(hand_id) & 1` halves (each half scored by the other half's model), three regularisation
  strengths for the pair embedding, the one with the best held-out likelihood kept; 6 epochs; 1× A10 GPU.
- Commands (in `<root>/cloud`, after `build_tables.py` above):
  ```bash
  python3 pairpol.py prep --tables $TPDS_CLOUD_ROOT/tables --data $TPDS_CLOUD_ROOT/data --prep $TPDS_CLOUD_ROOT/prep
  python3 pairpol.py run --tables $TPDS_CLOUD_ROOT/tables --data $TPDS_CLOUD_ROOT/data --prep $TPDS_CLOUD_ROOT/prep \
      --out $TPDS_CLOUD_ROOT/out --shards $TPDS_CLOUD_ROOT/shards --epochs 6 --bs 8192
  ```
  (stage `full` of `run_pairpol.sh`, which then runs the checks in `pairpol_accept.py`; those read two research-time tables from
  `<root>/data/extra` and can be skipped).
- Outputs used: `<root>/out/{pairpol_pair.parquet,pairpol_g0.pt,pairpol_g1.pt}` → `5_outputs/pairpol_0913/` (per ordered pair and
  phase: log-likelihood-ratio statistics, embedding norms; weights) and `<root>/shards/{development,evaluation}/T*.parquet` →
  `1_data/processed/pairpol/{development,evaluation}/`.
- Used by: `tools/build_pairpol_feats.py`, `5_outputs/research_0919/band/tw_pairpol.py`.

## 3. PokerBench-SFT decision scoring: log-probabilities of the legal actions

- Model: [`YiPz/llama3-8b-pokerbench-sft`](https://huggingface.co/YiPz/llama3-8b-pokerbench-sft) (Llama 3 Community License).
- Prompts (CPU), in the PokerBench format (positions, stacks, hole cards of the acting player, board, action history):
  - development: `python 5_outputs/pokerbench_0915/build_prompts.py` → `prompts.parquet`, `prompts_unique.parquet`: every decision of a
    pair member in the candidate hands of the 372 positive pairs (keys from `tools/build_gate_keys.py`, step s31a);
  - evaluation: `build_prompts_eval.py 8000` → `prompts_eval_top8000.parquet`, `prompts_eval_unique_top8000.parquet`: the candidate hands
    of the 8,000 pairs ranked highest by a research-time risk file. Without the two research-time files it reads the resulting (pair, hand)
    keys from `5_outputs/models/v5nb/evidence_eval_pfonscvxpb4.parquet` (in this repository).
  - Rebuilt this way on 2026-09-22 (from the raw tables, the step-s01 tables and the step-s31a keys), both prompt tables match the
    scored ones row for row.
- Scoring on 8× A100 (`vllm==0.11.0`, `transformers==4.57.1`; temperature 0, top-25 log-probs of the first token after `<action>`):
  ```bash
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python score_vllm.py --in prompts_unique.parquet --shard $i/8 --out scores_shard$i.parquet & done; wait
  python -c "import pandas as pd, glob; pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('scores_shard*.parquet'))]).to_parquet('scores.parquet', index=False)"
  ```
  (evaluation: `--in prompts_eval_unique_top8000.parquet`, concatenated to `scores_eval_top8000.parquet`).
- Outputs used: `5_outputs/pokerbench_0915/{prompts,prompts_eval_top8000,scores,scores_eval_top8000}.parquet`.
- Used by: `tools/build_pb_dev.py`, `5_outputs/pokerbench_0915/pokerbench_features_eval.py`.
