# Cloud stages (their outputs are shipped as frozen inputs)

Three feature sources were computed on rented machines during the competition. `run_all.sh` does not re-run them; it reads their
outputs from the frozen-inputs dataset (md5 in `frozen_inputs.md5`). The code is included so that they can be regenerated.
None of them uses any label.

## 1. Normal-behaviour model (`gbnll`) — per-action surprisal and entropy

- Code: `7_reproduce/lambda_0912/cloud/` — `build_tables.py` (decision-state tables from the raw hands/seats/actions and the development
  candidate pairs), `gbfeat.py` (features), `gbnll.py` (LightGBM multiclass models of action type and size), `outputs.py`, `accept.py`.
  `train.py` / `infer.py` / `model.py` / `engine.py` belong to a Transformer variant that was tested and not used.
- Cross-fitting: two halves by `crc32(hand_id) & 1`; each half is scored by the model trained on the other half.
- Command (as run): `build_tables.py --data <data> --out tables` then
  `gbnll.py --tables tables --out out --jobs 22 --train-rows 6000000 --rounds 1000` (stage `gbnll_full` of `run_stage.sh`; learning rate 0.05,
  127 leaves, seed 0, early stopping on a 3% validation split; recorded in `gbnll_train_info.json`).
- Outputs used: `5_outputs/seqnll_0912/gbnll_player.parquet` (per hand × player), `gbnll_action.parquet` (per action:
  `nll_type_N`, `nll_size_N`, `entropy_N`, street index).
- Used by: `tools/build_nllx.py`, `5_outputs/eqx_0913/build_presence_contrast*.py`, `5_outputs/research_0919/band/tw_pc.py`.

## 2. Pair-conditional policy (`pairpol`) — does knowing *who else is at the hand* improve the prediction of a player's actions?

- Code: `7_reproduce/lambda_0913/cloud/pairpol.py` (+ `run_pairpol.sh`, `pairpol_accept.py`, reusing `lambda_0912/cloud/build_tables.py`,
  `gbfeat.py`). An MLP trunk (256-256-128, GELU) on the decision state, an 8-d drift embedding per (player, phase, time quintile) and an 8-d
  embedding per ordered (player, other player, phase), summed over the other players dealt in; additive heads for action type and size.
  Trained on the two `crc32(hand_id) & 1` halves (each half scored by the other half's model), three regularisation strengths for the
  pair embedding, the one with the best held-out likelihood kept; 6 epochs; 1× A10 GPU.
- Command: `run_pairpol.sh full` (`pairpol.py prep ...` then `pairpol.py run ... --epochs 6 --bs 8192`).
- Outputs used: `5_outputs/pairpol_0913/pairpol_pair.parquet` (per ordered pair and phase: log-likelihood-ratio statistics, embedding
  norms) and the per-hand shards `1_data/processed/pairpol/{development,evaluation}/T*.parquet`; weights `pairpol_g{0,1}.pt`.
- Used by: `tools/build_pairpol_feats.py`, `5_outputs/research_0919/band/tw_pairpol.py`.

## 3. PokerBench-SFT decision scoring — log-probabilities of the legal actions

- Model: [`YiPz/llama3-8b-pokerbench-sft`](https://huggingface.co/YiPz/llama3-8b-pokerbench-sft) (Llama 3 Community License).
- Prompts: `5_outputs/pokerbench_0915/build_prompts.py` (development: every decision of a pair member in the candidate hands of the 372
  positive pairs) and `build_prompts_eval.py 8000` (evaluation: the candidate hands of the 8,000 pairs ranked highest by an earlier risk file);
  both serialise the hand in the PokerBench format (positions, stacks, hole cards of the acting player, board, action history).
- Scoring: `score_vllm.py --in <prompts> --shard i/8 --out scores_shard<i>.parquet` on 8× A100 (vllm 0.11.0, transformers 4.57.1,
  temperature 0, top-25 log-probs of the first token after `<action>`), shards concatenated.
- Outputs used: `5_outputs/pokerbench_0915/{prompts,prompts_eval_top8000,scores,scores_eval_top8000}.parquet`.
- Used by: `tools/build_pb_dev.py`, `5_outputs/pokerbench_0915/pokerbench_features_eval.py`.
