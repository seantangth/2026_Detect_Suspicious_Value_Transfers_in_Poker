#!/usr/bin/env bash
# Tier-1 check: re-run the final model layer of the selected submissions from the published checkpoint and compare md5.
#   re-trained here: stage-1 pair ensemble (LightGBM + CatBoost, arms F and FW), stage-2 head re-ranker + D' rule, family classifier;
#   re-applied here: evidence-listing decoder (12-seed detectors, evaluation inference and decoding), final assembly.
# Inputs: raw competition data (1_data/raw/...) + checkpoint (unpacked at the repository root, see README.md).
# Expected: all three md5 checks report "identical to the submitted file". Runtime ~40 min on 10 cores (Apple M4), ~2 h 10 min on a 4-core Kaggle CPU notebook.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-6}
mkdir -p logs_replay 5_outputs/submissions 5_outputs/revise_0917
RAWD=1_data/raw/detect-suspicious-value-transfers-in-poker
SUB=5_outputs/submissions
step() {
  local name=$1; shift
  echo "[$(date '+%H:%M:%S')] >>> $name"
  local t0=$SECONDS
  if "$@" > "logs_replay/$name.log" 2>&1; then echo "[$(date '+%H:%M:%S')] <<< $name ok ($((SECONDS - t0)) s)"
  else echo "!!! $name FAILED (see logs_replay/$name.log)"; tail -20 "logs_replay/$name.log"; exit 1; fi
}
md5of() { "$PY" -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
expect_md5() { local got; got=$(md5of "$1"); if [ "$got" = "$2" ]; then echo "[identical to the submitted file] $3: $got"; else echo "[DIFFERENT] $3: got $got, expected $2"; fi; }

step r0_check_inputs   "$PY" tools/check_inputs.py raw_inputs.md5 checkpoint.md5
step r1_attach_labels  "$PY" tools/attach_dev_labels.py
step r2_tw_stage1      "$PY" 5_outputs/research_0919/band/tw_chain1.py full F,FW
step r3_tw_stage2_F    "$PY" 5_outputs/research_0919/band/tw_chain2.py F "$RAWD/sample_submission.csv"
step r4_family_clf     "$PY" 5_outputs/revise_0917/family_clf_eval.py
step r5_final_d2       "$PY" 5_outputs/audit_0917/build_final.py "$SUB/submission_TWFDp_sample_submission.csv" "$SUB/submission_FINAL_D2_v040ev_famclf.csv" 5_outputs/models/v5nb/submission_v040cand.csv 0.2
step r6_hb_apply       "$PY" 5_outputs/research_0919/hb_apply3.py tag=hb3s12 base=submission_FINAL_D2_v040ev_famclf.csv out=submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv topn=3000 w=0.9 kdt=3
step r7_tw_stage2_FW   "$PY" 5_outputs/research_0919/band/tw_chain2.py FW "$SUB/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv"
cp "$SUB/submission_TWFWDp_HB3k12_FINALD2_split_f2p_kdt3_w90.csv" submission.csv
echo "================ checks"
expect_md5 "$SUB/submission_FINAL_D2_v040ev_famclf.csv"            92de95cce5ae8d8451f9f77b6f5e1a09 "base file FINAL-D2 (not submitted)"
expect_md5 "$SUB/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv" f17b3b4de833994b8ec994546f3f802b "selected submission B (private 0.92414)"
expect_md5 submission.csv                                           c303970ed74faa833cf671ebdfd4efb6 "selected submission A (private 0.92677) = submission.csv"
