#!/usr/bin/env bash
# Full rebuild of the two selected submissions from the raw competition data plus the frozen inputs (see README.md, "Full rebuild").
#   selected A = 5_outputs/submissions/submission_TWFWDp_HB3k12_FINALD2_split_f2p_kdt3_w90.csv  (submitted file: md5 c303970ed74faa833cf671ebdfd4efb6)
#   selected B = 5_outputs/submissions/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv         (submitted file: md5 f17b3b4de833994b8ec994546f3f802b)
# The full rebuild is statistically equivalent to the submitted files, not byte-identical: polars' multi-threaded group_by returns rows in a
# run-dependent order and sums floats in a run-dependent order, and several learners (sampling of unlabelled pairs, LightGBM bagging and
# bin sampling, CatBoost Bernoulli subsampling) depend on row order. replay.sh reproduces the submitted files byte for byte from the checkpoint.
# Steps are idempotent: a finished step leaves logs/<step>.done and is skipped on the next run (delete the marker to redo it).
# Run one heavy step at a time; peak RSS is about 6 GB on a 16 GB machine.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-6}
mkdir -p logs 5_outputs/submissions 5_outputs/eqx_0913 5_outputs/revise_0917 5_outputs/revise_0915
RAWD=1_data/raw/detect-suspicious-value-transfers-in-poker
SUB=5_outputs/submissions

step() {   # step <name> <command...>
  local name=$1; shift
  if [ -f "logs/$name.done" ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date '+%H:%M:%S')] >>> $name: $*"
  local t0=$SECONDS
  if "$@" > "logs/$name.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] <<< $name ok ($((SECONDS - t0)) s)"; touch "logs/$name.done"
  else
    echo "[$(date '+%H:%M:%S')] !!! $name FAILED (see logs/$name.log)"; tail -20 "logs/$name.log"; exit 1
  fi
}
md5of() { "$PY" -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
expect_md5() {   # expect_md5 <file> <md5> <label>
  local got; got=$(md5of "$1")
  if [ "$got" = "$2" ]; then echo "[identical to the submitted file] $3: $got"; else echo "[rebuilt, differs from the submitted file as expected for a full rebuild] $3: $got (submitted: $2); compare with tools/compare_submissions.py"; fi
}

step s00_check_inputs      "$PY" tools/check_inputs.py

# ---------------------------------------------------------------- 1. base tables (hands, seats, actions -> per pair-hand rows)
step s01_features          "$PY" 3_src/tpds_features.py l2
# ---------------------------------------------------------------- 2. equity (Monte-Carlo world and exact world)
step s02_equity            "$PY" 3_src/tpds_equity.py
step s03_preflop_eq        "$PY" 3_src/tpds_preflop_eq.py
step s04_equity_exact      env TPDS_EQTAG=x "$PY" 3_src/tpds_equity_exact.py development evaluation workers=5 nmc=10000
# ---------------------------------------------------------------- 3. population action (policy) models -> per-action surprise
step s05_policy            "$PY" 3_src/tpds_policy.py
step s06_policy_nb         env TPDS_VARIANT=nb "$PY" 3_src/tpds_policy.py drop_n_board=1
step s07_policy_nbcf       env TPDS_VARIANT=nbcf "$PY" 3_src/tpds_policy.py drop_n_board=1 crossfit=1
# ---------------------------------------------------------------- 4. per pair-hand feature rows with equity + surprise columns
step s08a_augment_default  "$PY" 3_src/tpds_augment.py development evaluation
step s08_augment_x         env TPDS_EQTAG=x "$PY" 3_src/tpds_augment.py development evaluation
step s09_augment_nb        env TPDS_VARIANT=nb "$PY" 3_src/tpds_augment.py
step s10_augment_nb_x      env TPDS_VARIANT=nb TPDS_EQTAG=x "$PY" 3_src/tpds_augment.py development evaluation
# ---------------------------------------------------------------- 5. side tables
step s11_betsize           "$PY" 3_src/tpds_betsize.py
step s12_direction         "$PY" 3_src/tpds_direction.py
step s13_callvalue_x       env TPDS_EQTAG=x "$PY" 3_src/tpds_callvalue.py development evaluation
# ---------------------------------------------------------------- 6. hand-level models + pooled pair features (two worlds)
step s14_model_v5x         env TPDS_EQTAG=x TPDS_HAND_FEATS=5_outputs/models/v5/hand_features.json "$PY" 3_src/tpds_model.py v5x unknown_per_table=30 hand_rounds=350 use_equity=2 use_surprise=2
step s15_model_v5nb        env TPDS_VARIANT=nb "$PY" 3_src/tpds_model.py v5nb unknown_per_table=30 hand_rounds=350 use_equity=2 use_surprise=2
step s16_pair_v5nb         env TPDS_VARIANT=nb "$PY" 3_src/tpds_pair_stage.py v5nb bs w_unknown=0.5 pu_stage2=0 seeds=3 use_pairstats=0 drop_suspects=1 suspect_w=0.0 use_betsize=1
# ---------------------------------------------------------------- 7. pair-level feature families
step s17_dom               "$PY" tools/build_dom.py
step s18_nllx              "$PY" tools/build_nllx.py
step s19_presence_x        env PC_TAG=_x PC_NLLX=1 "$PY" 5_outputs/eqx_0913/build_presence_contrast.py
step s20_presence_v2       "$PY" 5_outputs/eqx_0913/build_presence_contrast_v2.py
step s21_pairpol_feats     "$PY" tools/build_pairpol_feats.py
# ---------------------------------------------------------------- 8. pair model: truncated-window training world, two-stage LGB+CatBoost chain
step s22_tw_base_W1        "$PY" 5_outputs/research_0919/band/tw_base.py W1 0 2000
step s23_tw_base_W2        "$PY" 5_outputs/research_0919/band/tw_base.py W2 1000 3000
step s24_tw_presence       "$PY" 5_outputs/research_0919/band/tw_pc.py pc
step s25_tw_presence_v2    "$PY" 5_outputs/research_0919/band/tw_pc.py pc2
step s26_tw_pairpol        "$PY" 5_outputs/research_0919/band/tw_pairpol.py
step s27_tw_stage1         "$PY" 5_outputs/research_0919/band/tw_chain1.py full F,FW
step s28_tw_stage2_F       "$PY" 5_outputs/research_0919/band/tw_chain2.py F "$RAWD/sample_submission.csv"
# ---------------------------------------------------------------- 9. behaviour (family classifier)
step s29_family_clf        "$PY" 5_outputs/revise_0917/family_clf_eval.py
# ---------------------------------------------------------------- 10. evidence hand features: PokerBench surprisal + corrected N/I
step s30_pb_dev            "$PY" tools/build_pb_dev.py
step s31_pb_eval           "$PY" 5_outputs/pokerbench_0915/pokerbench_features_eval.py
step s31a_gate_keys       "$PY" tools/build_gate_keys.py
step s32_ni_dev            "$PY" 5_outputs/revise_0915/ni_corrected2.py
step s33_pbni_dev          "$PY" 5_outputs/pb_variants_0916/pbni_dev_table.py
step s34_ni_eval           "$PY" 5_outputs/pb_variants_0916/ni_eval.py
# ---------------------------------------------------------------- 11. evidence rankers (LambdaRank, generic + per family, 2 seed sets each)
EV="env TPDS_VARIANT=nb TPDS_EQTAG=x TPDS_CALLVAL=1 TPDS_PB=1 TPDS_PB_TAG=ni"
RK="use_equity=2 use_surprise=2 stack=0 seeds=2 leaves=63 rounds=500 lr=0.05 onset=3"
step s35_ranker_gen_s42    $EV "$PY" 3_src/tpds_evidence.py v5nb $RK seed0=42 tag=onscvxpbni
step s36_ranker_pf_s42     $EV "$PY" 3_src/tpds_evidence.py v5nb $RK per_family=1 seed0=42 tag=pfonscvxpbni
step s37_ranker_gen_s44    $EV "$PY" 3_src/tpds_evidence.py v5nb $RK seed0=44 tag=onscvxpbni_s2
step s38_ranker_pf_s44     $EV "$PY" 3_src/tpds_evidence.py v5nb $RK per_family=1 seed0=44 tag=pfonscvxpbni_s2
step s39_merge_gen         "$PY" 7_reproduce/merge_seed_sets.py v5nb onscvxpbni onscvxpbni_s2 onscvxpbni4
step s40_merge_pf          "$PY" 7_reproduce/merge_seed_sets.py v5nb pfonscvxpbni pfonscvxpbni_s2 pfonscvxpbni4
step s41_apply_gen         $EV "$PY" 3_src/tpds_evidence_apply.py v5nb evonscvxpbni4 use_equity=2 use_surprise=2 model_tag=onscvxpbni4 onset=3
step s42_apply_pf          $EV "$PY" 3_src/tpds_evidence_apply.py v5nb evpfonscvxpbni4 use_equity=2 use_surprise=2 model_tag=pfonscvxpbni4 onset=3
step s43_evidence_v040     $EV "$PY" 3_src/tpds_submit.py v5nb v040cand risk_tag=bs ev=1 active_frac=0.2 ev_tag=pfonscvxpbni4 ev2_tag=onscvxpbni4 w_rank=0.55 w_fam=0.15 w_gen=0.0 w_rank2=0.30 rerank=0 dir_k=5
# ---------------------------------------------------------------- 12. assembly of the base file (production risk + v040 evidence + family labels)
step s44_final_d2          "$PY" 5_outputs/audit_0917/build_final.py "$SUB/submission_TWFDp_sample_submission.csv" "$SUB/submission_FINAL_D2_v040ev_famclf.csv" 5_outputs/models/v5nb/submission_v040cand.csv 0.2
# ---------------------------------------------------------------- 13. evidence-listing decoder (HB3k, 12 seeds) -> selected submission B
step s45_hb_train          "$PY" 5_outputs/research_0919/hb_train3.py tag=hb3s12 seed0=42 seeds=12 fit=1 split=1 f2p=1 kdt=3 w=0.9
step s46_hb_apply          "$PY" 5_outputs/research_0919/hb_apply3.py tag=hb3s12 base=submission_FINAL_D2_v040ev_famclf.csv out=submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv topn=3000 w=0.9 kdt=3
# ---------------------------------------------------------------- 14. truncated-window risk (FW arm + D' rule) spliced onto B -> selected submission A
step s47_tw_stage2_FW      "$PY" 5_outputs/research_0919/band/tw_chain2.py FW "$SUB/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv"

cp "$SUB/submission_TWFWDp_HB3k12_FINALD2_split_f2p_kdt3_w90.csv" submission.csv
echo "================ checks"
expect_md5 "$SUB/submission_FINAL_D2_v040ev_famclf.csv"               92de95cce5ae8d8451f9f77b6f5e1a09 "base file FINAL-D2 (not submitted)"
expect_md5 "$SUB/submission_HB3k12_FINALD2_split_f2p_kdt3_w90.csv"    f17b3b4de833994b8ec994546f3f802b "selected submission B (private 0.92414)"
expect_md5 submission.csv                                              c303970ed74faa833cf671ebdfd4efb6 "selected submission A (private 0.92677) = submission.csv"
"$PY" 3_src/validate_submission.py submission.csv
