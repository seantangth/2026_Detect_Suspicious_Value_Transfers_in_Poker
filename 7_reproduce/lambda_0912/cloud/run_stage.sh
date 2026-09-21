#!/usr/bin/env bash
# 用法: run_stage.sh smoke|throughput|full|gbnll_full
set -euo pipefail
STAGE="${1:?smoke|throughput|full|gbnll_full}"
R=/home/ubuntu/tpds
D="$R/data"
cd "$R/cloud"
export PYTHONUNBUFFERED=1
T4ARGS_COMMON="--hands $D/raw/hands.parquet --seats $D/raw/seats.parquet \
 --cand-dev $D/processed/cand_pairs_development.parquet --cand-eval $D/raw/evaluation_pairs.csv"

case "$STAGE" in
  smoke)
    rm -rf "$R/tables_s" "$R/ckpt_s" "$R/out_s"; mkdir -p "$R/tables_s" "$R/ckpt_s" "$R/out_s"
    python3 build_tables.py --data "$D" --out "$R/tables_s" --limit-hands 20000
    for g in 0 1; do
      python3 train.py --tables "$R/tables_s" --out "$R/ckpt_s" --group $g --epochs 1 --bs 256 --eval-hands 2000
    done
    python3 infer.py --tables "$R/tables_s" --ckpt "$R/ckpt_s" --out "$R/out_s" --bs 256 --chunk-hands 20000
    python3 checks.py --tables "$R/tables_s" --ckpt "$R/ckpt_s" --data "$D" --out "$R/out_s" \
            --recon-hands 200 --causal-hands 50 --other-hands 128
    python3 accept.py --data "$D" --out "$R/out_s" --tables "$R/tables_s" --ckpt "$R/ckpt_s"
    python3 gbnll.py --tables "$R/tables_s" --out "$R/out_s" --train-rows 200000 --rounds 60 --jobs 24
    python3 accept.py --data "$D" --out "$R/out_s" --tables "$R/tables_s" --ckpt "$R/ckpt_s" --prefix gbnll
    for P in seqnll gbnll; do
      python3 t4_shards.py --t2 "$R/out_s/${P}_player.parquet" --t3 "$R/out_s/${P}_pair.parquet" \
              $T4ARGS_COMMON --out "$R/out_s" --prefix $P || echo "  （smoke 子集列數本來就對不上，忽略）"
    done
    python3 - <<'PY'
import pyarrow.parquet as pq, json, glob, os
R='/home/ubuntu/tpds'; m=json.load(open(R+'/tables_s/meta.json'))
for P in ('seqnll','gbnll'):
    for f,e in [(f'{P}_action.parquet',m['N_A']),(f'{P}_player.parquet',m['N_S']),(f'{P}_pair.parquet',m['N_H']*30)]:
        n=pq.ParquetFile(f'{R}/out_s/{f}').metadata.num_rows
        assert n==e,(f,n,e); print('smoke 列數 OK',f,n)
    for ph in ('development','evaluation'):
        g=glob.glob(f'{R}/out_s/nll_{P}/{ph}/*.parquet'); print(f'  {P} {ph} 分片 {len(g)} 個')
print('meta invested_bb 交叉驗證:', m.get('invested_bb_max_abs_diff'), m.get('invested_bb_mismatch_gt_1e3'))
print('meta 交叉擬合:', json.dumps(m.get('crossfit')))
PY
    echo SMOKE_OK
    ;;
  throughput)
    mkdir -p "$R/tables" "$R/ckpt"
    python3 train.py --tables "$R/tables" --out "$R/ckpt" --group 0 --epochs 1 --bs 1024 --throughput-only
    ;;
  full)
    mkdir -p "$R/tables" "$R/ckpt" "$R/out"
    [ -f "$R/tables/meta.json" ] || python3 build_tables.py --data "$D" --out "$R/tables"
    for g in 0 1; do
      python3 train.py --tables "$R/tables" --out "$R/ckpt" --group $g \
              --epochs "${EPOCHS:-6}" --bs 1024 --eval-hands 50000
    done
    python3 infer.py --tables "$R/tables" --ckpt "$R/ckpt" --out "$R/out" --bs 1024
    python3 checks.py --tables "$R/tables" --ckpt "$R/ckpt" --data "$D" --out "$R/out" \
            --recon-hands 1000 --causal-hands 100 --other-hands 512
    python3 accept.py --data "$D" --out "$R/out" --tables "$R/tables" --ckpt "$R/ckpt" --strict
    python3 t4_shards.py --t2 "$R/out/seqnll_player.parquet" --t3 "$R/out/seqnll_pair.parquet" \
            $T4ARGS_COMMON --out "$R/out" --prefix seqnll
    cp "$R/ckpt"/g0_hist.json "$R/ckpt"/g1_hist.json "$R/tables"/meta.json "$R/out"/ 2>/dev/null || true
    mv "$R/out/meta.json" "$R/out/tables_meta.json" 2>/dev/null || true
    echo FULL_SEQNLL_OK
    ;;
  gbnll_full)
    mkdir -p "$R/out"
    python3 gbnll.py --tables "$R/tables" --out "$R/out" --jobs "${JOBS:-24}" \
            --train-rows "${TRAINROWS:-6000000}" --rounds "${ROUNDS:-1000}"
    python3 accept.py --data "$D" --out "$R/out" --tables "$R/tables" --ckpt "$R/ckpt" --prefix gbnll --strict
    python3 t4_shards.py --t2 "$R/out/gbnll_player.parquet" --t3 "$R/out/gbnll_pair.parquet" \
            $T4ARGS_COMMON --out "$R/out" --prefix gbnll
    echo FULL_GBNLL_OK
    ;;
  *) echo "未知階段 $STAGE" >&2; exit 64;;
esac
