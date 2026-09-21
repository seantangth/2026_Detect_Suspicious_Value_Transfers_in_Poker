#!/usr/bin/env bash
# 用法: run_pairpol.sh smoke|full
# 失敗一律非 0 退出；不用 echo 判定成功。
set -euo pipefail
STAGE="${1:?smoke|full}"
R="${TPDS_CLOUD_ROOT:-/home/ubuntu/tpds}"   # (release) machine root, configurable
D="$R/data"
cd "$R/cloud"
export PYTHONUNBUFFERED=1

case "$STAGE" in
  smoke)
    rm -rf "$R/tables_s" "$R/prep_s" "$R/out_s" "$R/shards_s"
    mkdir -p "$R/tables_s" "$R/prep_s" "$R/out_s" "$R/shards_s"
    python3 build_tables.py --data "$D" --out "$R/tables_s" --limit-hands 20000
    python3 pairpol.py prep --tables "$R/tables_s" --data "$D" --prep "$R/prep_s"
    python3 pairpol.py run --tables "$R/tables_s" --data "$D" --prep "$R/prep_s" \
            --out "$R/out_s" --shards "$R/shards_s" --epochs 2 --bs 4096 \
            --score-bs 32768 --eval-actions 100000 --allow-missing-players
    python3 pairpol_accept.py --out "$R/out_s" --prep "$R/prep_s" --tables "$R/tables_s" \
            --data "$D" --extra "$D/extra" --check-hands 60
    echo SMOKE_OK
    ;;
  full)
    mkdir -p "$R/tables" "$R/prep" "$R/out" "$R/shards"
    [ -f "$R/tables/meta.json" ] || python3 build_tables.py --data "$D" --out "$R/tables"
    [ -f "$R/prep/prep_meta.json" ] || \
      python3 pairpol.py prep --tables "$R/tables" --data "$D" --prep "$R/prep"
    python3 pairpol.py run --tables "$R/tables" --data "$D" --prep "$R/prep" \
            --out "$R/out" --shards "$R/shards" --epochs "${EPOCHS:-6}" --bs 8192
    python3 pairpol_accept.py --out "$R/out" --prep "$R/prep" --tables "$R/tables" \
            --data "$D" --extra "$D/extra" --check-hands 100
    cd "$R/out" && md5sum pairpol_pair.parquet manifest.json lambda_selection.json \
        pairpol_acceptance.json pairpol_g0.pt pairpol_g1.pt > md5.txt
    cd "$R" && find shards -name '*.parquet' -print0 | sort -z | xargs -0 md5sum >> "$R/out/md5.txt"
    echo FULL_OK
    ;;
  *) echo "未知階段 $STAGE" >&2; exit 64;;
esac
