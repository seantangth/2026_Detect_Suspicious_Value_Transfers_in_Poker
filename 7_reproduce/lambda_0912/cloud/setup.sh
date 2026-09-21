#!/usr/bin/env bash
# 雲端環境準備。失敗就非 0 退出，不靠 echo 判定成功。
set -euo pipefail
echo "== nvidia-smi =="; nvidia-smi
echo "== python/torch =="
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
PIP=(python3 -m pip install --quiet --no-input)
"${PIP[@]}" -U polars pyarrow numba 2>/dev/null || "${PIP[@]}" --break-system-packages -U polars pyarrow numba
python3 -c "import polars,pyarrow,numba,numpy;print('polars',polars.__version__,'pyarrow',pyarrow.__version__,'numba',numba.__version__,'numpy',numpy.__version__)"
echo "== 資料 md5（與本機核對）=="
cd /home/ubuntu/tpds/data && md5sum raw/*.parquet raw/*.csv processed/*.parquet
echo "== 磁碟 =="; df -h /home/ubuntu | tail -1
echo SETUP_OK
