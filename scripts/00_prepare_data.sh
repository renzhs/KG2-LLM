#!/usr/bin/env bash
# 数据准备：自建数据集 → .pkl；公开基准按需下载 → .pkl
#
# 用法:
#   bash scripts/00_prepare_data.sh            # 只处理自建 DNUI-IKG（离线可用）
#   bash scripts/00_prepare_data.sh --public   # 额外下载并切分 4 个公开基准
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"

echo "===== [1/3] 检查目录 ====="
$PY experiments/paths.py

echo
echo "===== [2/3] 自建数据集 DNUI-IKG ====="
$PY experiments/data/prep_datasets.py --only dnui

if [[ "${1:-}" == "--public" ]]; then
  echo
  echo "===== [3/3] 公开基准（ASSIST09/15/17、Statics2011）====="
  echo "  提示：原始数据受各自许可约束，脚本只做镜像下载；请自行确认许可条款。"
  $PY experiments/tools/download_datasets.py
  echo "  AKT 官方 5 折划分需要 data/public/AKT_repo/data/{assist2009_pid,assist2015,assist2017_pid,statics2011}/"
  echo "  若缺失，请按 https://github.com/arghosh/AKT 的说明放置后再执行下一行。"
  $PY experiments/data/prep_bench.py --only assist09,assist15,assist17,statics11
else
  echo
  echo "===== 跳过公开基准（加上 --public 可一并处理）====="
fi

echo
echo "===== 产物 ====="
ls -lh data/processed | tail -n +2
