#!/usr/bin/env bash
# 完整主表复现：调参 → 5 折 × 11 模型 → 聚合。单张 V100 约数天，建议在 tmux/nohup 中运行。
#
# 用法:
#   bash scripts/04_full_bench.sh
#   PAR=1 bash scripts/04_full_bench.sh        # 显式指定并行度（默认 1，见下）
#   FOLDS="1 2 3 4 5" MODELS=all bash scripts/04_full_bench.sh
#
# 为什么默认 PAR=1:
#   单卡并发会在显存紧张时触发 CUDA error（实测 GKT 在 22 个"数据集×折"中有 16 个 OOM），
#   而 OOM 期间该折可能只写入前若干模型，形成"矩阵残缺但文件存在"的静默失效。
#   有时间就调大 PAR，但请务必让 run_bench.sh 的完整性门禁跑完再聚合。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PAR="${PAR:-1}"

echo "===== 前置检查：公开基准是否就绪 ====="
missing=0
for d in assist09_akt_f1 assist15_f1 assist17_f1 statics11_f1 dnui_syn; do
  [[ -f "data/processed/${d}.pkl" ]] || { echo "  缺少 data/processed/${d}.pkl"; missing=1; }
done
if [[ $missing -eq 1 ]]; then
  echo "请先执行：bash scripts/00_prepare_data.sh --public"
  exit 1
fi
echo "  OK"

echo
echo "===== 开始 run_bench.sh（阶段A 调参 → 阶段A2 基线调参 → 阶段B 5折×11模型 → 阶段C 聚合）====="
echo "  并行度 PAR=${PAR}；日志写入 logs/（不入库）"
echo "  过程中任何门禁失败都会中止并打印缺口，不要绕过它聚合。"
mkdir -p logs
bash experiments/run/run_bench.sh 2>&1 | tee "logs/run_bench_$(date +%Y%m%d_%H%M%S).log"

echo
echo "===== 完成：结果在 results/kt_bench_*.json 与 results/table_kt_*.tex ====="
echo "  接下来可跑 bash scripts/03_tables.sh 复核表格一致性。"
