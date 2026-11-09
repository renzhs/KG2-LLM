#!/usr/bin/env bash
# 实验收尾：聚合各数据集结果 + 生成论文用 LaTeX 表。
#
# 与 run_bench.sh 阶段C 的差别：这里额外调用 make_paper_tables.py，
# 产出 table_dataset.tex / table_kt_bench.tex / table_kt_pub.tex 与对应的 Markdown 表。
#
# 用法（远端）: bash experiments/run/make_tables.sh
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"   # 可用 PY=/path/to/python 覆盖

DS_LIST_5F="${DS_LIST_5F:-assist09_akt assist15 assist17 statics11}"
DS_LIST_1F="${DS_LIST_1F:-dnui_syn assist09 assist12}"

echo "===== 聚合 5 折数据集 ====="
for DS in $DS_LIST_5F; do
  $PY experiments/tools/aggregate_bench.py --dataset "$DS" --folds 1,2,3,4,5 2>&1 | tail -16
done

echo
echo "===== 聚合单划分数据集 ====="
for DS in $DS_LIST_1F; do
  $PY experiments/tools/aggregate_bench.py --dataset "$DS" --single 2>&1 | tail -16
done

echo
echo "===== 生成论文表 ====="
$PY experiments/tools/make_paper_tables.py 2>&1 | tail -20

echo
echo "===== 产物 ====="
ls -la results/table_*.tex results/table_*.md results/kt_bench_*.json 2>/dev/null
echo "===== DONE ====="
