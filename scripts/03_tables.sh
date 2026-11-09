#!/usr/bin/env bash
# 从随仓库发布的 results/ 快照重建论文表 1-4，并与 results/paper_tables/ 逐行比对。
# 全过程不需要 GPU、不需要重新训练（约 10 秒）。
#
# 用法:
#   bash scripts/03_tables.sh                 # 复核快照（默认，推荐）
#   bash scripts/03_tables.sh --reaggregate   # 刚重跑过实验，先重新聚合逐折结果再建表
#
# 说明:
#   · 快照模式下会跳过"多种子产物是否比主结果新"的 mtime 守卫——git clone 后
#     所有文件的 mtime 都是检出时间，该守卫会失效并误判为陈旧（见 make_paper_tables.py）。
#   · 建表会覆盖 results/table_*.tex 与 results/kt_signif.json；
#     若要还原：git checkout -- results/
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
AGG="${1:-}"

if [[ "$AGG" == "--reaggregate" ]]; then
  echo "===== [可选] 重新聚合逐折结果 ====="
  for DS in assist09_akt assist15 assist17 statics11; do
    $PY experiments/tools/aggregate_bench.py --dataset "$DS" --folds 1,2,3,4,5 2>&1 | tail -3
  done
  for DS in dnui_syn assist09 assist12; do
    $PY experiments/tools/aggregate_bench.py --dataset "$DS" --single 2>&1 | tail -3
  done
  echo
fi

echo "===== [1/3] 写表：dataset / kt_pub / kt_bench / kt_ours ====="
$PY experiments/tools/make_paper_tables.py --frozen 2>&1 | tail -12

echo
echo "===== [2/3] 重算配对检验（DeltaAUC / 95%CI / p / MDES）====="
$PY experiments/tools/kt_significance.py 2>&1 | tail -18

echo
echo "===== [3/3] 与投稿版快照比对 ====="
status=0
for f in table_dataset.tex table_kt_pub.tex table_kt_bench.tex table_kt_ours.tex; do
  if [[ -f "results/paper_tables/$f" && -f "results/$f" ]]; then
    if diff -q "results/paper_tables/$f" "results/$f" >/dev/null; then
      echo "  [一致]   $f"
    else
      echo "  [不一致] $f  —— 逐行差异：diff results/paper_tables/$f results/$f"
      status=1
    fi
  else
    echo "  [跳过]   $f（缺文件）"
  fi
done

echo
if [[ $status -eq 0 ]]; then
  echo "===== 全部一致：仓库内保存的逐折结果可以重建论文表格 ====="
else
  echo "===== 存在差异：请检查是否改动了 results/ 下的逐折结果，或聚合口径（折数/单划分）====="
fi
exit $status
