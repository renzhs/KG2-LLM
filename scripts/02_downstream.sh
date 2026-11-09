#!/usr/bin/env bash
# 下游任务：资源推荐 / 学习路径 / 重规划 / 模块消融 / 稀疏分组 / 编码器对照
#
# 说明:
#   · run_rec.py 的 --graph-scope：KT 用 train，推荐/路径用 cut（去泄漏口径），all 为早期口径。
#   · 所有命令都带 --tag，避免覆盖 results/ 里保存的论文报告值。
#   · GES（--task ges）需要 DEEPSEEK_API_KEY，默认不跑；需要时 bash scripts/02_downstream.sh ges
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DS="${DS:-dnui_syn}"
WHAT="${1:-all}"

if [[ ! -f "data/processed/${DS}.pkl" ]]; then
  echo "缺少 data/processed/${DS}.pkl —— 先跑 bash scripts/00_prepare_data.sh"
  exit 1
fi

run_rec() {
  echo; echo "===== 资源推荐：${DS}（graph-scope=all，主口径）====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task rec --graph-scope all --tag repro
  echo; echo "===== CSG 归因：门控是否只做『槽位释放』====="
  $PY experiments/tools/csg_attribution.py --dataset "$DS"
}

run_path() {
  echo; echo "===== 学习路径推荐：${DS} ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task path --graph-scope cut --tag repro
  echo; echo "===== 动态重规划：${DS} ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task mdrl --tag repro
}

run_abl() {
  echo; echo "===== 模块消融（3 种子）：${DS} ====="
  $PY experiments/run/run_abl_multi.py --dataset "$DS" --seeds 42,43,44
  echo; echo "===== 稀疏/稠密知识点分组评测：${DS} ====="
  $PY experiments/run/run_sparse_eval.py --dataset "$DS" --seeds 42,43,44
}

run_enc() {
  echo; echo "===== 序列通路编码器对照 ====="
  $PY experiments/run/run_enc_ablation.py --datasets dnui_syn,assist09
  $PY experiments/tools/make_enc_table.py
}

run_ges() {
  echo; echo "===== GES（需 DEEPSEEK_API_KEY 或命中磁盘缓存）====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task ges --llm-students 40 --tag repro
}

case "$WHAT" in
  all)  run_rec; run_path; run_abl; run_enc ;;
  rec)  run_rec ;;
  path) run_path ;;
  abl)  run_abl ;;
  enc)  run_enc ;;
  ges)  run_ges ;;
  *) echo "未知参数：$WHAT（可选 all|rec|path|abl|enc|ges）"; exit 1 ;;
esac

echo
echo "===== DONE：结果见 results/*_repro.json ====="
