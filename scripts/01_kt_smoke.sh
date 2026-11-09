#!/usr/bin/env bash
# 冒烟训练：DNUI-IKG 上把 11 个知识追踪模型各跑 8 epoch，验证环境可用。
# CPU 也能跑（约数分钟），GPU 更快。
#
# 用法:
#   bash scripts/01_kt_smoke.sh                 # 8 epoch，全部模型
#   EPOCHS=2 bash scripts/01_kt_smoke.sh        # 更快
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
EPOCHS="${EPOCHS:-8}"

if [[ ! -f data/processed/dnui_syn.pkl ]]; then
  echo "未找到 data/processed/dnui_syn.pkl，先跑数据准备 ..."
  $PY experiments/data/prep_datasets.py --only dnui
fi

echo "===== 冒烟：dnui_syn x 11 模型 x ${EPOCHS} epoch ====="
# --tag smoke 保证结果写到 results/kt_dnui_syn_smoke.json，不覆盖论文报告值
$PY experiments/run/run_kt.py --dataset dnui_syn --models all \
    --epochs "$EPOCHS" --patience "$EPOCHS" --tag smoke

echo
echo "===== 结果（results/kt_dnui_syn_smoke.json）====="
$PY - <<'EOF'
import json, os
p = "results/kt_dnui_syn_smoke.json"
d = json.load(open(p, encoding="utf-8"))
rows = d.get("results") or d.get("models") or d
print(f"{'model':<10}{'AUC':>8}{'ACC':>8}{'RMSE':>8}")
for k, v in rows.items():
    if not isinstance(v, dict) or "AUC" not in v:
        continue
    print(f"{k:<10}{v['AUC']:>8.4f}{v.get('ACC', float('nan')):>8.4f}{v.get('RMSE', float('nan')):>8.4f}")
EOF
