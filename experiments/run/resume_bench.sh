#!/usr/bin/env bash
# 阶段B 缺口补跑：只重算「缺模型/失败模型」的数据集×折，其余模型靠 run_kt.py 的
# 模型级续跑（--no-reuse 关闭时默认开启）原样复用，因此重入一个折的代价只是补缺口。
#
# 为什么需要它
#   run_kt.py 逐模型增量落盘，任一模型抛异常（最典型 CUDA OOM）也会生成 JSON，
#   旧的阶段B 判据"JSON 存在即跳过"于是把该折永久跳过 —— 表现为表里凭空少一个基线，
#   且**完全静默**。本脚本用 tools/check_bench.py 定位缺口，逐个补齐。
#
# 用法:
#   bash experiments/run/resume_bench.sh                       # 补所有缺口
#   MODELS=GKT PAR=3 bash experiments/run/resume_bench.sh      # 只补 GKT，3 路并行
#   ONLY=assist15 PAR=3 bash experiments/run/resume_bench.sh   # 只补某数据集
#
# 并行度建议: GKT 是**内核启动受限**（单批耗时与批大小无关、与 T 成正比），
#   在多进程共享单卡时会被严重拖慢 —— 实测 8 路并行下单折 3.9h，
#   而单批微基准外推仅约 440s。故补跑 GKT 时请用 PAR=2~3，不要用 8。
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-python3}"   # 可用 PY=/path/to/python 覆盖
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
PAR="${PAR:-3}"
MODELS="${MODELS:-}"
ONLY="${ONLY:-}"
DS5="${DS5:-assist15 assist09_akt assist17 statics11}"
DS1="${DS1:-assist12 assist09 dnui_syn}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------------------------------
# 1. 定位缺口：check_bench.py --quiet 逐行输出「缺口 <unit> k/n 缺:A,B」
# --------------------------------------------------------------------------
FORCE="${FORCE:-0}"
if [ "$FORCE" = "1" ]; then
  # 强制重跑全部单元（忽略"已完整"状态）。用于 GKT 这类**实现层提速后需要整行重算**
  # 以保证同一模型在各数据集/折上的批大小一致（否则同一 5 折均值的各折口径不同）。
  NEED="${MODELS:-GKT}"
  GAPS=""
  for u in $DS5; do for k in 1 2 3 4 5; do GAPS="$GAPS ${u}_f${k}|$NEED"$'\n'; done; done
  for u in $DS1; do GAPS="$GAPS ${u}|$NEED"$'\n'; done
  echo "[resume] FORCE=1 强制重跑全部单元，models=$NEED"
else
  GAPS=$($PY experiments/tools/check_bench.py --datasets "$DS5" --single-datasets "$DS1" \
          --quiet 2>/dev/null | sed -n 's/^\[check\] 缺口 \([^ ]*\) .*缺:\(.*\)$/\1|\2/p')
  if [ -z "$GAPS" ]; then
    echo "[resume] 无缺口，阶段B 已完整"
    exit 0
  fi
fi

# 按**单次开销**降序下发，使长杆先占用并行槽（否则长尾会把 total 拖长一整段）。
# 开销 ≈ 用户数 × K 量级：assist12 18998×198 最大，其次 assist15 11452×100、
# assist09_akt 2329×110、statics11 201×1223（K 极大，按 K 当量折算）、
# assist17 1026×102、assist09 2696×149、dnui_syn 1502×66。
cost() {
  case "$1" in
    assist12*)      echo 3762000 ;;
    assist15*)      echo 1145200 ;;
    statics11*)     echo  245900 ;;
    assist09_akt*)  echo  256190 ;;
    assist09)       echo  401700 ;;
    assist17*)      echo  104650 ;;
    dnui_syn*)      echo   99100 ;;
    *)              echo       0 ;;
  esac
}
GAPS=$(while IFS='|' read -r u n; do
         [ -z "${u:-}" ] && continue
         echo "$(cost "$u")|$u|$n"
       done <<< "$GAPS" | sort -t'|' -k1,1nr | cut -d'|' -f2-)
echo "[resume] 待补单元（按开销降序）:"
echo "$GAPS" | sed 's/^/   /'

# --------------------------------------------------------------------------
# 2. 逐个补齐（模型级续跑保证只算缺口模型）
# --------------------------------------------------------------------------
do_unit() {
  unit="$1"; need="$2"
  if [[ "$unit" =~ _f[0-9]+$ ]]; then
    base="${unit%_f*}"
    cfg="results/kt_model_cfg_${base}_f1.json"
  else
    cfg="results/kt_model_cfg_${unit}.json"
  fi
  extra=""
  [ -f "$cfg" ] && extra="--model-cfg-file $cfg"
  # 只跑缺口模型；其余模型由 run_kt.py 的复用分支原样保留。
  # `--redo-models` 是关键：缺口模型往往**已有**带 AUC 的旧条目（例如本次因
  # GKT 实现提速而需要整行重算），不加它会被 [reuse] 原样跳过、静默不生效。
  echo "[resume-run] $unit models=$need  cfg=$cfg  (强制重算)"
  $PY experiments/run/run_kt.py --dataset "$unit" --models $need --epochs "$EPOCHS" \
      --patience "$PATIENCE" --tag bench $extra --redo-models $need \
      >> "logs/resume_bench_${unit}.log" 2>&1
  echo "[resume-done] $unit"
}
export -f do_unit
export PY EPOCHS PATIENCE

JOBS=""
while IFS='|' read -r unit need; do
  [ -z "${unit:-}" ] && continue
  if [ -n "$ONLY" ] && [[ "$unit" != *"$ONLY"* ]]; then continue; fi
  if [ -n "$MODELS" ]; then
    # 交集：只补用户指定的模型
    keep=""
    for m in $(echo "$need" | tr ',' ' '); do
      for w in $MODELS; do [ "$m" = "$w" ] && keep="$keep $m"; done
    done
    need="$(echo $keep | tr ' ' ',')"
    [ -z "${need//,/}" ] && continue
  fi
  # 按缺口规模与数据集规模排序交给 xargs 无法稳定控制，这里按出现顺序下发，
  # 而 check_bench 的输出顺序 = DS5 顺序（已按单次开销降序），故长杆先启动。
  JOBS="$JOBS $unit|$need"
done <<< "$GAPS"

printf '%s\n' $JOBS | xargs -P "$PAR" -I{} bash -c 'do_unit "${1%%|*}" "${1##*|}"' _ {}

echo
echo "===== 补跑后复核 ====="
$PY experiments/tools/check_bench.py --datasets "$DS5" --single-datasets "$DS1" --quiet 2>/dev/null
echo "===== RESUME DONE ====="
