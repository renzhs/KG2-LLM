#!/usr/bin/env bash
# 统一知识追踪对比实验编排（7 数据集 × 11 模型）。
#
# 两组数据集:
#   A. 公开基准（AKT 官方 5 折划分）: assist09_akt / assist15 / assist17 / statics11
#      —— 与 AKT(2020)、pykt/SimpleKT(ICLR2023) 等已发表数字同划分，可直接对照
#   B. 自建划分: dnui_syn（自建半合成）/ assist09 / assist12
#      —— 无对应已发表数字，全部对比方法由我们自行跑出
#
# 流程:
#   阶段A  本文方法 DGCL-KT 的分阶段链式调参（stage 4→5→1→3，验证集选参）
#   阶段A2 各**基线**的两阶段超参搜索（lr → 容量，验证集选参）
#          ——保证"我方与对手同等调参"，避免"本方精调、对手用默认 lr"的不对等比较
#          ——A 与 A2 互不依赖，**并行执行**（见下方注释），缩短墙钟时间
#   阶段B  同一并行队列跑「5 折 × 全模型」与「单划分 × 全模型」，--tag bench
#   阶段C  聚合（mean±std）并出跨数据集 LaTeX 对比表
#
# 实验纪律:
#   · 所有选择（结构、容量、学习率）只在**验证集**上做，测试集只用于最终报告；
#   · 长序列按 AKT 协议**切块**（run_kt.py 默认 chunk），评测覆盖全部交互，
#     使各模型评测样本量一致、且与已发表数字可比；
#   · 题目 id 全程贯通（AKT/SAINT/SimpleKT 依赖 Rasch 难度/题目嵌入）；
#   · 逐折落盘，可中断续跑（已存在的 kt_<ds>_<tag>.json 会被跳过）。
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-python3}"   # 可用 PY=/path/to/python 覆盖
# 顺序按**单次运行开销**从大到小排，使长杆任务尽早进入并行槽，避免最后才开始而拖长
# 整体工期。注意两个列表的规模排序**不同**，不能照抄：
#   5 折（AKT 官方划分）用户数：assist15 11452 > assist09_akt 2329 > assist17 1026 > statics11 201
#   单划分：assist12 18998 > assist09 2696 > dnui_syn 1502
# 旧版 DS_LIST_5F 写成 "assist17 statics11 assist15 assist09_akt"，把**第二大数据集**
# assist15 排到第 3 位，实测使其基线调参比实际开跑时间晚 6h 才进入并行槽，
# 尾段只剩 2 路并行，白白拉长整体工期（单卡吞吐固定时，重排可省掉这段尾）。
DS_LIST_5F="${DS_LIST_5F:-assist15 assist09_akt assist17 statics11}"
DS_LIST_1F="${DS_LIST_1F:-assist12 assist09 dnui_syn}"
FOLDS="${FOLDS:-1 2 3 4 5}"
MODELS="${MODELS:-all}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
PAR="${PAR:-5}"
PAR_TUNE="${PAR_TUNE:-4}"       # 兼容旧调用：未分别指定时用作 A/A2 的默认并行度
PAR_A="${PAR_A:-${PAR_TUNE}}"   # 阶段A（本文方法链式调参）并行度
PAR_B="${PAR_B:-${PAR_TUNE}}"   # 阶段A2（基线超参搜索）并行度
# 调参阶段的训练集用户上限（0=全量）。只截断**训练集**，验证/测试保持全量，
# 阶段B 评测时恒用全量训练集重训。
# 取 2700 的依据：实测只有 ASSIST2012(18998 用户) 与 ASSIST2015(11452 用户) 属
# 真正的超大数据集（单次搜索耗时是其余数据集的 5~20 倍），其余数据集
# （assist09 2696 / assist09_akt 2329 / dnui_syn 1502 / assist17 1026 / statics11 201）
# 本就不超过该值 —— 因此该上限**只对两个大数据集生效**，其余数据集的搜索
# 完全等价于全量训练集，其已有试验行可直接续用，避免重复劳动。
# 这一上限对"我方与对手"同时生效，不破坏公平比较协议。
TUNE_TRAIN_USERS="${TUNE_TRAIN_USERS:-2700}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PY MODELS EPOCHS PATIENCE TUNE_TRAIN_USERS

# --------------------------------------------------------------------------
# 阶段A: 本文方法 DGCL-KT 的分阶段链式调参（stage4 结构 -> 5 容量/lr -> 1 图结构 -> 3 先验）
#        公平性要点：基线会搜 lr+容量，我方也必须搜；故用链式 --base 累积，
#        得到"联合最优"配置写入 kt_maincfg_<ds>.json。
# --------------------------------------------------------------------------
tune_main() {
  ds="$1"
  # 注意：**不能**用"kt_maincfg 是否存在"来整数据集跳过。链式调参的复用粒度是
  # **单个 stage**（由 base + 网格指纹 + 生效训练集规模三者判等）。一旦我们改动
  # 了某个 stage 的搜索网格（例如把 stage5 的学习率从 2 点扩到与基线一致的 5 点），
  # 整数据集跳过就会让"旧网格搜出来的配置"原样留在 kt_maincfg 里，修复静默失效。
  # 因此这里始终进链，交给 stage 级判等决定跑哪些；全部命中时开销仅为加载数据。
  $PY experiments/tools/tune_main_chain.py --dataset "$ds" --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --train-max-users "$TUNE_TRAIN_USERS" > "logs/tune_${ds}.log" 2>&1
  # 判据是**产物**而非退出码：曾出现"脚本正常退出但中途抛异常"导致
  # kt_maincfg 缺失却被记为 [done] 的情况（见 tune_main_chain.py 中复用分支的注释）。
  if [ -f "results/kt_maincfg_${ds}.json" ]; then
    echo "[done] tune-main ${ds}"
  else
    echo "[FAIL] tune-main ${ds} —— 未产出 kt_maincfg_${ds}.json，见 logs/tune_${ds}.log"
  fi
}
export -f tune_main

echo "===== 阶段A: 本文方法分阶段链式调参 (验证集选参) ====="
TASKS_A=""
for DS in $DS_LIST_1F; do TASKS_A="$TASKS_A ${DS}"; done
for DS in $DS_LIST_5F; do TASKS_A="$TASKS_A ${DS}_f1"; done
echo "待搜 ($(echo $TASKS_A | wc -w) 个, 并行度 $PAR_A):$TASKS_A"
# 阶段A 与 阶段A2 **互不依赖**：A2 只需要数据集本身（只有 A3 合并才需要两者齐备）。
# 故二者同时开跑；否则 A2 会被最慢的 assist12 链式调参（约 2 小时）白白阻塞，
# 使 7 路并行度在后半段闲置。实测两者并行后阶段A+A2 的总墙钟时间约为 max 而非 sum。
printf '%s\n' $TASKS_A | xargs -P "$PAR_A" -I{} bash -c 'tune_main {}' &
PID_STAGE_A=$!

# --------------------------------------------------------------------------
# 阶段A2: 基线两阶段超参搜索（同一代表划分）
# --------------------------------------------------------------------------
tune_base() {
  ds="$1"
  # 判据必须是「每个可调基线的 lr 与 cap 两个阶段都跑满整个网格」，而**不能**是
  # 「kt_baseline_cfg_<ds>.json 里有没有 8 个模型键」。后者会误判：
  # tune_baselines.py 在 **stage-lr 结束时就写 best[name]**，cfg 文件随即包含全部
  # 8 个模型键 —— 于是「lr 跑满、容量一次未搜」的数据集也被判为已完成并**永久跳过**。
  # 实测 dnui_syn 正是如此：40 行试验（8 模型 lr 全满 + 仅 DKT/DKT+ 的容量）即含 8 键，
  # 被 [skip]，缺 11 个容量试验（DKVMN/SAKT/AKT/SAINT/SimpleKT 各 2 + GKT 1）。
  # 而阶段 A3 闸门用的是同一条弱判据，同样放行 —— 形成**静默的不公平比较**。
  # 现统一交给 check_basetune.py（与 tune_baselines.py 内部 _stage_done 判据同源）。
  if $PY experiments/tools/check_basetune.py --datasets "$ds" --quiet >/dev/null 2>&1; then
    echo "[skip] basetune ${ds}"
    return
  fi
  $PY experiments/tools/tune_baselines.py --dataset "$ds" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --train-max-users "$TUNE_TRAIN_USERS" \
      > "logs/basetune_${ds}.log" 2>&1
  if [ -f "results/kt_baseline_cfg_${ds}.json" ]; then
    echo "[done] basetune ${ds}"
  else
    echo "[FAIL] basetune ${ds} —— 未产出 kt_baseline_cfg_${ds}.json，见 logs/basetune_${ds}.log"
  fi
}
export -f tune_base

echo
echo "===== 阶段A2: 基线两阶段超参搜索 (lr → 容量, 验证集选参；与阶段A 并行) ====="
echo "待搜 ($(echo $TASKS_A | wc -w) 个, 并行度 $PAR_B):$TASKS_A"
printf '%s\n' $TASKS_A | xargs -P "$PAR_B" -I{} bash -c 'tune_base {}' &
PID_STAGE_A2=$!

echo
echo "[wait] 阶段A (pid=$PID_STAGE_A) 与 阶段A2 (pid=$PID_STAGE_A2) 并行执行中..."
wait "$PID_STAGE_A";  echo "[done] 阶段A 全部完成"
wait "$PID_STAGE_A2"; echo "[done] 阶段A2 全部完成"

echo
echo "===== 阶段A3 前置检查: 模型配置齐备性 ====="
MISS_MAIN=""
MISS_BASE=""
for DS in $TASKS_A; do
  [ -f "results/kt_maincfg_${DS}.json" ] || MISS_MAIN="$MISS_MAIN $DS"
  # 与 tune_base 的跳过判据共用 check_basetune.py：必须真的"lr+cap 两阶段都跑满"，
  # 而不是"cfg 里出现过 8 个模型键"（后者会把只搜过 lr 的数据集放行）。
  $PY experiments/tools/check_basetune.py --datasets "$DS" --quiet >/dev/null 2>&1 \
    || MISS_BASE="$MISS_BASE $DS"
done
if [ -n "${MISS_MAIN// /}" ] || [ -n "${MISS_BASE// /}" ]; then
  [ -n "${MISS_MAIN// /}" ] && echo "[FATAL] 缺少本文方法配置 kt_maincfg，阶段B 会漏跑本文方法:$MISS_MAIN"
  [ -n "${MISS_BASE// /}" ] && echo "[FATAL] 基线配置不完整（未覆盖全部 8 个可调基线 / 容量阶段未跑满），会形成不公平比较:$MISS_BASE"
  $PY experiments/tools/check_basetune.py --datasets "$TASKS_A" 2>/dev/null | sed -n '1,60p'
  echo "        请修复 logs/ 中的报错后重跑；已完成的 stage 与试验结果会被复用。"
  exit 1
fi
echo "全部 $(echo $TASKS_A | wc -w) 个数据集的「本文方法配置 + 8 个基线配置」齐备。"

# 只检查"文件是否存在"是不够的：tune_main_chain 只在**链跑完**时才写 kt_maincfg，
# 于是"链中途失败/被中断"时上一轮遗留的旧文件仍在，闸门会放行、阶段B 用一份过时
# 配置跑本文方法 —— 完全静默。这里再用 check_maincfg 做**内容级**校验：网格指纹、
# 链式 base 累积、maincfg 与各 stage 自洽，三者缺一即拒绝进入阶段B。
echo "----- 阶段A3 前置检查: 配置内容级校验（网格指纹/链式继承/产物自洽）-----"
$PY experiments/tools/check_maincfg.py --datasets "$(echo $TASKS_A | tr ' ' ',')" || {
  echo "[FATAL] 本文方法配置未通过内容级校验，拒绝进入阶段B（见上方明细）。"
  exit 1
}

# --------------------------------------------------------------------------
# 阶段A3: 合并「基线配置 + DGCL-KT 配置」为单一 model-cfg 文件
# --------------------------------------------------------------------------
echo
echo "===== 阶段A3: 合并模型配置 ====="
for DS in $TASKS_A; do
  $PY experiments/tools/tune_main_chain.py --dataset "$DS" --merge-only
done

# --------------------------------------------------------------------------
# 阶段B: 全模型对比
# --------------------------------------------------------------------------
run_one() {
  t="$1"
  if [[ "$t" =~ _f[0-9]+$ ]]; then
    base="${t%_f*}"
    cfgfile="results/kt_model_cfg_${base}_f1.json"
  else
    cfgfile="results/kt_model_cfg_${t}.json"
  fi
  extra=""
  [ -f "$cfgfile" ] && extra="--model-cfg-file $cfgfile"
  echo "[run] $t  cfg=$cfgfile"
  $PY experiments/run/run_kt.py --dataset "$t" --models $MODELS --epochs "$EPOCHS" \
      --patience "$PATIENCE" --tag bench $extra > "logs/bench_${t}.log" 2>&1
  echo "[done] $t"
}
export -f run_one

echo
echo "===== 阶段B: 全模型对比 ====="
# 判据必须是「该折是否**完整**（每个模型都有非 error 的 AUC）」而不是「JSON 是否存在」。
# run_kt.py 逐模型增量落盘，任一模型抛异常（最典型是 CUDA OOM）也会生成 JSON ——
# 旧判据"文件存在即跳过"于是把该折**永久**跳过，表现为表里凭空少一个基线（完全静默）。
# 实测 GKT 在 16/22 个「数据集×折」上 OOM：assist15 与 assist09_akt 全缺、
# assist17 缺 1 折、statics11 只剩 1 折，而 GKT 恰是 statics11 上最强的基线。
# 现改为：不完整就重入该折，由 run_kt.py 的**模型级续跑**只补缺口模型（其余复用）。
# 同时阶段C 前加门禁，杜绝残缺矩阵被静默写进论文。
TASKS=""
for DS in $DS_LIST_1F; do
  if $PY experiments/tools/check_bench.py --datasets "" --single-datasets "$DS" --quiet >/dev/null 2>&1; then
    echo "[skip] ${DS}"; continue
  fi
  TASKS="$TASKS ${DS}"
done
for DS in $DS_LIST_5F; do
  for k in $FOLDS; do
    if $PY experiments/tools/check_bench.py --datasets "$DS" --folds "$k" --quiet >/dev/null 2>&1; then
      echo "[skip] ${DS}_f${k}"; continue
    fi
    TASKS="$TASKS ${DS}_f${k}"
  done
done
echo "待跑任务 ($(echo $TASKS | wc -w) 个, 并行度 $PAR):"
echo "  $TASKS"
if [ -n "${TASKS// /}" ]; then
  printf '%s\n' $TASKS | xargs -P "$PAR" -I{} bash -c 'run_one {}'
fi

# --------------------------------------------------------------------------
# 阶段C: 聚合
# --------------------------------------------------------------------------
echo
echo "===== 阶段C 前置门禁: 对比矩阵完整性 ====="
# 聚合脚本对残缺矩阵是**静默容忍**的（只写「有的那几折」），因此必须在聚合前拦一道。
# 实测教训：GKT 只跑成 1 折的 statics11 被写成 0.8214±0.0000，粗看像"方差极小"。
if ! $PY experiments/tools/check_bench.py --datasets "$DS_LIST_5F" \
        --single-datasets "$DS_LIST_1F" ; then
  echo "[FATAL] 阶段B 对比矩阵不完整，拒绝聚合出表。"
  echo "        补跑缺口: bash experiments/run/resume_bench.sh"
  exit 1
fi

echo "===== 阶段C: 聚合 ====="
for DS in $DS_LIST_5F; do
  $PY experiments/tools/aggregate_bench.py --dataset "$DS" --folds 1,2,3,4,5 2>&1 | tail -16
done
for DS in $DS_LIST_1F; do
  $PY experiments/tools/aggregate_bench.py --dataset "$DS" --single 2>&1 | tail -16
done
$PY experiments/tools/aggregate_bench.py --all 2>&1 | tail -3
echo "===== ALL DONE ====="
