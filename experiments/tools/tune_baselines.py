#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基线模型的**两阶段**超参搜索（保证"公平比较"的实验纪律）。

动机
----
本文方法 DGCL-KT 的结构/容量超参是**在验证集上搜出来的**（run_kt_tune.py 的
stage1~5）。若对比方法一律用同一个默认学习率/容量跑，就会形成"我方精调、对手
不调"的不对等比较——审稿人很容易据此质疑增益来源。

因此本脚本对**每个数据集**的**代表划分**（5 折数据集取 fold1，单划分数据集取自身）
为每个基线单独搜索超参，选择依据严格为**验证集最优 AUC**；测试集 AUC 只记录、
不参与任何选择，最终报告时才使用。

两阶段搜索
----------
stage lr  ：学习率 lr ∈ {5e-4, 1e-3, 2e-3, 4e-3, 8e-3}（其余沿用默认容量）
stage cap ：在 stage-lr 选出的最优 lr 上，搜模型容量
            （DKT/DKT+ 的 hidden；DKVMN 的 mem/key/val；SAKT/AKT/SAINT/SimpleKT
              的 d 与层数；GKT 的 hidden 与图传播步数）

第一轮只搜 3 个 lr 时暴露出两个问题，本版已修正：
  · 多个模型（DKT/DKT+/AKT/SimpleKT/SAINT）在 4e-3 处取得最优 → 网格上界过低；
  · DKVMN/GKT 的 epochs_run 顶到上限 30、验证集仍在上升 → 训练预算不足。
故把网格扩到 5 点、epochs 提到 60、patience 提到 10。

产物
----
    results/kt_basetune_<ds>.json      —— 全部试验行（含 stage 标记）+ 验证集最优
    results/kt_baseline_cfg_<ds>.json  —— 直接喂给 run_kt.py --model-cfg-file 的配置

用法
----
    python experiments/tools/tune_baselines.py --dataset statics11_f1
    python experiments/tools/tune_baselines.py --dataset assist17_f1 --models AKT DKVMN
    python experiments/tools/tune_baselines.py --dataset assist17_f1 --stage lr
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "run"))
sys.path.insert(0, os.path.dirname(HERE))

from paths import PROCESSED as PROC, RESULTS                      # noqa: E402
from run_kt import train_one, build_graph_arrays, build_scope_mask, ALL_MODELS  # noqa: E402

# 需要搜索的基线（BKT 是贝叶斯计数模型，无梯度、无学习率，单独跳过）
TUNE_MODELS = ["DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT", "SAINT", "SimpleKT"]
LR_GRID = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

# 容量搜索空间：每个模型给少量最有信息量的候选（避免搜索预算膨胀到不可接受）
CAP_GRID = {
    "DKT":      [dict(hidden=256)],
    "DKT+":     [dict(hidden=256)],
    "DKVMN":    [dict(mem_size=64, key_dim=128, val_dim=128), dict(mem_size=32, key_dim=64, val_dim=64)],
    "SAKT":     [dict(d=256, layers=1), dict(d=128, layers=2)],
    "AKT":      [dict(d=256, layers=1), dict(d=128, layers=2)],
    "SAINT":    [dict(d=256, layers=1), dict(d=128, layers=2)],
    "SimpleKT": [dict(d=256, layers=1), dict(d=128, layers=2)],
    "GKT":      [dict(hidden=128, steps=3)],
}

# ---------------------------------------------------------------------------
# 「逐时刻循环实现」模型的批量策略
# ---------------------------------------------------------------------------
# DKVMN 每步读写记忆、GKT 每步做一次图传播，单次前向要发起十余个小内核。实测
# （experiments/tools/diag_mem.py，T=200、V100-32G）：
#     · 单次前反向耗时几乎与批大小无关（DKVMN 0.38~0.45s、GKT 0.5~1.0s）
#       —— 成本由 200 步循环的内核启动决定，而非浮点量；
#     · GKT 峰值显存 ≈ 13.5·B·K·h·T·4 字节（K=198/h=128/B=64 实测 17.0GB，
#       K=1223/h=64/B=64 直接 OOM）。
# 推论：**批越大越省时间**（前向次数 ≈ 序列数/B），但 GKT 的批大小必须由显存预算
# 反推。曾把 GKT 批大小一律提到 160，结果是它独占 24GB 把 32G 卡吃满、同批启动的
# 其它任务全部 CUDA OOM（assist12 的基线搜索整轮零产出）——故改为按预算计算。
#
# 该覆盖**同时用于搜索阶段与最终阶段B**（通过写进 best.cfg 落盘），避免"选参时的
# 批大小"与"报告时的批大小"不一致导致 lr 迁移失真。模型容量、数据划分、评测口径
# 与训练轮数（60 epochs / patience 10）全部不变。
GKT_MEM_BYTES = 13.5            # 经验系数：每 (B·K·h·T) 元素折算的字节数
GKT_BUDGET_GB = 18.0            # 单个 GKT 任务的显存预算
# 注：GKT 的**单批耗时与批大小基本无关**（内核启动受限，B=8→128 仅 +8%），
#     故吞吐 ≈ 正比于 B ⇒ 把预算从早先的 4 GiB 提高到 18 GiB 是近乎免费的加速。
#     18 GiB 是**每个 GKT 进程**的预算；并行度 PAR≤1~2 时不会超出单卡 32 GiB。
DKVMN_BATCH = 128              # 实测 K=1223/B=160 仅 1.04GB，可放心放大（2× 提速）
SEQ_LEN = 200                   # 与 max_len 一致（切块后的典型长度）
# 各模型的"基础配置"（不进搜索网格，但会写入 best.cfg 以保证阶段B 一致）
MODEL_BASE = {"GKT": dict(hidden=64, steps=3)}
# GKT 的逐步图传播使单次试验开销约为其它基线的 10~40 倍，故其学习率网格收窄为
# 3 点（覆盖其余模型最优值所在区间）；其余模型仍为 5 点全网格。
LR_GRID_OVERRIDE = {"GKT": [1e-3, 2e-3, 4e-3]}
HEAVY_LOOP_MODELS = ("DKVMN", "GKT")


def lr_grid_of(model):
    return LR_GRID_OVERRIDE.get(model, LR_GRID)


def gkt_batch(K, hidden):
    """按显存预算反推 GKT 的批大小（向下取整到 2 的倍数，范围 [2,128]）。"""
    per_sample = GKT_MEM_BYTES * K * max(1, int(hidden)) * SEQ_LEN * 4
    b = int(GKT_BUDGET_GB * 2 ** 30 / max(per_sample, 1))
    b = max(2, min(128, b))
    return max(2, (b // 2) * 2)


def search_extra(model, K, extra):
    """为搜索阶段注入模型的**固定**配置（区别于网格：会一并写入 best.cfg）。

    当前用于：给逐时刻循环模型换更合适的批大小（DKVMN 放大以省时间；GKT 按显存
    预算取值，并在容量搜索把 hidden 翻倍时自动减半）。其余模型返回原网格项不变。
    """
    d = dict(MODEL_BASE.get(model) or {})
    d.update(extra or {})
    if model == "GKT":
        d["batch"] = gkt_batch(K, d.get("hidden", 64))
    elif model == "DKVMN":
        d["batch"] = DKVMN_BATCH
    return d or None


def build_args(ds, model, lr, epochs, patience, batch, seed, extra=None, train_max_users=0):
    """装成 run_kt.train_one 需要的 args 命名空间（与 run_kt_tune.build_args 同构）。"""
    a = argparse.Namespace()
    a.dataset = ds
    a.models = [model]
    a.epochs = epochs
    a.batch = batch
    a.lr = lr
    a.patience = patience
    a.max_len = 200
    a.min_len = 3
    a.max_users = 0
    # 搜索阶段的训练集用户上限（0=全量）。见 run_kt.train_one 的说明：
    # 只截断训练集，验证/测试保持全量，从而"选参依据"不被稀释。
    a.train_max_users = int(train_max_users or 0)
    a.seed = seed
    a.tag = "basetune"
    a.ablation = False
    a.ab_no_graph = False
    a.ab_no_calib = False
    a.ab_no_scope = False
    a.ab_struct_only = False
    a.lr_map = {}
    a.model_cfg = {model: dict(extra or {})} if extra else {}
    # DGCL-KT 专用字段：train_one 用 getattr 取缺省，这里给全以便工厂分支可读
    a.d = 128
    a.layers = 1
    a.steps = 2
    a.hops = 0
    a.dropout = 0.2
    a.gated_fuse = False
    a.topm = 0
    a.wprior = 0.0
    a.graph_in = False
    a.seq_enc = "transformer"
    a.hidden = 128
    a.mem_size = 64
    a.key_dim = 64
    a.val_dim = 64
    return a


def one_run(ds, model, lr, epochs, patience, batch, seed, obj, adj, ac, scope, s2i, extra,
            train_max_users=0):
    """单次试验。返回 dict=成功；字符串 "OOM"=显存不足（调用方把批大小减半重试）；
    None=其它异常（不重试，避免用 3 倍时间掩盖真实 bug）。"""
    t0 = time.time()
    try:
        r = train_one(model, build_args(ds, model, lr, epochs, patience, batch, seed, extra,
                                        train_max_users),
                      obj, adj, ac, scope, s2i, seed)
    except torch.cuda.OutOfMemoryError:                                # noqa: BLE001
        print(f"      lr={lr:g} cfg={extra or '-'} → CUDA OOM")
        torch.cuda.empty_cache()
        return "OOM"
    except Exception as e:                                             # noqa: BLE001
        print(f"      lr={lr:g} cfg={extra or '-'} → 失败: {type(e).__name__}: {e}")
        return None
    return dict(model=model, lr=lr, cfg=dict(extra or {}),
                val=r.get("best_valid_auc"), test=r["AUC"], acc=r["ACC"],
                epochs_run=r.get("epochs_run"), params=r.get("params"),
                sec=round(time.time() - t0, 1))


def run_with_oom_retry(ds, model, lr, epochs, patience, batch, seed, obj, adj, ac,
                       scope, s2i, extra, train_max_users=0):
    """带 OOM 退避的试验执行：显存不足时把显式批大小减半重试（最多 3 次）。

    GKT 的显存随 B·K·h·T 线性增长，预算公式是经验式的；即使公式偏乐观，这里也能
    自动退回可行批大小，而不至于整轮数据集零产出（此前一轮就是这样丢掉 assist12 的
    全部基线搜索结果）。注意退避后的批大小会写进返回行的 cfg，从而贯穿到阶段B。
    """
    for attempt in range(3):
        row = one_run(ds, model, lr, epochs, patience, batch, seed, obj, adj, ac,
                      scope, s2i, extra, train_max_users)
        if row != "OOM":
            return row
        if not isinstance(extra, dict) or "batch" not in extra or int(extra["batch"]) <= 4:
            return None
        extra = dict(extra, batch=max(4, int(extra["batch"]) // 2))
        print(f"      [retry {attempt + 1}] 显存不足 → 批大小降为 {extra['batch']}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=None,
                    help="默认搜全部可调基线；也可只给子集")
    ap.add_argument("--stage", default="both", choices=("lr", "cap", "both"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-max-users", type=int, default=0,
                    help="搜索阶段训练集用户上限（0=全量）。为控制超大数据集"
                         "（如 ASSIST2012 的 18998 名用户）的搜索开销，只截断训练集；"
                         "验证/测试集保持全量，最终阶段B 用全量训练集重训。")
    a = ap.parse_args()
    tmu = int(a.train_max_users or 0)

    models = a.models or TUNE_MODELS
    for m in models:
        assert m in ALL_MODELS, f"未知模型 {m}"
        assert m in TUNE_MODELS, f"{m} 不在可调基线集合内"

    out_path = os.path.join(RESULTS, f"kt_basetune_{a.dataset}.json")
    cfg_path = os.path.join(RESULTS, f"kt_baseline_cfg_{a.dataset}.json")
    rows_all, best = [], {}
    if os.path.exists(out_path):                       # 支持续跑
        with open(out_path, encoding="utf-8") as f:
            old = json.load(f)
        rows_all = old.get("rows") or []
        best = old.get("best") or {}
        print(f"[basetune] 续跑：已有 {len(rows_all)} 行试验、{len(best)} 个模型结果")

    with open(os.path.join(PROC, f"{a.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    n_train = len(obj["train"])
    K = obj["n_skill"]
    # 续跑一致性：比较**实际生效**的训练集规模，而不是原始上限值。
    # 例如旧结果用全量（2696 用户）、新上限 4000 —— 该上限并不生效（4000>2696），
    # 实际训练集仍是 2696，旧试验行完全可比，无需丢弃。只有真正改变了参与训练
    # 的用户数时才丢弃，避免把两种预算的结果混进同一份配置。
    eff_now = min(tmu, n_train) if tmu > 0 else n_train
    prev_tmu = int(old.get("train_max_users", 0) or 0) if os.path.exists(out_path) else eff_now
    eff_prev = min(prev_tmu, n_train) if prev_tmu > 0 else n_train
    if os.path.exists(out_path) and eff_prev != eff_now:
        print(f"[basetune] 生效训练集规模 {eff_prev} → {eff_now} 变化，丢弃旧试验行重跑")
        rows_all, best = [], {}
    adj, ac = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)

    # 批大小策略变化时，旧试验行不可比（"选参时的批大小"必须与"报告时的批大小"
    # 一致，否则 lr 迁移失真）。逐模型比对旧行里记录的 batch 与当前策略，不一致才
    # 丢弃 —— 该判据是幂等的，不会每次重启都重跑。
    for m in models:
        want = (search_extra(m, K, None) or {}).get("batch")
        if want is None:
            continue
        rows_of = [r for r in rows_all if r.get("model") == m]
        if rows_of and any((r.get("cfg") or {}).get("batch") != want for r in rows_of):
            old_b = sorted({(r.get("cfg") or {}).get("batch") for r in rows_of})
            print(f"[basetune] {m} 批大小策略 {old_b}→{want}，丢弃其 {len(rows_of)} 行旧试验重跑")
            rows_all = [r for r in rows_all if r.get("model") != m]
            best.pop(m, None)
    print(f"[basetune] dataset={a.dataset} K={obj['n_skill']} q={obj['n_question']} "
          f"train/valid/test={n_train}/{len(obj['valid'])}/{len(obj['test'])}")
    if eff_now != n_train:
        print(f"[basetune] 搜索用训练用户上限 {tmu}（训练集 {n_train} → {eff_now}，"
              f"验证/测试保持全量）")
    print(f"[basetune] epochs={a.epochs} patience={a.patience} stage={a.stage} "
          f"网格(默认)={LR_GRID} 覆盖={LR_GRID_OVERRIDE}")

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dict(dataset=a.dataset, epochs=a.epochs, patience=a.patience,
                           batch=a.batch, seed=a.seed, train_max_users=tmu,
                           lr_grid=LR_GRID,
                           cap_grid=CAP_GRID, n_trials=len(rows_all),
                           rows=rows_all, best=best), f, ensure_ascii=False, indent=1)
        cfg = {m: dict(v.get("cfg") or {}, lr=v["lr"]) for m, v in best.items()}
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)

    # 续跑判据必须是"该模型的该阶段**全部**试验都跑完"，而不是"有过记录"。
    # 反例：某模型的 lr 网格跑到第 3/5 个点被中断，若按"有过记录"就跳过，
    # 该模型会带着不完整的网格进入最终对比，形成"对手没调好"的不公平比较。
    # 失败的试验不会写入 rows，因此计数小于期望值时会被正确重跑。
    def _stage_done(stage, expected_of):
        done = set()
        for m in models:
            exp = expected_of(m)
            if exp <= 0:
                done.add(m)
                continue
            n = sum(1 for r in rows_all
                    if r.get("model") == m and r.get("stage") == stage)
            if n >= exp:
                done.add(m)
        return done

    # ------------------------- stage lr -------------------------
    lr_stage_done = _stage_done("lr", lambda m: len(lr_grid_of(m)))
    if a.stage in ("lr", "both"):
        for name in models:
            if name in lr_stage_done:
                print(f"--- {name}: stage-lr 已有记录，跳过")
                continue
            print(f"--- [stage-lr] {name} on {a.dataset} ---")
            ovr = dict(search_extra(name, K, None) or {})
            rows = []
            for lr in lr_grid_of(name):
                row = run_with_oom_retry(a.dataset, name, lr, a.epochs, a.patience, a.batch,
                                         a.seed, obj, adj, ac, scope, s2i, ovr or None, tmu)
                if row is None:
                    continue
                row["stage"] = "lr"
                rows.append(row); rows_all.append(row)
                v = row["val"] if row["val"] is not None else float("nan")
                print(f"      lr={lr:g} → val={v:.4f} test={row['test']:.4f} "
                      f"({row['sec']:.0f}s, {row['epochs_run']} ep)")
            if rows:
                b = max(rows, key=lambda r: (r["val"] if r["val"] is not None else -1))
                # cfg 里带上该模型的固定覆盖（如重模型的 batch），使其贯穿到阶段B
                best[name] = dict(lr=b["lr"], cfg=dict(ovr), val=b["val"], test=b["test"])
                print(f"  [{name}] stage-lr 最优 lr={b['lr']:g} val={b['val']:.4f}"
                      + (f" 固定覆盖={ovr}" if ovr else ""))
            save()

    # ------------------------- stage cap ------------------------
    # 容量网格各模型长度不同（DKT 1 个、DKVMN/SAKT/AKT/SAINT/SimpleKT 2 个…），
    # 故期望值按模型各自的 CAP_GRID 长度计算。
    cap_done = _stage_done("cap", lambda m: len(CAP_GRID.get(m, [])))
    if a.stage in ("cap", "both"):
        for name in models:
            if name in cap_done:
                print(f"--- {name}: stage-cap 已有记录，跳过")
                continue
            if name not in best:
                print(f"--- {name}: 缺 stage-lr 结果，跳过容量搜索")
                continue
            lr = best[name]["lr"]
            print(f"--- [stage-cap] {name} on {a.dataset} @ lr={lr:g} ---")
            rows = []
            for cfg in CAP_GRID.get(name, []):
                row = run_with_oom_retry(a.dataset, name, lr, a.epochs, a.patience, a.batch,
                                         a.seed, obj, adj, ac, scope, s2i,
                                         search_extra(name, K, cfg), tmu)
                if row is None:
                    continue
                row["stage"] = "cap"
                rows.append(row); rows_all.append(row)
                v = row["val"] if row["val"] is not None else float("nan")
                print(f"      cfg={cfg} → val={v:.4f} test={row['test']:.4f} "
                      f"({row['sec']:.0f}s, {row['epochs_run']} ep)")
            if rows:
                b = max(rows, key=lambda r: (r["val"] if r["val"] is not None else -1))
                # 只在新容量确实更好时替换
                if b["val"] is not None and b["val"] > (best[name]["val"] or -1):
                    best[name] = dict(lr=b["lr"], cfg=b["cfg"], val=b["val"], test=b["test"])
                    print(f"  [{name}] 容量更新为 {b['cfg']}（val {b['val']:.4f}）")
                else:
                    print(f"  [{name}] 保留默认容量（容量搜索未带来验证集提升）")
            save()

    save()
    print(f"\n[basetune] saved {out_path}")
    print(f"[basetune] saved {cfg_path}")
    for m, v in best.items():
        print(f"    {m:<9} lr={v['lr']:<8g} cfg={v.get('cfg') or {}} val={v['val']:.4f}")


if __name__ == "__main__":
    main()
