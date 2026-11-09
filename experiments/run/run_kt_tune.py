#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DGCL-KT 超参搜索（严格在验证集上选择）。

动机
----
DGCL-KT 在 DNUI-IKG 上取得最优，但在 ASSIST09（**长尾短序列**：median 23 / avg 72、
K=149 大技能空间）上略逊于 GKT。原实现把容量/结构超参**硬编码**在模型构造里、无法调，
因此先把它们暴露成 CLI，再做两阶段搜索：

  阶段 1（结构）：gated_fuse × steps × hops
  阶段 2（容量）：d × dropout × lr（固定阶段 1 选出的最优结构）

选择依据统一为「验证集最优 AUC」(`best_valid_auc`)；**测试集 AUC 只对最终选定组合报告**，
避免在测试集上调参。结果写入 results/kt_tune_<ds>_stage<N>.json。

用法
----
  python experiments/run/run_kt_tune.py --dataset assist09 --stage 1
  python experiments/run/run_kt_tune.py --dataset assist09 --stage 2 \
      --base '{"gated_fuse": true, "steps": 2, "hops": 2}'
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))            # -> experiments/
from paths import PROCESSED as PROC, RESULTS          # noqa: E402
from run_kt import train_one, build_graph_arrays, build_scope_mask  # noqa: E402


# 阶段 1：结构（图证据怎么用）。hops=0 表示沿用默认 steps+1；topm=0 表示不稀疏。
STAGE1 = [dict(gated_fuse=g, steps=s, hops=h, topm=t)
          for g in (False, True) for s in (1, 2) for h in (1, 2) for t in (0, 8)]
# 学习率网格：必须与基线侧 `tune_baselines.LR_GRID` **逐点一致**。
# 这两处曾经不一致 —— stage2/stage5 只用 (1e-3, 2e-3) 两点，而基线搜 5 点
# （含 4e-3、8e-3）。后果是 7 个数据集里有 **5 个**把 DGCL-KT 的 lr 选在 2e-3
# 这个**网格上界**上（最优值显然在网格之外），而基线却搜到了 8e-3；等价于
# "我方被窄网格卡住、对手可以充分调参"，是典型的协议不对称。
# 注意：改动本网格会使已有的 stage 产物失效（`tune_main_chain.run_stage` 用
# 网格指纹判等），这是期望行为 —— 否则会复用窄网格搜出来的旧配置。
LR_GRID = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

# 阶段 2：容量（d × dropout × lr）。
STAGE2 = [dict(d=d, dropout=dr, lr=lr)
          for d in (64, 128, 256) for dr in (0.1, 0.2, 0.3) for lr in LR_GRID]
# 阶段 3：结构先验强度 β（在注意力 logits 上叠加 β·log P(v|u)）。
STAGE3 = [dict(wprior=w, topm=t) for w in (0.0, 0.5, 1.0, 2.0) for t in (0, 8)]
# 阶段 4：序列通路（编码器类型 × 层数 × 输入嵌入是否做图平滑）。
STAGE4 = [
    dict(seq_enc="gru", layers=1, graph_in=False),
    dict(seq_enc="gru", layers=1, graph_in=True),
    dict(seq_enc="gru", layers=2, graph_in=False),
    dict(seq_enc="lstm", layers=1, graph_in=False),
    dict(seq_enc="lstm", layers=1, graph_in=True),
    dict(seq_enc="transformer", layers=2, graph_in=False),
    dict(seq_enc="transformer", layers=2, graph_in=True),
    dict(seq_enc="transformer", layers=3, graph_in=False),
]

# 阶段 5：在最终序列编码器下复核容量（d × dropout × 学习率）。
# lr 用与基线一致的 5 点网格 —— 实测 5/7 数据集原先都顶到 2e-3 上界。
STAGE5 = [dict(d=d, dropout=dr, lr=lr)
          for d in (64, 128, 256) for dr in (0.1, 0.2) for lr in LR_GRID]

STAGES = {1: STAGE1, 2: STAGE2, 3: STAGE3, 4: STAGE4, 5: STAGE5}


def build_args(ds, cfg, epochs, batch, seed, train_max_users=0, patience=10):
    """把超参字典装成 run_kt.train_one 需要的 args 命名空间。"""
    a = argparse.Namespace()
    a.dataset = ds
    a.models = ["DGCLKT"]
    a.epochs = epochs
    a.batch = batch
    a.lr = cfg.get("lr", 2e-3)
    # 早停耐心值：曾硬编码为 5，而基线侧用 10 —— 实测 DGCL-KT 的 epochs_run
    # 中位数只有 10~18（几乎都停在 10 轮上下），基线是 19~28。虽因按"验证集最优
    # 轮次"取权重而不至于直接损失，但过小的耐心会截断"验证集先降后升"的再优化
    # 机会。与基线对齐为 10。
    a.patience = patience
    a.max_len = 200
    a.min_len = 3
    a.max_users = 0
    # 搜索阶段的训练集用户上限（0=全量）；只截断训练集，验证/测试保持全量。
    a.train_max_users = int(train_max_users or 0)
    a.seed = seed
    a.tag = "tune"
    a.ablation = False
    a.ab_no_graph = False
    a.ab_no_calib = False
    a.ab_no_scope = False
    a.ab_struct_only = False
    a.d = cfg.get("d", 128)
    a.layers = cfg.get("layers", 1)
    a.steps = cfg.get("steps", 2)
    a.hops = cfg.get("hops", 0)
    a.dropout = cfg.get("dropout", 0.2)
    a.gated_fuse = cfg.get("gated_fuse", False)
    a.layers = cfg.get("layers", 1)
    a.topm = cfg.get("topm", 0)
    a.wprior = cfg.get("wprior", 0.0)
    a.graph_in = cfg.get("graph_in", False)
    a.seq_enc = cfg.get("seq_enc", "transformer")
    return a


def _show_main(ds):
    """打印主实验里各模型的 (验证集, 测试集) 作为对照。"""
    mp = os.path.join(RESULTS, f"kt_{ds}_main.json")
    if not os.path.exists(mp):
        return
    with open(mp, encoding="utf-8") as f:
        m = json.load(f).get("results", {})
    print("[tune] 主实验对照 (model: val, test) =")
    for k, v in m.items():
        print(f"        {k:<8} val={v.get('best_valid_auc', float('nan')):.4f} "
              f"test={v['AUC']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3, 4, 5))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="", help="阶段 2 的基础配置（JSON 字符串）")
    ap.add_argument("--grid", default="", help="自定义网格（JSON 数组），留空则用内置")
    ap.add_argument("--train-max-users", type=int, default=0,
                    help="搜索阶段训练集用户上限（0=全量），只截断训练集")
    ap.add_argument("--patience", type=int, default=10,
                    help="早停耐心值（须与基线侧一致，默认 10）")
    a = ap.parse_args()

    with open(os.path.join(PROC, f"{a.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    adj, ac = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)
    print(f"[tune] dataset={a.dataset} K={obj['n_skill']} "
          f"train/val/test={len(obj['train'])}/{len(obj['valid'])}/{len(obj['test'])} "
          f"CSG={'on' if scope is not None else 'off'}")
    _show_main(a.dataset)

    if a.grid:
        grid = json.loads(a.grid)
        base = json.loads(a.base) if a.base else {}
    else:
        grid, base = STAGES[a.stage], (json.loads(a.base) if a.base else {})

    rows = []
    for i, cfg in enumerate(grid, 1):
        full = {**base, **cfg}
        t0 = time.time()
        try:
            r = train_one("DGCLKT",
                                 build_args(a.dataset, full, a.epochs, a.batch, a.seed,
                                            a.train_max_users, a.patience),
                                 obj, adj, ac, scope, s2i, a.seed)
        except Exception as e:                        # noqa: BLE001
            print(f"  [{i}/{len(grid)}] {full} → 失败: {type(e).__name__}: {e}")
            continue
        row = dict(**full, val=r.get("best_valid_auc"), test=r["AUC"], acc=r["ACC"],
                   epochs_run=r.get("epochs_run"), sec=round(time.time() - t0, 1))
        rows.append(row)
        v = row["val"] if row["val"] is not None else float("nan")
        print(f"  [{i}/{len(grid)}] {full} → val={v:.4f} test={row['test']:.4f} "
              f"({row['sec']:.0f}s)")

    if not rows:
        print("[tune] 无有效试验"); return
    best = max(rows, key=lambda r: (r["val"] if r["val"] is not None else -1))
    sel = {k: best[k] for k in (grid[0].keys())}
    print(f"\n[tune] 验证集最优：{sel}  val={best['val']:.4f} test={best['test']:.4f}")
    out = dict(dataset=a.dataset, stage=a.stage, base=base, epochs=a.epochs,
               seed=a.seed, n_trials=len(rows), rows=rows, best=best)
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, f"kt_tune_{a.dataset}_stage{a.stage}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[tune] saved", p)


if __name__ == "__main__":
    main()
