#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学习率/容量搜索的**边界审计器**（只读）。

为什么需要它
------------
"我方与对手同等调参"是本实验协议的核心。但网格搜索有个隐蔽失效模式：
**最优值落在网格端点**（顶格或触底）—— 此时真实最优在网格之外，搜索没做完。
它不会报任何错，只会让被搜的那一侧**系统性偏弱**：

  · 若端点出现在**基线**上 → 基线被欠调 → 本文方法的领先幅度虚高（审稿人会质疑）；
  · 若端点出现在**本文方法**上 → 本文方法被自缚 → 等于自己让分。
  两种都发生过：本文方法 stage5 的 lr 曾只有 2 个候选且 5/7 数据集顶到 8e-3；
  基线侧 AKT 在 assist15 上单调偏好更小的 lr，最优落在下界 5e-4（真实最优在网格外）。

本脚本扫描 results/kt_basetune_*.json 的**每一次试验**，对每个 (数据集, 模型) 判断：
  1. 最优 lr 是否等于网格的最小/最大值（触底/顶格）；
  2. 最优容量是否等于容量候选的第一个/最后一个（升序时同样意味着端点）；
  3. 端点是否伴随**单调趋势**（相邻步长改善量未收敛 → 强烈提示最优在网格外）。
输出可直接用于决定"是否需要扩网格重搜"。

退出码: 0 = 无端点命中；1 = 存在端点命中。
"""
import argparse
import json
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
from paths import RESULTS                          # noqa: E402

META = ("model", "stage", "val", "test", "acc", "epochs_run", "sec",
        "cfg", "params", "tags")


def norm_lr(v):
    return None if v is None else round(float(v), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9, help="端点判定容差")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    hit = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "kt_basetune_*.json"))):
        ds = os.path.basename(p)[12:-5]
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            continue
        grid = sorted({norm_lr(r.get("lr")) for r in (d.get("rows") or [])
                       if r.get("lr") is not None})
        if len(grid) < 3:
            continue
        lo, hi = grid[0], grid[-1]
        by_model = {}
        for r in d.get("rows") or []:
            if r.get("val") is None or r.get("lr") is None:
                continue
            by_model.setdefault(r["model"], []).append(r)
        for m, rows in by_model.items():
            lr_rows = [r for r in rows if r.get("stage") == "lr"]
            if len(lr_rows) < 3:
                continue
            best = max(lr_rows, key=lambda r: r["val"])
            blr = norm_lr(best["lr"])
            if abs(blr - lo) < a.tol or abs(blr - hi) < a.tol:
                side = "触底(下界)" if abs(blr - lo) < a.tol else "顶格(上界)"
                # 端点是否伴随单调趋势：把 lr 升序排列看 val 是否单调
                ser = sorted(((norm_lr(r["lr"]), r["val"]) for r in lr_rows),
                             key=lambda t: t[0])
                vals = [v for _, v in ser]
                mono_dec = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
                mono_inc = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
                trend = ""
                if side.startswith("触底") and mono_dec:
                    trend = "（且 lr 越小 val 单调越好 → 最优极可能在网格外）"
                if side.startswith("顶格") and mono_inc:
                    trend = "（且 lr 越大 val 单调越好 → 最优极可能在网格外）"
                hit.append((ds, m, side, blr, best["val"], len(lr_rows), trend,
                            [(x, round(v, 4)) for x, v in ser]))
    if not hit:
        print("[check] LR 边界审计：无端点命中（各模型最优 lr 均落在网格内部）")
        return 0
    print(f"[check] LR 边界命中 {len(hit)} 处（端点意味着该侧搜索未做完）：")
    for ds, m, side, blr, val, n, trend, ser in hit:
        print(f"   {ds:<16} {m:<9} 最优 lr={blr:<8} val={val:.4f}  {side}{trend}")
        if not a.quiet:
            print(f"        lr→val: {ser}")
    print("\n处置建议：对这些 (数据集, 模型) 用更宽的网格重搜该模型的 lr 阶段"
          "（下方/上方各补 1~2 个点），再比较验证集；" 
          "若更优则更新 results/kt_baseline_cfg_<ds>.json 并重跑该数据集主表。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
