#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基线调参**完整性**校验（只读，不修改任何产物）。

为什么需要它
------------
run_bench.sh 的 tune_base 原本用「results/kt_baseline_cfg_<ds>.json 是否含全部 8 个
基线键」来判断该数据集是否已调好。但 tune_baselines.py 在 **stage-lr 结束时就写入
best[name]**（见其 main() 的 stage-lr 分支），cfg 文件随之立刻包含全部 8 个模型键——
于是「lr 阶段跑满、容量阶段一次都没跑」的数据集也满足旧判据，被**永久 [skip]**。
后果：同一张对比表里各数据集的基线调参力度不一致（部分基线从未搜过容量），
而阶段 A3 的前置闸门用的是同一条判据，同样放行 —— 形成**静默的不公平比较**
（我方 DGCL-KT 是「结构+容量+lr」全搜，对手却只有 lr）。

实测发生在 dnui_syn：40 行试验（8 个模型的 lr 全满 + 仅 DKT/DKT+ 的容量），
cfg 里 8 个模型键齐全 → 被 [skip]，缺 11 个容量试验（DKVMN/SAKT/AKT/SAINT/
SimpleKT 各 2 + GKT 1）。

判据
----
对每个可调基线 m，要求
    lr  行数 ≥ len(lr_grid_of(m))      （GKT 为 3，其余为 5）
    cap 行数 ≥ len(CAP_GRID[m])        （DKT/DKT+ 1、GKT 1，其余 2）
与 tune_baselines.py 内部的 _stage_done 判据**同源**（直接复用其常量），
因此不会随网格改动而失配。

用法
----
    python experiments/tools/check_basetune.py --datasets dnui_syn
    python experiments/tools/check_basetune.py --datasets a,b,c --quiet

退出码：0 = 全部跑满；1 = 存在缺口（明细打到 stdout）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "run"))
sys.path.insert(0, os.path.dirname(HERE))

from paths import RESULTS                                    # noqa: E402
from tune_baselines import TUNE_MODELS, CAP_GRID, lr_grid_of  # noqa: E402


def expected(m):
    """该模型 lr / cap 两个阶段各自应完成的行数。"""
    return (("lr", len(lr_grid_of(m))), ("cap", len(CAP_GRID.get(m, []))))


def check(ds):
    """返回 (是否跑满, 明细行列表)。"""
    p = os.path.join(RESULTS, "kt_basetune_%s.json" % ds)
    if not os.path.exists(p):
        return False, ["  %-16s [缺失] 无 kt_basetune 产物" % ds]
    with open(p, encoding="utf-8") as f:
        rows = json.load(f).get("rows") or []

    ok, gaps, lines = True, [], []
    for m in TUNE_MODELS:
        parts = []
        for stage, exp in expected(m):
            n = sum(1 for r in rows
                    if r.get("model") == m and r.get("stage") == stage)
            if n < exp:
                ok = False
                gaps.append("%s/%s=%d/%d" % (m, stage, n, exp))
            parts.append("%s=%d/%d" % (stage, n, exp))
        lines.append("      %-9s %s" % (m, "  ".join(parts)))
    head = ("  %-16s [OK]   rows=%d" % (ds, len(rows)) if ok
            else "  %-16s [缺口] %s" % (ds, ", ".join(gaps)))
    return ok, [head] + lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="逗号分隔的数据集名（接受带空格的 DS_LIST 形式）")
    ap.add_argument("--quiet", action="store_true",
                    help="只打印有缺口的数据集（闸门用，保持日志干净）")
    a = ap.parse_args()

    ds_list = [d for d in a.datasets.replace(",", " ").split() if d]
    all_ok = True
    for ds in ds_list:
        ok, lines = check(ds)
        if ok and a.quiet:
            continue
        for ln in lines:
            print(ln)
        all_ok = all_ok and ok

    if all_ok:
        print("[check_basetune] 全部 %d 个数据集的 8 个可调基线均已跑满 lr+cap 两阶段。"
              % len(ds_list))
        return 0
    print("[check_basetune] 存在未跑满的数据集：基线调参力度与其余数据集不一致，"
          "会形成不公平比较。请补跑（tune_baselines.py 会按行续跑，只补缺口）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
