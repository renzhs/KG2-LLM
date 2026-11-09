#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段B 全模型对比的**完整性校验器**（只读）。

为什么需要它
------------
阶段B 的编排判据曾经是「`results/kt_<ds>_f<k>_bench.json` 文件是否存在」。
但 run_kt.py 是**逐模型增量落盘**的：任一模型抛异常（最典型是 CUDA OOM）
都会被 `except` 吞掉、只往 JSON 里写一条 `{"error": ...}`，**文件照样生成**。
于是这个折会被永久判为"已完成"而不再重试 —— 表现为对比表里某个基线凭空变成
`--`，或更糟：某个模型只跑了 1 折，却被聚合脚本当成 5 折均值输出
（实测 GKT 在 statics11 上正是 1/5 折 → 表里显示 `0.8214±0.0000`，
粗看像"标准差极小"，实为**单折**，且它恰是该数据集上最强的基线）。

本脚本把"完整性"从「文件存在」提升为「每个 (数据集, 折, 模型) 都有非 error 的 AUC」，
供两处使用：
  1. run_bench.sh 阶段B 的跳过判据（不完整则重进该折，靠 run_kt.py 的模型级续跑只补缺口）；
  2. run_bench.sh 阶段C 的闸门（不完整则拒绝出表，杜绝残缺矩阵被静默写进论文）。

退出码: 0 = 全部完整；1 = 存在缺口。
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
REPO = os.path.dirname(EXPROOT)
sys.path.insert(0, EXPROOT)
from paths import RESULTS                          # noqa: E402

RUN_KT = os.path.join(EXPROOT, "run", "run_kt.py")


def expected_models():
    """从 run_kt.py 源码里取 ALL_MODELS —— 单一事实来源，避免两处列表漂移。"""
    src = open(RUN_KT, encoding="utf-8").read()
    m = re.search(r"ALL_MODELS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        raise RuntimeError("无法从 run_kt.py 解析 ALL_MODELS")
    return re.findall(r'"([^"]+)"', m.group(1))


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                              # noqa: BLE001
        return None


def check_unit(ds, fold, models, allow_single):
    """返回 (ok_models, bad: {model: 原因}, 是否整体缺失)。"""
    fn = (f"kt_{ds}_bench.json" if fold is None
          else f"kt_{ds}_f{fold}_bench.json")
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        return [], {}, "缺文件"
    d = load(p)
    if not isinstance(d, dict):
        return [], {}, "文件不可解析"
    res = d.get("results") or {}
    ok, bad = [], {}
    for m in models:
        v = res.get(m)
        if not isinstance(v, dict):
            bad[m] = "未运行"
        elif "error" in v:
            bad[m] = "error: " + str(v["error"])[:70]
        elif v.get("AUC") is None:
            bad[m] = "无 AUC"
        else:
            ok.append(m)
    return ok, bad, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="",
                    help="5 折数据集（逗号或空格分隔），如 'assist15,assist09_akt'")
    ap.add_argument("--single-datasets", default="", dest="singles",
                    help="单划分数据集，如 'assist12,assist09,dnui_syn'")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--models", default="", help="期望的模型（默认取 run_kt.py 的 ALL_MODELS）")
    ap.add_argument("--quiet", action="store_true", help="只输出结论行")
    a = ap.parse_args()

    models = ([x for x in re.split(r"[,\s]+", a.models) if x] or expected_models())
    folds = [int(x) for x in re.split(r"[,\s]+", a.folds) if x]
    dss5 = [x for x in re.split(r"[,\s]+", a.datasets) if x]
    dss1 = [x for x in re.split(r"[,\s]+", a.singles) if x]

    n_gap = 0
    rows = []
    for ds in dss5:
        for k in folds:
            ok, bad, whole = check_unit(ds, k, models, False)
            if whole:
                n_gap += 1
                rows.append((f"{ds}_f{k}", 0, len(models), whole, bad))
            elif bad:
                n_gap += 1
                rows.append((f"{ds}_f{k}", len(ok), len(models), None, bad))
    for ds in dss1:
        ok, bad, whole = check_unit(ds, None, models, True)
        if whole:
            n_gap += 1
            rows.append((ds, 0, len(models), whole, bad))
        elif bad:
            n_gap += 1
            rows.append((ds, len(ok), len(models), None, bad))

    nunit = len(dss5) * len(folds) + len(dss1)
    if not n_gap:
        print(f"[check] BENCH OK —— {nunit} 个「数据集×折」全部覆盖 "
              f"{len(models)} 个模型（{len(models) * nunit} 格无缺口）")
        return 0
    print(f"[check] BENCH 缺口 {n_gap}/{nunit} 个「数据集×折」：")
    if not a.quiet:
        for unit, nok, ntot, whole, bad in rows:
            print(f"   {unit:<18} {nok}/{ntot} 模型"
                  + (f"  ({whole})" if whole else ""))
            for m, why in bad.items():
                print(f"        - {m:<9} {why}")
    else:
        for unit, nok, ntot, whole, bad in rows:
            print(f"[check] 缺口 {unit:<18} {nok}/{ntot} 模型"
                  + (f" ({whole})" if whole else "")
                  + ("" if whole else "  缺:" + ",".join(bad)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
