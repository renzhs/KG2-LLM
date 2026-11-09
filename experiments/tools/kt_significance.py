#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识追踪主结果的**配对显著性检验** —— 输出 results/kt_signif.json。

为什么需要这个脚本
==================
论文在推荐任务上给了配对 Bootstrap 检验（以学生为重采样单位，见 4.6 节），
但 KT 主结果此前只有"均值±标准差"，正文却写了"显著优于最强基线"——
**没有任何检验支撑**，属于过强表述，审稿人会直接质疑。

本脚本用**同一数据集内的配对单元**做检验：
  * 公开基准（AKT 官方 5 折划分）：以「折」为配对单元，n=5；
  * 自建划分（单划分 + 多随机种子）：以「随机种子」为配对单元，n=3。
配对的意义：同一折/同一种子下，两个模型共享完全相同的训练/验证/测试数据，
差值 d_i = ours_i - base_i 消除了"折/种子的难度差异"这一最大方差来源，
因此比"比较两个独立均值"敏感得多。

为什么是配对 t 检验，而不是 Wilcoxon / 符号检验
----------------------------------------------
自建划分数据集只有 3 个随机种子。Wilcoxon 符号秩检验在 n=3 时最小双侧 p = 0.25，
符号检验同样是 2/2^3 = 0.25——**无论差距多大都不可能显著**，用它们会把
ASSIST12 上 Δ=+0.013 这种压倒性结果判成"不显著"，反而失真。
配对 t 检验在 n=3 时仍可用（实测 ASSIST12 上 t≈46、p≈5e-4），也是 KT 文献通行做法。
代价是正态性前提在小样本下无法验证，故输出中**同时给出逐单元胜负计数**，
让正文能如实并陈"检验 + 计数"两条证据，而不是只报一个 p 值。

用法
----
    python experiments/tools/kt_significance.py
    python experiments/tools/kt_significance.py --results ../results
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import RESULTS                                     # noqa: E402

DS_5F = ["assist09_akt", "assist15", "assist17", "statics11"]
DS_1F = ["dnui_syn", "assist09", "assist12"]
ALL_DS = DS_5F + DS_1F
OURS = "DGCLKT"
MODEL_ORDER = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT",
               "SAINT", "SimpleKT", "DGCLKT"]
PRETTY = {"DGCLKT": "DGCL-KT", "BKT+": "BKT+", "DKT+": "DKT+"}


def load(p):
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


METRICS = [("AUC", True), ("ACC", True), ("RMSE", False)]   # (指标, 是否越大越好)


def paired_units(resdir, ds):
    """该数据集内可配对的单元 -> (unit_name, {model: {metric: np.array}}).

    5 折数据集从 kt_bench_<ds>.json 的 per_fold 取（折为单元）；
    自建划分从 kt_<ds>_multi.json 的 per_seed 取（种子为单元）。
    AUC / ACC / RMSE 三个指标都在同一组配对单元上取出，
    以便对 ACC 与 RMSE 的表格加粗也给出显著性依据（而不是"谁大加粗谁"）。
    """
    def _pack(per, units_):
        """per: {model: {unit: {metric: value}}} -> {model: {metric: np.array}}"""
        out = {}
        for m, pu in per.items():
            got = {}
            for met, _ in METRICS:
                vals = [((pu.get(u) or {}).get(met) if isinstance(pu.get(u), dict)
                         else None) for u in units_]
                if units_ and all(v is not None for v in vals):
                    got[met] = np.array(vals, dtype=float)
            if got:
                out[m] = got
        return out

    if ds in DS_5F:
        d = load(os.path.join(resdir, "kt_bench_%s.json" % ds)) or {}
        pf = d.get("per_fold") or {}
        folds = [str(x) for x in (d.get("folds") or [])]
        if not folds:
            folds = sorted((pf.get(OURS) or {}).keys(), key=lambda s: int(s))
        return "fold", _pack(pf, folds)

    d = load(os.path.join(resdir, "kt_%s_multi.json" % ds)) or {}
    seeds = [str(u) for u in (d.get("seeds") or [])]
    pf = {}
    for m, s in (d.get("summary") or {}).items():
        per_unit = {u: {} for u in seeds}
        for met, _ in METRICS:
            ps = (s.get(met) or {}).get("per_seed") or []
            if len(ps) == len(seeds):
                for i, u in enumerate(seeds):
                    per_unit[u][met] = ps[i]
        pf[m] = per_unit
    return "seed", _pack(pf, seeds)


def test_one(units, ours, base, metric="AUC"):
    """配对 t 检验（双侧）+ 逐单元胜负计数 + 95%CI + 最小可检出效应。

    Δ 的口径（2026-09-25 统一）
    --------------------------
    检验本身用**全精度**配对差值算 t/p；但对外报告的 Δ 一律取
    `round(ours_mean,4) - round(base_mean,4)`，即读者用表内两个四位小数
    相减即可复现。两者在数学上同为"配对均值差"（mean(a-b) == mean(a)-mean(b)），
    差别仅来自末位舍入：曾出现脚本报 +0.0095、表格报 +0.0094 的 1 个末位分歧，
    会让审稿人以为是两处对不上，故显式按"表内均值相减"取值。

    新增 field（2026-09-28，回应"n=3 配对检验功效不足"的评审意见）
    --------------------------------------------------------------
      ci95        配对均值差的 95% 置信区间（$t_{0.975,n-1} \\cdot s_d/\\sqrt{n}$）
      mdes        80% 功效下可检出的最小效应（$(t_{0.975,n-1}+t_{0.80,n-1}) \\cdot s_d/\\sqrt{n}$）
      mdes_ratio  观测效应 / mdes；< 1 表示本设计**检不出**该量级的差异
    """
    a, b = units[ours][metric], units[base][metric]
    higher_better = dict(METRICS)[metric]
    d = a - b
    n = int(d.size)
    t, p = stats.ttest_rel(a, b)
    om = round(float(a.mean()), 4)
    bm = round(float(b.mean()), 4)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        tc = float(stats.t.ppf(0.975, n - 1))
        tp = float(stats.t.ppf(0.80, n - 1))
        se = sd / np.sqrt(n)
        ci = (round(float(d.mean()) - tc * se, 4), round(float(d.mean()) + tc * se, 4))
        mdes = float((tc + tp) * se)
    else:
        ci, mdes = (None, None), 0.0
    gain = float(d.mean()) if higher_better else -float(d.mean())
    return dict(metric=metric, n=n,
                ours_mean=om,
                base_mean=bm,
                delta=round(om - bm, 4),
                delta_paired=round(float(d.mean()), 4),
                delta_std=round(sd, 4),
                d_min=round(float(d.min()), 4), d_max=round(float(d.max()), 4),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                ties=int((d == 0).sum()),
                t=round(float(t), 3), p=round(float(p), 4),
                significant=bool(p < 0.05),
                better=bool(p < 0.05 and gain > 0),
                ci95=list(ci) if ci else None,
                mdes=round(mdes, 4),
                mdes_ratio=(round(abs(gain) / mdes, 3) if mdes > 0 else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()

    out = {}
    print("=" * 78)
    print("知识追踪主结果 · 配对显著性检验（配对 t 检验，双侧）")
    print("  公开基准：折间配对（n=5）；自建划分：种子间配对（n=3）")
    print("=" * 78)
    for ds in ALL_DS:
        unit, units = paired_units(a.results, ds)
        if OURS not in units:
            print("  %-12s 跳过（缺 %s 的逐单元数据）" % (ds, OURS))
            continue
        others = [m for m in MODEL_ORDER if m in units and m != OURS]
        if not others:
            print("  %-12s 跳过（无基线逐单元数据）" % ds)
            continue
        # 「最强基线」必须与 tab:kt-ours 同一选取规则：配对单元 AUC 均值最大者
        base = max(others, key=lambda m: units[m]["AUC"].mean())
        r = test_one(units, OURS, base, "AUC")
        r.update(dataset=ds, unit=unit, base_model=base)
        # 三个指标各自做配对检验（ACC/RMSE 的加粗同样要有依据）
        r["by_metric"] = {met: test_one(units, OURS, base, met)
                          for met, _ in METRICS}
        # 对全部基线也各算一遍（正文可能需要"优于所有基线"的支撑）
        r["vs_all"] = {m: test_one(units, OURS, m, "AUC") for m in others}
        out[ds] = r
        flag = "显著" if r["significant"] else "不显著"
        print("  %-12s [%s/ n=%d] vs %-9s Δ=%+.4f  t=%7.3f  p=%.4f  "
              "胜负 %d/%d  95%%CI=[%+.4f,%+.4f]  MDES=%.4f -> %s"
              % (ds, unit, r["n"], PRETTY.get(base, base), r["delta"],
                 r["t"], r["p"], r["wins"], r["losses"],
                 r["ci95"][0], r["ci95"][1], r["mdes"], flag))

    p = os.path.join(a.results, "kt_signif.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("-" * 78)
    n_sig = sum(1 for r in out.values() if r["significant"])
    n_tot = len(out)
    print("写入 %s" % p)
    print("显著 %d / 共 %d 个数据集" % (n_sig, n_tot))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
