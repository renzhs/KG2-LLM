#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚合基准实验的 5 折结果，产出 mean±std 与论文用 LaTeX 表。

输入: results/kt_{ds}_f{k}_bench.json（run_kt.py --tag bench 逐折输出）
输出: results/kt_bench_{ds}.json         聚合结果（含逐折原始值）
      results/table_kt_bench.tex        所有数据集的合并对比表（若已全部就绪）
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import RESULTS                          # noqa: E402

MODEL_ORDER = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT",
               "SAINT", "SimpleKT", "DGCLKT"]
PRETTY = {"DGCLKT": "DGCL-KT (本文)", "BKT": "BKT", "BKT+": "BKT+", "DKT": "DKT", "DKT+": "DKT+",
          "DKVMN": "DKVMN", "SAKT": "SAKT", "AKT": "AKT", "GKT": "GKT",
          "SAINT": "SAINT", "SimpleKT": "SimpleKT"}
PRETTY_DS = {"assist09_akt": "ASSIST09", "assist15": "ASSIST15", "assist17": "ASSIST17",
             "statics11": "Statics11", "assist09": "ASSIST09(自建划分)",
             "assist12": "ASSIST12", "dnui_syn": "DNUI-IKG-S(自建)",
             "ednet": "EdNet"}


def load_fold(ds, k):
    """k 为折号；k=None 表示单划分数据集（读 kt_<ds>_bench.json）。"""
    cands = ([os.path.join(RESULTS, f"kt_{ds}_bench.json"),
              os.path.join(RESULTS, f"kt_{ds}_main.json")] if k is None else
             [os.path.join(RESULTS, f"kt_{ds}_f{k}_bench.json"),
              os.path.join(RESULTS, f"kt_{ds}_f{k}_main.json")])
    for p in cands:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def aggregate(ds, folds):
    per_model = {}
    per_fold = {}
    used = []
    for k in folds:
        d = load_fold(ds, k)
        if not d:
            continue
        used.append(k if k is not None else "all")
        for m, v in d.get("results", {}).items():
            if not isinstance(v, dict) or "AUC" not in v:
                continue
            per_model.setdefault(m, []).append(v)
            per_fold.setdefault(m, {})[(k if k is not None else "all")] = {
                kk: v.get(kk) for kk in ("AUC", "ACC", "RMSE")}
    summary = {}
    for m, vs in per_model.items():
        s = {}
        for kk in ("AUC", "ACC", "RMSE"):
            arr = np.array([v[kk] for v in vs if v.get(kk) is not None], dtype=float)
            s[kk] = dict(mean=round(float(arr.mean()), 4),
                         std=round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
                         n=len(arr))
        s["params"] = vs[0].get("params")
        s["train_sec_mean"] = round(float(np.mean([v.get("train_sec", 0) for v in vs])), 1)
        summary[m] = s
    # 每个数据集上的最优（按 AUC 均值）
    best = None
    if summary:
        best = max(summary, key=lambda m: summary[m]["AUC"]["mean"])
    # ---- 完整性：**必须**显式记录覆盖了多少折/多少模型 ----
    # 旧实现只写 summary（每个模型只留 n），下游无法知道"某模型是否只跑了 1 折"，
    # 于是 GKT 在 statics11 上只跑成 1 折，却被当成 5 折均值输出 0.8214±0.0000，
    # 粗看像"方差极小"，实为单折；而它恰是该数据集上最强的基线 —— 结论方向会被带偏。
    n_exp = len(folds)
    n_used = len(used)
    missing = [m for m in MODEL_ORDER if m not in summary]
    partial = {m: summary[m]["AUC"]["n"] for m in summary
               if summary[m]["AUC"]["n"] != n_used}
    return dict(dataset=ds, folds=used, expect_folds=n_exp, n_folds_used=n_used,
                complete=(not missing and not partial),
                missing_models=missing, partial_models=partial,
                summary=summary, per_fold=per_fold, best_model=best)


def completeness_report(dss):
    """汇总各数据集完整性，返回 (是否全部完整, 明细行)。"""
    ok_all, rows = True, []
    for ds in dss:
        p = os.path.join(RESULTS, f"kt_bench_{ds}.json")
        if not os.path.exists(p):
            ok_all = False
            rows.append((ds, "缺聚合产物", "", ""))
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("complete", False):
            rows.append((ds, "OK", f"{d.get('n_folds_used')}/{d.get('expect_folds')} 折",
                         f"{len(d.get('summary') or {})} 模型"))
            continue
        ok_all = False
        for m, n in sorted((d.get("partial_models") or {}).items()):
            rows.append((ds, f"折数不足 {m}", f"{n}/{d.get('n_folds_used')} 折", ""))
        for m in d.get("missing_models") or []:
            rows.append((ds, f"缺模型 {m}", "", ""))
    return ok_all, rows


def order_models(ms):
    known = [m for m in MODEL_ORDER if m in ms]
    others = sorted(m for m in ms if m not in MODEL_ORDER)
    return known + others


def write_table(dss):
    """跨数据集合并对比表（行=模型，列=数据集 AUC mean±std）。"""
    data = {}
    for ds in dss:
        p = os.path.join(RESULTS, f"kt_bench_{ds}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data[ds] = json.load(f)
    if not data:
        return
    allm = []
    for d in data.values():
        for m in d["summary"]:
            if m not in allm:
                allm.append(m)
    models = order_models(allm)
    # 每个数据集的最优均值，用于加粗
    bestv = {ds: max(d["summary"][m]["AUC"]["mean"] for m in d["summary"])
             for ds, d in data.items()}

    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{各数据集知识追踪性能对比（AUC，5 折均值$\\pm$标准差；"
             "AKT 官方划分）。粗体为该列最优，$\\dagger$ 表示本文方法；"
             "上标 $n/N$ 表示该单元格仅由 $n$ 折聚得（缺折或未跑满时不得写入论文）。}",
             "\\label{tab:kt_bench}",
             "\\begin{tabular}{l" + "c" * len(data) + "}", "\\toprule",
             "模型 & " + " & ".join(PRETTY_DS.get(ds, ds) for ds in data) + " \\\\",
             "\\midrule"]
    for m in models:
        cells = []
        for ds in data:
            s = data[ds]["summary"].get(m)
            if not s:
                cells.append("--")
                continue
            mu, sd = s["AUC"]["mean"], s["AUC"]["std"]
            txt = f"{mu:.4f}$\\pm${sd:.4f}"
            # 折数不足时必须显式标注，否则"单折"会被读成"方差极小"
            n, ntot = s["AUC"]["n"], data[ds].get("n_folds_used", s["AUC"]["n"])
            if n != ntot:
                txt += f"$^{{{n}/{ntot}}}$"
            if abs(mu - bestv[ds]) < 1e-9:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        name = PRETTY.get(m, m)
        if m == "DGCLKT":
            name = "\\textbf{" + name + "}$^\\dagger$"
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    p = os.path.join(RESULTS, "table_kt_bench.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--single", action="store_true", help="单划分数据集（无折号）")
    ap.add_argument("--all", action="store_true", help="聚合所有已有分折结果的基准数据集")
    ap.add_argument("--allow-partial", action="store_true", dest="allow_partial",
                    help="允许在数据不完整时仍写表（仅在排查阶段使用）。"
                         "默认拒绝：残缺矩阵（某模型缺折/缺模型）绝不允许静默写进论文。")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",") if x]
    if a.single:
        folds = [None]

    single_ds = []
    if a.all or not a.dataset:
        found = set()
        for p in glob.glob(os.path.join(RESULTS, "kt_*_f*_bench.json")):
            m = re.match(r"kt_(.+)_f\d+_bench\.json$", os.path.basename(p))
            if m:
                found.add(m.group(1))
        dss = sorted(found)
        # 单划分数据集（kt_<ds>_bench.json，无折号）同样须进主表，否则论文声称的
        # "7 个数据集构成 11×7 矩阵"与产物（4 列）不符。
        for p in glob.glob(os.path.join(RESULTS, "kt_*_bench.json")):
            m = re.match(r"kt_(.+)_bench\.json$", os.path.basename(p))
            if not m:
                continue
            g = m.group(1)
            # 排除：① 折文件（含 _f<k>）；② 本脚本自己的聚合产物 kt_bench_<ds>.json
            if re.search(r"_f\d+$", g) or "_f" in g or g.startswith("bench_"):
                continue
            single_ds.append(g)
        single_ds = sorted(single_ds)
    else:
        dss = [a.dataset]
        if a.single:
            single_ds, dss = [a.dataset], []

    for ds in dss:
        agg = aggregate(ds, folds)
        if not agg["summary"]:
            print(f"[{ds}] 无可用分折结果")
            continue
        p = os.path.join(RESULTS, f"kt_bench_{ds}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=1)
        flag = "" if agg["complete"] else "  [不完整!]"
        print(f"\n===== {PRETTY_DS.get(ds, ds)}  ({agg['n_folds_used']} 折: "
              f"{agg['folds']}){flag} =====")
        for m in order_models(agg["summary"]):
            s = agg["summary"][m]
            star = "  <== best" if m == agg["best_model"] else ""
            print(f"  {PRETTY.get(m, m):<16} AUC={s['AUC']['mean']:.4f}±{s['AUC']['std']:.4f}  "
                  f"ACC={s['ACC']['mean']:.4f}  RMSE={s['RMSE']['mean']:.4f}"
                  f"  n={s['AUC']['n']}{star}")
        if not agg["complete"]:
            for m, n in sorted(agg["partial_models"].items()):
                print(f"  [缺口] {m}: 仅 {n}/{agg['n_folds_used']} 折")
            for m in agg["missing_models"]:
                print(f"  [缺口] {m}: 未跑")
        print(f"  -> {p}")

    for ds in single_ds:
        agg = aggregate(ds, [None])
        if not agg["summary"]:
            continue
        p = os.path.join(RESULTS, f"kt_bench_{ds}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=1)
        print(f"\n===== {PRETTY_DS.get(ds, ds)}  (单划分) =====")
        for m in order_models(agg["summary"]):
            s = agg["summary"][m]
            star = "  <== best" if m == agg["best_model"] else ""
            print(f"  {PRETTY.get(m, m):<16} AUC={s['AUC']['mean']:.4f}  "
                  f"ACC={s['ACC']['mean']:.4f}  RMSE={s['RMSE']['mean']:.4f}{star}")
        print(f"  -> {p}")

    all_out = dss + single_ds
    if len(all_out) >= 2:
        ok, rows = completeness_report(all_out)
        if not ok:
            print("\n----- 完整性门禁：发现缺口 -----")
            for ds, kind, a1, a2 in rows:
                print(f"  {ds:<14} {kind:<16} {a1:<12} {a2}")
            if not a.allow_partial:
                print("[FATAL] 数据不完整，拒绝写出跨数据集对比表（tab:kt_bench）。")
                print("        先补跑缺口：bash experiments/run/resume_bench.sh")
                print("        或加 --allow-partial 仅用于排查（表内会以上标标注折数）。")
                return 1
            print("[warn] --allow-partial 生效：仍写表，但残缺单元格带 n/N 上标。")
        write_table(all_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
