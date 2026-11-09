#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""M10 敏感性（知识追踪侧）：把先修图的归纳范围从"全量序列"改成"只用训练学生"
（run_kt.py --graph-scope train，产物 *_gstrain.json）后，KT 精度指标是否变化。

对照口径
  main : results/kt_<ds>_f<k>_bench.json     —— 主表口径（全量序列归纳，全部模型共用同一张图）
  train: results/kt_<ds>_f<k>_gstrain.json   —— 去泄漏口径（仅训练学生归纳）

每个数据集输出
  (a) 逐模型 (k 折均值±标准差) 两口径并列 + Δ
  (b) 本文方法(DGCLKT) vs 该口径下最强基线的折均值差
  (c) 本文方法在该口径下是否为最优（argmax）

用法（在仓库根目录）
    ./env/bin/python experiments/tools/kt_gstrain_cmp.py [数据集名 ...]
产物
    results/kt_gstrain_cmp.json
    （stdout 另有一份人类可读的表）
"""
import glob
import json
import os
import re
import statistics as st
import sys

METHOD = "DGCLKT"
OUT = "results/kt_gstrain_cmp.json"


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def folds_of(ds):
    fs = set()
    for p in glob.glob("results/kt_%s_f*_bench.json" % ds):
        m = re.search(r"_f(\d+)_bench", p)
        if m:
            fs.add(int(m.group(1)))
    return sorted(fs)


def gather(ds, fs, tag):
    """tag: '' -> bench(main) ; 'gstrain' -> 去泄漏"""
    suffix = "_bench" if tag == "" else "_gstrain"
    per, used = {}, []
    for f in fs:
        d = load("results/kt_%s_f%d%s.json" % (ds, f, suffix))
        if not d:
            continue
        used.append(f)
        for m, v in (d.get("results") or {}).items():
            if v.get("AUC") is None:
                continue
            per.setdefault(m, []).append(float(v["AUC"]))
    return per, used


def ms(xs):
    if not xs:
        return None, None
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    dses = sys.argv[1:] or ["assist09_akt", "assist15", "assist17", "statics11"]
    out = {}
    for ds in dses:
        fs = folds_of(ds)
        rec = {"folds_main": fs, "reason": ""}
        if not fs:
            rec["reason"] = "无主口径 bench 分折文件"
            out[ds] = rec
            continue
        mainp, used_m = gather(ds, fs, "")
        gstrp, used_g = gather(ds, fs, "gstrain")
        rec["folds_main_used"] = used_m
        rec["folds_gstrain_used"] = used_g
        if not gstrp:
            rec["reason"] = "尚无 *_gstrain.json（批处理未完成）"
            out[ds] = rec
            print("== %-13s [跳过] %s" % (ds, rec["reason"]))
            continue
        only = sorted(set(mainp) & set(gstrp))
        table = {}
        for m in only:
            a, sa = ms(mainp[m])
            b, sb = ms(gstrp[m])
            table[m] = {"main_mean": a, "main_std": sa, "main_n": len(mainp[m]),
                        "gstrain_mean": b, "gstrain_std": sb, "gstrain_n": len(gstrp[m]),
                        "delta": (b - a) if (a is not None and b is not None) else None}
        rec["per_model"] = table
        # ---- 覆盖度自检（必须先于"读数字"）----
        # 动机（2026-09-28 发现）：首轮 batch_m10_kt.sh 用 PAR=4 在单卡上并发，
        # 多折触发 `CUDA error: unknown error`，该折只写入了前若干个模型，
        # 结果 JSON 里留下大量 `AUC: null`。gather() 把这些 null 当缺失**静默丢弃**，
        # 于是"某模型只有 1 折参与均值"被无声接受——论文若引用就成了
        # "用 1 折均值代表 5 折"的隐性过度声明。故这里必须显式告警。
        # 注意：n_full 必须在下方的 vs_baseline 循环**之前**算出（那里要用它）。
        n_full = len(used_g)
        short = {m: t["gstrain_n"] for m, t in table.items()
                 if t["gstrain_n"] < n_full}
        rec["gstrain_folds_full"] = n_full
        rec["gstrain_short"] = short
        # (b)(c) 两口径下的 本文方法 vs 最强基线
        for tag in ("main", "gstrain"):
            src = mainp if tag == "main" else gstrp
            if METHOD not in src:
                rec["vs_baseline_" + tag] = {"reason": "缺 %s" % METHOD}
                continue
            mine, _ = ms(src[METHOD])
            base = None
            for m, v in src.items():
                if m == METHOD:
                    continue
                mu, _ = ms(v)
                if mu is None:
                    continue
                if base is None or mu > base[1]:
                    base = (m, mu)
            rec["vs_baseline_" + tag] = {
                "method_mean": mine, "best_baseline": base[0] if base else None,
                "best_baseline_mean": base[1] if base else None,
                "gap": (mine - base[1]) if (base and mine is not None) else None,
                "method_is_top": bool(base and mine is not None and mine >= base[1]),
                # 覆盖度：折数不足时"本文最优/差距"都很可能是欠采样造成的假象，
                # 故把两边的折数一并落盘，供引用时判断能否作强主张。
                "method_n": len(src[METHOD]) if METHOD in src else 0,
                "best_baseline_n": len(src[base[0]]) if base else 0,
                "coverage_full": bool(base and len(src[METHOD]) == n_full
                                      and len(src[base[0]]) == n_full),
            }
        out[ds] = rec
        # ---- 人读表 ----
        print("=" * 78)
        print("== %s   折: main=%s gstrain=%s" % (ds, used_m, used_g))
        if short:
            print("   !! 覆盖不足：下列模型只用了 %d 折中的部分折（其余折 AUC 为 null，"
                  "疑 CUDA 错误中断）:" % n_full)
            for m, n in sorted(short.items(), key=lambda kv: kv[1]):
                print("        %-9s %d/%d 折" % (m, n, n_full))
            print("      解法：bash experiments/run/batch_m10b_kt.sh（PAR=2 重跑不全的折）")
        print("%-10s %18s %18s %9s" % ("模型", "main 均值±std", "train 均值±std", "Δ"))
        for m in sorted(table, key=lambda k: -(table[k]["main_mean"] or 0)):
            t = table[m]
            mark = " <== 本文" if m == METHOD else ""
            if short.get(m):
                mark = " [!%d/%d折]%s" % (t["gstrain_n"], n_full, mark)
            print("%-10s %10.4f±%.4f %10.4f±%.4f %+9.4f%s"
                  % (m, t["main_mean"], t["main_std"], t["gstrain_mean"], t["gstrain_std"],
                     t["delta"], mark))
        for tag in ("main", "gstrain"):
            b = rec.get("vs_baseline_" + tag) or {}
            if b.get("gap") is None:
                print("  [%s] %s" % (tag, b.get("reason", "n/a")))
            else:
                print("  [%s] 本文 %.4f  vs  最强基线 %s %.4f   Δ=%+.4f   本文最优=%s"
                      % (tag, b["method_mean"], b["best_baseline"], b["best_baseline_mean"],
                         b["gap"], b["method_is_top"]))
    # 防呆：本工具是**在远端跑**的（main/gstrain 分折文件都在远端 results/）。
    # 本地 if 只有 bench_raw/ 而无 results/kt_*_f*_bench.json，跑一遍会得到空 dict
    # 并把 results/kt_gstrain_cmp.json **静默清空**（2026-09-28 实际发生过一次）。
    if not any((r.get("per_model") for r in out.values())):
        print("[skip] 没有算出任何数据集（本机缺 main 口径分折文件？）"
              "——为防误清空，不写 %s" % OUT)
        return 2
    if os.path.dirname(OUT):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[write] %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
