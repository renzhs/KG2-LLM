#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""去泄漏（--graph-scope train）复核的**同折配对**对照器。

为什么需要它
------------
``kt_gstrain_cmp.py`` 对每个模型各自取"已完成折"的均值。当不同模型的完成折数
不同（实测：ASSIST15 的 GKT 只有 2 折、ASSIST09-AKT 的 GKT 只有 3 折，
而本文方法 5 折齐全）时，跨模型的均值**不可比**——拿 5 折均值去比 2 折均值，
差值里混进了"折的构成"而非"方法差异"，这正是审稿人会抓的点。

本脚本改为**先对齐折、再比较**：本文方法与每个基线只在**两者都完成了的折**
上取均值（配对比较），并同时给出该折集上 main 口径的均值，于是
Δ = gstrain − main 是同折、同模型的差，干净可比。

输出
----
* ``results/kt_gstrain_pair.json``：机器可读；
* 终端表格：可直接抄进论文 §4.7.4(4)。

用法：``python experiments/tools/kt_gstrain_pair.py``
"""

import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "kt_gstrain_pair.json")
METHOD = "DGCLKT"
DATASETS = ["assist09_akt", "assist15", "assist17", "statics11"]
PRETTY = {"assist09_akt": "ASSIST09（AKT 划分）", "assist15": "ASSIST15",
          "assist17": "ASSIST17", "statics11": "Statics11"}

# 自建划分（ASSIST09S / ASSIST12S）不是 K 折交叉，而是单折 + 多种子。
# 这类数据集没有 ``kt_<ds>_f*_bench.json``，主口径在 ``kt_<ds>_multi.json``，
# 去泄漏口径在 ``kt_<ds>_gstrain.json``；两者都是「模型 → summary.per_seed」，
# 因此以 **seed 42** 作配对单位（表 14 的 folds 字段记作 [42]）。
SEED_DATASETS = ["assist09", "assist12"]
SEED_PRETTY = {"assist09": "ASSIST09S（自建划分）",
               "assist12": "ASSIST12S（自建划分）"}
SEED_UNIT = 42


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _auc(d, model):
    """从结果 JSON 里取某模型的 AUC。两种文件结构都兼容：
    ``{model: {AUC: x}}`` 与 ``{results: {model: {AUC: x}}}``。"""
    if not d:
        return None
    r = d.get("results") if isinstance(d.get("results"), dict) else d
    v = (r or {}).get(model)
    if isinstance(v, dict):
        a = v.get("AUC")
        return float(a) if a is not None else None
    return None


def folds_of(pattern, ds):
    """{fold: {model: auc}}，只保留有值的折。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, pattern % ds))):
        m = re.search(r"_f(\d+)_", os.path.basename(p))
        if not m:
            continue
        d = _load(p)
        if not d:
            continue
        r = d.get("results") if isinstance(d.get("results"), dict) else d
        cell = {}
        for k, v in (r or {}).items():
            if isinstance(v, dict) and v.get("AUC") is not None:
                cell[k] = float(v["AUC"])
        if cell:
            out[int(m.group(1))] = cell
    return out


def seed_cell(d, unit=SEED_UNIT):
    """从多模型汇总文件里取某一个种子的 {model: AUC}。

    主口径（``kt_<ds>_multi.json``）与去泄漏口径（``kt_<ds>_gstrain.json``）
    都是 ``run_kt_multi.py`` 的产物，summary 里每个模型有 ``AUC.per_seed``，
    其顺序与顶层 ``seeds`` 一致。
    """
    if not d:
        return {}
    seeds = d.get("seeds") or []
    if unit not in seeds:
        return {}
    i = seeds.index(unit)
    cell = {}
    for m, v in (d.get("summary") or {}).items():
        ps = ((v or {}).get("AUC") or {}).get("per_seed") or []
        if i < len(ps) and ps[i] is not None:
            cell[m] = float(ps[i])
    return cell


def seed_record(ds):
    """单折 + 多种子数据集的一条配对记录（配对单位 = seed 42）。"""
    main_d = _load(os.path.join(RES, "kt_%s_multi.json" % ds))
    gs_d = _load(os.path.join(RES, "kt_%s_gstrain.json" % ds))
    if not main_d or not gs_d:
        return None
    main_c = seed_cell(main_d)
    gs_c = seed_cell(gs_d)
    models = sorted(set(main_c) & set(gs_c))
    if METHOD not in models:
        return None
    rows = {}
    for m in models:
        rows[m] = {
            "folds": [SEED_UNIT], "n": 1,
            "main_mean": main_c[m], "main_std": 0.0,
            "gstrain_mean": gs_c[m], "gstrain_std": 0.0,
            "delta": gs_c[m] - main_c[m],
        }
    mine = rows[METHOD]
    base = [(m, r) for m, r in rows.items() if m != METHOD]
    base.sort(key=lambda kv: -kv[1]["gstrain_mean"])
    best = base[0] if base else None
    return {
        "dataset": ds, "pretty": SEED_PRETTY[ds],
        "unit": "seed（单折、单种子配对）",
        "anchor_folds": [SEED_UNIT], "n_anchor": 1,
        "method": mine,
        "best_baseline": best[0] if best else None,
        "best_baseline_row": best[1] if best else None,
        "gap_main": (mine["main_mean"] - best[1]["main_mean"]) if best else None,
        "gap_gstrain": (mine["gstrain_mean"] - best[1]["gstrain_mean"]) if best else None,
        "rows": rows,
        "incomplete": {},
    }


def report(rec):
    """打印一条配对记录的终端表格。"""
    anchor = rec["anchor_folds"]
    print("=" * 78)
    print("== %s  配对单位 = %s（%d）" % (rec["pretty"], anchor, len(anchor)))
    print("%-9s %4s %16s %16s %9s" % ("模型", "n", "main", "train", "Δ"))
    for m, r in sorted(rec["rows"].items(), key=lambda kv: -kv[1]["gstrain_mean"]):
        mark = "  <= 本文" if m == METHOD else ""
        print("%-9s %4d %8.4f±%.4f %8.4f±%.4f %+9.4f%s"
              % (m, r["n"], r["main_mean"], r["main_std"],
                 r["gstrain_mean"], r["gstrain_std"], r["delta"], mark))
    b = rec["best_baseline"]
    if b:
        print("  最强基线 %s：gap main %+.4f -> train %+.4f"
              % (b, rec["gap_main"], rec["gap_gstrain"]))
    if rec["incomplete"]:
        print("  [!] 折数不足（相对配对折集 %d 折）：%s"
              % (len(anchor), rec["incomplete"]))


def main():
    out = {}
    for ds in DATASETS:
        main_f = folds_of("kt_%s_f*_bench.json", ds)
        gs_f = folds_of("kt_%s_f*_gstrain.json", ds)
        if not main_f or not gs_f:
            continue
        models = sorted(set().union(*[set(c) for c in main_f.values()])
                        | set().union(*[set(c) for c in gs_f.values()]))
        if METHOD not in models:
            continue
        # 本文方法在两种口径下都完成了的折 = 配对基准折集
        anchor = sorted(set(f for f in gs_f if METHOD in gs_f[f])
                        & set(f for f in main_f if METHOD in main_f[f]))
        rows = {}
        for m in models:
            if m == METHOD:
                fs = anchor
            else:
                # 与本文方法对齐：只在两者都完成的折上比
                fs = sorted(set(f for f in anchor
                                if m in gs_f.get(f, {}) and m in main_f.get(f, {})))
            if not fs:
                continue
            mv = [main_f[f][m] for f in fs if m in main_f.get(f, {})]
            gv = [gs_f[f][m] for f in fs if m in gs_f.get(f, {})]
            if not mv or not gv:
                continue
            # 标准差用**样本标准差（ddof=1）**：表4、表5 与 kt_gstrain_cmp.py
            # 都用这个口径（n=5 时与总体标准差差 11.8%），混用会让读者
            # 拿本文的 ±std 与表4 对不上。
            rows[m] = {
                "folds": fs, "n": len(fs),
                "main_mean": st.mean(mv),
                "main_std": st.stdev(mv) if len(mv) > 1 else 0.0,
                "gstrain_mean": st.mean(gv),
                "gstrain_std": st.stdev(gv) if len(gv) > 1 else 0.0,
                "delta": st.mean(gv) - st.mean(mv),
            }
        mine = rows.get(METHOD)
        if not mine:
            continue
        base = [(m, r) for m, r in rows.items() if m != METHOD]
        # 最强基线：按去泄漏口径均值，并列时取靠前者（DGCLKT 之外的原表顺序）
        base.sort(key=lambda kv: -kv[1]["gstrain_mean"])
        best = base[0] if base else None
        rec = {
            "dataset": ds, "pretty": PRETTY[ds],
            "anchor_folds": anchor, "n_anchor": len(anchor),
            "method": mine,
            "best_baseline": best[0] if best else None,
            "best_baseline_row": best[1] if best else None,
            "gap_main": (mine["main_mean"] - best[1]["main_mean"]) if best else None,
            "gap_gstrain": (mine["gstrain_mean"] - best[1]["gstrain_mean"]) if best else None,
            "rows": rows,
            "incomplete": {m: r["n"] for m, r in rows.items() if r["n"] < len(anchor)},
        }
        out[ds] = rec
        report(rec)

    # 自建划分（单折 + 多种子）走 seed 配对分支
    for ds in SEED_DATASETS:
        rec = seed_record(ds)
        if rec is None:
            print("=" * 78)
            print("== %s  [跳过] 缺 kt_%s_multi.json 或 kt_%s_gstrain.json"
                  % (SEED_PRETTY[ds], ds, ds))
            continue
        out[ds] = rec
        report(rec)

    if not out:
        print("[abort] 没有可用的折文件，未写出 %s" % OUT)
        return 1
    # Windows 下直接 open("w") 覆盖已存在文件可能 PermissionError，
    # 故先写临时文件再原子替换。
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if os.path.exists(OUT):
        os.remove(OUT)
    os.replace(tmp, OUT)
    print("\n[write] %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
