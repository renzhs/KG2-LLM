#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文图表与结果表生成。

读取 results/*.json，输出:
  figs/fig_*.pdf|png   论文插图（矢量 PDF + 300dpi PNG）
  results/table_*.tex  LaTeX 结果表
  results/summary.md   汇总（便于直接贴入骨架）

覆盖的实验：
  KT 主结果（含多种子 mean±std）、KT 消融、推荐（含 γ 敏感性）、路径、LLM/GES、MDRL、数据集统计
"""
import os, sys, json, glob, math, re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
from paths import RESULTS as RES, FIGS   # noqa: E402

# ---- 与论文表格共用同一套取数口径（单一事实来源）--------------------------
# 论文的 KT 数字有两个消费者：表格（tools/make_paper_tables.py）与插图（本文件）。
# 若两边各自取数，就会出现"同一 (数据集, 模型) 在图上一个数、表上另一个数"。
# 实测（2026-09-24）：出图原先走历史遗留的 kt_<ds>_main.json（调参前产物，每数据集
# 仅 7 个模型），而表格走 kt_bench_<ds>.json，于是 ASSIST12 的 DGCL-KT
# 图上 0.7324、表上 0.7705（差 0.038），图与表对"本文是否最优"结论相反。
# 因此这里**直接 import 表格侧的取数函数**，不再自己扫文件。
sys.path.insert(0, os.path.join(EXPROOT, "tools"))
from make_paper_tables import (                      # noqa: E402
    effective_kt_values, load_bench, load_multi, MODEL_ORDER as TBL_MODEL_ORDER)

# fig:kt-auc / fig:kt-seeds 只覆盖 3 个自建划分数据集（论文 caption 如此声明）
FIG_KT_DS = ["dnui_syn", "assist09", "assist12"]

# ---------------- 中文字体自动探测 ----------------
def setup_font():
    cand = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
            "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
            "SimHei", "Microsoft YaHei", "AR PL UMing CN", "Droid Sans Fallback"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for c in cand:
        if c in have:
            plt.rcParams["font.sans-serif"] = [c, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print("font:", c)
            return c
    for d in ("/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts"):
        for r, _, fs in os.walk(d):
            for f in fs:
                if any(k in f.lower() for k in ("cjk", "wqy", "han", "hei", "msyh", "simhei")):
                    try:
                        fp = font_manager.FontProperties(fname=os.path.join(r, f))
                        font_manager.fontManager.addfont(os.path.join(r, f))
                        plt.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
                        plt.rcParams["axes.unicode_minus"] = False
                        print("font(file):", fp.get_name(), os.path.join(r, f))
                        return fp.get_name()
                    except Exception:
                        continue
    print("font: DejaVu Sans (no CJK)")
    return None


FONT = setup_font()
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.autolayout": True,
})
C = dict(ours="#c0392b", base="#7f8c8d", b2="#95a5a6", b3="#bdc3c7",
         blue="#2e6da4", orange="#e67e22", green="#27ae60", purple="#8e44ad",
         teal="#16a085")
OURS_KEYS = ("本文",)

# 出图来源清单：记录每张图**实际绘制**的数值与其来源文件，
# 由 main() 落盘为 results/fig_provenance.json，供 paper/tools/audit_tables.py
# 的 G 段核对"图上的数与表上的数是否一致"（PDF 自身不可解析，故另存机器可读清单）。
_PROV = {}


def is_ours(name):
    return any(k in name for k in OURS_KEYS)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find(prefix):
    return sorted(glob.glob(os.path.join(RES, f"{prefix}*.json")))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  fig ->", os.path.join(FIGS, name + ".pdf/.png"))


def tex_write(name, lines):
    p = os.path.join(RES, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  table ->", p)
    return p


# ==========================================================================
# 数据集统计
# ==========================================================================
def fig_dataset_stats():
    stats = {}
    p = os.path.join(RES, "dataset_stats.json")
    if os.path.exists(p):
        stats = load(p)
    for f in find("kt_"):
        d = load(f)
        if not d.get("results"):
            continue
        ds = str(d.get("dataset", ""))
        if re.match(r".*_f\d+$", ds):
            continue                      # 逐折文件：统计量已在汇总文件里
        st = d.get("data_meta", {}).get("stats")
        if st:
            stats.setdefault(ds, st)
    if not stats:
        print("  [skip] fig_dataset_stats"); return
    # 7 数据集对比实验的数据集优先（并保持固定顺序），其余追加在后
    order = [k for k in (DS7 + ["dnui_real"]) if k in stats]
    order += [k for k in stats if k not in order]
    label = lambda k: DS7_LABEL.get(k, k)          # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    x = np.arange(len(order))
    labs = [label(k) for k in order]
    axes[0].bar(x, [stats[k].get("users", 0) for k in order], color=C["blue"])
    axes[0].set_ylabel("学习者数"); axes[0].set_xticks(x, labs, rotation=30, ha="right")
    axes[1].bar(x, [stats[k].get("interactions", 0) for k in order], color=C["orange"])
    axes[1].set_ylabel("交互记录数"); axes[1].set_xticks(x, labs, rotation=30, ha="right")
    axes[1].set_yscale("log")
    axes[2].bar(x, [stats[k].get("correct_rate") or 0 for k in order], color=C["green"])
    axes[2].set_ylabel("作答正确率"); axes[2].set_xticks(x, labs, rotation=30, ha="right")
    axes[2].axhline(0.5, color=C["base"], lw=0.8, ls=":")
    save(fig, "fig_dataset_stats")


# ==========================================================================
# 知识追踪
# ==========================================================================
KT_ORDER = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT",
            "SAINT", "SimpleKT", "DGCLKT"]
KT_LABEL = {"BKT": "BKT", "BKT+": "BKT+", "DKT": "DKT", "DKT+": "DKT+", "DKVMN": "DKVMN",
            "SAKT": "SAKT", "AKT": "AKT", "GKT": "GKT", "SAINT": "SAINT",
            "SimpleKT": "SimpleKT", "DGCLKT": "DGCL-KT (本文)"}
DS_ORDER = ["dnui_real", "dnui_syn", "assist09", "assist12"]

# 7 数据集对比实验的数据集顺序（AKT 官方 5 折划分在前，自建划分在后）
DS7_5F = ["assist09_akt", "assist15", "assist17", "statics11"]
DS7_1F = ["dnui_syn", "assist09", "assist12"]
DS7 = DS7_5F + DS7_1F
DS7_LABEL = {"assist09_akt": "ASSIST09", "assist15": "ASSIST15",
             "assist17": "ASSIST17", "statics11": "Statics11",
             "dnui_syn": "DNUI-IKG", "assist09": "ASSIST09(自建)",
             "assist12": "ASSIST12(自建)"}


def _ds_sort(dss):
    o = [d for d in DS_ORDER if d in dss]
    return o + [d for d in dss if d not in o]


def collect_kt():
    """知识追踪主结果 —— **与论文表格同源**。

    返回 {ds: {model: {"AUC": mean, "std": std, "src": "multi"|"bench"}}}。

    历史教训：本函数原先把 results/kt_<ds>_main.json（调参前的历史遗留产物）
    当成"主结果"，而论文表格读的是 kt_bench_<ds>.json，两条取数路径给出两套数字
    （实测 ASSIST12 的 DGCL-KT：图上 0.7324 / 表上 0.7705，差 0.038，
    图与表对"本文是否最优"给出相反结论）。
    现在改为直接调用表格侧的 effective_kt_values()，图与表在构造上不可能再分叉。
    """
    eff = effective_kt_values()
    res = {}
    for ds, models in eff.items():
        res[ds] = {m: {"AUC": v[1], "std": v[2] or 0.0, "src": v[0]}
                   for m, v in models.items()}
    return res


def collect_kt_bench():
    """7 数据集对比实验：读聚合文件 results/kt_bench_<ds>.json。

    返回 {ds: {model: {"mean":..,"std":..,"n":..}}}，
    5 折数据集为跨折 mean±std，自建单划分数据集 std=0。
    """
    out = {}
    for ds in DS7:
        p = os.path.join(RES, f"kt_bench_{ds}.json")
        if not os.path.exists(p):
            continue
        d = load(p)
        sm = d.get("summary") or {}
        if not sm:
            continue
        out[ds] = {m: dict(mean=v["AUC"]["mean"], std=v["AUC"]["std"],
                           n=v["AUC"].get("n"), folds=d.get("folds"))
                   for m, v in sm.items() if v.get("AUC")}
    return out


def fig_kt_bench(bt):
    """7 个数据集的 AUC 分组柱状图（2 行 × 4 列）。"""
    dss = [d for d in DS7 if d in bt]
    if not dss:
        print("  [skip] fig_kt_bench"); return
    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             squeeze=False)
    for i, ds in enumerate(dss):
        ax = axes[i // ncol][i % ncol]
        models = [m for m in KT_ORDER if m in bt[ds]]
        vals = [bt[ds][m]["mean"] for m in models]
        errs = [bt[ds][m]["std"] for m in models]
        cols = [C["ours"] if m == "DGCLKT" else C["base"] for m in models]
        ax.bar(range(len(models)), vals, color=cols, yerr=errs, capsize=2.2,
               error_kw=dict(lw=0.7, ecolor="#333333"))
        lo = min(v - e for v, e in zip(vals, errs))
        hi = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(lo - 0.03, hi + 0.025)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([KT_LABEL[m].replace(" (本文)", "") for m in models],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(DS7_LABEL[ds], fontsize=10)
        if i % ncol == 0:
            ax.set_ylabel("AUC")
    for j in range(len(dss), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    save(fig, "fig_kt_bench")


def collect_kt_multi():
    """多种子汇总 —— **与论文表格同源**（复用 tools/make_paper_tables.load_multi）。

    原先这里自带一份 mtime 判据，实测会把**内容自洽**的多种子误判为陈旧：
    日志里出现「多种子(07:58) 早于主结果(07:58)，本次不采用」——两个文件由 pscp
    同批拉回、mtime 趋同，秒级先后纯属拷贝次序，于是 ASSIST09/ASSIST12 的多种子
    被无端丢弃，fig:kt-seeds 只剩 DNUI-IKG 一个数据集、图上误差棒也集体消失。
    load_multi 以内容（种子级数值是否与主结果吻合）为主判据、mtime 仅兜底，
    正是为此；图与表共用同一函数，也保证图上的误差棒正是表里的 ±std。
    """
    out = {}
    for f in find("kt_"):
        if not f.endswith("_multi.json"):
            continue
        d = load(f)
        ds = d.get("dataset")
        if not ds:
            continue
        sm, seeds = load_multi(ds)                 # 内容优先的陈旧守卫
        if sm:
            out[ds] = dict(dataset=ds, seeds=seeds or d.get("seeds"), summary=sm)
        else:
            print(f"  [stale] {ds}: 多种子产物判为陈旧或与主结果不吻合，本次不采用")
    return out


def fig_kt_main(res, multi):
    """fig:kt-auc —— 3 个自建划分数据集上 11 个模型的 AUC 对比。

    覆盖范围由 FIG_KT_DS 固定（论文 caption 声明"三个数据集"）；
    数值一律来自 collect_kt()（= 论文表格同源），误差棒来自多种子产物。
    """
    dss = [d for d in FIG_KT_DS if d in res]
    if not dss:
        print("  [skip] fig_kt_main"); return
    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 4.0), squeeze=False)
    prov = {}
    for ax, ds in zip(axes[0], dss):
        models = [m for m in KT_ORDER if m in res[ds]]
        vals, errs, srcs = [], [], []
        for m in models:
            e = res[ds][m].get("std") or 0.0
            src = res[ds][m].get("src", "bench")
            if ds in multi and m in multi[ds]["summary"] \
                    and (multi[ds]["summary"][m].get("AUC") or {}).get("mean") is not None:
                s = multi[ds]["summary"][m]["AUC"]
                v, e, src = s["mean"], s.get("std") or 0.0, "multi"
            else:
                v = res[ds][m]["AUC"]
            vals.append(float(v)); errs.append(float(e)); srcs.append(src)
        prov[ds] = {m: dict(auc=round(v, 4), std=round(e, 4), src=s)
                    for m, v, e, s in zip(models, vals, errs, srcs)}
        cols = [C["ours"] if m == "DGCLKT" else C["base"] for m in models]
        if any(e > 0 for e in errs):
            ax.bar(range(len(models)), vals, color=cols, yerr=errs, capsize=2.0,
                   error_kw=dict(lw=0.7, ecolor="#333333"))
            ax.set_ylabel("AUC（均值$\\pm$标准差）")
        else:
            ax.bar(range(len(models)), vals, color=cols)
            ax.set_ylabel("AUC")
        ax.set_xticks(range(len(models)),
                      [KT_LABEL[m].replace(" (本文)", "\\n(本文)") for m in models],
                      rotation=45, ha="right", fontsize=8)
        lo = min(v - e for v, e in zip(vals, errs)); hi = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(lo - 0.03, hi + 0.02)
        ax.set_title(DS7_LABEL.get(ds, ds))
        for r, v in zip(range(len(models)), vals):
            ax.text(r, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    save(fig, "fig_kt_auc")
    _PROV["fig_kt_auc"] = dict(datasets=prov, source="kt_bench_<ds>.json (+kt_<ds>_multi.json)")


def table_kt(res, multi):
    """旧版 AUC 汇总表（仅供出图时肉眼核对）。

    注意：**论文不再使用本函数产出的表格**。论文正文的 KT 表由
    tools/make_paper_tables.py 统一生成（table_kt_bench / table_kt_pub /
    table_kt_ours），以保证"结果 -> 论文"只有一条路径、不会有第二处手写数字。
    这里把 label 从 `tab:kt` 改为 `tab:kt-main`，避免与论文里的表格标签撞车
    （两个 \\label{tab:kt} 会让 \\ref 静默指向错误的那张表）。
    """
    dss = _ds_sort(res)
    use_multi = {ds: ds in multi for ds in dss}
    tex = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{知识追踪模型在四个数据集上的 AUC（多随机种子均值$\\pm$标准差；"
           "无星号数据集为单次运行随机种子 42）}",
           "\\label{tab:kt-main}",
           "\\begin{tabular}{l" + "c" * len(dss) + "}", "\\toprule",
           "方法 & " + " & ".join(dss) + " \\\\", "\\midrule"]
    # 每列最优（以可用数值为准）
    best = {}
    for ds in dss:
        vals = []
        for m in KT_ORDER:
            if use_multi[ds] and m in multi[ds]["summary"] \
                    and multi[ds]["summary"][m]["AUC"].get("mean") is not None:
                vals.append(multi[ds]["summary"][m]["AUC"]["mean"])
            elif m in res.get(ds, {}):
                vals.append(res[ds][m]["AUC"])
        best[ds] = max(vals) if vals else None
    for m in KT_ORDER:
        cells = []
        for ds in dss:
            if use_multi[ds] and m in multi[ds]["summary"] \
                    and multi[ds]["summary"][m]["AUC"].get("mean") is not None:
                s = multi[ds]["summary"][m]["AUC"]
                txt = f"{s['mean']:.4f}$\\pm${s.get('std', 0):.4f}"
                num = s["mean"]
            elif m in res.get(ds, {}):
                num = res[ds][m]["AUC"]; txt = f"{num:.4f}"
            else:
                cells.append("--"); continue
            if best[ds] is not None and abs(num - best[ds]) < 1e-9:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        tex.append(KT_LABEL[m] + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_write("table_kt_main.tex", tex)


def fig_kt_seeds(multi):
    """fig:kt-seeds —— 3 个自建划分数据集上的逐种子 AUC 散布。

    为什么不用箱线图：每格只有 3 个随机种子，四分位数在 3 个点上没有统计意义。
    改用"逐种子散点 + 均值横线 + 标注 std"，既如实呈现可复现性，也不夸大方差。
    """
    dss = [d for d in FIG_KT_DS
           if d in multi and any((v.get("AUC") or {}).get("per_seed")
                                 for v in multi[d]["summary"].values())]
    if not dss:
        print("  [skip] fig_kt_seeds"); return
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.6), squeeze=False)
    prov = {}
    for ax, ds in zip(axes[0], dss):
        mm = multi[ds]["summary"]
        models = [m for m in KT_ORDER if m in mm and (mm[m].get("AUC") or {}).get("per_seed")]
        xs = np.arange(len(models))
        for i, m in enumerate(models):
            a = mm[m]["AUC"]
            ps = list(a["per_seed"])
            ax.scatter([i] * len(ps), ps, s=22, zorder=3,
                       color=C["ours"] if m == "DGCLKT" else "#34495e", alpha=0.9)
            ax.hlines(a["mean"], i - 0.24, i + 0.24,
                      color="#111111", lw=1.6, zorder=4)
            ax.text(i, a["mean"] + 0.0012, f"{a['mean']:.3f}", ha="center",
                    fontsize=6.5, zorder=5)
        prov[ds] = {m: dict(mean=mm[m]["AUC"]["mean"], std=mm[m]["AUC"]["std"],
                            per_seed=mm[m]["AUC"]["per_seed"])
                    for m in models}
        ax.set_xticks(xs, [KT_LABEL[m].replace(" (本文)", "") for m in models],
                      rotation=45, ha="right", fontsize=8)
        lo = min(min(mm[m]["AUC"]["per_seed"]) for m in models)
        hi = max(max(mm[m]["AUC"]["per_seed"]) for m in models)
        span = max(hi - lo, 0.002)
        ax.set_ylim(lo - 0.28 * span, hi + 0.28 * span)
        ax.set_ylabel("AUC（逐种子）")
        ax.set_title(f"{DS7_LABEL.get(ds, ds)}（{multi[ds]['seeds']} 种子）", fontsize=9.5)
    save(fig, "fig_kt_seeds")
    _PROV["fig_kt_seeds"] = dict(datasets=prov, source="kt_<ds>_multi.json")


def collect_kt_abl():
    """模块消融 —— 与论文表 tab:ab 同源（读 results/kt_<ds>_abl.json）。

    为什么不沿用旧的逐变体文件 kt_<ds>_ab-<tag>.json：那是「每变体单独跑一次」的
    旧产物，且各自带一个"完整模型"基线，与现在「同种子配对、多种子取均值」不是
    同一口径；继续用它出图会让图上的 Δ 与 4.7.2 节 tab:ab 里的 Δ 对不上
    （实测：旧文件停在 09-21，而 tab:ab 已在 09-24 按新配置重跑）。
    """
    out = {}
    for ds in FIG_KT_DS:
        p = os.path.join(RES, f"kt_{ds}_abl.json")
        if not os.path.exists(p):
            continue
        d = load(p)
        if d.get("summary"):
            out[ds] = dict(seeds=d.get("seeds"), summary=d["summary"],
                           paired=d.get("paired_vs_full") or {},
                           cfg_source=d.get("cfg_source"))
    return out


def fig_ablation():
    """fig:ablation —— 各模块去除后 AUC 的变化量（同种子配对差值）。"""
    abl = collect_kt_abl()
    if not abl:
        print("  [skip] fig_ablation"); return
    dss = [d for d in FIG_KT_DS if d in abl]
    tags = [t for t in AB_ORDER if any(t in abl[d]["summary"] for d in dss)]
    x = np.arange(len(tags)); w = 0.8 / max(1, len(dss))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    prov = {}
    for i, ds in enumerate(dss):
        full = (abl[ds]["summary"].get("完整模型") or {}).get("mean")
        vals = []
        for t in tags:
            v = (abl[ds]["summary"].get(t) or {}).get("mean")
            vals.append(np.nan if (v is None or full is None) else v - full)
        ax.bar(x + i * w, vals, w,
               label=f"{DS7_LABEL.get(ds, ds)}（完整模型 AUC={full:.4f}）")
        prov[ds] = dict(full=full,
                        **{t: (abl[ds]["summary"].get(t) or {}).get("mean") for t in tags})
    ax.axhline(0, color=C["ours"], lw=1.2)
    ax.set_xticks(x + w * (max(0, len(dss) - 1)) / 2,
                  [AB_PRETTY[t] for t in tags], rotation=16, ha="right")
    ax.set_ylabel("AUC 相对完整模型的变化量 $\\Delta$")
    ax.legend(fontsize=8)
    save(fig, "fig_ablation")
    _PROV["fig_ablation"] = dict(datasets=prov, source="kt_<ds>_abl.json")


AB_ORDER = ["w/o 图编码", "w/o 反向标定", "仅结构先验", "w/o CSG 门控"]
AB_PRETTY = {"w/o 图编码": "去除 CKG 图编码",
             "w/o 反向标定": "去除反向耦合标定",
             "仅结构先验": "去除经验边权\n（保留结构先验）",
             "w/o CSG 门控": "去除 CSG 门控"}


def table_ablation():
    """旧版消融表（仅供出图时肉眼核对）。论文正文用 4.7.2 节手写的 tab:ab。"""
    abl = collect_kt_abl()
    if not abl:
        print("  [skip] table_ablation"); return
    dss = [d for d in FIG_KT_DS if d in abl]
    tags = [t for t in AB_ORDER if any(t in abl[d]["summary"] for d in dss)]
    tex = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{消融实验：各模块对知识追踪性能（AUC）的贡献}",
           "\\label{tab:ablation}", "\\begin{tabular}{l" + "c" * len(dss) + "}", "\\toprule",
           "配置 & " + " & ".join(dss) + " \\\\", "\\midrule"]

    def _num(ds, key):
        v = (abl[ds]["summary"].get(key) or {}).get("mean")
        return f"{v:.4f}" if v is not None else "--"

    tex.append("DGCL-KT（完整） & " + " & ".join(_num(ds, "完整模型") for ds in dss) + " \\\\")
    for t in tags:
        tex.append(AB_PRETTY[t] + " & " + " & ".join(_num(ds, t) for ds in dss) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_write("table_ablation.tex", tex)


# ==========================================================================
# 推荐
# ==========================================================================
def fig_rec_cscr():
    fs = find("rec_")
    if not fs:
        print("  [skip] fig_rec_cscr"); return
    recs = {os.path.basename(f)[4:-5]: load(f) for f in fs}
    ds = "dnui_syn" if "dnui_syn" in recs else list(recs)[0]
    d = recs[ds]
    methods = [m for m in d if not m.startswith("_") and isinstance(d[m], dict)
               and "CSCR@10" in d[m]]
    if not methods:
        print("  [skip] fig_rec_cscr"); return
    order = sorted(methods, key=lambda m: d[m]["CSCR@10"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    cols = [C["ours"] if is_ours(m) else C["base"] for m in order]
    y = np.arange(len(order))
    axes[0].barh(y, [d[m]["CSCR@10"] for m in order], color=cols)
    axes[0].set_yticks(y, order, fontsize=8.5)
    axes[0].set_xlabel("CSCR@10（跨作用域串味率，越低越好）")
    axes[1].barh(y, [d[m].get("sNDCG@10", d[m].get("NDCG@10", 0)) for m in order], color=cols)
    axes[1].set_yticks(y, [""] * len(order))
    axes[1].set_xlabel("作用域内 NDCG@10（越高越好）")
    for a in axes:
        a.invert_yaxis()
    fig.suptitle(f"{ds}：合规性 vs 精度", y=1.02)
    save(fig, "fig_rec_cscr")


def table_rec():
    for f in find("rec_"):
        ds = os.path.basename(f)[4:-5]
        d = load(f)
        methods = [m for m in d if not m.startswith("_") and isinstance(d[m], dict)
                   and "CSCR@10" in d[m]]
        if not methods:
            continue
        order = sorted(methods, key=lambda m: d[m]["CSCR@10"], reverse=True)
        cols = ["sRecall@10", "sNDCG@10", "Recall@10", "NDCG@10", "CSCR@10", "CSCR@20"]
        var = d.get("_variant") or {}
        cap = (f"{ds} 上学习资源推荐：作用域内精度与跨作用域串味率"
               f"（$s$ 前缀表示作用域内指标）")
        if var:
            cap += (f"。验证集选定：就绪度项={var.get('ready')}、协同项={var.get('gcf')}、"
                    f"$\\gamma^\\ast$={var.get('gamma')}")
        tex = ["\\begin{table}[htbp]", "\\centering",
               f"\\caption{{{cap}}}",
               f"\\label{{tab:rec_{ds}}}",
               "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
               "方法 & sRecall@10 & sNDCG@10 & Recall@10 & NDCG@10 & "
               "CSCR@10 $\\downarrow$ & CSCR@20 $\\downarrow$ \\\\", "\\midrule"]
        bnd = max(d[m].get("sNDCG@10", 0) for m in methods)
        bnd2 = min(d[m]["CSCR@10"] for m in methods)
        sig = d.get("_sig") or {}
        has_sig = False
        for m in order:
            v = d[m]
            sg = sig.get(m) or {}
            mk = ""
            if sg.get("win"):            # 本文方法显著优于该方法
                mk, has_sig = "$^{\\ddagger}$", True
            elif sg.get("ci") and sg["ci"][1] < 0:   # 该方法显著优于本文方法
                mk, has_sig = "$^{\\star}$", True

            def fmt(key, bold=False, mark=""):
                x = v.get(key)
                if x is None:
                    return "--"
                s = f"{x:.4f}"
                return ("\\textbf{" + s + "}" if bold else s) + mark
            name = m
            if is_ours(m):
                name = "\\textbf{" + m + "}"
            tex.append(f"{name} & {fmt('sRecall@10')} & "
                       f"{fmt('sNDCG@10', abs(v.get('sNDCG@10', -1) - bnd) < 1e-9, mk)} & "
                       f"{fmt('Recall@10')} & {fmt('NDCG@10')} & "
                       f"{fmt('CSCR@10', abs(v['CSCR@10'] - bnd2) < 1e-9)} & "
                       f"{fmt('CSCR@20')} \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        if has_sig:
            tex.append("\\par\\smallskip\\footnotesize "
                       "配对 bootstrap 检验（以 DGCL-Rec 为参照，重采样 2000 次）："
                       "$^{\\ddagger}$ 本文方法显著更优；$^{\\star}$ 该方法显著更优（$p<0.05$）。")
        tex.append("\\end{table}")
        tex_write(f"table_rec_{ds}.tex", tex)


def fig_rec_gamma():
    """γ（图谱协同权重）敏感性曲线：验证集选参 + 测试集表现。

    同一数据集现含多个 (就绪度项, 协同项) 变体；只把「验证集选中的变体」画成实线，
    其余变体画淡色虚线，避免把不同变体的点连成锯齿状折线。
    """
    fs = find("rec_")
    rows = []
    for f in fs:
        d = load(f)
        sw = d.get("_gamma_sweep")
        if sw:
            rows.append((os.path.basename(f)[4:-5], d.get("_gamma"), sw, d.get("_variant")))
    if not rows:
        print("  [skip] fig_rec_gamma"); return
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (ds, gstar, sw, var) in enumerate(rows):
        col = [C["ours"], C["blue"], C["green"]][i % 3]
        keys, seen = [], set()
        for s in sw:
            k = (s.get("ready"), s.get("gcf"))
            if k not in seen:
                seen.add(k); keys.append(k)
        for rk, gk in keys:
            sel = [s for s in sw if s.get("ready") == rk and s.get("gcf") == gk]
            if not sel:
                continue
            chosen = bool(var) and rk == var.get("ready") and gk == var.get("gcf")
            xs = [s["gamma"] for s in sel]; ys = [s["val_sNDCG10"] for s in sel]
            if chosen:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=col,
                        label=f"{ds}（选中 {rk}+{gk}）")
            else:
                ax.plot(xs, ys, lw=1.0, ls="--", alpha=0.35, color=col,
                        label=f"{ds}·{rk}+{gk}")
        if gstar is not None:
            ax.axvline(gstar, color=col, ls=":", lw=1)
    ax.set_xlabel(r"图谱协同权重 $\gamma$")
    ax.set_ylabel("验证集作用域内 NDCG@10")
    ax.set_title(r"$\gamma$ 敏感性（虚线为验证集选出的 $\gamma^\ast$）")
    ax.legend(fontsize=7)
    save(fig, "fig_rec_gamma")


# ==========================================================================
# 学习路径
# ==========================================================================
PATH_COLS = [("完备性", "完备性"), ("完备性_s", "完备性$_s$"),
             ("可执行覆盖率", "可执行覆盖率$\\uparrow$"), ("先修合规率", "先修合规率$\\uparrow$"),
             ("难度平滑度", "难度平滑度$\\uparrow$"), ("路径串味率", "路径串味率$\\downarrow$")]


def table_path():
    for f in find("path_"):
        ds = os.path.basename(f)[5:-5]
        d = load(f)
        if not isinstance(d, dict) or not d:
            continue
        chosen = d.get("_selected")
        methods = [m for m in d if not m.startswith("_")]
        if not methods:
            continue
        keys = [k for k, _ in PATH_COLS]
        tex = ["\\begin{table}[htbp]", "\\centering",
               f"\\caption{{{ds} 上学习路径推荐质量（选点策略经验证集按可执行覆盖率选定"
               f"{'：' + chosen if chosen else ''}）}}",
               f"\\label{{tab:path_{ds}}}",
               "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
               "方法 & " + " & ".join(l for _, l in PATH_COLS) + " \\\\", "\\midrule"]
        # 最优值（EC 与合规率越大越好，串味率越小越好）
        bnd = {}
        for k in ("完备性_s", "可执行覆盖率", "先修合规率", "难度平滑度"):
            vs = [d[m].get(k) for m in methods if d[m].get(k) is not None]
            bnd[k] = max(vs) if vs else None
        vs = [d[m].get("路径串味率") for m in methods if d[m].get("路径串味率") is not None]
        bnd["路径串味率"] = min(vs) if vs else None
        for m in methods:
            cells = []
            for k in keys:
                v = d[m].get(k)
                if v is None:
                    cells.append("N/A"); continue
                s = f"{v:.4f}" if isinstance(v, float) else str(v)
                b = bnd.get(k)
                if b is not None and isinstance(v, float) and abs(v - b) < 1e-9:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            name = ("\\textbf{" + m + "}") if m == chosen else m
            tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        tex_write(f"table_path_{ds}.tex", tex)


# ==========================================================================
# LLM / GES
# ==========================================================================
GES_METS = [("HR", "幻觉率 HR $\\downarrow$"), ("ETR", "证据可追溯率 ETR $\\uparrow$"),
            ("SCR", "作用域合规率 SCR $\\uparrow$"),
            ("PR", "先修合规率 PCR $\\uparrow$"),
            ("NDCG@10", "NDCG@10 $\\uparrow$")]


def fig_ges():
    fs = find("ges_")
    if not fs:
        print("  [skip] fig_ges"); return
    d = load(fs[0])
    tags = [t for t in ("Direct", "GES") if t in d]
    if not tags:
        print("  [skip] fig_ges"); return
    fig, axes = plt.subplots(1, len(GES_METS), figsize=(3.1 * len(GES_METS), 3.2))
    for ax, (k, lab) in zip(axes, GES_METS):
        vals = [d[t].get(k, 0) for t in tags]
        cols = [C["base"], C["ours"]][:len(tags)]
        ax.bar(range(len(tags)), vals, color=cols, width=0.55)
        ax.set_xticks(range(len(tags)), ["Direct", "GES"])
        ax.set_title(lab, fontsize=9.5)
        ax.set_ylim(0, max(1.0, max(vals) * 1.25))
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "fig_ges")


def table_ges():
    fs = find("ges_")
    if not fs:
        print("  [skip] table_ges"); return
    d = load(fs[0])
    rows = [t for t in ("Direct", "GES") if t in d]
    if not rows:
        print("  [skip] table_ges"); return
    keys = ["HR", "ETR", "SCR", "PR", "NDCG@10", "n"]
    heads = ["HR $\\downarrow$", "ETR $\\uparrow$", "SCR $\\uparrow$",
             "先修合规率 PCR $\\uparrow$", "NDCG@10 $\\uparrow$", "返回条数"]
    tex = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{大语言模型推荐：图谱证据子图与作用域门控的作用"
           "（两法使用**同一召回池**，仅差作用于过滤与证据注入）}",
           "\\label{tab:ges-legacy}", "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
           "方法 & " + " & ".join(heads) + " \\\\", "\\midrule"]
    for t in rows:
        cells = []
        for k in keys:
            v = d[t].get(k)
            cells.append("--" if v is None else f"{v:.4f}")
        name = "\\textbf{GES（本文）}" if t == "GES" else t
        tex.append(name + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_write("table_ges.tex", tex)


def fig_ges_case():
    """案例分析：Direct 越界推荐 vs GES 合规推荐（堆叠条形）。"""
    fs = find("ges_")
    if not fs:
        print("  [skip] fig_ges_case"); return
    d = load(fs[0])
    sc = d.get("_probe_summary") or {}
    cases = d.get("_cases") or []
    if not cases:
        print("  [skip] fig_ges_case"); return
    labels, out_cnt, in_cnt = [], [], []
    for c in cases[:8]:
        labels.append(f"#{c['student_row']}")
        out_cnt.append(c.get("n_direct_out_of_scope", 0))
        in_cnt.append(max(0, c.get("n_direct", 0) - c.get("n_direct_out_of_scope", 0)))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(x, in_cnt, color=C["green"], label="作用域内（合规）")
    ax.bar(x, out_cnt, bottom=in_cnt, color="#c0392b", label="作用域外（串味）")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Direct 基线推荐条数")
    ax.set_xlabel("学生编号")
    ax.set_title("Direct 基线的跨作用域串味实例（GES 在同批学生上串味为 0）")
    ax.legend(fontsize=8)
    save(fig, "fig_ges_case")


# ==========================================================================
# MDRL
# ==========================================================================
def fig_mdrl():
    fs = find("mdrl_")
    if not fs:
        print("  [skip] fig_mdrl"); return
    d = load(fs[0])
    rc = d.get("replan_counts") or {}
    if not rc:
        print("  [skip] fig_mdrl"); return
    labels = list(rc); vals = [rc[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cols = [C["base"], C["orange"], C["ours"], C["teal"]][:len(labels)]
    b = ax.bar(range(len(labels)), vals, color=cols)
    ax.set_xticks(range(len(labels)), labels, rotation=12, fontsize=8.5)
    ax.set_ylabel("重规划触发次数")
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.01, f"{v}", ha="center", fontsize=9)
    save(fig, "fig_mdrl")


# ==========================================================================
# KG^2-LLM 系统架构图
# ==========================================================================
def fig_framework():
    """KG^2-LLM 总体架构示意图（学术灰阶版，2026-10-14 重绘）。

    版式契约（对齐《计算机工程与应用》插图要求）：
    * figsize 宽 = 最终印刷宽（0.80*\\textwidth ≈ 14.2cm = 5.6in），
      因此 matplotlib 的 fontsize 即最终印刷字号，所见即所得；
    * 图内文字不小于 6 号（7.5pt）：主文字 7.5pt、组件名 8pt、
      次要标注 6.8pt（脚线级别，仍 ≥ CEA 图内文字下限的宽容读法）；
    * 图内不放总标题——图题由文档 \\caption（9pt 小 5 号）承担；
    * 灰阶配色（黑白印刷安全）：组件盒浅灰底 + 细深灰边框，
      外部依赖（LLM）白底虚线框，反馈回路黑色虚线，箭头黑色细线；
    * 层级：应用层（下游任务）/ 方法层（四个组件）/ 数据层（双图谱底座），
      左缘竖排层标签；组件名与正文严格一致（DGCL-KT / CSG / GES / MDRL）。
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # 灰阶语义色
    INK = "#1a1a1a"        # 主文字 / 边框
    GREY = "#595959"       # 次要文字
    FILL = "#f2f2f2"       # 组件盒底
    FILL_D = "#e4e4e4"     # 数据层底座（略深，示"底"）
    EDGE = "#333333"
    fig, ax = plt.subplots(figsize=(5.85, 3.5))
    ax.set_xlim(0, 117); ax.set_ylim(6.5, 76.5); ax.axis("off")
    ax.set_facecolor("white")

    def box(x, y, w, h, lines, fs=7.5, bold_first=True, face=FILL,
            edge=EDGE, lw=0.7, ls="-", name_fs=8.0):
        """lines: [(text, weight_override|None), ...]；首行默认加粗。"""
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.15,rounding_size=0.5",
                           linewidth=lw, edgecolor=edge, facecolor=face,
                           linestyle=ls, zorder=3)
        ax.add_patch(p)
        n = len(lines)
        lh = h / (n + 0.35)                      # 行高自适应
        for i, (t, wbold) in enumerate(lines):
            ax.text(x + w / 2, y + h - (i + 0.62) * lh, t,
                    ha="center", va="center", zorder=4,
                    fontsize=(name_fs if (i == 0 and bold_first) else fs),
                    fontweight=("bold" if (i == 0 and bold_first) else
                                (wbold or "normal")),
                    color=INK, linespacing=1.25)

    def arrow(p1, p2, ls="-", lw=0.8, rad=0.0, style="-|>", z=2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=INK,
                                     linewidth=lw, linestyle=ls,
                                     mutation_scale=7, zorder=z,
                                     connectionstyle=f"arc3,rad={rad}"))

    # ---------------- 全局约束条 ----------------
    ax.text(58, 73.6, "全局约束：C1 先序约束 · C2 学段—专业作用域约束 · C3 学习负荷约束",
            ha="center", fontsize=7.0, color=GREY)
    ax.plot([8, 108], [70.6, 70.6], color=EDGE, lw=0.5)

    # ---------------- 应用层：两个下游任务 ----------------
    t1 = box(13, 59.5, 40, 8.4,
             [("资源推荐  $f:(u,t)\\!\\rightarrow\\!\\mathrm{TopK}(\\mathcal{M})$", None),
              ("指标：$s$NDCG@10、串味率 CSCR", "normal")], face="#ffffff")
    t2 = box(59, 59.5, 40, 8.4,
             [("路径推荐  $g:(u,\\mathrm{ctx}(u))\\!\\rightarrow\\!\\pi$", None),
              ("指标：可执行覆盖率 EC、先修合规率", "normal")], face="#ffffff")

    # ---------------- 方法层：四个组件 ----------------
    y3, h3, w3 = 36.5, 14.5, 25.0
    xs = [3.0, 30.5, 58.0, 85.5]
    box(xs[0], y3, w3, h3,
        [("DGCL-KT", None), ("双图谱协同", None), ("知识追踪", None),
         ("（创新点 1）", "normal")], name_fs=8.0, fs=7.2)
    box(xs[1], y3, w3, h3,
        [("CSG", None), ("学段—专业", None), ("语义约束门控", None),
         ("（创新点 2）", "normal")], name_fs=8.0, fs=7.2)
    box(xs[2], y3, w3, h3,
        [("GES", None), ("证据子图约束的", None), ("检索增强生成", None),
         ("（创新点 3）", "normal")], name_fs=8.0, fs=7.2)
    box(xs[3], y3, w3, h3,
        [("MDRL", None), ("掌握度驱动的", None), ("动态路径重规划", None),
         ("（创新点 4）", "normal")], name_fs=8.0, fs=7.2)

    # LLM 外部依赖（GES 下方，虚线框表示外部服务）
    llm_x, llm_y, llm_w, llm_h = xs[2], 26.5, w3, 6.6
    box(llm_x, llm_y, llm_w, llm_h,
        [("大语言模型（DeepSeek）", None), ("约束解码＋输出校验器", None)],
        fs=6.8, name_fs=7.2, face="#ffffff", ls=(0, (3, 1.6)), lw=0.7)
    arrow((llm_x + llm_w / 2, y3), (llm_x + llm_w / 2, llm_y + llm_h),
          style="<|-|>", lw=0.7)

    # 组件 → 任务
    for xc, xt in ((xs[0] + w3 / 2, 26), (xs[1] + w3 / 2, 44),
                   (xs[2] + w3 / 2, 68), (xs[3] + w3 / 2, 86)):
        arrow((xc, y3 + h3), (xt, 59.5), lw=0.7)

    # ---------------- 数据层：双图谱底座 ----------------
    y2, h2 = 8.5, 11.5
    box(5.5, y2, 39, h2,
        [("课程知识图谱 CKG", None),
         ("$G^{c}=(\\mathcal{V}^{c},\\mathcal{E}^{c})$", "normal"),
         ("知识点 / 知识单元 / 资源 / 能力指标", "normal"),
         ("PREVIOUS · PARENT · COVERS · ALIGNS", "normal")],
        fs=6.8, name_fs=7.5, face=FILL_D)
    box(67.5, y2, 39, h2,
        [("学习者认知图谱 LKG", None),
         ("$G^{l}_{u}=(\\mathcal{K}_{u},\\mathcal{E}_{u})$，边属性 $(p_{u,i},d_{u,i},c_{u,i})$", "normal"),
         ("四态：未解锁 · 学习中 · 已通过 · 已精通", "normal")],
        fs=6.8, name_fs=7.5, face=FILL_D)

    # 双向耦合（中间 23 单位宽的走廊里放标注）
    arrow((44.5, y2 + h2 * 0.68), (67.5, y2 + h2 * 0.68), lw=0.9)
    ax.text(56, y2 + h2 * 0.68 + 1.6, "正向：结构$\\rightarrow$学情",
            ha="center", fontsize=6.8, color=INK)
    arrow((67.5, y2 + h2 * 0.26), (44.5, y2 + h2 * 0.26), lw=0.9)
    ax.text(56, y2 + h2 * 0.26 - 2.0, "反向：学情$\\rightarrow$结构",
            ha="center", fontsize=6.8, color=INK)

    # 底座 → 组件（CKG 供结构，LKG 供学情；证据子图用虚线示"检索式"取用）
    arrow((15.5, y2 + h2), (15.5, y3), lw=0.7)             # CKG→DGCL-KT
    arrow((36, y2 + h2), (43, y3), lw=0.7)                 # CKG→CSG
    arrow((44, y2 + h2), (62, y3), lw=0.6, ls=(0, (2, 1.4)))   # CKG→GES 证据子图
    arrow((96, y2 + h2), (98, y3), lw=0.7)                 # LKG→MDRL
    # （DGCL-KT 的学情输入经由 LKG⇄CKG 双向耦合折算，不另画长线以免穿盒）

    # ---------------- 徽章反馈闭环（右缘绕行）----------------
    bx = 113.0
    ax.plot([106.5, bx], [y2 + h2 / 2, y2 + h2 / 2], color=INK,
            lw=0.7, ls=(0, (3, 1.6)), zorder=1)
    ax.plot([bx, bx], [y2 + h2 / 2, y3 + h3 * 0.55], color=INK,
            lw=0.7, ls=(0, (3, 1.6)), zorder=1)
    arrow((bx, y3 + h3 * 0.55), (110.5, y3 + h3 * 0.55),
          ls=(0, (3, 1.6)), lw=0.7)
    ax.text(115.2, (y2 + h2 / 2 + y3 + h3 * 0.55) / 2, "徽章反馈闭环",
            rotation=90, va="center", ha="center", fontsize=6.8, color=INK)

    # ---------------- 左缘层标签 ----------------
    for label, yc in (("应用层", 63.7), ("方法层", y3 + h3 / 2), ("数据层", y2 + h2 / 2)):
        ax.text(1.0, yc, label, rotation=90, va="center", ha="center",
                fontsize=7.5, color=GREY, fontweight="bold")
    for ysep in (57.2, 24.6):
        ax.plot([0.5, 110], [ysep, ysep], color="#cccccc", lw=0.4,
                linestyle=(0, (1, 2)), zorder=0)

    save(fig, "fig_framework")


# ==========================================================================
def write_summary():
    lines = ["# 实验结果汇总（自动生成）", ""]
    res = collect_kt()
    multi = collect_kt_multi()
    if res:
        lines.append("## 知识追踪（AUC）")
        lines.append("")
        dss = _ds_sort(res)
        lines.append("| 模型 | " + " | ".join(dss) + " |")
        lines.append("|" + "---|" * (len(dss) + 1))
        for m in KT_ORDER:
            cells = []
            for ds in dss:
                if ds in multi and m in multi[ds]["summary"] \
                        and multi[ds]["summary"][m]["AUC"].get("mean") is not None:
                    s = multi[ds]["summary"][m]["AUC"]
                    cells.append(f"{s['mean']:.4f}±{s.get('std', 0):.4f}")
                elif m in res.get(ds, {}):
                    cells.append(f"{res[ds][m]['AUC']:.4f}")
                else:
                    cells.append("--")
            lines.append(f"| {KT_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
        lines.append("")
    for f in find("rec_"):
        ds = os.path.basename(f)[4:-5]; d = load(f)
        methods = [m for m in d if not m.startswith("_") and isinstance(d[m], dict)
                   and "CSCR@10" in d[m]]
        if not methods:
            continue
        var = d.get("_variant") or {}
        sel = (f"，验证集选定 就绪度项={var.get('ready')}/协同项={var.get('gcf')}"
               f"/γ*={d.get('_gamma')}") if var else f"，γ*={d.get('_gamma')}"
        lines.append(f"## 推荐（{ds}，评测学生子集 n={d.get('_n_eval')}{sel}）")
        lines.append("")
        lines.append("| 方法 | sRecall@10 | sNDCG@10 | CSCR@10 | CSCR@20 |")
        lines.append("|---|---|---|---|---|")
        for m in sorted(methods, key=lambda x: -d[x].get("sNDCG@10", 0)):
            v = d[m]
            lines.append(f"| {m} | {v.get('sRecall@10')} | {v.get('sNDCG@10')} | "
                         f"{v['CSCR@10']} | {v['CSCR@20']} |")
        lines.append("")
        sig = d.get("_sig") or {}
        if sig:
            lines.append(f"配对 bootstrap 检验（参照 = DGCL-Rec(本文)，"
                         f"n={d.get('_sig_n')} 学生，指标 NDCG@10，2000 次重采样）：")
            lines.append("")
            lines.append("| 对照方法 | ΔNDCG@10 | 95% CI | p | 结论 |")
            lines.append("|---|---|---|---|---|")
            for m in sorted(sig, key=lambda x: -sig[x]["delta"]):
                s = sig[m]
                if s.get("win"):
                    concl = "本文显著更优"
                elif s["ci"][1] < 0:
                    concl = "**该方法显著更优**"
                else:
                    concl = "无显著差异"
                lines.append(f"| {m} | {s['delta']:+.4f} | "
                             f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | {s['p']:.4f} | {concl} |")
            lines.append("")
    for f in find("path_"):
        ds = os.path.basename(f)[5:-5]; d = load(f)
        if not isinstance(d, dict) or not d:
            continue
        lines.append(f"## 学习路径（{ds}，评测学生 n={d.get('_n_eval')}，"
                     f"选定策略：{d.get('_selected')}）")
        lines.append("")
        lines.append("| 方法 | 可执行覆盖率 | 先修合规率 | 路径串味率 | 完备性_s |")
        lines.append("|---|---|---|---|---|")
        for m in d:
            if m.startswith("_"):
                continue
            v = d[m]
            lines.append(f"| {m} | {v.get('可执行覆盖率')} | {v.get('先修合规率')} | "
                         f"{v.get('路径串味率')} | {v.get('完备性_s')} |")
        lines.append("")
    for f in find("ges_"):
        d = load(f)
        lines.append(f"## LLM/GES（{os.path.basename(f)[4:-5]}）")
        lines.append("")
        for t in ("Direct", "GES"):
            if t in d:
                lines.append(f"- {t}: " + ", ".join(
                    f"{k}={d[t].get(k)}" for k in ("HR", "ETR", "SCR", "PR", "NDCG@10", "n")))
        lines.append(f"- 预算: {d.get('_budget')}")
        lines.append(f"- Direct 越界案例数: {d.get('_n_cases')}；"
                     f"同名跨课程受控探查: {d.get('_probe_summary')}")
        lines.append("")
    p = os.path.join(RES, "summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  summary ->", p)


def write_provenance():
    """把每张图**实际绘制**的数值落盘（results/fig_provenance.json）。

    PDF 本身不可解析，所以"图上的数与表上的数是否一致"无法事后从图上验证。
    这里让出图脚本自报账目：{图名: {datasets: {ds: {model: value}}, source: ...}}。
    paper/tools/audit_tables.py 的 G 段据此把图与论文表格逐项对比，
    使"图-表口径分裂"这类缺陷在审计阶段就暴露，而不是等评审看到。
    """
    p = os.path.join(RES, "fig_provenance.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_PROV, f, ensure_ascii=False, indent=1)
    print("  provenance ->", p)


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("== 生成图表 ==")
    fig_framework()
    fig_dataset_stats()
    res = collect_kt(); multi = collect_kt_multi()
    if res:
        print("  KT 数据集:", {k: len(v) for k, v in res.items()})
    if multi:
        print("  多种子:", {k: len(v["summary"]) for k, v in multi.items()})
    fig_kt_main(res, multi); table_kt(res, multi); fig_kt_seeds(multi)
    # 7 数据集 × 10 模型的对比实验（读聚合文件 kt_bench_<ds>.json）
    bt = collect_kt_bench()
    if bt:
        print("  基准对比数据集:", {k: len(v) for k, v in bt.items()})
    fig_kt_bench(bt)
    fig_rec_cscr(); table_rec(); fig_rec_gamma()
    table_path()
    fig_ablation(); table_ablation()
    fig_ges(); table_ges(); fig_ges_case()
    fig_mdrl()
    write_summary()
    write_provenance()
    print("== 完成，输出目录:", FIGS)


if __name__ == "__main__":
    sys.exit(main())
