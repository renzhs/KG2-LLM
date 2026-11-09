#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把基准实验结果 + 已发表文献数字，组装成论文可直接引用的 LaTeX 表。

产物（全部写入 results/）:
  table_dataset.tex      7 个数据集统计表
  table_kt_bench.tex     主对比表：11 个模型 × 7 个数据集（AUC）
  table_kt_pub.tex       一致性核验表：4 个公开数据集上"本文复现值 vs AKT(2020) 报告值"
  table_kt_bench.md      同上内容的 Markdown 版本（便于在对话/文档里直接看）

设计约定:
  · 5 折数据集报 mean±std；自建划分（单划分）只报点估计，表头加注说明；
  · 已发表数字读自 experiments/data/published_benchmarks.json（坐标级核对过）；
  · 只在文献与本文都有的 (数据集, 模型) 上计算 Δ，避免拿不同协议的数字硬比。

用法:
  python experiments/tools/make_paper_tables.py
  python experiments/tools/make_paper_tables.py --only assist09_akt,statics11
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import RESULTS, REPO                      # noqa: E402

PUB_JSON = os.path.join(REPO, "experiments", "data", "published_benchmarks.json")

# 5 折数据集（报 mean±std）与自建单划分数据集（报点估计）
DS_5F = ["assist09_akt", "assist15", "assist17", "statics11"]
DS_1F = ["dnui_syn", "assist09", "assist12"]
ALL_DS = DS_5F + DS_1F

PRETTY_DS = {
    "assist09_akt": "ASSIST09", "assist15": "ASSIST15", "assist17": "ASSIST17",
    "statics11": "Statics11", "dnui_syn": "DNUI-IKG", "assist09": "ASSIST09$^\\S$",
    "assist12": "ASSIST12$^\\S$",
}
PLAIN_DS = {
    "assist09_akt": "ASSIST09", "assist15": "ASSIST15", "assist17": "ASSIST17",
    "statics11": "Statics11", "dnui_syn": "DNUI-IKG", "assist09": "ASSIST09(自建划分)",
    "assist12": "ASSIST12(自建划分)",
}

MODEL_ORDER = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT",
               "SAINT", "SimpleKT", "DGCLKT"]
PRETTY_M = {"DGCLKT": "DGCL-KT", "BKT+": "BKT$+$", "DKT+": "DKT$+$",
            "SimpleKT": "SimpleKT"}

OURS = "DGCLKT"

# 本文模型名 -> AKT(2020) 表里的列名。
# 说明：AKT 原文区分 AKT-R（用题目 id 的 Rasch 嵌入）与 AKT-NR（不用）。
# 本文在含题目 id 的数据集（ASSIST09 / ASSIST17）上启用题目级建模，故对应 AKT-R；
# 在不含题目 id 的数据集（ASSIST15 / Statics11）上题目索引退化为技能索引，对应 AKT-NR。
PUB_MAP_AKT = {
    "assist09_akt": {"DKT": "DKT", "DKT+": "DKT+", "DKVMN": "DKVMN",
                     "SAKT": "SAKT", "AKT": "AKT-R"},
    "assist15": {"DKT": "DKT", "DKT+": "DKT+", "DKVMN": "DKVMN",
                 "SAKT": "SAKT", "AKT": "AKT-NR"},
    "assist17": {"DKT": "DKT", "DKT+": "DKT+", "DKVMN": "DKVMN",
                 "SAKT": "SAKT", "AKT": "AKT-R"},
    "statics11": {"DKT": "DKT", "DKT+": "DKT+", "DKVMN": "DKVMN",
                  "SAKT": "SAKT", "AKT": "AKT-NR"},
}


def load_bench(ds):
    p = os.path.join(RESULTS, f"kt_bench_{ds}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pub():
    with open(PUB_JSON, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 表 1：数据集统计
# --------------------------------------------------------------------------
DS_STATS_FALLBACK = {
    # 远端 dataset_audit.py 的实测输出（无结果文件时用于占位，保证表结构完整）
    "assist09_akt": dict(users=3862, n_skill=110, n_question=16891, inter=325181,
                         avg=84.2, med=26, correct=0.6586, edges=256, pub="ASSIST09"),
    "assist15": dict(users=19093, n_skill=100, n_question=100, inter=682797,
                     avg=35.8, med=21, correct=0.7325, edges=353, pub="ASSIST15"),
    "assist17": dict(users=1708, n_skill=102, n_question=3162, inter=942814,
                     avg=552.0, med=441, correct=0.3727, edges=322, pub="ASSIST17"),
    "statics11": dict(users=333, n_skill=1223, n_question=1223, inter=189297,
                      avg=568.5, med=634, correct=0.7654, edges=658, pub="Statics11"),
    "dnui_syn": dict(users=2147, n_skill=66, n_question=467, inter=117865,
                     avg=54.9, med=57, correct=0.6270, edges=88, pub="--"),
    "assist09": dict(users=3852, n_skill=149, n_question=17751, inter=278390,
                     avg=72.3, med=23, correct=0.6576, edges=384, pub="--"),
    "assist12": dict(users=27141, n_skill=198, n_question=50983, inter=2504175,
                     avg=92.3, med=40, correct=0.6957, edges=820, pub="--"),
}


def write_dataset_table():
    lines = [
        "% 由 tools/make_paper_tables.py 自动生成",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{实验数据集统计。前四行为 AKT 官方 5 折划分（取 fold 1）的公开基准，"
        "其中 ASSIST09/ASSIST17 含题目 id；后三行为自建划分，其中仅 DNUI-IKG 带"
        "学段—专业标签（CSG 门控的作用域监督来源）。"
        "``题目数'' 与知识点数相同的行表示该数据集无题目粒度信息，"
        "此时题目索引退化为技能索引。``先修边'' 为按 3.2.1 节方法由答题序列归纳得到、"
        "且通过无环性校验的先修有向边数。}",
        "\\label{tab:dataset}",
        "\\small",
        "\\begin{tabular}{llrrrrrrr}", "\\toprule",
        "数据集 & 划分协议 & 学生数 & 知识点数 & 题目数 & 交互数 & 平均序列长 & 正确率 & 先修边 \\\\",
        "\\midrule",
    ]
    for i, ds in enumerate(ALL_DS):
        if i == 4:
            lines.append("\\midrule")
        s = DS_STATS_FALLBACK[ds]
        proto = "AKT 5 折（fold 1）" if ds in DS_5F else "自建划分"
        lines.append(
            f"{PLAIN_DS[ds]} & {proto} & {s['users']:,} & {s['n_skill']:,} & "
            f"{s['n_question']:,} & {s['inter']:,} & {s['avg']:.1f} & "
            f"{s['correct']:.4f} & {s['edges']} \\\\".replace(",", "\\,"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    p = os.path.join(RESULTS, "table_dataset.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] -> {p}")


# --------------------------------------------------------------------------
# 表 2：主对比表（10 模型 × 7 数据集）
# --------------------------------------------------------------------------
def write_main_table():
    data = {ds: load_bench(ds) for ds in ALL_DS}
    ready = [ds for ds in ALL_DS if data[ds] and data[ds].get("summary")]
    if not ready:
        print("[table] 主对比表：尚无聚合结果，跳过")
        return None
    missing = [ds for ds in ALL_DS if ds not in ready]
    if missing:
        print(f"[table] 主对比表：以下数据集结果缺失，将以 -- 占位: {missing}")

    # 口径统一（2026-09-24）：见 effective_kt_values() 的说明。
    eff = effective_kt_values()
    best = {ds: max(v[1] for v in eff[ds].values()) for ds in ready if eff[ds]}

    # 表注里的「多种子覆盖范围」由数据推导，而非手写。
    # 手写的后果（实测）：补跑把某个自建划分的 11 个模型都变成多种子后，
    # 表注仍写着"多种子仅覆盖本文方法与最强基线"，与表体自相矛盾。
    b_ds = [ds for ds in ALL_DS if ds not in DS_5F and eff.get(ds)]
    full = [ds for ds in b_ds
            if all(eff[ds].get(m, ("",))[0] == "multi" for m in MODEL_ORDER)]
    part = [ds for ds in b_ds if ds not in full]
    seedtxt = "3 个随机种子（42/43/44）"
    # 确定性算法（BKT/BKT+ 的 EM 拟合不依赖随机种子）种子间 std 恒为 0，
    # 若照常渲染会得到 "0.5966$\pm$0.0000$^m$"——噪声且易被误读为"精度极高"。故只报均值。
    # 哪些模型属于这一类同样由数据判定，不写死。
    det = [m for m in MODEL_ORDER
           if any(eff[ds].get(m, ("", "", 0))[0] == "multi"
                  and not ((eff[ds].get(m, ("", "", 0))[2] or 0) > 0) for ds in b_ds)]
    det_sent = ("；其中 " + "、".join(PRETTY_M.get(m, m) for m in det)
                + " 为确定性算法、种子间标准差为 0，故只报均值") if det else ""
    if not b_ds:
        multi_sent = ""
    elif not part:
        multi_sent = f"自建划分的三列均为 {seedtxt}的均值$\\pm$标准差{det_sent}。"
    elif full:
        multi_sent = (
            f"自建划分中带上标 $^m$ 的单元格为 {seedtxt}均值$\\pm$标准差："
            f"其中 {'、'.join(PLAIN_DS[d] for d in full)} 已覆盖全部 "
            f"{len(MODEL_ORDER)} 个模型，"
            f"{'、'.join(PLAIN_DS[d] for d in part)} 暂仅覆盖本文方法与最强基线，"
            f"其余单元格为单次运行点估计{det_sent}；相应显著性判定见表~\\ref{{tab:kt-ours}}。")
    else:
        multi_sent = (
            f"自建划分中带上标 $^m$ 的单元格为 {seedtxt}均值$\\pm$标准差{det_sent}；"
            "按 4.1.2 节约定，多种子产物仅覆盖本文方法与最强基线，"
            "其余单元格为单次运行点估计；相应显著性判定见表~\\ref{tab:kt-ours}。")

    lines = [
        "% 由 tools/make_paper_tables.py 自动生成",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{知识掌握预测性能对比（AUC）。前四列为 AKT 官方 5 折划分的公开基准，"
        "报告 5 折均值$\\pm$标准差；后三列为自建划分（$^\\S$ 标记）。"
        + multi_sent +
        "每列粗体为该列最高值，$\\dagger$ 标出本文方法。"
        "所有模型在同一数据集上共享完全相同的训练/验证/测试划分与评测样本。}",
        "\\label{tab:kt-bench}",
        "\\footnotesize",
        # 11 行 × 8 列在双栏排版下必定超宽：用 resizebox 精确缩到版心宽度
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(ALL_DS) + "}", "\\toprule",
        "\\multirow{2}{*}{模型} & \\multicolumn{4}{c}{公开基准（AKT 官方 5 折划分）}"
        " & \\multicolumn{3}{c}{自建划分} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
        " & " + " & ".join(PRETTY_DS[ds] for ds in ALL_DS) + " \\\\",
        "\\midrule",
    ]
    for m in MODEL_ORDER:
        cells = []
        for ds in ALL_DS:
            if m not in eff.get(ds, {}):
                cells.append("--")
                continue
            src, mu, sd = eff[ds][m]
            if src == "multi":
                c = (f"{mu:.4f}$\\pm${sd:.4f}$^m$" if (sd or 0) > 0
                     else f"{mu:.4f}$^m$")
            else:
                c = f"{mu:.4f}$\\pm${sd:.4f}" if ds in DS_5F else f"{mu:.4f}"
            if ds in best and abs(mu - best[ds]) < 1e-9:
                c = "\\textbf{" + c + "}"
            cells.append(c)
        name = PRETTY_M.get(m, m)
        if m == OURS:
            name = "\\textbf{" + name + "（本文）}$^\\dagger$"
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    p = os.path.join(RESULTS, "table_kt_bench.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] -> {p}")

    # Markdown 版（方便直接看）
    md = ["| 模型 | " + " | ".join(PLAIN_DS[ds] for ds in ALL_DS) + " |",
          "|" + "---|" * (len(ALL_DS) + 1)]
    for m in MODEL_ORDER:
        row = []
        for ds in ALL_DS:
            if m not in eff.get(ds, {}):
                row.append("--")
                continue
            src, mu, sd = eff[ds][m]
            if src == "multi":
                txt = (f"{mu:.4f}±{sd:.4f}^m" if (sd or 0) > 0 else f"{mu:.4f}^m")
            else:
                txt = f"{mu:.4f}±{sd:.4f}" if ds in DS_5F else f"{mu:.4f}"
            if ds in best and abs(mu - best[ds]) < 1e-9:
                txt = f"**{txt}**"
            row.append(txt)
        md.append(f"| {PLAIN_DS.get(m, m)} | " + " | ".join(row) + " |")
    pmd = os.path.join(RESULTS, "table_kt_bench.md")
    with open(pmd, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[table] -> {pmd}")
    return data


# --------------------------------------------------------------------------
# 表 3：与 AKT(2020) 报告值的一致性核验
# --------------------------------------------------------------------------
def write_pub_table(data):
    """一致性核验表：**两套**发表值并列 + 本文复现值 + 逐套 Δ。

    早期版本只对照 AKT(2020) 一套，与 4.1 节正文"把两套并列给出、不与任何一套
    混用"的表述不符（正文承诺两套，表里只有一套）。这里补齐第二套
    （pykt / SimpleKT 同划分数字），并自动统计"复现值落在两套之外"的单元格 ——
    那是复现管线存在系统性偏差或基线欠调的直接证据，必须如实暴露、不能藏。
    """
    pub_akt = load_pub()["akt_2020_table2_auc"]["rows"]
    sk = load_pub()["simplekt_iclr2023_table2_auc"]
    inv = {v: k for k, v in sk["map_to_ours"].items()}
    sk_ds = sk["datasets"]

    def pykt_of(ds, m):
        col = inv.get(ds)
        if col is None or col not in sk_ds:
            return None
        row = sk["rows"].get(m)
        if not row:
            return None
        i = sk_ds.index(col)
        if i >= len(row):
            return None
        try:
            return float(row[i])
        except (TypeError, ValueError):
            return None

    lines = [
        "% 由 tools/make_paper_tables.py 自动生成",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{本文复现值与两套独立公开报告值的逐项对照（AUC）。"
        "``pykt'' 一栏为 pykt 基准库与 SimpleKT 论文报告的同划分数字；"
        "``AKT(2020)'' 一栏为 AKT 原文表~2 的数字，在含题目 id 的数据集上取 AKT-R，"
        "否则取 AKT-NR。两套均与本文共用同一份官方 5 折划分，差异来自实现细节与"
        "调参预算。$\\Delta$ 为本文减对应报告值；$^\\ast$ 标出本文复现值"
        "\\textbf{低于两套报告值}的单元格（复现偏弱，多为该基线欠调，需修），"
        "$^\\dag$ 标出\\textbf{高于两套报告值}的单元格（复现强于文献，属实现或预算增益）。}",
        "\\label{tab:kt-pub}",
        "\\footnotesize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}", "\\toprule",
        "数据集 & 模型 & pykt 报告 & AKT(2020) 报告 & 本文复现 & $\\Delta$(pykt) "
        "& $\\Delta$(AKT) & 本文 std \\\\",
        "\\midrule",
    ]
    worst, n, outside, missing = 0.0, 0, [], []
    for ds in DS_5F:
        akt_map = PUB_MAP_AKT.get(ds, {})
        sm = (data.get(ds) or {}).get("summary") or {}
        first = True
        for m in MODEL_ORDER:
            pubcol = akt_map.get(m)
            pv = pykt_of(ds, m)
            if pubcol is None and pv is None:
                continue
            label = PLAIN_DS[ds] if first else ""
            first = False
            ref = pub_akt.get(ds, {}).get("values", {}).get(pubcol, [None])[0] \
                if pubcol else None
            mine = sm.get(m, {})
            mv = (mine.get("AUC") or {}).get("mean")

            star = ""
            if pv is not None and ref is not None and mv is not None:
                lo, hi = min(pv, ref), max(pv, ref)
                if not (lo - 1e-9 <= mv <= hi + 1e-9):
                    # 区分方向：低于两套 = 复现偏弱（典型是基线欠调，会虚高本文优势，
                    # 是审稿人可攻击点）；高于两套 = 复现强于文献（我们自己实现/预算更好，
                    # 不构成公平性问题，但必须在正文说明，否则读者会以为数字抄错）。
                    d = "below" if mv < lo else "above"
                    star = "$^\\ast$" if d == "below" else "$^\\dag$"
                    outside.append((ds, m, mv, pv, ref, d))

            def fmt(v):
                if v is None or mv is None:
                    return "--"
                return f"${mv - v:+.4f}$"

            if mv is None:
                missing.append((ds, m))

            def cell(v):
                return f"{v:.4f}" if v is not None else "--"

            if pv is not None and mv is not None:
                n += 1
            worst = max([worst] + [abs(mv - x) for x in (pv, ref)
                                   if x is not None and mv is not None])
            sd = (mine.get("AUC") or {}).get("std")
            sd_txt = f"$\\pm${sd:.4f}" if sd is not None else "--"
            lines.append(
                f"{label} & {PRETTY_M.get(m, m)}{star} & {cell(pv)} & {cell(ref)} & "
                f"{cell(mv)} & {fmt(pv)} & {fmt(ref)} & {sd_txt} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    p = os.path.join(RESULTS, "table_kt_pub.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] -> {p}")
    if missing:
        print(f"[table] 一致性核验：{len(missing)} 个单元格无本文复现值（结果未聚合）: "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if n:
        print(f"[table] 一致性核验：{n} 个 (数据集, 模型) 对，最大 |ΔAUC| = {worst:.4f}")
    if outside:
        below = [o for o in outside if o[5] == "below"]
        above = [o for o in outside if o[5] == "above"]
        print(f"[table] ⚠ 复现值**落在两套报告值区间之外**的单元格 {len(outside)} 个"
              f"（低于两套 {len(below)} / 高于两套 {len(above)}）")
        if below:
            print("[table]   ↓ 低于两套（复现偏弱 ⇒ 疑似基线欠调，会虚高本文优势，"
                  "必须修复或如实披露）:")
            for ds, m, mv, pv, ref, _ in below:
                print(f"        {PLAIN_DS[ds]:<22} {m:<9} 本文={mv:.4f} "
                      f"pykt={pv:.4f} AKT={ref:.4f}")
        if above:
            print("[table]   ↑ 高于两套（复现强于文献 ⇒ 非公平性问题，但正文需说明）:")
            for ds, m, mv, pv, ref, _ in above:
                print(f"        {PLAIN_DS[ds]:<22} {m:<9} 本文={mv:.4f} "
                      f"pykt={pv:.4f} AKT={ref:.4f}")
    else:
        print("[table] 全部复现值均落在两套报告值区间内 ✓")

    # Markdown 版
    md = ["| 数据集 | 模型 | pykt | AKT(2020) | 本文 | Δ(pykt) | Δ(AKT) |",
          "|---|---|---|---|---|---|---|"]
    cur = None
    for ds in DS_5F:
        akt_map = PUB_MAP_AKT.get(ds, {})
        sm = (data.get(ds) or {}).get("summary") or {}
        for m in MODEL_ORDER:
            pubcol = akt_map.get(m)
            pv = pykt_of(ds, m)
            if pubcol is None and pv is None:
                continue
            ref = pub_akt.get(ds, {}).get("values", {}).get(pubcol, [None])[0] \
                if pubcol else None
            mv = ((sm.get(m, {}).get("AUC") or {}).get("mean"))
            f = lambda v: ("--" if v is None else f"{v:.4f}")            # noqa: E731
            d = lambda v: ("--" if (v is None or mv is None)             # noqa: E731
                           else f"{mv - v:+.4f}")
            flag = ""
            if pv is not None and ref is not None and mv is not None:
                lo2, hi2 = min(pv, ref), max(pv, ref)
                if not (lo2 - 1e-9 <= mv <= hi2 + 1e-9):
                    flag = " ▽低于两套" if mv < lo2 else " △高于两套"
            md.append(f"| {PLAIN_DS[ds] if ds != cur else ''} | {m}{flag} | {f(pv)} | "
                      f"{f(ref)} | **{f(mv)}** | {d(pv)} | {d(ref)} |")
            cur = ds
    pmd = os.path.join(RESULTS, "table_kt_pub.md")
    with open(pmd, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[table] -> {pmd}")


# --------------------------------------------------------------------------
# 表 4：三个指标上的"本文 vs 最强基线"摘要
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 快照模式开关（--frozen）：results/ 是投稿版快照时，文件 mtime 由 clone/拷贝顺序
# 决定、不含任何信息，此时跳过"多种子是否比主结果新"的新鲜度守卫。
# 重跑实验后请去掉该开关，让守卫照常拦截陈旧派生产物。
FROZEN = False


def _ts(p):
    import time
    return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))


def load_multi(ds):
    """多种子汇总 results/kt_<ds>_multi.json -> (summary, seeds)；缺文件返回 ({}, None)。

    结构: {dataset, seeds, summary: {model: {AUC: {mean, std, per_seed}, ...}}}

    **陈旧守卫（2026-09-23 加）**：只有当多种子文件**比主结果文件新**时才采用。
    多种子是主结果的派生产物；一旦配置变更（例如把序列编码器纳入验证集搜索），
    主结果会被重跑而多种子不会自动失效，此时旧多种子的均值会与主表**自相矛盾**
    （实测：kt_assist09_multi.json 停在 0.7482，而更新后的主结果是 0.7656）。
    宁可报点估计，也不能把两个口径的数字混进同一张表。
    """
    pm = os.path.join(RESULTS, f"kt_{ds}_multi.json")
    pb = os.path.join(RESULTS, f"kt_bench_{ds}.json")
    if not os.path.exists(pm):
        return {}, None
    try:
        with open(pm, encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, OSError):
        return {}, None
    sm = d.get("summary") or {}
    b = load_bench(ds) or {}
    bsm = b.get("summary") or {}

    # 新鲜度判定：**内容优先**，mtime 仅作兜底。
    # 为什么不能只看 mtime：本地结果由远端 pscp 拉回，所有文件 mtime 会趋同
    # （实测：主结果与多种子同为 09-24 07:58，秒级差异纯属拷贝顺序），
    # "多种子是否比主结果新"于是变成随机；而在远端又会把**内容自洽**的多种子
    # 误判为陈旧（DNUI-IKG 多种子 09-23 17:31 < 主结果 21:14，但三粒子里
    # seed-42 单值 0.7426 与主结果聚合值 0.7426 完全吻合，实为同一操作点）。
    # 判据：本文模型（DGCLKT）的多种子里只要有一次单次值与主结果聚合值吻合，
    # 即视为同配置、可采用；否则回落 mtime，再用"均值量级"兜底。
    o_m = ((sm.get(OURS) or {}).get("AUC") or {})
    o_b = ((bsm.get(OURS) or {}).get("AUC") or {}).get("mean")
    content_ok = o_b is not None and any(
        v is not None and abs(v - o_b) <= 5e-5 for v in (o_m.get("per_seed") or []))
    if not content_ok and FROZEN:
        # 快照模式：随仓库发布的 results/ 里，文件 mtime 由 clone 顺序决定、不含信息，
        # 因此直接采用快照中的多种子结果（重跑实验时不要加 --frozen）。
        content_ok = True
    if not content_ok:
        if os.path.exists(pb) and os.path.getmtime(pm) < os.path.getmtime(pb):
            print(f"[table] ⚠ {ds}: 多种子产物({_ts(pm)}) 早于主结果({_ts(pb)})、"
                  f"且种子级数值与主结果不吻合，判为**陈旧派生结果**、本次不采用；"
                  f"需在本轮配置下重跑多种子。")
            return {}, None
        if o_b is not None and o_m.get("mean") is not None \
                and abs(o_m["mean"] - o_b) > 0.010:
            print(f"[table] ⚠ {ds}/{OURS}: 多种子均值 {o_m['mean']:.4f} 与主结果 "
                  f"{o_b:.4f} 相差过大，判为陈旧派生结果、本次不采用。")
            return {}, None
    # 交叉提示（保留原诊断）：任一模型多种子均值与主结果相差过大都值得排查
    for m, v in sm.items():
        a = (v.get("AUC") or {}).get("mean")
        bb = ((bsm.get(m) or {}).get("AUC") or {}).get("mean")
        if a is not None and bb is not None and abs(a - bb) > 0.010:
            print(f"[table] ⚠ {ds}/{m}: 多种子均值 {a:.4f} 与主结果 {bb:.4f} "
                  f"相差 {abs(a - bb):.4f}，超出种子噪声量级，请核查是否同配置。")
    return sm, d.get("seeds") or d.get("n_seeds")


def load_signif():
    """配对显著性检验结果 results/kt_signif.json -> {ds: {...}}（缺文件返回 {}）。

    由 experiments/tools/kt_significance.py 生成：以「折」（公开 5 折）或
    「随机种子」（自建划分）为配对单元，对 DGCL-KT 与最强基线做配对 t 检验。
    tab:kt-ours 的「显著更优」判定必须**以本文件为唯一依据**，
    不能只凭"点估计更大"——否则正文的"显著"二字没有检验支撑。
    """
    p = os.path.join(RESULTS, "kt_signif.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def effective_kt_values():
    """论文表与插图**共用**的唯一取数口径：{ds: {model: (src, mean, std)}}。

    为什么必须共用
    --------------
    论文的 KT 数字有两个消费者——表格（本模块）与插图（figs/make_figures.py）。
    若两边各自取数，就会出现"同一 (数据集, 模型) 单元格在图上一个数、表上另一个数"。
    实测（2026-09-24 审计）：出图走了历史遗留的 `kt_<ds>_main.json`（调参前产物，
    每数据集只有 7 个模型），而表格走 `kt_bench_<ds>.json`，于是
    ASSIST12 的 DGCL-KT 在图上 0.7324、表上 0.7705，差 0.038——
    图与表对"本文是否最优"给出相反结论。故此处收敛为单一事实来源，
    出图侧 import 本函数而非另写一套。

    返回值的第 3 个字段可能为 None（5 折数据集之外的 bench 点估计仍有 std，
    但多种子缺失时 std 可能为 0.0）；调用方按需格式化。
    """
    data = {ds: load_bench(ds) for ds in ALL_DS}
    msm_cache = {ds: (load_multi(ds)[0] or {}) for ds in ALL_DS}
    eff = {}
    for ds in ALL_DS:
        sm = (data[ds] or {}).get("summary") or {}
        eff[ds] = {}
        for m in MODEL_ORDER:
            if m not in sm or not sm[m].get("AUC"):
                continue
            mv = ((msm_cache[ds].get(m) or {}).get("AUC") or {})
            if ds not in DS_5F and mv.get("mean") is not None:
                eff[ds][m] = ("multi", mv["mean"], mv.get("std"))
            else:
                eff[ds][m] = ("bench", sm[m]["AUC"]["mean"], sm[m]["AUC"]["std"])
    return eff


def write_ours_table(data):
    """本文 vs 各数据集最强基线，三指标逐数据集对照。

    说明（2026-09-23 扩展）：原版只报点估计，与摘要"多种子均值±标准差"的口径不符；
    现优先取 results/kt_<ds>_multi.json 的多种子均值，并在本文 AUC 一栏给出 ±std。
    单种子/单划分数据集（无 multi 产物）仍报点估计，并在表注中说明，
    避免用"看起来更严谨"的符号掩盖实际是单次运行的事实。
    """
    # 表注里"多种子覆盖范围"改为**由数据推导**（2026-09-25）：
    # 补跑进度一变，表注自动改口，避免手写的"仅覆盖本文方法与最强基线"
    # 与表体出现矛盾。全部覆盖时不再出现"其余为点估计"这类空句子。
    _cov, _cov_part = [], []
    for _ds in ALL_DS:
        if _ds in DS_5F:
            continue
        _sm, _ = load_multi(_ds)
        if len(_sm or {}) >= len(MODEL_ORDER):
            _cov.append(PLAIN_DS[_ds])
        elif _sm:
            _cov_part.append("%s（%d 个模型）" % (PLAIN_DS[_ds], len(_sm)))
        else:
            _cov_part.append(PLAIN_DS[_ds])
    if _cov_part:
        seed_sent = ("自建划分中 %s 取 3 种子均值$\\pm$种子标准差，"
                     "%s 仅部分模型有 3 种子产物、其余为点估计。"
                     % ("、".join(_cov), "、".join(_cov_part)))
    else:
        seed_sent = ("自建划分的三个数据集均取 3 个随机种子（42/43/44）的"
                     "均值$\\pm$种子标准差；其中 BKT、BKT$+$ 为确定性算法、"
                     "种子间标准差为 0，故只报均值。")

    lines = [
        "% 由 tools/make_paper_tables.py 自动生成",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{本文方法（DGCL-KT）与各数据集最强基线的三指标对比。"
        "``最强基线''为该数据集上基线中 AUC 最高者（多位并列时取表中靠前者）；"
        "$\\Delta$ 为本文减最强基线"
        "（按表内报告的四位小数均值相减，读者可直接复核）。"
        "标记规则\\textbf{对三个指标完全一致}："
        "$\\ddagger$（粗体）表示本文在该指标上经\\textbf{配对检验显著更优}"
        "（AUC/ACC 越大越好、RMSE 越小越好；公开基准以 5 折为配对单元、"
        "自建划分以 3 个随机种子为配对单元，配对 $t$ 检验双侧 $p<0.05$，见 4.1.5 节）；"
        "\\underline{下划线}表示差异\\textbf{不显著}，此时本文不主张占优。"
        "故表中不存在``数值更大即加粗''的隐式规则——"
        "如 DNUI-IKG 一行三个指标的下划线即表示该数据集上不主张任何优势。"
        + seed_sent + "}",
        "\\label{tab:kt-ours}",
        "\\small",
        # 9 列在双栏版心下仍可能超出 30pt 量级：与另两张宽表一致，缩到版心宽度
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}", "\\toprule",
        "数据集 & 最强基线 & 基线 AUC & 本文 AUC & $\\Delta$AUC & 基线 ACC & 本文 ACC "
        "& 基线 RMSE & 本文 RMSE \\\\",
        "\\midrule",
    ]
    SIG = load_signif()
    multi_note = []
    sig_pairs = []
    sig_detail = []
    sig_ci = []
    n_win, n_tie, n_lose = 0, 0, 0
    for ds in ALL_DS:
        sm = (data.get(ds) or {}).get("summary") or {}
        if not sm:
            continue
        msm, nseed = load_multi(ds)
        if msm:
            multi_note.append(f"{PLAIN_DS[ds]} {nseed} 种子")

        def val(model, key):
            """优先多种子均值；返回 (mean, std_or_None, src)。

            src == "fold" 表示数值来自 k 折聚合（std 是**折间**标准差），
            src == "multi" 表示来自多随机种子（std 是**种子间**标准差）。
            两者含义不同，绝不能混着标 ±：单划分数据集只有 1 折时折间 std 恒为 0，
            直接打印会写出 ``0.7426$\\pm$0.0000'' 这种**看起来像零方差**的假象。
            """
            m_multi = ((msm.get(model) or {}).get(key) or {})
            if m_multi.get("mean") is not None:
                return m_multi["mean"], m_multi.get("std"), "multi"
            r = (sm.get(model) or {}).get(key) or {}
            return r.get("mean"), r.get("std"), "fold"

        others = [m for m in sm if m != OURS and val(m, "AUC")[0] is not None]
        if not others or val(OURS, "AUC")[0] is None:
            continue
        bm = max(others, key=lambda m: val(m, "AUC")[0])
        b_a = val(bm, "AUC")[0]
        o_a, o_a_sd, o_src = val(OURS, "AUC")
        b_c, _, _ = val(bm, "ACC")
        o_c, o_c_sd, o_c_src = val(OURS, "ACC")
        b_r = val(bm, "RMSE")[0]
        o_r = val(OURS, "RMSE")[0]

        a_txt = f"{o_a:.4f}"
        show_sd = o_a_sd is not None and (o_src == "multi" or ds in DS_5F)
        if show_sd and o_a_sd > 0:
            a_txt += f"$\\pm${o_a_sd:.4f}"
        # ---- 判胜/平/负：三个指标一律**以配对检验为准**（results/kt_signif.json）----
        # 2026-09-28 修订（回应评审意见 8）：此前 AUC 用配对检验、ACC/RMSE 却用
        # "与基线差值得分超过容差就加粗"的隐式判据，导致 DNUI-IKG 上 AUC 标"不显著"
        # 而 ACC 却加粗，同一行内自相矛盾。现在 ACC/RMSE 也做配对检验
        # （kt_significance.py 已输出 by_metric），三者共用同一标记规则。
        sig = SIG.get(ds) or {}
        bym = sig.get("by_metric") or {}
        # 校验检验对象与表格取到的"最强基线"是同一个（避免静默错配）
        if bym and abs(bym.get("AUC", {}).get("delta", 9) - (o_a - b_a)) >= 5e-4:
            bym = {}
        if not bym:
            print(f"[table]   ⚠ {PLAIN_DS[ds]}: 无配对检验结果，ACC/RMSE 退回描述性判据")

        def mark(val_txt, metric, diff, tol, lower_better=False):
            """按配对检验标记；无检验结果时退回描述性判据并打印告警。"""
            m = bym.get(metric) or {}
            if m:
                return ("\\textbf{" + val_txt + "}$^{\\ddagger}$" if m.get("better")
                        else "\\underline{" + val_txt + "}")
            gain = (-diff if lower_better else diff)
            return ("\\textbf{" + val_txt + "}" if gain > tol else val_txt)

        auc_txt = mark(a_txt, "AUC", o_a - b_a, 0.0 if bym else 1e-3)
        if bym:
            m_auc = bym["AUC"]
            if m_auc.get("better"):
                n_win += 1
            else:
                n_tie += 1
            sig_pairs.append("%s $p=%.4f$" % (PLAIN_DS[ds], m_auc["p"]))
            unit_cn = "折" if m_auc["n"] == 5 else "种子"
            sig_detail.append("%s %d/%d %s胜" % (PLAIN_DS[ds], m_auc["wins"],
                                                m_auc["n"], unit_cn))
            if m_auc.get("ci95"):
                sig_ci.append("%s $[%+.4f,%+.4f]$"
                              % (PLAIN_DS[ds], m_auc["ci95"][0], m_auc["ci95"][1]))
        else:
            tol = o_a_sd if (o_src == "multi" and o_a_sd) else 1e-3
            if (o_a - b_a) > tol:
                n_win += 1
            elif abs(o_a - b_a) <= tol:
                n_tie += 1
            else:
                n_lose += 1
        if bym and not bym["AUC"].get("better") and abs(o_a - b_a) < 5e-3:
            print(f"[table]   · {PLAIN_DS[ds]}: 本文 {o_a:.4f} vs 最强基线 "
                  f"{PRETTY_M.get(bm, bm)} {b_a:.4f}（差 {o_a - b_a:+.4f}，"
                  f"配对检验 p={bym['AUC']['p']:.4f}，判为不显著）")
        c_txt = mark(f"{o_c:.4f}", "ACC", o_c - b_c, 1e-3)
        r_txt = mark(f"{o_r:.4f}", "RMSE", o_r - b_r, 1e-3, lower_better=True)
        lines.append(
            f"{PLAIN_DS[ds]} & {PRETTY_M.get(bm, bm)} & {b_a:.4f} & {auc_txt} & "
            f"${o_a - b_a:+.4f}$ & {b_c:.4f} & {c_txt} & "
            f"{b_r:.4f} & {r_txt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    if sig_pairs:
        lines.append("\\par\\smallskip\\footnotesize 配对 $t$ 检验（双侧）$p$ 值："
                     + "，".join(sig_pairs) + "。"
                     + "（此处 $p$ 值对应 AUC；ACC 与 RMSE 的配对检验结论见表内标记。）"
                     + "$\\Delta$AUC 的 95\\% 置信区间："
                     + "，".join(sig_ci) + "。"
                     + "逐配对单元胜负（本文胜/总单元数）："
                     + "，".join(sig_detail) + "。")
        lines.append("\\par\\footnotesize 功效说明：配对 $t$ 检验的自由度分别为 4（5 折）与 2"
                     "（3 种子），对应的双侧临界值 $|t|_{0.025}$ 为 2.776 与 \\textbf{4.303}。"
                     "据此计算 80\\% 功效下的\\textbf{最小可检出效应}"
                     " $\\mathrm{MDES}=(t_{0.025,n-1}+t_{0.80,n-1})\\,s_d/\\sqrt{n}$"
                     "（$s_d$ 为配对差值的标准差）：DNUI-IKG 上为 $0.0023$，"
                     "ASSIST09（自建划分）上为 $0.0104$。"
                     "DNUI-IKG 的观测效应（$+0.0003$）仅为该下限的 13\\%，故``不显著''反映的是"
                     "\\textbf{样本量不足以分辨该量级的差异}，\\textbf{不能解读为两方法等价}；"
                     "ASSIST09（自建划分）的观测效应（$+0.0094$）亦略低于其下限，"
                     "故该处 $p=0.0393$ 需谨慎对待，本文不据此单独作强主张。")
    lines.append("\\end{table*}")
    if multi_note:
        lines.insert(4, f"% 多种子口径: {', '.join(multi_note)}")
    p = os.path.join(RESULTS, "table_kt_ours.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] -> {p}")
    print(f"[table] 本文 vs 最强基线：胜 {n_win} / 平 {n_tie} / 负 {n_lose}"
          f"{'（多种子: ' + ', '.join(multi_note) + '）' if multi_note else ''}")
    if n_lose:
        print("[table] ⚠ 存在本文 AUC 不及最强基线的数据集 —— "
              "摘要与结论**不得**使用「全部数据集均最优」的表述，需按实际改写。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只处理给定数据集（逗号分隔），默认全部")
    ap.add_argument("--frozen", action="store_true",
                    help="快照模式：跳过多种子新鲜度守卫，用于复核随仓库发布的论文报告值")
    a = ap.parse_args()
    global FROZEN
    FROZEN = a.frozen
    if FROZEN:
        print("[table] frozen 模式：results/ 视为投稿版快照，跳过 mtime 新鲜度守卫")

    write_dataset_table()
    data = write_main_table()
    if data:
        write_pub_table(data)
        write_ours_table(data)


if __name__ == "__main__":
    sys.exit(main())
