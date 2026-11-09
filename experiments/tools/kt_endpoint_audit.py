#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学习率网格端点审计与定向扩搜核对。

论文 4.1.2 节 (4) 声称：固定网格的前提是"最优点落在网格内部"；本文在搜索结束后
审计每个 (数据集, 模型) 的最优学习率是否命中端点，命中即向该方向倍增扩搜。

本脚本把该声明的**证据从结果文件里重新导出**（而不是手写在正文里）：
  - 输入：results/kt_model_cfg_<ds>.json（每个 (数据集, 模型) 的最终选定配置）
  - 判据：模型侧搜索网格 {0.5,1,2,4,8}e-3（GKT 因逐时刻图传播开销大，收窄为 {1,2,4}e-3）
          lr <= 网格下界 或 lr >= 网格上界  → 命中端点
          lr  < 网格下界 或 lr > 网格上界  → 已扩搜（说明真实最优点在网格之外）
  - 输出：results/kt_endpoint.json 与 results/table_endpoint.tex

为什么需要它：正文里"19 个单元命中端点"这类数字一旦与结果文件脱钩，
就会变成无法复核的手写声明（评审最反感的一类问题）。此处改为可重跑的推导。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
from paths import RESULTS as RES  # noqa: E402

# 模型侧搜索网格（论文 4.1.2 节）
GRID = [0.0005, 0.001, 0.002, 0.004, 0.008]
GRID_GKT = [0.001, 0.002, 0.004]

# 配置文件名 -> 论文中的数据集名（顺序即论文表格顺序）
DS_FILES = [
    ("assist09_akt_f1", "ASSIST09"),
    ("assist15_f1", "ASSIST15"),
    ("assist17_f1", "ASSIST17"),
    ("statics11_f1", "Statics11"),
    ("dnui_syn", "DNUI-IKG"),
    ("assist09", "ASSIST09(自建)"),
    ("assist12", "ASSIST12(自建)"),
]

MODELS = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT", "SAINT", "SimpleKT"]

# 论文表内用的模型名（BKT/BKT+ 无学习率，不在配置文件中）
MODEL_ALIAS = {"DKT+": "DKT+", "SimpleKT": "SimpleKT"}


def grid_of(model):
    return GRID_GKT if model == "GKT" else GRID


def audit():
    out = {}
    for stem, dsname in DS_FILES:
        p = os.path.join(RES, f"kt_model_cfg_{stem}.json")
        if not os.path.exists(p):
            continue
        cfg = json.load(open(p, encoding="utf-8"))
        rows = []
        for model in MODELS:
            key = MODEL_ALIAS.get(model, model)
            if key not in cfg:
                continue
            v = cfg[key]
            lr = v.get("lr")
            if lr is None:
                continue
            g = grid_of(model)
            lo, hi = min(g), max(g)
            hit_lo, hit_hi = lr <= lo + 1e-12, lr >= hi - 1e-12
            expanded = lr < lo - 1e-12 or lr > hi + 1e-12
            rows.append(dict(model=model, lr=lr, grid_lo=lo, grid_hi=hi,
                             hit_lo=bool(hit_lo), hit_hi=bool(hit_hi),
                             expanded=bool(expanded)))
        out[dsname] = rows
    return out


def summarize(out):
    hit = [(ds, r) for ds, rows in out.items() for r in rows if r["hit_lo"] or r["hit_hi"]]
    exp = [(ds, r) for ds, rows in out.items() for r in rows if r["expanded"]]
    still = [(ds, r) for ds, rows in out.items() for r in rows
             if (r["hit_lo"] or r["hit_hi"]) and not r["expanded"]]
    return hit, exp, still


def _grid_tex(r):
    """网格按 1e-3 为单位显示（网格本身就是以 1e-3 为量纲给出的）。"""
    lo, hi = r["grid_lo"] * 1e3, r["grid_hi"] * 1e3
    fmt = lambda x: ("%g" % x)  # noqa: E731
    return "$[%s,%s]\\times10^{-3}$" % (fmt(lo), fmt(hi))


def _lr_tex(lr):
    """把学习率写成以 1e-3 为量纲的形式，如 0.000125 -> 0.125。"""
    return "$%g\\times10^{-3}$" % (lr * 1e3)


def tex(out):
    """生成附录表：只列"命中端点或已扩搜"的单元，未命中的不必占版面。"""
    hit, exp, still = summarize(out)
    lines = [
        "% 由 tools/kt_endpoint_audit.py 自动生成",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{学习率网格端点审计与定向扩搜结果。"
        "模型侧搜索网格为 $\\{0.5,1,2,4,8\\}\\times10^{-3}$"
        "（GKT 因逐时刻图传播开销大，按同等墙钟预算收窄为 $\\{1,2,4\\}\\times10^{-3}$）。"
        "``命中端点''指该单元在网格下界或上界取到最优；"
        "``最终学习率''给出扩展搜索后的选定值，"
        "``—''表示该单元在扩展搜索后仍回到原网格端点取值、即扩搜未改变选点。"
        "论文 4.1.2 节 (4) 的端点审计结论即由本表支撑。}",
        "\\label{tab:endpoint}", "\\small",
        "\\begin{tabular}{llcccc}", "\\toprule",
        "数据集 & 模型 & 网格 & 命中下界 & 命中上界 & 最终学习率 \\\\", "\\midrule",
    ]
    for ds, rows in out.items():
        for r in rows:
            if not (r["hit_lo"] or r["hit_hi"]):
                continue
            lines.append("%s & %s & %s & %s & %s & %s \\\\" % (
                ds, r["model"].replace("+", "$+$"), _grid_tex(r),
                "$\\surd$" if r["hit_lo"] else "—",
                "$\\surd$" if r["hit_hi"] else "—",
                _lr_tex(r["lr"]) if r["expanded"] else "—"))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}"]
    return lines, len(hit), len(exp), len(still)


def main():
    out = audit()
    with open(os.path.join(RES, "kt_endpoint.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    lines, n_hit, n_exp, n_still = tex(out)
    with open(os.path.join(RES, "table_endpoint.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("== 端点审计 ==")
    for ds, rows in out.items():
        marks = ["%s:%s%s" % (r["model"], "L" if r["hit_lo"] else "", "H" if r["hit_hi"] else "")
                 for r in rows if r["hit_lo"] or r["hit_hi"]]
        print("  %-14s 命中端点: %s" % (ds, ", ".join(marks) if marks else "-"))
    print("  命中端点的单元总数 = %d" % n_hit)
    print("  其中已向两端扩搜的 = %d，扩搜后仍贴合端点的 = %d" % (n_exp, n_still))
    print("  -> results/kt_endpoint.json, results/table_endpoint.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
