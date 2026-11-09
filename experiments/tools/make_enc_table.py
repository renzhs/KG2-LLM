#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把编码器对照结果（results/kt_<ds>_encabl.json）生成 4.7.1 节的 LaTeX 表。

表由 run/run_enc_ablation.py 的产物驱动，**不再手工维护**：
手工维护正是此前出错的来源——论文里写"ASSIST09/12 选 GRU 2 层"，
而实际配置是 GRU 1 层 / LSTM 1 层。

用法:
  python experiments/tools/make_enc_table.py --datasets dnui_syn,assist09
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import RESULTS                              # noqa: E402

PLAIN = {"dnui_syn": "DNUI-IKG", "assist09": "ASSIST09", "assist12": "ASSIST12",
         "assist09_akt": "ASSIST09(公开)", "assist15": "ASSIST15",
         "assist17": "ASSIST17", "statics11": "Statics11"}
ORDER = ["Transformer(2层)", "Transformer(3层)", "LSTM(1层)", "GRU(1层)",
         "GRU(1层)+图平滑输入", "GRU(2层)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dnui_syn,assist09")
    a = ap.parse_args()
    dss = [x.strip() for x in a.datasets.split(",") if x.strip()]

    data = {}
    for ds in dss:
        p = os.path.join(RESULTS, f"kt_{ds}_encabl.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data[ds] = json.load(f)
        else:
            print(f"[enc-table] 缺 {p}，跳过 {ds}")
    if not data:
        print("[enc-table] 无可用结果")
        return 1

    dss = [ds for ds in dss if ds in data]
    # 种子数由结果决定，caption 里不能写占位符（曾误留 `$N$`、`**最终配置**`，
    # 前者在 LaTeX 里渲染成数学符号 N，后者渲染成一对星号）。
    nseeds = max((len(data[ds].get("seeds") or []) for ds in dss), default=0)
    nseeds_txt = f"{nseeds} 个随机种子" if nseeds else "多个随机种子"
    lines = [
        "% 由 tools/make_enc_table.py 自动生成（勿手工编辑）",
        "\\begin{table}[t]", "\\centering",
        "\\caption{序列通路编码器对照。固定各数据集\\textbf{最终配置}的其余超参，"
        f"只替换编码器主体，报 {nseeds_txt}的测试集 AUC 均值$\\pm$标准差；"
        "括注为该变体相对最终选中编码器的逐种子平均差。}",
        "\\label{tab:enc}", "\\small", "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(dss) + "}", "\\toprule",
        "编码器 & " + " & ".join(PLAIN.get(ds, ds) for ds in dss) + " \\\\",
        "\\midrule",
    ]
    sel_note = []
    for name in ORDER:
        cells = []
        for ds in dss:
            d = data[ds]
            s = (d.get("summary") or {}).get(name)
            sel = d.get("selected_variant_guess")
            if not s or s.get("mean") is None:
                cells.append("--")
                continue
            txt = f"{s['mean']:.4f}$\\pm${s.get('std', 0):.4f}"
            dv = (d.get("paired_vs_ref") or {}).get(name, {}).get("mean_delta")
            if name == sel:
                txt = "\\best{" + txt + "}"
                if ds not in [n for n, _ in sel_note]:
                    sel_note.append((ds, sel))
            elif dv is not None:
                txt += f" ({dv:+.4f})"
            cells.append(txt)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    if sel_note:
        lines.insert(4, "% 最终选中的编码器：" +
                     "；".join(f"{PLAIN.get(d, d)}={s}" for d, s in sel_note))

    p = os.path.join(RESULTS, "table_enc.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[enc-table] -> {p}")
    for ds, sel in sel_note:
        d = data[ds]
        bs = max((v["mean"], k) for k, v in d["summary"].items()
                 if v.get("mean") is not None)
        print(f"[enc-table]   {PLAIN.get(ds, ds)}: 最终选中 {sel}；"
              f"本表最优 {bs[1]} {bs[0]:.4f}")
        if bs[1] != sel:
            print(f"[enc-table]   ⚠ 本表最优({bs[1]}) 与最终配置选中的编码器({sel}) 不一致 ——"
                  f"说明选参序列（先定编码器、后调容量）与固定最终配置的对照存在落差，"
                  f"正文必须把这一落差写清楚，不能两处都声称最优。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
