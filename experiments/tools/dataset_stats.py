#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总所有已预处理数据集的统计量 -> results/dataset_stats.json

用途：论文表 1（数据集统计）与 figs/fig_dataset_stats 的输入。
数据来源：experiments/data/processed/*.pkl 里的 meta.stats（由 prep_datasets.py 写入）。
不需要 GPU，秒级完成。

用法: python experiments/tools/dataset_stats.py
"""
import os, sys, json, glob, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
from paths import PROCESSED as PROC, RESULTS as RES   # noqa: E402

ORDER = ["dnui_real", "dnui_syn", "assist09", "assist12", "ednet"]


def main():
    out = {}
    for p in sorted(glob.glob(os.path.join(PROC, "*.pkl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        st = (obj.get("meta") or {}).get("stats")
        if not st:
            # 兜底：从 train/valid/test 序列现场统计（正常路径不会走到这里）
            seqs = [s for sp in ("train", "valid", "test") for s in (obj.get(sp) or [])]
            L = [len(s["skill"]) for s in seqs] or [0]
            st = dict(name=name, users=len(seqs),
                      n_skill=int(obj.get("n_skill") or 0),
                      n_question=int(obj.get("n_question") or 0),
                      interactions=int(sum(L)),
                      avg_len=round(float(sum(L) / max(1, len(L))), 2),
                      correct_rate=None)
        st = dict(st)
        st.setdefault("name", name)
        out[name] = st
        print(f"  {name:<12} users={st.get('users')} skill={st.get('n_skill')} "
              f"q={st.get('n_question')} inter={st.get('interactions')} "
              f"cr={st.get('correct_rate')}")

    # 按固定顺序重排
    ordered = {k: out[k] for k in ORDER if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})

    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "dataset_stats.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=1)
    print("SAVED", p)

    # 同时导出 LaTeX 表（论文表 1）
    keys = [("users", "学习者数"), ("n_skill", "知识点数"), ("n_question", "题目数"),
            ("interactions", "交互记录数"), ("avg_len", "平均序列长度"),
            ("correct_rate", "作答正确率")]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{实验数据集统计}", "\\label{tab:datasets}",
             "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
             "数据集 & " + " & ".join(n for _, n in keys) + " \\\\", "\\midrule"]
    for ds, st in ordered.items():
        cells = []
        for k, _ in keys:
            v = st.get(k)
            if v is None:
                cells.append("--")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(f"{int(v):,}")
        lines.append(ds + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p2 = os.path.join(RES, "table_dataset_stats.tex")
    with open(p2, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SAVED", p2)


if __name__ == "__main__":
    main()
