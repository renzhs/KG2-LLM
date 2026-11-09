#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计每个数据集的先修边数与作用域信息，供论文数据集表使用。

用法: python experiments/tools/dataset_edges.py
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import PROCESSED as PROC                      # noqa: E402
from models.kg_induct import dag_report                  # noqa: E402

DS = ["assist09_akt_f1", "assist15_f1", "assist17_f1", "statics11_f1",
      "dnui_syn", "assist09", "assist12"]

print(f"{'dataset':<16}{'K':>6}{'prereq_edges':>14}  dag_report")
for ds in DS:
    p = os.path.join(PROC, ds + ".pkl")
    if not os.path.exists(p):
        print(f"{ds:<16}  MISSING")
        continue
    with open(p, "rb") as f:
        o = pickle.load(f)
    pr = o.get("prereq")
    K = o["n_skill"]
    ne = int(sum(len(x) for x in pr)) if pr else 0
    rep = ""
    try:
        rep = dag_report(pr, K) if pr else "无"
    except Exception as e:                                   # noqa: BLE001
        rep = f"(report 失败: {e})"
    print(f"{ds:<16}{K:>6}{ne:>14}  {rep}")
    # 作用域信息
    ss = o.get("student_scope")
    ks = o.get("skill_scope")
    if ss is not None or ks is not None:
        print(f"{'':<16}  student_scope={'有' if ss is not None else '无'} "
              f"skill_scope={'有' if ks is not None else '无'}")
