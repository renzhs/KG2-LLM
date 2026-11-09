#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先修图「归纳数据范围」泄漏审计（回应审稿意见 M10）。

问题
----
`models/kg_induct.induce_prereq` 由**答题序列**归纳有向先修 DAG。
但调用方此前一律传 `train + valid + test` 的全部序列：

  · KT 任务是「按学生划分」，测试学生的序列本不该参与建图；
  · 推荐/路径任务是「按位置三段切分」，[cut2,1) 段的交互是评测真值，
    用它归纳先修边会给图谱类方法带来相对无图谱基线的信息优势。

本脚本对每个「图靠归纳得到」的数据集，分别用三种数据范围归纳先修图并比较：

  A. all  : train + valid + test 全序列            <- 现稿口径
  B. train: 仅 train 划分（KT 任务的正确范围）
  C. cut  : 每条序列前 cut1 比例的交互（推荐/路径任务的正确范围）

输出边数、重叠召回率(recall=|E∩E_ref|/|E_ref|)与 Jaccard，
用于判断「改用训练段建图」会改变多少结构。
DNUI-IKG / DNUI-IKG-S 的先修边来自教学大纲（`prereq` 非空），不在本审计范围。
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)

import numpy as np                                        # noqa: E402
from paths import PROCESSED, RESULTS                      # noqa: E402
from models.kg_induct import induce_prereq                # noqa: E402

STEMS = ["assist09_akt_f1", "assist15_f1", "assist17_f1", "statics11_f1",
         "assist09", "assist12"]
CUT1 = 0.6
MIN_LEN = 3


def eset(pre):
    return {(u, int(v)) for u, lst in enumerate(pre) for v, _ in (lst or [])}


def cut_seqs(seqs, frac):
    out = []
    for s in seqs:
        sk = s["skill"]
        if len(sk) < MIN_LEN:
            continue
        n = max(1, int(round(len(sk) * frac)))
        out.append(dict(skill=np.asarray(sk[:n], dtype=np.int64)))
    return out


def main():
    rep = {}
    for stem in STEMS:
        p = os.path.join(PROCESSED, stem + ".pkl")
        if not os.path.exists(p):
            print("[skip] 缺 %s" % p)
            continue
        with open(p, "rb") as f:
            obj = pickle.load(f)
        K = obj["n_skill"]
        alls = obj["train"] + obj["valid"] + obj["test"]
        pre_all, ne_all, _ = induce_prereq(alls, K)
        pre_tr, ne_tr, _ = induce_prereq(obj["train"], K)
        pre_cu, ne_cu, _ = induce_prereq(cut_seqs(alls, CUT1), K)

        E_all, E_tr, E_cu = eset(pre_all), eset(pre_tr), eset(pre_cu)
        # 校验：pkl 里存的 prereq 是否就是"全序列归纳"的结果
        # （A 组由 prep_bench 写入、B 组由 add_induced_prereq 写入，都应为归纳所得）
        E_store = eset(obj.get("prereq") or [])
        stored_matches = (E_store == E_all)

        def cmp(E, ref):
            if not ref:
                return 0.0, 0.0
            inter = len(E & ref)
            return (round(inter / len(ref), 4),
                    round(inter / max(1, len(E | ref)), 4))

        r_tr, j_tr = cmp(E_tr, E_all)
        r_cu, j_cu = cmp(E_cu, E_all)
        # 反向：现口径的边有多少落在 train 范围内（= 泄漏了多少）
        leak = len(E_all - E_tr)
        rep[stem] = dict(n_skill=K, n_students=len(alls) and len(
            {s.get("sid") for s in alls if s.get("sid") is not None}),
            n_edges_all=ne_all, n_edges_train=ne_tr, n_edges_cut=ne_cu,
            n_edges_stored=len(E_store), stored_matches_all=stored_matches,
            recall_train_vs_all=r_tr, jaccard_train_vs_all=j_tr,
            recall_cut_vs_all=r_cu, jaccard_cut_vs_all=j_cu,
            n_edges_only_in_all=leak,
            leak_share=round(leak / max(1, ne_all), 4))
        print("%-16s K=%-5d E(stored)=%-4d E(all)=%-5d E(train)=%-5d E(cut)=%-5d "
              "train∩all=%5.1f%%  cut∩all=%5.1f%%  仅 all 含=%d(%.1f%%)  stored==all:%s"
              % (stem, K, len(E_store), ne_all, ne_tr, ne_cu, 100 * r_tr, 100 * r_cu,
                 leak, 100 * leak / max(1, ne_all), stored_matches))

    out = os.path.join(RESULTS, "graph_leak.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("\n写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
