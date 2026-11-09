#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「无先修标注」的公开数据集归纳一个有向先修图。

为什么需要：
  ASSIST09/12、EdNet 等公开 KT 数据集只给出「学生—题目—对错」序列，没有课程先修标注。
  于是：① 图模型只能用对称共现图（无方向、无先修语义）；
        ② 路径推荐里的「先修合规率 / 可执行覆盖率」直接无从计算（分母为 0）。
  本文用一个简单且可辩护的启发式补上这一环——「共现 + 位置序」：
        · 先统计同一学生序列中**相邻**技能的无向共现次数，保留强共现对；
        · 再统计每个技能在序列中的**平均相对位置** pos[k] ∈ [0,1]（越靠前越小）；
        · 对每个保留的共现对，把 pos 较小者当作先修，边权 w = |pos[a]-pos[b]|。
  由于方向完全由标量 pos 决定，只要同一对只处理一次，得到的图必然**无环**（是 DAG），
  因而可以安全用于拓扑排序与先修合规性度量。

  这是"数据驱动归纳课程先后关系"的做法，与 DNUI-IKG 中由教学大纲给出的真实
  PREREQUISITE_OF 边性质一致（只是来源不同），论文中需明确标注两类数据集的图来源差异。
"""
import numpy as np


def skill_mean_position(seqs, K, min_len=3):
    """每个技能在序列中的平均相对位置（0=序列开头, 1=序列结尾）。"""
    s = np.zeros(K, dtype=np.float64)
    n = np.zeros(K, dtype=np.float64)
    for sq in seqs:
        sk = sq["skill"]
        L = len(sk)
        if L < min_len:
            continue
        denom = max(1, L - 1)
        for i, a in enumerate(sk):
            a = int(a)
            if 0 <= a < K:
                s[a] += i / denom
                n[a] += 1
    out = np.full(K, 0.5, dtype=np.float64)
    ok = n > 0
    if ok.any():
        out[ok] = s[ok] / n[ok]
        if (~ok).any():
            out[~ok] = float(out[ok].mean())
    return out


def induce_prereq(seqs, K, min_len=3, top_q=0.9, min_count=5, symmetric_co=True):
    """归纳先修 DAG。

    返回 (prereq, n_edges, pos)
      prereq[u] = [(v, w), ...]  表示 u 是 v 的先修，w∈[0,1] 为置信/强度
    """
    co = np.zeros((K, K), dtype=np.float64)
    for sq in seqs:
        sk = sq["skill"]
        if len(sk) < min_len:
            continue
        for a, b in zip(sk[:-1], sk[1:]):
            a = int(a); b = int(b)
            if a == b or not (0 <= a < K and 0 <= b < K):
                continue
            co[a, b] += 1
            if symmetric_co:
                co[b, a] += 1

    pos = skill_mean_position(seqs, K, min_len)
    prereq = [[] for _ in range(K)]
    if co.max() <= 0:
        return prereq, 0, pos

    vals = co[co > 0]
    thr = float(np.quantile(vals, top_q)) if vals.size > 10 else 0.0
    thr = max(thr, float(min_count), 1e-9)
    ij = np.argwhere(co >= thr)
    seen = set()
    for a, b in ij:
        a = int(a); b = int(b)
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue                      # 同一对只处理一次 → 保证无环
        seen.add(key)
        u, v = (a, b) if pos[a] < pos[b] else (b, a)
        w = float(abs(pos[a] - pos[b]))
        prereq[u].append((v, w))
    ne = sum(len(x) for x in prereq)
    return prereq, ne, pos


def dag_report(prereq, K, pos=None):
    """体检：边数 / 是否有环 / 拓扑序长度。"""
    from collections import deque
    indeg = [0] * K
    adj = [[] for _ in range(K)]
    for u, lst in enumerate(prereq):
        for v, w in lst:
            adj[u].append(v); indeg[v] += 1
    dq = deque([i for i in range(K) if indeg[i] == 0]); seen = 0
    while dq:
        x = dq.popleft(); seen += 1
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    return dict(n_edges=sum(len(x) for x in prereq), acyclic=(seen == K), reached=seen)


if __name__ == "__main__":
    import os, sys, pickle
    HERE = os.path.dirname(os.path.abspath(__file__))
    EXPROOT = os.path.dirname(HERE)
    sys.path.insert(0, EXPROOT)
    from paths import PROCESSED as PROC
    for name in (sys.argv[1:] or ["assist09", "assist12"]):
        p = os.path.join(PROC, f"{name}.pkl")
        if not os.path.exists(p):
            print("missing", p); continue
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if obj.get("prereq"):
            print(f"{name}: 已有先修边 {sum(len(x) for x in obj['prereq'])}，跳过")
            continue
        pre, ne, pos = induce_prereq(obj["train"] + obj["valid"] + obj["test"], obj["n_skill"])
        print(f"{name}: 归纳 {ne} 条先修边, {dag_report(pre, obj['n_skill'])}")
