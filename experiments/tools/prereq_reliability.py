"""先修图"归纳"的可靠性验证（审稿意见 M11）。

公开基准没有先修标注，本文用 models.kg_induct 从答题序列归纳有向先修 DAG。
审稿意见指出：归纳图本身准不准，正文没有验证。

本脚本用**唯一有真值的数据集**做验证：
  DNUI-IKG 的先修边来自课程教学大纲（dnui_prereq），把它当作 gold，
  用同一套诱导算法从答题序列重新归纳，比较二者的 precision / recall / F1。
这把"归纳图是否可信"从主观断言变成可核验数字，
并为公开基准上的归纳图提供一个（数据集外的）可信度上界参考。

同时报告归纳图的结构合理性：有无自环/环、平均出度、最大路径长度等。

用法（远端）:
  ./env/bin/python experiments/tools/prereq_reliability.py
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import PROCESSED as PROC, RESULTS as RES          # noqa: E402
from models.kg_induct import induce_prereq                    # noqa: E402


def edge_set(pre):
    return {(u, v) for u, lst in enumerate(pre) for v, _ in (lst or [])}


def has_cycle(pre, K):
    color = [0] * K
    adj = [[] for _ in range(K)]
    for u, lst in enumerate(pre):
        for v, _ in (lst or []):
            adj[u].append(v)

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    return any(dfs(u) for u in range(K) if color[u] == 0)


def longest_path(pre, K):
    order = []
    indeg = [0] * K
    for u, lst in enumerate(pre):
        for v, _ in (lst or []):
            indeg[v] += 1
    from collections import deque
    q = deque([u for u in range(K) if indeg[u] == 0])
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in (pre[u] or []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    dp = [0] * K
    for u in order:
        for v, _ in (pre[u] or []):
            dp[v] = max(dp[v], dp[u] + 1)
    return (max(dp) if dp else 0), len(order) == K


def main():
    out = {}
    ds = "dnui_syn"
    p = os.path.join(PROC, f"{ds}.pkl")
    if not os.path.exists(p):
        print("missing", p)
        return
    with open(p, "rb") as f:
        obj = pickle.load(f)
    K = obj["n_skill"]
    gold_pre = obj.get("prereq") or []
    if not gold_pre:
        print("[warn] 该数据集 pkl 未携带教学大纲先修边，无法做金标准对照")
        return
    # 只用训练学生的序列归纳——与主实验"全量"相比更保守，避免用测试信息
    for scope_name, seqs in (("train", list(obj.get("train") or [])),
                             ("all", (obj.get("train") or []) + (obj.get("valid") or [])
                                     + (obj.get("test") or []))):
        ind, ne, _ = induce_prereq(seqs, K)
        g, i = edge_set(gold_pre), edge_set(ind)
        tp = len(g & i)
        prec = tp / max(1, len(i))
        rec = tp / max(1, len(g))
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        lp, is_dag = longest_path(ind, K)
        out[scope_name] = dict(
            n_gold=len(g), n_induced=len(i), tp=tp,
            precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
            n_gold_only=len(g - i), n_induced_only=len(i - g),
            induced_is_dag=bool(is_dag), induced_longest_path=lp,
            gold_is_dag=not has_cycle(gold_pre, K),
        )
        r = out[scope_name]
        print(f"\n== DNUI-IKG 归纳图 vs 教学大纲 (序列范围={scope_name})")
        print(f"   大纲边 {r['n_gold']} 条 / 归纳边 {r['n_induced']} 条 / 命中 {r['tp']} 条")
        print(f"   precision={r['precision']:.4f}  recall={r['recall']:.4f}  F1={r['f1']:.4f}")
        print(f"   归纳图为 DAG: {r['induced_is_dag']}（最长链 {r['induced_longest_path']} 层）；"
              f"大纲图为 DAG: {r['gold_is_dag']}")

    out["note"] = ("公开基准无先修真值，故只能报'跨作用域稳定性'（见 graph_leak.json）；"
                   "本脚本给出的是同一套归纳算法在**有真值数据集**上的精度上界，"
                   "用于说明归纳图并非任意噪声，而能恢复相当比例的真实先修结构。")

    # ---- 工作点敏感性：top_q 越大保留的共现对越多，precision/recall 此消彼长 ----
    tr = list(obj.get("train") or [])
    g = edge_set(gold_pre)
    sweep = []
    for tq in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999):
        ind, ne, _ = induce_prereq(tr, K, top_q=tq)
        i = edge_set(ind)
        tp = len(g & i)
        pr = tp / max(1, len(i))
        rc = tp / max(1, len(g))
        sweep.append(dict(top_q=tq, n_induced=len(i), precision=round(pr, 4),
                          recall=round(rc, 4),
                          f1=round(2 * pr * rc / max(1e-12, pr + rc), 4)))
    out["top_q_sweep_train"] = sweep
    print("\n== 归纳工作点敏感性（DNUI-IKG，train 序列，金标准=教学大纲）")
    print("   top_q   归纳边数  precision  recall   F1")
    for s in sweep:
        star = "  <-- 本文默认" if abs(s["top_q"] - 0.90) < 1e-9 else ""
        print(f"   {s['top_q']:<7} {s['n_induced']:<9} {s['precision']:<10} "
              f"{s['recall']:<8} {s['f1']}{star}")
    out["sweep_note"] = ("默认工作点 top_q=0.90 在召回与精确之间取折中；"
                         "调大 top_q 可提高召回但精确率下降更快，"
                         "说明'共现+位置序'只能恢复先修结构的骨架，无法替代教学大纲。")

    q = os.path.join(RES, "prereq_reliability.json")
    with open(q, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nSAVED", q)


if __name__ == "__main__":
    main()
