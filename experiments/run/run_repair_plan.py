#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2  阻塞诊断与最小补修 (Blocking Diagnosis & Minimal Repair)

问题：学习目标当前不可执行时，如何用最少的补修把它变成可执行；
      预算不够时，如何给出「可达的阶段目标」而不是一句失败。

设定
----
课程图 66 个知识点 / 88 条 PREREQUISITE 边（无环，最大深度 27，23 个节点有双父）。
学生的已掌握集合 M 由真实作答给出（正确率 ≥ 0.6 且至少 2 次作答）。
路径可执行 <=> 路径上每个节点，其全部直接前置都已掌握或在路径更早处出现。

三种策略
  FullReplan  忽略预算：直接输出到目标的全部缺失前置（可能超预算 -> 不可行）
  LocalFix    只补目标的直接前置，不递归（便宜，但常留残余阻塞）
  Ours        传递闭包求最小补修集；超预算时回退到「预算内可达的最深阶段目标」

指标
  Feasible         方案在预算内可执行的比例
  n_fix            补修知识点数
  Depth            实际可达目标在图上的深度
  ResidualBlock    修复后路径上仍然缺前置的节点数
  Time             求解耗时 (ms)

只依赖标准库。
用法：python experiments/run/run_repair_plan.py --scenarios 300
"""
import os, sys, csv, json, time, random, argparse
from collections import defaultdict
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "dnui_ikg")
OUT = os.environ.get("E2_OUT", os.path.join(ROOT, "results", "repair_plan.json"))
MASTER = 0.60


def _rows(n):
    with open(os.path.join(DATA, n), encoding="utf-8") as f:
        return list(csv.DictReader(f))


KTS = {int(r["kt_id"]): r for r in _rows("kts.csv")}
PAR = defaultdict(list)
CH = defaultdict(list)
for r in _rows("edges.csv"):
    a, b = int(r["src_kt"]), int(r["dst_kt"])
    PAR[b].append(a); CH[a].append(b)
ACC = defaultdict(lambda: [0, 0])
for r in _rows("interactions.csv"):
    a = ACC[(int(r["student_id"]), int(r["kt_id"]))]
    a[0] += int(r["correct"]); a[1] += 1
NODES = sorted(set(PAR) | set(CH))


@lru_cache(None)
def ancestors(k):
    """全部传递前置（不含自身）。"""
    s = set()
    for p in PAR[k]:
        s.add(p); s |= ancestors(p)
    return frozenset(s)


@lru_cache(None)
def depth(k):
    return 0 if not PAR[k] else 1 + max(depth(p) for p in PAR[k])


def mastered_set(sid):
    return {k for k in NODES
            if (lambda c, n: n >= 2 and c / n >= MASTER)(*ACC.get((sid, k), [0, 0]))}


def topo(sub):
    """子图上的拓扑序。"""
    sub = set(sub)
    indeg = {k: sum(1 for p in PAR[k] if p in sub) for k in sub}
    q = sorted([k for k in sub if indeg[k] == 0])
    out = []
    while q:
        u = q.pop(0); out.append(u)
        for v in CH[u]:
            if v in sub:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return out


# --------------------------------------------------------------------------
def plan_full(sid, t, M):
    """忽略预算，给出全部缺失前置。"""
    need = ancestors(t) - M
    return dict(target=t, add=sorted(need), feasible=None)


def plan_local(sid, t, M):
    """只补直接前置。"""
    add = [p for p in PAR[t] if p not in M]
    still = ancestors(t) - M - set(add)
    return dict(target=t, add=add, feasible=len(still) == 0, residual=len(still))


def plan_ours(sid, t, M, B):
    """最小补修 + 预算内阶段目标回退。"""
    need = ancestors(t) - M
    if len(need) <= B:
        return dict(target=t, add=sorted(need), feasible=True, residual=0, fallback=False)
    # 在「目标 + 其全部前置」里，找补修需求不超过预算、且深度最大的节点
    cand = [k for k in (ancestors(t) | {t}) if len(ancestors(k) - M) <= B]
    if not cand:
        return dict(target=None, add=[], feasible=False, residual=len(need), fallback=True)
    best = max(cand, key=lambda k: (depth(k), -len(ancestors(k) - M)))
    nd = ancestors(best) - M
    return dict(target=best, add=sorted(nd), feasible=True, residual=0, fallback=best != t)


# --------------------------------------------------------------------------
def scenarios(n, seed=42, min_inter=12):
    """目标取自该生所在课程内、尚未掌握且确有前置的节点；按深度分层抽样，
    使「预算够」与「预算不够」两类情形都能出现。"""
    rnd = random.Random(seed)
    enroll = defaultdict(set)
    for r in _rows("enrolled.csv"):
        enroll[int(r["student_id"])].add(r["course_id"])
    sids = sorted({s for (s, _) in ACC})
    rnd.shuffle(sids)
    out = []
    for sid in sids:
        tot = sum(ACC[(sid, k)][1] for k in NODES if (sid, k) in ACC)
        if tot < min_inter:
            continue
        M = mastered_set(sid)
        if not M:
            continue
        course = KTS[list(M)[0]]["course_id"]
        cands = [k for k in NODES
                 if PAR[k] and k not in M and KTS[k]["course_id"] == course]
        if not cands:
            continue
        # 深度分层：一半取最深的，一半随机，避免只落在浅层
        cands.sort(key=lambda k: -depth(k))
        pick = cands[0] if len(out) % 2 == 0 else rnd.choice(cands)
        out.append(dict(student_id=sid, target=pick, M=M))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budgets", default="1,3,5,10")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]

    scs = scenarios(args.scenarios, args.seed)
    print(f"场景 {len(scs)}｜图 {len(NODES)} 节点 / {sum(len(v) for v in PAR.values())} 边"
          f"｜最大深度 {max(depth(k) for k in NODES)}")

    res = dict(n_scenarios=len(scs), budgets=budgets, by_budget={})
    for B in budgets:
        agg = defaultdict(lambda: dict(n=0, feasible=0, fix=0, depth=0, resid=0, ms=0.0, fb=0))
        for sc in scs:
            sid, t, M = sc["student_id"], sc["target"], sc["M"]

            t0 = time.perf_counter(); p = plan_full(sid, t, M); d1 = time.perf_counter() - t0
            t0 = time.perf_counter(); q = plan_local(sid, t, M); d2 = time.perf_counter() - t0
            t0 = time.perf_counter(); r = plan_ours(sid, t, M, B); d3 = time.perf_counter() - t0

            a = agg["FullReplan"]; a["n"] += 1
            a["feasible"] += int(len(p["add"]) <= B)
            a["fix"] += len(p["add"]); a["depth"] += depth(p["target"])
            a["resid"] += max(0, len(p["add"]) - B); a["ms"] += d1 * 1000

            a = agg["LocalFix"]; a["n"] += 1
            a["feasible"] += int(q["feasible"])
            a["fix"] += len(q["add"]); a["depth"] += depth(q["target"])
            a["resid"] += q.get("residual", 0); a["ms"] += d2 * 1000

            a = agg["Ours"]; a["n"] += 1
            a["feasible"] += int(r["feasible"])
            a["fix"] += len(r["add"])
            a["depth"] += depth(r["target"]) if r["target"] is not None else 0
            a["resid"] += r.get("residual", 0); a["ms"] += d3 * 1000
            a["fb"] += int(r.get("fallback", False))

        rows = {}
        for m, a in agg.items():
            n = max(1, a["n"])
            rows[m] = dict(Feasible=round(a["feasible"] / n, 4),
                           n_fix=round(a["fix"] / n, 3),
                           Depth=round(a["depth"] / n, 1),
                           ResidualBlock=round(a["resid"] / n, 3),
                           FallbackRate=round(a["fb"] / n, 4),
                           Time_ms=round(a["ms"] / n, 4))
        res["by_budget"][str(B)] = rows

        print(f"\n=== 预算 B = {B} 个补修知识点 ===")
        print(f"  {'策略':<12}{'可行率':>8}{'补修数':>8}{'可达深度':>9}{'残余阻塞':>9}{'回退率':>8}{'耗时ms':>9}")
        for m in ("FullReplan", "LocalFix", "Ours"):
            d = rows[m]
            print(f"  {m:<12}{d['Feasible']:>8.3f}{d['n_fix']:>8.2f}{d['Depth']:>9.1f}"
                  f"{d['ResidualBlock']:>9.2f}{d['FallbackRate']:>8.3f}{d['Time_ms']:>9.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n写出:", OUT)


if __name__ == "__main__":
    main()
