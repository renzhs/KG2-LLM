#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构造 DNUI-IKG 的学习资源层（结构映射版，非行为仿真）

背景
----
原始数据里 materials.csv 只有表头，资源层为空。E1 的天花板效应
（题目正确率中位 0.997、前置完备度均值 0.99、序列先修违反率落在零模型内、
先修完备度对练习量的 kt 内效应量 1.001）说明：任何从行为结果派生的资源层
都会继承同一个天花板，无法为「作用域判定」提供可验证信号。

因此本脚本不做行为仿真，只做结构映射：
    1. 知识点 -> 资源（讲义 / 习题集 / 项目任务），资源继承 kt 的作用域与先修闭包；
    2. 题目实体化：以「题干+选项+答案」规范化后哈希为题目实体，聚合同一实体在多门
       课程上的全部挂载，得到 item -> {kt} 的多对多结构。

第 2 点是本层的关键。实测 86/188 的 6 道题中有 5 道逐字相同，全库 33 组重复题、
其中 21 组跨课程。也就是说「这道题属于哪个知识点」在数据里本来就不唯一，
标注不完整不是「丢失了唯一真值」，而是「多挂载中的某一条未被记录」。

输出
----
data/dnui_ikg/resource_layer.json
    resources   资源表
    items       题目实体表（含 mounts）
    closure     每个 kt 的先修闭包
    diag        诊断读数

用法
----
    python experiments/kg/build_resource_layer.py
"""
import os, csv, json, hashlib, re
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data", "dnui_ikg")
OUT = os.path.join(DATA, "resource_layer.json")

RTYPES = ("lec", "ex", "proj")


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def norm(s):
    """题目文本规范化：去空白、统一全角括号与逗号，用于判定「同一道题」"""
    s = (s or "").strip()
    s = s.replace("（", "(").replace("）", ")").replace("，", ",")
    s = re.sub(r"\s+", "", s)
    return s


def item_key(r):
    """题目实体指纹：题干 + 四个选项 + 答案"""
    parts = [norm(r["stem"])] + [norm(r.get("option_" + c, "")) for c in "ABCD"] + [norm(r.get("answer", ""))]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]


def load_graph():
    kts = {}
    for r in read_csv("kts.csv"):
        kts[int(r["kt_id"])] = dict(
            kt_id=int(r["kt_id"]),
            name=r["kt_name"].strip(),
            course=r["course_id"],
            unit=r["unit_id"],
            grades=sorted(int(x) for x in r["grade_scope"].split("|") if x.strip()),
            majors=sorted(x.strip() for x in r["major_scope"].split("|") if x.strip()),
            is_hard=int(r["is_hard"] or 0),
            n_questions=int(r["n_questions"] or 0),
            n_inter=int(r["n_interactions"] or 0),
        )
    pre = defaultdict(set)
    for r in read_csv("edges.csv"):
        if r["relation"] == "PREREQUISITE":
            pre[int(r["dst_kt"])].add(int(r["src_kt"]))
    courses = {r["course_id"]: dict(name=r["name"], grade=r["grade_scope"], major=r["major_scope"])
               for r in read_csv("courses.csv")}
    return kts, pre, courses


def closure_of(k, pre, memo):
    """先修闭包（迭代 + 记忆化，带环保护）"""
    if k in memo:
        return memo[k]
    seen, stack = set(), [k]
    while stack:
        x = stack.pop()
        for p in pre.get(x, ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    memo[k] = seen
    return seen


def build():
    kts, pre, courses = load_graph()
    memo = {}
    clo = {k: closure_of(k, pre, memo) for k in kts}

    # ---------------- 题目实体与多挂载 ----------------
    qrows = read_csv("questions.csv")
    items = {}
    for r in qrows:
        kt = int(r["kt_id"])
        key = item_key(r)
        it = items.setdefault(key, dict(iid=key, stem=norm(r["stem"]), qids=[], mounts=[]))
        it["qids"].append(r["q_id"])
        if kt not in [m["kt_id"] for m in it["mounts"]]:
            it["mounts"].append(dict(kt_id=kt, course=kts[kt]["course"], name=kts[kt]["name"]))

    multi = {k: v for k, v in items.items() if len(v["mounts"]) > 1}
    cross = {k: v for k, v in multi.items()
             if len({m["course"] for m in v["mounts"]}) > 1}
    same_name = {k: v for k, v in multi.items()
                 if len({m["name"] for m in v["mounts"]}) == 1}

    # ---------------- 资源表 ----------------
    problems_by_kt = Counter(int(r["kt_id"]) for r in qrows)
    resources = []
    for k, v in sorted(kts.items()):
        for rt in RTYPES:
            if rt == "ex" and problems_by_kt[k] == 0:
                continue
            if rt == "proj" and not pre.get(k) and not any(k in s for s in clo.values()):
                continue
            resources.append(dict(
                rid="R%d-%s" % (k, rt),
                kt_id=k, kt_name=v["name"], course=v["course"], rtype=rt,
                scope_grade=v["grades"], scope_major=v["majors"],
                prereq=sorted(clo[k]),
                n_prereq=len(clo[k]),
                n_problem=problems_by_kt[k] if rt == "ex" else 0,
                est_min=2 * problems_by_kt[k] if rt == "ex" else (25 if rt == "lec" else 60),
            ))

    # ---------------- 诊断 ----------------
    per_course = Counter(v["course"] for v in kts.values())
    mounts_dist = Counter(len(v["mounts"]) for v in items.values())
    diag = dict(
        n_kt=len(kts),
        n_resource=len(resources),
        n_question_row=len(qrows),
        n_item=len(items),
        n_multi_mount=len(multi),
        n_multi_cross_course=len(cross),
        n_multi_same_name=len(same_name),
        mount_distribution=dict(sorted(mounts_dist.items())),
        kt_per_course=dict(per_course),
        dup_examples=[dict(stem=v["stem"][:40], mounts=[m["course"] + ":" + str(m["kt_id"]) for m in v["mounts"]])
                      for v in list(cross.values())[:5]],
        prereq_edges=sum(len(s) for s in pre.values()),
        cross_course_edges=sum(1 for dst in pre for src in pre[dst]
                               if kts[src]["course"] != kts[dst]["course"]),
    )

    out = dict(
        meta=dict(
            source="data/dnui_ikg",
            note="结构映射构造，非行为仿真；resources 由 kt 与先修图派生，items 的 mounts 取自原始题目挂载",
            courses=courses,
        ),
        resources=resources,
        items=list(items.values()),
        closure={str(k): sorted(v) for k, v in clo.items()},
        diag=diag,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("== 资源层构造完成 ==")
    print("  知识点 %d -> 资源 %d 条（%s）" % (diag["n_kt"], diag["n_resource"], "/".join(RTYPES)))
    print("  题目行 %d -> 题目实体 %d（去重 %d）"
          % (diag["n_question_row"], diag["n_item"], diag["n_question_row"] - diag["n_item"]))
    print("  多挂载实体 %d，其中跨课程 %d、同名 %d"
          % (diag["n_multi_mount"], diag["n_multi_cross_course"], diag["n_multi_same_name"]))
    print("  挂载数分布 %s" % diag["mount_distribution"])
    print("  先修边 %d，跨课程边 %d" % (diag["prereq_edges"], diag["cross_course_edges"]))
    print("\n  跨课程重复题示例：")
    for e in diag["dup_examples"]:
        print("    %-42s %s" % (e["stem"], e["mounts"]))
    print("\n写入", OUT)


if __name__ == "__main__":
    build()
