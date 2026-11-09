#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCL 双图谱构建：生产库(已导入本地 MySQL) -> 规范化数据集 + Neo4j 双图谱

产出:
  data/dnui_ikg/nodes_*.csv, edges_*.csv   —— 规范化数据集
  kg/cypher/10_ckg.cypher, 20_lkg.cypher   —— Neo4j 导入脚本
  data/dnui_ikg/stats.json                 —— 数据集统计

概念:
  CKG (Course Knowledge Graph)  客观课程知识结构: 课程/单元/知识点/题目/资源 + 先修关系
  LKG (Learner Knowledge Graph) 学习者认知状态: 学生节点 + 掌握度/作答/相似关系
  双向耦合:
    正向 CKG->LKG : 先修结构约束掌握度传播
    反向 LKG->CKG : 用真实掌握数据重标定先修边权 w(u,v) = P(m_u=1,m_v=1)-P(m_u=0,m_v=1)
                    (即 u 的掌握对 v 掌握的"条件增益", 可加性)
"""
import os, sys, json, csv, math, argparse, itertools
from collections import defaultdict, Counter

import pymysql

EXPROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(EXPROOT)
OUT = os.path.join(REPO, "data", "dnui_ikg")
CY = os.path.join(EXPROOT, "kg", "cypher")

DB = dict(host=os.environ.get("IKG_DB_HOST", "127.0.0.1"),
          port=int(os.environ.get("IKG_DB_PORT", "3406")),
          user=os.environ.get("IKG_DB_USER", "root"),
          password=os.environ.get("IKG_DB_PASS", "Ikg@Dnui2026"),
          database=os.environ.get("IKG_DB_NAME", "ikg_dnui"), charset="utf8mb4")

# ----------------------------------------------------------------------------
# 课程 / 学段 / 专业 作用域配置
# 依据: (a) knowledge_lesson 与 knowledge_unit 的真实归属;
#       (b) exam_question 中恢复出的第三门课程(机器学习与深度学习);
#       (c) 学号第3-4位=学院码、第5-6位=专业码 的真实结构;
#       (d) 培养方案中的开课学期.
# ----------------------------------------------------------------------------
L7 = [25] + list(range(77, 94))       # 程序设计基础(Java)  18 KT
L13 = list(range(178, 206))           # 面向对象程序设计(Java) 28 KT
LML = list(range(94, 114))            # 机器学习与深度学习  20 KT

COURSES = {
    "C1": dict(course_id="C1", name="程序设计基础(Java)", lesson_id=7,
               grade_scope=[1], major_scope=["软件工程", "计算机科学", "人工智能"],
               term="大一秋", kts=L7),
    "C2": dict(course_id="C2", name="面向对象程序设计(Java)", lesson_id=13,
               grade_scope=[2], major_scope=["软件工程", "计算机科学"],
               term="大二春", kts=L13),
    "C3": dict(course_id="C3", name="机器学习与深度学习导论", lesson_id=None,
               grade_scope=[3], major_scope=["人工智能", "计算机科学"],
               term="大三秋", kts=LML),
}

# 学号第5-6位 -> 专业（依据真实分布 + 与课程归属的交叉验证）
MAJOR_CODE = {
    "10": "软件工程", "11": "软件工程", "12": "软件工程", "13": "软件工程", "14": "软件工程",
    "20": "计算机科学", "21": "计算机科学", "22": "计算机科学", "23": "计算机科学", "24": "计算机科学",
    "31": "人工智能", "32": "人工智能", "33": "人工智能",
}
ALL_MAJORS = ["软件工程", "计算机科学", "人工智能"]
ALL_GRADES = [1, 2, 3, 4]

# KT 名称跨课程重名的"串味对"（由真实数据自动发现，见 find_scope_conflicts）
def kt2course():
    m = {}
    for cid, c in COURSES.items():
        for k in c["kts"]:
            m[k] = cid
    return m

KT2C = kt2course()


def q(cur, sql, p=()):
    cur.execute(sql, p)
    return cur.fetchall()


def connect():
    return pymysql.connect(**DB)


# ----------------------------------------------------------------------------
def load_raw(cur):
    """读取全部原始数据。"""
    d = {}
    d["lessons"] = {r[0]: r[1] for r in q(cur, "select lesson_id, lesson_name from knowledge_lesson")}
    d["units"] = [dict(unit_id=r[0], unit_code=r[1], unit_name=r[2], parent_unit_id=r[3],
                       lesson_id=r[4]) for r in q(cur, "select unit_id,unit_code,unit_name,"
                                                  "parent_unit_id,lesson_id from knowledge_unit")]
    d["kt"] = [dict(kt_id=r[0], kt_code=r[1], kt_name=r[2], kt_description=r[3],
                    is_important=r[4], is_hard=r[5], ability_metric_id=r[6],
                    unit_id=r[7], lesson_id=r[8])
               for r in q(cur, "select kt_id,kt_code,kt_name,kt_description,is_important,is_hard,"
                               "ability_metric_id,unit_id,lesson_id from knowledge_kt")]
    d["rel"] = [dict(from_kt=r[0], to_kt=r[1], relationship=r[2])
                for r in q(cur, "select from_kt_id,to_kt_id,relationship from knowledge_kt_relationship")]
    d["q"] = [dict(q_id=r[0], kt_id=int(r[1]), stem=r[2], oA=r[3], oB=r[4], oC=r[5], oD=r[6],
                   oE=r[7], oF=r[8], answer=r[9], score=r[10])
              for r in q(cur, "select id, knowledge_id, question, option_A, option_B, option_C,"
                              "option_D, option_E, option_F, answer, score from exam_question "
                              "where del_flag=0")]
    d["qn"] = {int(r[0]): r[1] for r in q(cur, "select knowledge_id, max(knowledge_name) "
                                               "from exam_question group by knowledge_id")}
    d["stu"] = {int(r[0]): dict(user_id=r[0], user_name=r[1], nick_name=r[2])
                for r in q(cur, "select user_id,user_name,nick_name from sys_user where del_flag='0'")}
    d["inter"] = [dict(student_id=int(r[0]), test_id=r[1], kt_id=int(r[2]),
                       answer=r[3], correct=int(r[4]))
                  for r in q(cur, "select student_id, test_id, knowledge_id, answer, is_correct "
                                  "from question_result")]
    d["mastery"] = [dict(student_id=int(r[0]), kt_id=int(r[1]), done=int(r[2]), correct=int(r[3]),
                         correct_rate=float(r[4] or 0), complete_rate=float(r[5] or 0), passed=int(r[6]))
                    for r in q(cur, "select student_id,kt_id,done_question_count,"
                                    "correct_question_count,correct_rate,complete_rate,passed "
                                    "from achievement_level")]
    d["master_rule"] = [dict(rule_level=r[0], object_id=r[1], master_level=r[2],
                             done_question_count=r[3], correct_rate=r[4])
                        for r in q(cur, "select rule_level,object_id,master_level,"
                                        "done_question_count,correct_rate from knowledge_master_rule")]
    d["ability"] = {r[0]: r[1] for r in q(cur, "select ability_metric_id, ability_metric_name "
                                              "from knowledge_ability_metric")}
    d["material"] = [dict(material_id=r[0], name=r[1], desc=r[2], mtype=r[3], ftype=r[4],
                          kt_id=r[5], unit_id=r[6], lesson_id=r[7])
                     for r in q(cur, "select material_id,material_name,material_description,"
                                     "material_type,file_type,kt_id,unit_id,lesson_id from material "
                                     "where status<>'DELETED'")]
    return d


def recover_missing_kts(d):
    """从 exam_question.knowledge_name 恢复 knowledge_kt 中缺失的 23 个知识点。"""
    have = {k["kt_id"] for k in d["kt"]}
    recovered = []
    for kt_id, name in sorted(d["qn"].items()):
        if kt_id in have:
            continue
        cid = KT2C.get(kt_id, "C3")
        # 单元归属: C3 按 20 个 KT 分为 4 个逻辑单元; C1 的 86/87/88 归入 unit 32(操作一组数据)
        if cid == "C3":
            idx = LML.index(kt_id)
            unit = {0: 71, 1: 71, 2: 71, 3: 71, 4: 72, 5: 72, 6: 72, 7: 72, 8: 72,
                    9: 72, 10: 72, 11: 73, 12: 73, 13: 73, 14: 73, 15: 73, 16: 74,
                    17: 74, 18: 74, 19: 74}[idx]
        else:
            unit = 32
        recovered.append(dict(kt_id=kt_id, kt_code=f"RKT{kt_id}", kt_name=name,
                              kt_description=name, is_important=-1, is_hard=-1,
                              ability_metric_id=0, unit_id=unit, lesson_id=7 if cid == "C1" else None,
                              recovered=True, course_id=cid))
    d["kt_recovered"] = recovered
    return recovered


def build_kts(d):
    for k in d["kt"]:
        k["course_id"] = KT2C.get(k["kt_id"], "C3")
        k["recovered"] = False
    return d["kt"] + d["kt_recovered"]


def build_units(d):
    units = {u["unit_id"]: dict(u) for u in d["units"]}
    # C3 的 4 个逻辑单元
    for uid, uname, code in [(71, "机器学习基础", "ML1"), (72, "神经网络与深度学习", "ML2"),
                             (73, "卷积神经网络", "ML3"), (74, "循环神经网络与模式识别", "ML4")]:
        units[uid] = dict(unit_id=uid, unit_code=code, unit_name=uname,
                          parent_unit_id=None, lesson_id=None)
    for u in units.values():
        u["course_id"] = "C3" if u["unit_id"] >= 71 else ("C1" if (u.get("lesson_id") == 7) else "C2")
    return list(units.values())


def extend_edges(d, kts):
    """先修边构建:
      (1) 剔除悬挂边: 关系表中有 28 条边指向已停开课程(KT 153~177)——
          该课程知识点既不在 knowledge_kt 也不在 exam_question 中, 无题目/资源关联;
      (2) 保留剩余真实边;
      (3) 按 单元内 kt_code 顺序 + 单元顺序 补全恢复知识点(86/87/88 及 C3 的 20 个 KT)的先修链;
      (4) 用并查集保证每门课程内部强连通(补最少的边)。
    """
    kt_ids = {k["kt_id"] for k in kts}
    raw = [r for r in d["rel"]]
    dangling = [r for r in raw if r["from_kt"] not in kt_ids or r["to_kt"] not in kt_ids]
    kept = [r for r in raw if r["from_kt"] in kt_ids and r["to_kt"] in kt_ids]
    edges = [dict(src=r["from_kt"], dst=r["to_kt"], rel="PREREQUISITE",
                  source="knowledge_kt_relationship") for r in kept]
    existing = {(e["src"], e["dst"]) for e in edges}
    print(f"  [edges] 原始={len(raw)} 剔除悬挂={len(dangling)} 保留真实={len(kept)}")

    added_infer = 0
    # (3) 课程内按 (unit 顺序, kt_code) 生成链
    bycourse = defaultdict(list)
    for k in kts:
        bycourse[k["course_id"]].append(k)
    for cid, ks in bycourse.items():
        def keyf(k):
            return (k.get("unit_id") or 0, str(k.get("kt_code") or ""))
        ks = sorted(ks, key=keyf)
        for a, b in zip(ks, ks[1:]):
            u, v = a["kt_id"], b["kt_id"]
            if (u, v) not in existing:
                edges.append(dict(src=u, dst=v, rel="PREREQUISITE",
                                  source="inferred:curriculum_order"))
                existing.add((u, v)); added_infer += 1
    print(f"  [edges] 课程内序列补全={added_infer}")

    # (4) 连通性检查（并查集）
    parent = {k: k for k in kt_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        a, b = find(e["src"]), find(e["dst"])
        if a != b: parent[a] = b
    comps = defaultdict(list)
    for k in kt_ids:
        comps[find(k)].append(k)
    big = sorted(comps.values(), key=len, reverse=True)
    print(f"  [edges] 连通分量数={len(big)}  各分量规模={[len(c) for c in big[:6]]}")
    # 合并孤立/小分量到同课程主分量
    added_conn = 0
    for comp in big[1:]:
        cid = KT2C.get(comp[0], "C3")
        same = [c for c in big[0] if KT2C.get(c, "C3") == cid]
        if not same:
            continue
        a, b = sorted(comp)[0], sorted(same)[-1]
        if (b, a) not in existing:
            edges.append(dict(src=b, dst=a, rel="PREREQUISITE", source="inferred:connectivity"))
            existing.add((b, a)); added_conn += 1
    print(f"  [edges] 连通性补全={added_conn} 总边数={len(edges)}")
    return edges


def derive_students(d):
    stu = {}
    for uid, s in d["stu"].items():
        un = s["user_name"]
        if len(un) == 11 and un.isdigit():
            yy = int(un[:2]); enroll = 2000 + yy
            college = un[2:4]; mcode = un[4:6]
            major = MAJOR_CODE.get(mcode, "其他")
        else:
            enroll, college, mcode, major = None, None, None, "其他"
        stu[uid] = dict(student_id=uid, stu_number=un, nick_name=s["nick_name"],
                        enroll_year=enroll, college_code=college, major_code=mcode,
                        major=major)
    # 年级：以数据采集学期(2025-2026 学年)为基准
    for s in stu.values():
        if s["enroll_year"]:
            # 2024 级 => 大二(2025学年), 2023 级 => 大三, 2025 级 => 大一
            g = 2025 - s["enroll_year"] + 1
            s["grade"] = max(1, min(4, g))
        else:
            s["grade"] = 1
    return stu


def empirical_edge_weights(mastery, courses):
    """反向耦合(LKG -> CKG): 用真实掌握数据重标定先修边权。

    采用「共观测学生的掌握度相关系数」作为经验依赖强度:
        w_emp(u,v) = max(0, Pearson( m_u , m_v )),  m 为 correct_rate
    选择依据(由 diag 脚本实测): 真实 passed 标签极度饱和(2x2 表 01/00 恒为 0,
    即"未掌握"侧无样本), 条件增益 P(v|u)-P(v|¬u) 不可估计; 而 correct_rate
    在共观测对上可计算且于真实先修边上显著为正(0.03~0.41)。
    小样本(n<8)回退为 0, 由结构先验主导。
    """
    m = defaultdict(dict)          # student -> kt -> correct_rate
    passed = defaultdict(dict)     # student -> kt -> passed
    for r in mastery:
        m[r["student_id"]][r["kt_id"]] = float(r["correct_rate"])
        passed[r["student_id"]][r["kt_id"]] = int(r["passed"])
    students = list(m.keys())
    return m, students, passed


def _pearson(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def edge_weight(edges, m, students, passed):
    """返回 {(u,v): (w_emp, n_obs)}"""
    # 预取: kt -> [(stu, rate)]
    bykt = defaultdict(list)
    for s in students:
        for kt, cr in m[s].items():
            bykt[kt].append((s, cr))
    out = {}
    for e in edges:
        u, v = e["src"], e["dst"]
        du = dict(bykt.get(u, []))
        xs, ys = [], []
        for s, crv in bykt.get(v, []):
            if s in du:
                xs.append(du[s]); ys.append(crv)
        r = _pearson(xs, ys)
        out[(u, v)] = (round(max(0.0, r), 6) if r is not None else 0.0, len(xs))
    return out


def find_scope_conflicts(kts):
    """自动发现跨课程同名知识点（串味的天然来源）。"""
    byname = defaultdict(list)
    for k in kts:
        byname[k["kt_name"].strip()].append(k)
    conf = []
    for name, ks in byname.items():
        cs = sorted({k["course_id"] for k in ks})
        if len(cs) > 1:
            conf.append(dict(name=name, courses=cs, kt_ids=[k["kt_id"] for k in ks]))
    return sorted(conf, key=lambda x: x["name"])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neo4j", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CY, exist_ok=True)

    conn = connect(); cur = conn.cursor()
    print("[1] 读取原始数据 ...")
    d = load_raw(cur)
    print(f"    kt={len(d['kt'])} rel={len(d['rel'])} question={len(d['q'])} "
          f"interaction={len(d['inter'])} mastery={len(d['mastery'])} material={len(d['material'])}")

    print("[2] 恢复缺失知识点 ...")
    rec = recover_missing_kts(d)
    kts = build_kts(d)
    units = build_units(d)
    print(f"    恢复 {len(rec)} 个知识点; KT 总数 = {len(kts)}")

    print("[3] 补全先修边 ...")
    edges = extend_edges(d, kts)

    print("[4] 学生画像标签 ...")
    students = derive_students(d)
    # 有活动的学生 = 有答题记录 或 有掌握度标签（后者是 KT 任务的标签来源，不能丢）
    active = ({r["student_id"] for r in d["inter"]} |
              {r["student_id"] for r in d["mastery"]})
    print(f"    学生总数={len(students)} 有活动={len(active)} "
          f"(答题={len({r['student_id'] for r in d['inter']})} "
          f"标签={len({r['student_id'] for r in d['mastery']})})")

    print("[5] 反向耦合：重标定先修边权 ...")
    m, ml_students, passed = empirical_edge_weights(d["mastery"], COURSES)
    ewp = edge_weight(edges, m, ml_students, passed)
    ew = {k: v[0] for k, v in ewp.items()}
    nobs = {k: v[1] for k, v in ewp.items()}
    nz = sum(1 for v in ew.values() if v > 0)
    print(f"    可计算边权 {nz}/{len(edges)}; "
          f"均值={sum(ew.values())/max(1,len(ew)):.4f}; "
          f"最大={max(ew.values()) if ew else 0:.4f}")

    print("[6] 作用域冲突发现 ...")
    conf = find_scope_conflicts(kts)
    print(f"    跨课程同名知识点对: {len(conf)}")
    for c in conf:
        print("       ", c["name"], c["courses"], c["kt_ids"])

    # ---------------- 写 CSV ----------------
    print("[7] 写出规范化数据集 ...")
    # 知识点级真实统计（题目数 / 交互数 / 经验正确率）
    qn = Counter(int(x["kt_id"]) for x in d["q"])
    _agg = defaultdict(lambda: [0, 0])
    for r in d["inter"]:
        a = _agg[r["kt_id"]]; a[0] += 1; a[1] += r["correct"]
    qi = {k: (v[0], v[1] / v[0]) for k, v in _agg.items()}

    def wcsv(name, cols, rows):
        p = os.path.join(OUT, name)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"    {name:<28} {len(rows):>8}")

    wcsv("courses.csv", ["course_id", "name", "grade_scope", "major_scope", "term"],
         [dict(course_id=c["course_id"], name=c["name"],
               grade_scope="|".join(map(str, c["grade_scope"])),
               major_scope="|".join(c["major_scope"]), term=c["term"])
          for c in COURSES.values()])
    wcsv("units.csv", ["unit_id", "unit_code", "unit_name", "course_id"],
         [dict(unit_id=u["unit_id"], unit_code=u["unit_code"], unit_name=u["unit_name"],
               course_id=u["course_id"]) for u in units])
    wcsv("kts.csv", ["kt_id", "kt_code", "kt_name", "course_id", "unit_id", "is_important",
                     "is_hard", "ability_metric_id", "recovered", "grade_scope", "major_scope",
                     "n_questions", "n_interactions", "emp_accuracy", "emp_difficulty"],
         [dict(k, grade_scope="|".join(map(str, COURSES[k["course_id"]]["grade_scope"])),
               major_scope="|".join(COURSES[k["course_id"]]["major_scope"]),
               unit_id=k.get("unit_id") or "", is_important=k.get("is_important", -1),
               is_hard=k.get("is_hard", -1), ability_metric_id=k.get("ability_metric_id", 0),
               n_questions=qn.get(k["kt_id"], 0),
               n_interactions=qi.get(k["kt_id"], (0, 0))[0],
               emp_accuracy=round(qi.get(k["kt_id"], (0, 0))[1], 6),
               emp_difficulty=round(1 - qi.get(k["kt_id"], (0, 0))[1], 6) if k["kt_id"] in qi else "")
          for k in kts])
    wcsv("edges.csv", ["src_kt", "dst_kt", "relation", "weight", "n_obs", "source"],
         [dict(src_kt=e["src"], dst_kt=e["dst"], relation="PREREQUISITE",
               weight=ew.get((e["src"], e["dst"]), 0.0),
               n_obs=nobs.get((e["src"], e["dst"]), 0),
               source=e.get("source", "knowledge_kt_relationship")) for e in edges])
    wcsv("questions.csv", ["q_id", "kt_id", "stem", "option_A", "option_B", "option_C",
                           "option_D", "answer", "score"],
         [dict(q_id=x["q_id"], kt_id=x["kt_id"], stem=(x["stem"] or "").replace("\n", " ")[:300],
               option_A=x["oA"], option_B=x["oB"], option_C=x["oC"], option_D=x["oD"],
               answer=x["answer"], score=x["score"]) for x in d["q"]])
    wcsv("students.csv", ["student_id", "stu_number", "nick_name", "enroll_year",
                          "college_code", "major_code", "major", "grade", "has_interaction"],
         [dict(s, has_interaction=int(s["student_id"] in active)) for s in students.values()])
    wcsv("interactions.csv", ["student_id", "kt_id", "test_id", "answer", "correct"],
         d["inter"])
    wcsv("mastery.csv", ["student_id", "kt_id", "done", "correct", "correct_rate",
                         "complete_rate", "passed"],
         d["mastery"])
    wcsv("scope_conflicts.csv", ["kt_name", "courses", "kt_ids"],
         [dict(kt_name=c["name"], courses="|".join(c["courses"]),
               kt_ids="|".join(map(str, c["kt_ids"]))) for c in conf])
    wcsv("materials.csv", ["material_id", "name", "mtype", "ftype", "kt_id", "unit_id", "lesson_id"],
         d["material"])

    # ---- LKG 衍生关系 ----
    agg = defaultdict(lambda: [0, 0])
    for r in d["inter"]:
        a = agg[(r["student_id"], r["kt_id"])]
        a[0] += 1; a[1] += r["correct"]
    _kt2c = {k["kt_id"]: k["course_id"] for k in kts}
    wcsv("attempted.csv", ["student_id", "kt_id", "n", "correct", "acc"],
         [dict(student_id=s, kt_id=k, n=v[0], correct=v[1],
               acc=round(v[1] / v[0], 6)) for (s, k), v in agg.items()])
    wcsv("enrolled.csv", ["student_id", "course_id"],
         [dict(student_id=s, course_id=c) for s, c in sorted({(s, _kt2c.get(k, "C3"))
                                                              for (s, k) in agg.keys()})])
    sn = []
    for c in conf:
        ids = [int(x) for x in c["kt_ids"]]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sn.append(dict(a_kt=min(ids[i], ids[j]), b_kt=max(ids[i], ids[j]),
                               kt_name=c["name"], course_pair="|".join(c["courses"])))
    wcsv("same_name_as.csv", ["a_kt", "b_kt", "kt_name", "course_pair"], sn)

    # 学生相似度 kNN（掌握向量余弦，top-10，标签数>=5）
    import numpy as np
    vec = defaultdict(dict)
    for r in d["mastery"]:
        vec[r["student_id"]][r["kt_id"]] = float(r["correct_rate"]) if r["passed"] == 2 else 0.0
    ids = [s for s, v in vec.items() if len(v) >= 5]
    allk = sorted({k for s in ids for k in vec[s]})
    kidx = {k: i for i, k in enumerate(allk)}
    M = np.zeros((len(ids), len(allk)), dtype=np.float32)
    for i, s in enumerate(ids):
        for k, v in vec[s].items():
            M[i, kidx[k]] = v
    nrm = np.linalg.norm(M, axis=1, keepdims=True); nrm[nrm == 0] = 1
    S = (M / nrm) @ (M / nrm).T
    np.fill_diagonal(S, -1)
    TOPK = min(10, max(1, len(ids) - 1))
    sim_rows = []
    for i in range(len(ids)):
        idx = np.argpartition(-S[i], TOPK - 1)[:TOPK]
        for j in idx:
            if S[i, j] > 0:
                sim_rows.append(dict(a_student=int(ids[i]), b_student=int(ids[j]),
                                     sim=round(float(S[i, j]), 6)))
    wcsv("similar_to.csv", ["a_student", "b_student", "sim"], sim_rows)

    stats = dict(
        courses=len(COURSES), units=len(units), kts=len(kts),
        kts_recovered=len(rec), edges=len(edges), edges_with_weight=nz,
        questions=len(d["q"]), students=len(students), students_active=len(active),
        interactions=len(d["inter"]), mastery_labels=len(d["mastery"]),
        scope_conflict_pairs=len(conf),
        interaction_correct_rate=round(sum(r["correct"] for r in d["inter"]) / len(d["inter"]), 6),
        mastery_passed_dist=dict(Counter(r["passed"] for r in d["mastery"])),
        mastery_correct_rate_is_1=sum(1 for r in d["mastery"] if r["correct_rate"] == 1.0),
        major_dist=dict(Counter(s["major"] for s in students.values())),
        grade_dist=dict(Counter(s["grade"] for s in students.values())),
        course_kt_count={c: len(v["kts"]) for c, v in COURSES.items()},
    )
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1600])

    if not args.no_neo4j:
        print("[8] 生成 Neo4j 导入脚本 ...")
        write_cypher(d, kts, units, edges, ew, students, COURSES)
        write_load_cypher()
    conn.close()
    print("DONE")


def write_load_cypher():
    """生成纯 Cypher 的 LOAD CSV 装载脚本（供 cypher-shell 执行，无需 Python 驱动）。"""
    steps = [
        ("Course", "courses.csv",
         "MERGE (c:Course {course_id:row.course_id}) "
         "SET c.name=row.name, c.grade_scope=row.grade_scope, "
         "    c.major_scope=row.major_scope, c.term=row.term"),
        ("Unit", "units.csv",
         "MERGE (u:Unit {unit_id:toInteger(row.unit_id)}) "
         "SET u.code=row.unit_code, u.name=row.unit_name, u.course_id=row.course_id"),
        ("KT", "kts.csv",
         "MERGE (k:KT {kt_id:toInteger(row.kt_id)}) "
         "SET k.code=row.kt_code, k.name=row.kt_name, k.course_id=row.course_id, "
         "    k.unit_id=toInteger(row.unit_id), k.is_important=toInteger(row.is_important), "
         "    k.is_hard=toInteger(row.is_hard), k.recovered=(row.recovered='True'), "
         "    k.grade_scope=row.grade_scope, k.major_scope=row.major_scope, "
         "    k.n_questions=toInteger(row.n_questions), "
         "    k.n_interactions=toInteger(row.n_interactions), "
         "    k.emp_accuracy=CASE WHEN row.emp_accuracy='' THEN null ELSE toFloat(row.emp_accuracy) END, "
         "    k.emp_difficulty=CASE WHEN row.emp_difficulty='' THEN null ELSE toFloat(row.emp_difficulty) END"),
        ("Question", "questions.csv",
         "MERGE (q:Question {q_id:row.q_id}) "
         "SET q.kt_id=toInteger(row.kt_id), q.stem=row.stem, q.answer=row.answer, "
         "    q.option_A=row.option_A, q.option_B=row.option_B, "
         "    q.option_C=row.option_C, q.option_D=row.option_D"),
        ("Student", "students.csv",
         "WITH row WHERE row.has_interaction='1' "
         "MERGE (s:Student {student_id:toInteger(row.student_id)}) "
         "SET s.stu_number=row.stu_number, s.name=row.nick_name, "
         "    s.enroll_year=toInteger(row.enroll_year), s.grade=toInteger(row.grade), "
         "    s.major=row.major, s.college_code=row.college_code, s.major_code=row.major_code"),
        ("HAS_UNIT", "units.csv",
         "MATCH (c:Course {course_id:row.course_id}), (u:Unit {unit_id:toInteger(row.unit_id)}) "
         "MERGE (c)-[:HAS_UNIT]->(u)"),
        ("HAS_KT", "kts.csv",
         "MATCH (u:Unit {unit_id:toInteger(row.unit_id)}), (k:KT {kt_id:toInteger(row.kt_id)}) "
         "MERGE (u)-[:HAS_KT]->(k)"),
        ("TESTS", "questions.csv",
         "MATCH (q:Question {q_id:row.q_id}), (k:KT {kt_id:toInteger(row.kt_id)}) "
         "MERGE (q)-[:TESTS]->(k)"),
        ("PREREQUISITE_OF", "edges.csv",
         "MATCH (a:KT {kt_id:toInteger(row.src_kt)}), (b:KT {kt_id:toInteger(row.dst_kt)}) "
         "MERGE (a)-[e:PREREQUISITE_OF]->(b) "
         "SET e.w=toFloat(row.weight), e.n_obs=toInteger(row.n_obs), e.src=row.source"),
        ("SAME_NAME_AS", "same_name_as.csv",
         "MATCH (a:KT {kt_id:toInteger(row.a_kt)}), (b:KT {kt_id:toInteger(row.b_kt)}) "
         "MERGE (a)-[e:SAME_NAME_AS]->(b) "
         "SET e.name=row.kt_name, e.course_pair=row.course_pair"),
        ("MASTERED", "mastery.csv",
         "MATCH (s:Student {student_id:toInteger(row.student_id)}), "
         "      (k:KT {kt_id:toInteger(row.kt_id)}) "
         "MERGE (s)-[e:MASTERED]->(k) "
         "SET e.prob=toFloat(row.correct_rate), e.done=toInteger(row.done), "
         "    e.correct=toInteger(row.correct), e.passed=toInteger(row.passed)"),
        ("ATTEMPTED", "attempted.csv",
         "MATCH (s:Student {student_id:toInteger(row.student_id)}), "
         "      (k:KT {kt_id:toInteger(row.kt_id)}) "
         "MERGE (s)-[e:ATTEMPTED]->(k) "
         "SET e.n=toInteger(row.n), e.acc=toFloat(row.acc), e.correct=toInteger(row.correct)"),
        ("ENROLLED", "enrolled.csv",
         "MATCH (s:Student {student_id:toInteger(row.student_id)}), "
         "      (c:Course {course_id:row.course_id}) "
         "MERGE (s)-[:ENROLLED]->(c)"),
        ("SIMILAR_TO", "similar_to.csv",
         "MATCH (a:Student {student_id:toInteger(row.a_student)}), "
         "      (b:Student {student_id:toInteger(row.b_student)}) "
         "MERGE (a)-[e:SIMILAR_TO]->(b) SET e.sim=toFloat(row.sim)"),
    ]
    L = ["// 自动生成: Neo4j LOAD CSV 批量装载 (由 cypher-shell 执行)",
         "// 前置: data/dnui_ikg/*.csv 已同步到 neo4j 容器的 /var/lib/neo4j/import/",
         "",
         "MATCH (n) DETACH DELETE n;", ""]
    for name, fn, body in steps:
        L.append(f"// ---- {name} <- {fn} ----")
        L.append(f"LOAD CSV WITH HEADERS FROM 'file:///{fn}' AS row")
        L.append(body + ";")
        L.append("")
    L += ["// ---- 反向耦合标定: w_calib = 1 + tanh(w) ----",
          "MATCH (:KT)-[e:PREREQUISITE_OF]->(:KT)",
          "SET e.w_calib = 1.0 + (exp(2*e.w)-1)/(exp(2*e.w)+1);", "",
          "// ---- 课程规模属性 ----",
          "MATCH (c:Course)",
          "CALL { WITH c MATCH (c)-[:HAS_UNIT]->(:Unit)-[:HAS_KT]->(k:KT) RETURN count(DISTINCT k) AS n }",
          "SET c.n_kt = n;", ""]
    os.makedirs(CY, exist_ok=True)
    with open(os.path.join(CY, "20_load.cypher"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("    LOAD CSV 脚本 -> kg/cypher/20_load.cypher")


def write_cypher(d, kts, units, edges, ew, students, COURSES):
    """输出 Neo4j 图模式定义（实际数据装载由 kg/load_neo4j.py 用 CSV + UNWIND 批量完成）。"""
    c = """// ============================================================
// DGCL 双图谱图模式 (Neo4j 5.x)
// 数据装载: experiments/kg/load_neo4j.py  (读取 data/dnui_ikg/*.csv, UNWIND 批量导入)
// ============================================================

// ---- 约束 ----
CREATE CONSTRAINT course_id   IF NOT EXISTS FOR (c:Course)   REQUIRE c.course_id   IS UNIQUE;
CREATE CONSTRAINT unit_id     IF NOT EXISTS FOR (u:Unit)     REQUIRE u.unit_id     IS UNIQUE;
CREATE CONSTRAINT kt_id       IF NOT EXISTS FOR (k:KT)       REQUIRE k.kt_id       IS UNIQUE;
CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.q_id        IS UNIQUE;
CREATE CONSTRAINT student_id  IF NOT EXISTS FOR (s:Student)  REQUIRE s.student_id  IS UNIQUE;

// ---- 索引 ----
CREATE INDEX kt_course IF NOT EXISTS FOR (k:KT) ON (k.course_id);
CREATE INDEX kt_scope  IF NOT EXISTS FOR (k:KT) ON (k.grade_scope, k.major_scope);
CREATE INDEX stu_scope IF NOT EXISTS FOR (s:Student) ON (s.grade, s.major);

// ============================================================
// CKG 课程知识图谱
//   (:Course)-[:HAS_UNIT]->(:Unit)-[:HAS_KT]->(:KT)
//   (:KT)-[:PREREQUISITE_OF {w,w_struct,w_calib,src}]->(:KT)
//   (:Question)-[:TESTS]->(:KT)
//   (:KT)-[:SAME_NAME_AS {course_pair}]->(:KT)     // 跨课程同名(串味源)
// ============================================================

// 反向耦合标定: w_calib = 1 + tanh(w)
// 装载后执行:
//   MATCH (a:KT)-[r:PREREQUISITE_OF]->(b:KT)
//   SET r.w_calib = 1.0 + (exp(2*r.w)-1)/(exp(2*r.w)+1);

// ============================================================
// LKG 学习者认知图谱
//   (:Student {student_id,stu_number,name,enroll_year,grade,major})
//   (:Student)-[:MASTERED  {prob,done,correct,passed,in_scope}]->(:KT)
//   (:Student)-[:ATTEMPTED {n,acc}]->(:KT)
//   (:Student)-[:ENROLLED  {course_id}]->(:Course)
//   (:Student)-[:SIMILAR_TO {sim}]->(:Student)     // 掌握向量 kNN
// ============================================================

// 常用查询: 取学生作用域内的先修闭包(供 GES 证据子图)
//   MATCH (s:Student {student_id:$sid})-[:MASTERED]->(k:KT)
//   WITH s, collect(k) AS known
//   UNWIND known AS k
//   MATCH (k)-[:PREREQUISITE_OF*1..3]->(t:KT)
//   WHERE t.grade_scope CONTAINS toString(s.grade)
//   RETURN DISTINCT t, count(*) AS support ORDER BY support DESC LIMIT $k;

// 常用查询: 候选资源集合(CSG 硬门控)
//   MATCH (k:KT)-[:PREREQUISITE_OF*1..2]->(t:KT)
//   WHERE k.kt_id IN $known
//     AND any(g IN split(t.grade_scope,',') WHERE toInteger(g)=$grade)
//     AND any(m IN split(t.major_scope,',') WHERE m=$major)
//   MATCH (q:Question)-[:TESTS]->(t)
//   RETURN t.kt_id, t.name, q.q_id LIMIT $k;
"""
    with open(os.path.join(CY, "01_schema.cypher"), "w", encoding="utf-8") as f:
        f.write(c)
    print(f"    图模式 -> kg/cypher/01_schema.cypher")


if __name__ == "__main__":
    sys.exit(main())
