#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data/dnui_ikg/*.csv 批量装载进 Neo4j，构成 CKG + LKG 双图谱。

用法:
  python kg/load_neo4j.py [--uri bolt://localhost:7687] [--user neo4j] [--password IkgDnui2026]
"""
import os, sys, csv, json, argparse, math
from collections import defaultdict, Counter

EXPROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(EXPROOT)
DATA = os.path.join(REPO, "data", "dnui_ikg")
CY = os.path.join(EXPROOT, "kg", "cypher")


def rd(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chunked(it, n=1000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf; buf = []
    if buf:
        yield buf


def run(driver, cypher, batch, label=""):
    with driver.session() as s:
        for b in chunked(batch, 1000):
            s.run(cypher, rows=b)
    print(f"  {label:<34} {len(batch):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASS", "IkgDnui2026"))
    args = ap.parse_args()
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    drv.verify_connectivity()
    print("connected:", args.uri)

    courses = rd("courses.csv"); units = rd("units.csv"); kts = rd("kts.csv")
    edges = rd("edges.csv"); questions = rd("questions.csv"); students = rd("students.csv")
    mastery = rd("mastery.csv"); inter = rd("interactions.csv"); conf = rd("scope_conflicts.csv")
    print(f"loaded csv: course={len(courses)} unit={len(units)} kt={len(kts)} "
          f"edge={len(edges)} q={len(questions)} stu={len(students)} "
          f"mastery={len(mastery)} inter={len(inter)}")

    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    # schema
    with open(os.path.join(CY, "01_schema.cypher"), encoding="utf-8") as f:
        body = [x.strip() for x in f.read().split(";")]
    with drv.session() as s:
        for st in body:
            st = "\n".join(l for l in st.splitlines() if not l.strip().startswith("//"))
            if st.strip():
                s.run(st)
    print("schema created")

    # ---------- CKG ----------
    run(drv, """UNWIND $rows AS r
        MERGE (c:Course {course_id:r.course_id})
        SET c.name=r.name, c.grade_scope=r.grade_scope,
            c.major_scope=r.major_scope, c.term=r.term""",
        courses, "Course")
    run(drv, """UNWIND $rows AS r
        MERGE (u:Unit {unit_id:toInteger(r.unit_id)})
        SET u.code=r.unit_code, u.name=r.unit_name, u.course_id=r.course_id""",
        units, "Unit")
    run(drv, """UNWIND $rows AS r
        MERGE (k:KT {kt_id:toInteger(r.kt_id)})
        SET k.code=r.kt_code, k.name=r.kt_name, k.course_id=r.course_id,
            k.unit_id=toInteger(r.unit_id), k.is_important=toInteger(r.is_important),
            k.is_hard=toInteger(r.is_hard), k.recovered=(r.recovered='True'),
            k.grade_scope=r.grade_scope, k.major_scope=r.major_scope,
            k.n_questions=toInteger(r.n_questions),
            k.n_interactions=toInteger(r.n_interactions),
            k.emp_accuracy=CASE WHEN r.emp_accuracy='' THEN null
                                ELSE toFloat(r.emp_accuracy) END,
            k.emp_difficulty=CASE WHEN r.emp_difficulty='' THEN null
                                  ELSE toFloat(r.emp_difficulty) END""",
        kts, "KT")
    run(drv, """UNWIND $rows AS r
        MERGE (q:Question {q_id:r.q_id})
        SET q.kt_id=toInteger(r.kt_id), q.stem=r.stem, q.answer=r.answer,
            q.option_A=r.option_A, q.option_B=r.option_B,
            q.option_C=r.option_C, q.option_D=r.option_D""",
        questions, "Question")
    run(drv, """UNWIND $rows AS r
        MATCH (c:Course {course_id:r.course_id}), (u:Unit {unit_id:toInteger(r.unit_id)})
        MERGE (c)-[:HAS_UNIT]->(u)""", units, "HAS_UNIT")
    run(drv, """UNWIND $rows AS r
        MATCH (u:Unit {unit_id:toInteger(r.unit_id)}), (k:KT {kt_id:toInteger(r.kt_id)})
        MERGE (u)-[:HAS_KT]->(k)""", kts, "HAS_KT")
    run(drv, """UNWIND $rows AS r
        MATCH (q:Question {q_id:r.q_id}), (k:KT {kt_id:toInteger(r.kt_id)})
        MERGE (q)-[:TESTS]->(k)""", questions, "TESTS")
    run(drv, """UNWIND $rows AS r
        MATCH (a:KT {kt_id:toInteger(r.src_kt)}), (b:KT {kt_id:toInteger(r.dst_kt)})
        MERGE (a)-[e:PREREQUISITE_OF]->(b)
        SET e.w=toFloat(r.weight), e.n_obs=toInteger(r.n_obs), e.src=r.source""",
        edges, "PREREQUISITE_OF")
    # 反向耦合标定 + 跨课程同名边
    with drv.session() as s:
        s.run("MATCH (:KT)-[e:PREREQUISITE_OF]->(:KT) "
              "SET e.w_calib = 1.0 + (exp(2*e.w)-1)/(exp(2*e.w)+1)")
    pairs = []
    for c in conf:
        ids = [int(x) for x in c["kt_ids"].split("|")]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append(dict(a=min(ids[i], ids[j]), b=max(ids[i], ids[j]),
                                  nm=c["kt_name"], cp=c["courses"]))
    run(drv, """UNWIND $rows AS r
        MATCH (a:KT {kt_id:r.a}), (b:KT {kt_id:r.b})
        MERGE (a)-[e:SAME_NAME_AS]->(b)
        SET e.name=r.nm, e.course_pair=r.cp""", pairs, "SAME_NAME_AS(串味源)")

    # ---------- LKG ----------
    run(drv, """UNWIND $rows AS r
        MERGE (s:Student {student_id:toInteger(r.student_id)})
        SET s.stu_number=r.stu_number, s.name=r.nick_name,
            s.enroll_year=toInteger(r.enroll_year), s.grade=toInteger(r.grade),
            s.major=r.major, s.college_code=r.college_code, s.major_code=r.major_code""",
        [x for x in students if x["has_interaction"] == "1"], "Student(有交互)")
    run(drv, """UNWIND $rows AS r
        MATCH (s:Student {student_id:toInteger(r.student_id)})
        MATCH (k:KT {kt_id:toInteger(r.kt_id)})
        MERGE (s)-[e:MASTERED]->(k)
        SET e.prob=toFloat(r.correct_rate), e.done=toInteger(r.done),
            e.correct=toInteger(r.correct), e.passed=toInteger(r.passed)""",
        mastery, "MASTERED")

    # ATTEMPTED: 聚合 (student, kt)
    agg = defaultdict(lambda: [0, 0])
    for r in inter:
        a = agg[(r["student_id"], r["kt_id"])]
        a[0] += 1; a[1] += int(r["correct"])
    att = [dict(student_id=s, kt_id=k, n=v[0], acc=v[1] / v[0],
                correct=v[1]) for (s, k), v in agg.items()]
    run(drv, """UNWIND $rows AS r
        MATCH (s:Student {student_id:toInteger(r.student_id)})
        MATCH (k:KT {kt_id:toInteger(r.kt_id)})
        MERGE (s)-[e:ATTEMPTED]->(k)
        SET e.n=toInteger(r.n), e.acc=toFloat(r.acc), e.correct=toInteger(r.correct)""",
        att, "ATTEMPTED")

    # ENROLLED: 学生 -> 课程（由其交互的知识点课程归属决定）
    kt2c = {r["kt_id"]: r["course_id"] for r in kts}
    enr = sorted({(s, kt2c.get(k, "C3")) for (s, k) in agg.keys()})
    run(drv, """UNWIND $rows AS r
        MATCH (s:Student {student_id:toInteger(r.sid)}), (c:Course {course_id:r.cid})
        MERGE (s)-[e:ENROLLED]->(c)""",
        [dict(sid=a, cid=b) for a, b in enr], "ENROLLED")

    # SIMILAR_TO: 掌握向量 kNN（top-10，仅标签数>=5 的学生）
    vec = defaultdict(dict)
    for r in mastery:
        vec[r["student_id"]][r["kt_id"]] = float(r["correct_rate"]) if r["passed"] == 2 else 0.0
    ids = [s for s, v in vec.items() if len(v) >= 5]
    allk = sorted({k for s in ids for k in vec[s]})
    kidx = {k: i for i, k in enumerate(allk)}
    import numpy as np
    M = np.zeros((len(ids), len(allk)), dtype=np.float32)
    for i, s in enumerate(ids):
        for k, v in vec[s].items():
            M[i, kidx[k]] = v
    nrm = np.linalg.norm(M, axis=1, keepdims=True); nrm[nrm == 0] = 1
    Mn = M / nrm
    S = Mn @ Mn.T
    np.fill_diagonal(S, -1)
    TOPK = 10
    sim_rows = []
    for i in range(len(ids)):
        idx = np.argpartition(-S[i], TOPK)[:TOPK]
        for j in idx:
            sim_rows.append(dict(a=int(ids[i]), b=int(ids[j]), sim=float(S[i, j])))
    run(drv, """UNWIND $rows AS r
        MATCH (a:Student {student_id:r.a}), (b:Student {student_id:r.b})
        MERGE (a)-[e:SIMILAR_TO]->(b) SET e.sim=r.sim""",
        sim_rows, "SIMILAR_TO")

    # ---------- 校验 ----------
    print("\n===== 图统计 =====")
    with drv.session() as s:
        for lbl in ["Course", "Unit", "KT", "Question", "Student"]:
            print(f"  {lbl:<10}", s.run(f"MATCH (n:{lbl}) RETURN count(n) AS c").single()["c"])
        for rel in ["HAS_UNIT", "HAS_KT", "TESTS", "PREREQUISITE_OF", "SAME_NAME_AS",
                    "MASTERED", "ATTEMPTED", "ENROLLED", "SIMILAR_TO"]:
            print(f"  :{rel:<16}", s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"])
    drv.close()
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
