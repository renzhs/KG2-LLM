// ============================================================
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
