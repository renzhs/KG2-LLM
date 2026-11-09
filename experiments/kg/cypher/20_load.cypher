// 自动生成: Neo4j LOAD CSV 批量装载 (由 cypher-shell 执行)
// 前置: data/dnui_ikg/*.csv 已同步到 neo4j 容器的 /var/lib/neo4j/import/

MATCH (n) DETACH DELETE n;

// ---- Course <- courses.csv ----
LOAD CSV WITH HEADERS FROM 'file:///courses.csv' AS row
MERGE (c:Course {course_id:row.course_id}) SET c.name=row.name, c.grade_scope=row.grade_scope,     c.major_scope=row.major_scope, c.term=row.term;

// ---- Unit <- units.csv ----
LOAD CSV WITH HEADERS FROM 'file:///units.csv' AS row
MERGE (u:Unit {unit_id:toInteger(row.unit_id)}) SET u.code=row.unit_code, u.name=row.unit_name, u.course_id=row.course_id;

// ---- KT <- kts.csv ----
LOAD CSV WITH HEADERS FROM 'file:///kts.csv' AS row
MERGE (k:KT {kt_id:toInteger(row.kt_id)}) SET k.code=row.kt_code, k.name=row.kt_name, k.course_id=row.course_id,     k.unit_id=toInteger(row.unit_id), k.is_important=toInteger(row.is_important),     k.is_hard=toInteger(row.is_hard), k.recovered=(row.recovered='True'),     k.grade_scope=row.grade_scope, k.major_scope=row.major_scope,     k.n_questions=toInteger(row.n_questions),     k.n_interactions=toInteger(row.n_interactions),     k.emp_accuracy=CASE WHEN row.emp_accuracy='' THEN null ELSE toFloat(row.emp_accuracy) END,     k.emp_difficulty=CASE WHEN row.emp_difficulty='' THEN null ELSE toFloat(row.emp_difficulty) END;

// ---- Question <- questions.csv ----
LOAD CSV WITH HEADERS FROM 'file:///questions.csv' AS row
MERGE (q:Question {q_id:row.q_id}) SET q.kt_id=toInteger(row.kt_id), q.stem=row.stem, q.answer=row.answer,     q.option_A=row.option_A, q.option_B=row.option_B,     q.option_C=row.option_C, q.option_D=row.option_D;

// ---- Student <- students.csv ----
LOAD CSV WITH HEADERS FROM 'file:///students.csv' AS row
WITH row WHERE row.has_interaction='1' MERGE (s:Student {student_id:toInteger(row.student_id)}) SET s.stu_number=row.stu_number, s.name=row.nick_name,     s.enroll_year=toInteger(row.enroll_year), s.grade=toInteger(row.grade),     s.major=row.major, s.college_code=row.college_code, s.major_code=row.major_code;

// ---- HAS_UNIT <- units.csv ----
LOAD CSV WITH HEADERS FROM 'file:///units.csv' AS row
MATCH (c:Course {course_id:row.course_id}), (u:Unit {unit_id:toInteger(row.unit_id)}) MERGE (c)-[:HAS_UNIT]->(u);

// ---- HAS_KT <- kts.csv ----
LOAD CSV WITH HEADERS FROM 'file:///kts.csv' AS row
MATCH (u:Unit {unit_id:toInteger(row.unit_id)}), (k:KT {kt_id:toInteger(row.kt_id)}) MERGE (u)-[:HAS_KT]->(k);

// ---- TESTS <- questions.csv ----
LOAD CSV WITH HEADERS FROM 'file:///questions.csv' AS row
MATCH (q:Question {q_id:row.q_id}), (k:KT {kt_id:toInteger(row.kt_id)}) MERGE (q)-[:TESTS]->(k);

// ---- PREREQUISITE_OF <- edges.csv ----
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (a:KT {kt_id:toInteger(row.src_kt)}), (b:KT {kt_id:toInteger(row.dst_kt)}) MERGE (a)-[e:PREREQUISITE_OF]->(b) SET e.w=toFloat(row.weight), e.n_obs=toInteger(row.n_obs), e.src=row.source;

// ---- SAME_NAME_AS <- same_name_as.csv ----
LOAD CSV WITH HEADERS FROM 'file:///same_name_as.csv' AS row
MATCH (a:KT {kt_id:toInteger(row.a_kt)}), (b:KT {kt_id:toInteger(row.b_kt)}) MERGE (a)-[e:SAME_NAME_AS]->(b) SET e.name=row.kt_name, e.course_pair=row.course_pair;

// ---- MASTERED <- mastery.csv ----
LOAD CSV WITH HEADERS FROM 'file:///mastery.csv' AS row
MATCH (s:Student {student_id:toInteger(row.student_id)}),       (k:KT {kt_id:toInteger(row.kt_id)}) MERGE (s)-[e:MASTERED]->(k) SET e.prob=toFloat(row.correct_rate), e.done=toInteger(row.done),     e.correct=toInteger(row.correct), e.passed=toInteger(row.passed);

// ---- ATTEMPTED <- attempted.csv ----
LOAD CSV WITH HEADERS FROM 'file:///attempted.csv' AS row
MATCH (s:Student {student_id:toInteger(row.student_id)}),       (k:KT {kt_id:toInteger(row.kt_id)}) MERGE (s)-[e:ATTEMPTED]->(k) SET e.n=toInteger(row.n), e.acc=toFloat(row.acc), e.correct=toInteger(row.correct);

// ---- ENROLLED <- enrolled.csv ----
LOAD CSV WITH HEADERS FROM 'file:///enrolled.csv' AS row
MATCH (s:Student {student_id:toInteger(row.student_id)}),       (c:Course {course_id:row.course_id}) MERGE (s)-[:ENROLLED]->(c);

// ---- SIMILAR_TO <- similar_to.csv ----
LOAD CSV WITH HEADERS FROM 'file:///similar_to.csv' AS row
MATCH (a:Student {student_id:toInteger(row.a_student)}),       (b:Student {student_id:toInteger(row.b_student)}) MERGE (a)-[e:SIMILAR_TO]->(b) SET e.sim=toFloat(row.sim);

// ---- 反向耦合标定: w_calib = 1 + tanh(w) ----
MATCH (:KT)-[e:PREREQUISITE_OF]->(:KT)
SET e.w_calib = 1.0 + (exp(2*e.w)-1)/(exp(2*e.w)+1);

// ---- 课程规模属性 ----
MATCH (c:Course)
CALL { WITH c MATCH (c)-[:HAS_UNIT]->(:Unit)-[:HAS_KT]->(k:KT) RETURN count(DISTINCT k) AS n }
SET c.n_kt = n;

