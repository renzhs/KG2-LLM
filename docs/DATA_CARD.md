# DNUI-IKG 数据集说明（Data Card）

> 本文件描述随仓库发布的 `data/dnui_ikg/` 目录。数据集名 **DNUI-IKG**（Dalian Neusoft University of
> Information – Intelligent Knowledge Graph）对应论文中的自建数据集；其中用于知识追踪训练的
> **DNUI-IKG-S** 为**校准式半合成**数据，生成规则见第 5 节，`experiments/data/prep_datasets.py`
> 可从本目录的 CSV 完整重建（随机种子固定为 42）。

## 1. 来源与授权

| 项 | 内容 |
|---|---|
| 采集场景 | 大连东软信息学院软件学院三门本科课程的真实教学平台（Java Web 应用开发等） |
| 采集方式 | 教师端导入知识点、单元与题库；学生端作答日志与掌握度统计由平台落库后只读导出 |
| 课程 | `C1` 程序设计基础(Java) / `C2` 面向对象程序设计(Java) / `C3` 机器学习与深度学习导论 |
| 授权 | 数据用于教学研究，任课教师授权公开；学生侧已做去标识处理（见第 4 节） |
| 许可 | 数据部分采用 **CC BY-NC 4.0**（仅限非商业研究使用，须署名）；代码部分采用 MIT |
| 引用 | 见仓库根目录 `CITATION.cff` |

## 2. 规模

| 实体 | 数量 | 说明 |
|---|---|---|
| 课程 | 3 | 每门课程带学段（grade_scope）与专业（major_scope）标签 |
| 单元 | 25 | 课程 → 单元 → 知识点 三层结构 |
| 知识点（KT） | 66 | 其中 23 个为平台导入后按答题记录补录（`recovered=True`） |
| 先修边 | 88 | 全部为 `PREREQUISITE`；43 条来自教学大纲人工标注，45 条由课程顺序推断；64 条带经验标定权重 |
| 题目 | 467 | 单选，含题干、四个选项、标准答案与分值 |
| 学习资源 | 0 | `materials.csv` 仅有表头（平台资源表在导出时为空） |
| 学生账号 | 1 139 | 其中 `has_interaction=1` 的 1 060 个；其余 79 个为无作答记录的账号（含 admin / teacher / test 等系统与测试账号） |
| 作答记录 | 169 351 | 知识点粒度的作答（`test_id` 为平台试卷实例 UUID） |
| 掌握度标签 | 17 862 | (学生, 知识点) 级：完成数、正确数、正确率、完成率、是否通过 |

## 3. 文件与字段

| 文件 | 行数（含表头） | 字段 |
|---|---|---|
| `courses.csv` | 4 | course_id, name, grade_scope, major_scope, term |
| `units.csv` | 26 | unit_id, unit_code, unit_name, course_id |
| `kts.csv` | 67 | kt_id, kt_code, kt_name, course_id, unit_id, is_important, is_hard, ability_metric_id, recovered, grade_scope, major_scope, n_questions, n_interactions, emp_accuracy, emp_difficulty |
| `edges.csv` | 89 | src_kt, dst_kt, relation, weight, n_obs, source |
| `questions.csv` | 695 | q_id, kt_id, stem, option_A~D, answer, score |
| `materials.csv` | 1 | material_id, name, mtype, ftype, kt_id, unit_id, lesson_id（空表） |
| `students.csv` | 1 140 | student_id, enroll_year, major, grade, has_interaction |
| `enrolled.csv` | 1 187 | student_id, course_id |
| `interactions.csv` | 169 352 | student_id, kt_id, test_id, answer, correct |
| `attempted.csv` | 21 832 | student_id, kt_id, n, correct, acc |
| `mastery.csv` | 17 863 | student_id, kt_id, done, correct, correct_rate, complete_rate, passed |
| `similar_to.csv` | 4 341 | a_student, b_student, sim（同作用域学生的相似度图，LKG 用） |
| `same_name_as.csv` | 4 | a_kt, b_kt, kt_name, course_pair（同名异义知识点对） |
| `scope_conflicts.csv` | 4 | kt_name, courses, kt_ids（跨课程同名冲突） |
| `stats.json` | — | 上述统计的机器可读汇总 |

字段口径要点：

- `grade_scope` / `major_scope` 用 `|` 分隔的集合，如 `1` 表示仅大一，`软件工程|计算机科学` 表示两个专业共享；
  这两个字段是 CSG（学段—专业语义约束门控）的**唯一标签来源**，也是本数据集区别于公开基准之处。
- `passed` 取 2/1 两值（2 = 已通过），论文中作为掌握度二值标签。
- `edges.csv` 的 `weight` 为用户侧经验先修强度（由真实通过率标定），`n_obs` 为标定所用观测数；
  `weight=0` 表示该边仅有结构信息、无经验标定。
- `mastery.csv` 的 `correct_rate` 与 `complete_rate` 已由平台按知识点聚合，非原始日志。

## 4. 隐私与去标识（已执行）

发布版相对平台导出**删除了以下四列**：

- `students.csv`：`stu_number`（学号 / 登录名）、`nick_name`（真实姓名）、`college_code`、`major_code`（院系专业代码）

保留的 `student_id` 为平台内部自增整数，**不含任何姓名、学号、手机号或邮箱**，无法与校内任何
记录直接关联；`interactions.csv` 的 `test_id` 为试卷实例 UUID，亦不含学生信息。

对全部 CSV 做过正则扫描（手机号、邮箱、11 位学号模式），发布版**零命中**。
若使用者通过组合字段仍能推断出个体身份，请立即停止使用并联系作者删除。

## 5. DNUI-IKG-S：为什么是"校准式半合成"

平台日志存在两个**结构性缺陷**，使其不能直接用于深度知识追踪的训练与评测：

1. **正确率饱和**：练习模式"答对为止"，落库的 `correct` 有 99.35% 为 1，几乎没有区分度；
2. **掌握度饱和**：92.3% 的 (学生, 知识点) 掌握标签恰为 1.0，同样缺乏方差。

因此 `experiments/data/prep_datasets.py::prep_dnui_syn()` 在**真实**的课程先修 DAG、**真实**的
学段—专业作用域标签、**真实**的交互量分布之上，用 2PL IRT + slip/guess 生成答题序列：

```
P(correct | θ_i, b_k, a_k) = g + (1 − s − g) · σ(a_k (θ_i − b_k))
  θ_i ~ N(0.45, 1)                      学生能力
  b_k : 由真实经验正确率的次序统计量线性映射到 [−1.2, 1.2]   知识点难度
  a_k : 由真实交互量映射到 [0.8, 1.6]                        区分度
  s = 0.08 (slip), g = 0.15 (guess)
  序列按先修拓扑序推进，每个知识点采样 1~3 次；按 10% 比例注入跨学段/跨专业越界作答
  n_students = 3000（其中 2 147 条序列通过"作用域内至少 3 次作答"的过滤），
  max_len = 120，seed = 42
```

生成结果的统计与论文表 1 的 DNUI-IKG 行**逐位一致**（2 147 条序列、117 865 次交互、
平均序列长 54.9、正确率 0.6270）；`prep_datasets.py` 的默认参数即为该口径。

> **使用与引用时的注意事项**：DNUI-IKG-S 的**图结构、作用域标签、规模分布是真实的**，
> **答题序列是半合成的**。因此在论文中它只能支撑"在真实课程结构与真实作用域约束下"的结论，
> 不能用来推断真实学生的学习增益；涉及学习增益的因果结论本文不作主张（见论文"局限"一节）。

## 6. 版本与校验

本版本对应论文投稿稿，文件均为 UTF-8 + LF 换行（`.gitattributes` 已强制）。
各文件 SHA-256 前 16 位（`sha256sum data/dnui_ikg/*` 可复核）：

| 文件 | SHA-256[:16] | 文件 | SHA-256[:16] |
|---|---|---|---|
| attempted.csv | `ea28e9cec92a5737` | mastery.csv | `315d68eadde759c9` |
| courses.csv | `3256274788822a3d` | materials.csv | `fc2c9fea0008395c` |
| edges.csv | `51f7bb708520e178` | questions.csv | `639a7c5b116f68c2` |
| enrolled.csv | `1eb0e3d45f40cd44` | same_name_as.csv | `7585b7447b5c2f8b` |
| interactions.csv | `0c9c1f9eabe4eec6` | scope_conflicts.csv | `30c506dcc00b72ca` |
| kts.csv | `153f28c4fe27ebae` | similar_to.csv | `69f2660d3e3989bf` |
| students.csv | `42cefc2baddcdc1a` | units.csv | `4112cec5ffa0c5df` |

## 7. 公开基准不在本仓库内

ASSIST09 / ASSIST15 / ASSIST17 / Statics2011 的原始数据受各自许可约束，**未随仓库分发**。
获取方式与目录约定见 `docs/EXPERIMENTS.md` 的"数据准备"一节；`experiments/tools/download_datasets.py`
提供了从 HuggingFace 镜像批量拉取的脚本（需自行确认各数据集的许可条款）。
