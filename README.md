# KG²-LLM：双知识图谱与大语言模型协同的个性化学习跟踪与资源推荐

本仓库是论文《融合双知识图谱与大语言模型的学习跟踪与资源推荐》的**配套代码与自建数据集**，
包含：11 种知识追踪模型的统一实现与公平调参管线、学习资源推荐 / 学习路径推荐 / 图谱证据子图
约束生成（GES）/ 掌握度驱动重规划（MDRL）的可执行实验、先修图去泄漏审计，以及论文全部
表图的生成脚本。

> 论文状态：投稿中（《计算机工程与应用》）。若本文被接收，将在此处补全卷期页码与 DOI。
>
> **仓库地址**：<https://github.com/renzhs/KG2-LLM> ·
> `git clone git@github.com:renzhs/KG2-LLM.git`

## 1. 一分钟了解

- **问题**：跨学段、跨专业的语义匹配会让推荐内容"串味"（跨作用域语义泄漏）。
- **方法**：课程知识图谱 CKG 与学习者认知图谱 LKG 双向耦合为底座，向上三条约束通路——
  学段—专业语义门控 CSG、图谱证据子图约束的 RAG（GES）、掌握度阈值触发的重规划（MDRL）。
- **主结论**：在图谱**不提升**知识追踪预测精度的前提下（消融贡献落在噪声量级），
  图谱对下游任务的**约束满足**有稳定而显著的收益：串味率 0.3604 → 0.0000，
  生成侧作用域合规率 0.3600 → 1.0000，重规划调用次数 −85.7%。
- **规模**：7 个数据集（4 个公开 5 折基准 + 3 个自建划分）× 11 种知识追踪方法，
  主结果以配对检验判定显著性。

## 2. 仓库结构

```
KG2-LLM/
├── data/
│   ├── dnui_ikg/            自建数据集 DNUI-IKG（脱敏 CSV，随仓库发布）
│   └── public/              公开基准（自行下载，不入库）
├── experiments/
│   ├── paths.py             统一的路径解析
│   ├── models/              kt_models.py（11 个 KT 模型）、rec_models.py、kg_induct.py、llm_ges.py
│   ├── data/                prep_datasets.py（统一成 .pkl）、prep_bench.py（AKT 5 折切分）
│   ├── run/                 run_kt.py、run_rec.py、run_abl_multi.py、run_bench.sh …
│   ├── tools/               聚合、配对检验、去泄漏审计、调参门禁、建表
│   ├── kg/                  双图谱构建与 Neo4j 装载（可选）
│   └── figs/                make_figures.py（论文插图）
├── results/                 论文报告值（JSON）+ 投稿版表格快照（paper_tables/）
├── figs/                    论文插图（PNG/PDF）
├── scripts/                 按阶段封装的一键脚本
└── docs/
    ├── DATA_CARD.md         数据集说明（来源、字段、隐私、半合成口径、校验和）
    ├── EXPERIMENTS.md       脚本 ↔ 论文表图 对照 + 三条复现路径
    ├── REPRODUCIBILITY.md   环境、种子、耗时、已知差异
    └── CODE_AVAILABILITY.md 论文里标注代码地址的表述模板（中英/匿名评审）
```

## 3. 安装

```bash
git clone <this-repo> && cd KG2-LLM
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Python ≥ 3.10（在 3.12.3 上验证）
- PyTorch ≥ 2.0（GPU 可选：CPU 也能跑通 DNUI-IKG 的全部流程，只是慢）
- 生成 GES 结果需 `DEEPSEEK_API_KEY`（见 `docs/REPRODUCIBILITY.md`）；复核已报告数值不需要

## 4. 快速开始（无需 GPU，约 10 秒）

仓库附带全部逐折/逐种子结果，因此**不训练也能重建论文的每一张表**，并与投稿版快照逐行比对：

```bash
bash scripts/03_tables.sh
# [一致]   table_dataset.tex
# [一致]   table_kt_pub.tex
# [一致]   table_kt_bench.tex
# [一致]   table_kt_ours.tex
# ===== 全部一致：仓库内保存的逐折结果可以重建论文表格 =====
```

跑通一次训练（DNUI-IKG × 11 个模型 × 8 epoch，单张 V100 实测 **171 s**；CPU 亦可在数分钟内完成）：

```bash
python experiments/data/prep_datasets.py --only dnui
python experiments/run/run_kt.py --dataset dnui_syn --models all --epochs 8 --tag smoke
```

## 5. 完整复现

见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)，按代价分三级：

| 级别 | 做什么 | 代价 |
|---|---|---|
| A | 从已存结果重建全部表格、复核显著性 | 无需 GPU，约 10 秒（实测） |
| B | 重跑单个数据集上 11 个模型 | 单卡 V100：DNUI-IKG 上 8 epoch 实测 171 s；60 epoch 视数据集规模约 20 分钟 ~ 数小时 |
| C | 从零复现主表（调参 + 5 折 × 11 模型 + 聚合） | 单卡 V100 数天量级，可断点续跑 |

`experiments/run/run_bench.sh` 内置四道**门禁**（基线调参完整性、本人配置内容级校验、
对比矩阵完整性、学习率端点审计），目的是防止"某一侧静默偏弱"造成的不公平比较——
它们本身也是论文"统一实现、公平调参"这一主张的证据，建议保留原样。

## 6. 自建数据集 DNUI-IKG

- **内容**：3 门本科课程、66 个知识点、88 条先修边、467 道题、1 060 名有作答记录的学生、
  169 351 条作答记录、17 862 条掌握度标签；**唯一带学段—专业标注**的数据集。
- **隐私**：已删除学号、姓名、院系与专业代码四列，保留的 `student_id` 为平台内部整数编号，
  无法与任何校内记录直接关联；全表做过手机号/邮箱/学号模式扫描，零命中。
- **半合成声明（务必阅读）**：平台日志"答对为止"，正确率 99.35%、掌握度 92.3% 恒为 1，
  缺乏方差。因此论文中用于知识追踪的 **DNUI-IKG-S** 在**真实先修 DAG + 真实作用域标签 +
  真实交互量分布**之上，用 2PL IRT + slip/guess 生成答题序列（`prep_datasets.py` 可完整重建，
  seed=42）。**图结构与标签是真实的，序列是半合成的**——它支撑"真实课程结构与真实作用域约束下"
  的结论，不能用于推断真实学生的学习增益。
- 细节、字段字典、校验和见 [`docs/DATA_CARD.md`](docs/DATA_CARD.md)。
- 公开基准（ASSIST09/15/17、Statics2011）受各自许可约束**未随仓库分发**，获取方式见
  `docs/EXPERIMENTS.md` 第 0 节。

## 7. 复现口径与已知差异

1. **随机性**：`torch.manual_seed` 等已固定（默认 seed=42），但未启用 `cudnn.deterministic`，
   跨硬件不保证逐位一致；单次 AUC 波动约 ±0.001。论文的显著性判断基于 5 折 / 3 种子的
   **配对检验**，对单次波动不敏感。
2. **单卡串行**：所有结果在单张 Tesla V100-PCIE-32GB 上以单进程串行跑出（并行会触发 CUDA OOM，
   见 `run_bench.sh` 中 GKT 的教训）；多卡并行不改变数值，但需要自行调 `batch`。
3. **LLM 非确定性**：GES 以 `temperature=0` 调用 DeepSeek，并带磁盘缓存；未命中缓存时重跑
   可能得到措辞不同但结构相同的生成结果，指标（HR/ETR/SCR/PCR）稳定。
4. **论文数值与重跑差异**：若重跑结果与论文相差超过上述波动范围，请先检查是否误用了
   `--graph-scope`（KT 用 `train`，推荐/路径用 `cut`）或未加载 `--model-cfg-file`。

## 8. 引用

```bibtex
@article{ren2026kg2llm,
  title   = {融合双知识图谱与大语言模型的学习跟踪与资源推荐},
  author  = {任志森 and 邵欣欣 and 任政},
  journal = {计算机工程与应用},
  year    = {2026},
  note    = {投稿中}
}
```

仓库元数据见 [`CITATION.cff`](CITATION.cff)（GitHub 会显示 "Cite this repository"）。

## 9. 许可

- **代码**：MIT，见 [`LICENSE`](LICENSE)
- **DNUI-IKG 数据**：CC BY-NC 4.0，见 [`LICENSE-DATA`](LICENSE-DATA)（仅限非商业研究，须署名）

## 10. 联系方式

任志森，大连东软信息学院 软件学院，辽宁 大连 116023
邮箱：renzhisen@neusoft.edu.cn ／ renzhisen@126.com

问题、复现困难或数据使用申请请开 Issue，或直接邮件联系。

---

## English (short)

**KG²-LLM** couples a course knowledge graph and a learner cognitive graph with an LLM for
personalized learning tracking and resource recommendation. This repository contains the
reference implementation of 11 knowledge-tracing baselines under a unified, fairly-tuned
protocol, the downstream experiments (resource recommendation, learning-path planning,
graph-evidence-subgraph constrained RAG, mastery-triggered replanning), the prerequisite-graph
de-leakage audit, and the self-built **DNUI-IKG** dataset.

Headline finding: the graph modules bring **no stable gain in knowledge-tracing accuracy**, yet
they impose **verifiable structural constraints** on downstream tasks (contamination ratio
0.3604 → 0.0000; generation scope-compliance 0.3600 → 1.0000; replanning calls −85.7%).

Quick check without a GPU: `bash scripts/03_tables.sh` rebuilds Tables 1–4 from the shipped
result files and diffs them against the camera-ready snapshot in `results/paper_tables/`.

**Data note**: DNUI-IKG-S (the sequence benchmark used for knowledge tracing) is *semi-synthetic* —
real curriculum DAG, real grade/major scope labels and real interaction volumes, with answer
sequences generated by a 2PL IRT + slip/guess model (see `docs/DATA_CARD.md` §5). It must not be
used to infer real students' learning gains.
