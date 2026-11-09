# 实验脚本 ↔ 论文表图 对照

每个脚本都只用命令行参数控制行为，不读硬编码路径；所有路径经 `experiments/paths.py` 解析。
下表"论文位置"的编号以投稿版为准（LaTeX 浮动体可能整体位移，`label` 名稳定）。

## 0. 数据准备

```bash
# (a) 公开基准（4 个），受各自许可约束，需自行下载到 data/public/
python experiments/tools/download_datasets.py          # HF 镜像批量拉取，可只取需要的子集
# 目录约定（AKT 官方 5 折划分）：
#   data/public/AKT_repo/data/{assist2009_pid,assist2015,assist2017_pid,statics2011}/

# (b) 自建数据集：已随仓库发布，无需下载
#   data/dnui_ikg/*.csv

# (c) 统一成 .pkl（data/processed/）
python experiments/data/prep_datasets.py --only dnui          # DNUI-IKG（真实掌握度 + 半合成序列）
python experiments/data/prep_bench.py --only assist09,assist15,assist17,statics11
```

`prep_datasets.py` 产出两个对象：`dnui_real`（(学生, 知识点) 级掌握度，真实标签）与
`dnui_syn`（DNUI-IKG-S，半合成序列，论文表 1 的 DNUI-IKG 行即此）。

## 1. 表图对照

| 论文位置 | 内容 | 脚本 | 关键产物 |
|---|---|---|---|
| 表 1 `tab:dataset` | 7 个数据集的规模与序列特征 | `tools/dataset_stats.py`、`tools/dataset_edges.py`、`tools/make_paper_tables.py` | `results/table_dataset.tex` |
| 表 2 `tab:kt-pub` | 本文复现值 vs 两套公开报告值（汇总） | `tools/make_paper_tables.py`（读 `experiments/data/published_benchmarks.json`） | `results/table_kt_pub.tex` |
| 表 3 `tab:kt-bench` | **11 个模型 × 7 个数据集**的 AUC 主对比 | `run/run_bench.sh` → `tools/aggregate_bench.py` → `tools/make_paper_tables.py` | `results/kt_bench_*.json`、`results/table_kt_bench.tex` |
| 表 4 `tab:kt-ours` | 本文方法 vs 各数据集最强基线的**配对检验**（ΔAUC/95%CI/p/MDES） | `tools/kt_significance.py` | `results/kt_signif.json` |
| 表 5 | 主要超参数设置 | `results/kt_model_cfg_*.json`（验证集选出的最终配置） | — |
| 表 6 `tab:enc` | 序列通路编码器对照（Transformer/GRU/LSTM × 层数） | `run/run_enc_ablation.py` → `tools/make_enc_table.py` | `results/kt_*_encabl.json` |
| 表 7–9 | 学习资源推荐：$s$NDCG@10、CSCR@10、各策略对照 | `run/run_rec.py --task rec`、`tools/csg_attribution.py` | `results/rec_*.json` |
| 表 10 | 学习路径推荐与动态重规划 | `run/run_rec.py --task path`、`--task mdrl` | `results/path_*.json`、`results/mdrl_*.json` |
| 表 11 | 检索增强生成（GES）可靠性：HR/ETR/SCR/PCR | `run/run_rec.py --task ges`（需 LLM 凭证或缓存） | `results/ges_*.json` |
| 表 12–13 | 模块消融与稀疏/稠密知识点分组评测 | `run/run_abl_multi.py`、`run/run_sparse_eval.py` | `results/kt_*_abl.json`、`kt_*_sparse.json` |
| 表 14 `tab:graph-rel` | 先修图可靠性、归纳范围与去泄漏审计 | `tools/prereq_reliability.py`、`tools/graph_leak_audit.py`、`tools/kt_gstrain_pair.py` | `results/prereq_induction.json`、`graph_leak.json` |
| 图 1 | 框架总体结构（示意图，脚本重绘需 matplotlib） | `figs/make_figures.py::fig_framework` | `figs/fig_framework.pdf` |
| 图 2–11 | 数据规模、AUC 误差棒、逐种子散点、CSCR、γ 敏感性、GES、MDRL、消融 | `figs/make_figures.py` | `figs/*.pdf` |

## 2. 三条复现路径（按代价从低到高）

### 路径 A：只复核论文数字（**无需 GPU，约 1 秒**）

仓库已附带全部逐折/逐种子结果 JSON，因此可以直接重建论文的每一张表：

```bash
bash   scripts/03_tables.sh                          # 一步到位：重建表 1-4 + 与快照 diff

# 等价的手工步骤（注意 --frozen：results/ 是投稿版快照，mtime 无意义）
python experiments/tools/make_paper_tables.py --frozen   # 表 1/2/3/4
python experiments/tools/kt_significance.py              # 表 4 的配对检验（写 kt_signif.json）
python experiments/tools/dataset_stats.py                # 表 1 的数据来源
```

产物写入 `results/table_*.tex`，可与 `results/paper_tables/`（投稿版快照）逐行 `diff` 核对。

### 路径 B：重跑单个数据集（**单卡 10 分钟 ~ 数小时**）

```bash
# 冒烟：DNUI-IKG，全部 11 个模型，8 epoch
python experiments/run/run_kt.py --dataset dnui_syn --models all --epochs 8 --tag smoke

# 正式：单数据集全模型 + 复用验证集选出的配置
python experiments/run/run_kt.py --dataset assist09_akt_f1 --models all --epochs 60 \
       --patience 10 --tag bench --model-cfg-file results/kt_model_cfg_assist09_akt_f1.json
```

`--use-tune` 会从 `results/kt_tune_<ds>_stage*.json` 载入**验证集**最优配置；
`--model-cfg-file` 载入已合并的最终配置（含基线）。测试集不参与任何选择。

### 路径 C：从零复现整张主表（**单卡 V100 约数天**）

```bash
bash experiments/run/run_bench.sh      # 阶段A 本文方法链式调参 → 阶段A2 基线调参 → 阶段B 5折×11模型 → 阶段C 聚合
```

该脚本内含**四道门禁**，是本文"统一实现、公平调参"结论的证据链，建议保留原样：

| 门禁 | 脚本 | 拦住什么 |
|---|---|---|
| 基线调参完整性 | `tools/check_basetune.py` | 只搜了 lr、容量阶段没跑满却被判"已完成"（会静默形成不公平比较） |
| 本文配置内容级校验 | `tools/check_maincfg.py` | 链式调参中途失败、`kt_maincfg` 是上一轮遗留的旧文件 |
| 对比矩阵完整性 | `tools/check_bench.py` | 某折因 CUDA OOM 只写入部分模型，聚合时被当成"方差极小" |
| 学习率端点审计 | `tools/kt_endpoint_audit.py`、`tools/check_lr_boundary.py` | 最优点落在网格端点 → 搜索其实没做完 |

## 3. 下游任务与消融

```bash
# 推荐（DNUI-IKG；--graph-scope all=主口径，cut=去泄漏口径，产物加 _gscut 后缀）
python experiments/run/run_rec.py --dataset dnui_syn --task rec  --graph-scope all
python experiments/run/run_rec.py --dataset assist09 --task rec  --graph-scope cut --tag gscut

# 路径 / 重规划
python experiments/run/run_rec.py --dataset dnui_syn --task path
python experiments/run/run_rec.py --dataset dnui_syn --task mdrl

# 生成（需 DEEPSEEK_API_KEY，或命中 logs/ges_cache 的磁盘缓存）
python experiments/run/run_rec.py --dataset dnui_syn --task ges --llm-students 40

# 模块消融（3 种子）/ 稀疏分组评测 / 编码器对照
python experiments/run/run_abl_multi.py    --dataset dnui_syn --seeds 42,43,44
python experiments/run/run_sparse_eval.py  --dataset dnui_syn --seeds 42,43,44
python experiments/run/run_enc_ablation.py --datasets dnui_syn,assist09
```

> `run_rec.py` / `run_kt.py` 默认把结果写进 `results/` 下与论文同名的文件。
> **重跑请加 `--tag`**，否则会覆盖仓库内保存的论文报告值。

## 4. 双图谱构建（可选，需 MySQL/Neo4j）

```bash
# 从教学平台库导出并规范化成 data/dnui_ikg/*.csv（需要可读的平台数据库）
python experiments/kg/build_dual_graph.py
# 装载进 Neo4j（容器内 /var/lib/neo4j/import 指向 data/dnui_ikg）
python experiments/kg/load_neo4j.py
```

这两步只用于重建知识图谱与提供可视化，**不影响**上面任何数值结果——所有实验读的都是 CSV。
