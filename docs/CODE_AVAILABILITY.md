# 论文中标注代码与数据地址（可直接粘贴）

> 仓库地址（已上线）：<https://github.com/renzhs/KG2-LLM>
> 克隆用 SSH：`git@github.com:renzhs/KG2-LLM.git`；克隆用 HTTPS：`https://github.com/renzhs/KG2-LLM.git`
> 下文表述已按该地址定稿，可直接粘贴到论文与投稿系统。

## 1. 中文表述（正式发表版）

**完整版**（建议放 §3.1 实验设置的第一段，或引言贡献列表之后）：

> 本文的全部实验代码、自建数据集 DNUI-IKG、各数据集的最终超参配置与逐折结果文件均已开源，
> 地址为 https://github.com/renzhs/KG2-LLM。仓库提供 11 种知识追踪方法在统一实现、
> 统一划分与统一调参预算下的完整管线，以及无需 GPU、约 10 s 即可重建本文表 1～表 4 并逐行
> 比对的复现脚本。

**精简版**（放脚注、致谢或"数据可用性"栏）：

> 代码与数据：https://github.com/renzhs/KG2-LLM

**只提数据**（若期刊把代码与数据分栏）：

> 自建数据集 DNUI-IKG（3 门课程、66 个知识点、88 条先修边、169 351 条作答记录）
> 已在去除学号与姓名后公开：https://github.com/renzhs/KG2-LLM/tree/main/data/dnui_ikg

## 2. 英文表述

**Full**:

> The code, the self-built DNUI-IKG dataset, all tuned hyper-parameter configurations and the
> per-fold result files are publicly available at https://github.com/renzhs/KG2-LLM.
> The repository contains the complete pipeline that implements and fairly tunes 11
> knowledge-tracing methods under identical data splits and tuning budgets, together with a
> GPU-free script that rebuilds Tables 1–4 of this paper in about 10 seconds and diffs them
> against the archived camera-ready values.

**Short**:

> Code and data: https://github.com/renzhs/KG2-LLM

## 3. 放置位置建议（《计算机工程与应用》口径）

| 优先级 | 位置 | 理由 |
|---|---|---|
| ① | §3.1「数据集与实验设置」首段末尾 | 审稿人读实验时第一眼能看到，且与"可复现性"直接相关 |
| ② | 引言最后的贡献列表之后（单独一句） | 与四点贡献并列，强调"统一基准"本身是贡献 |
| ③ | 首页脚注（作者简介附近） | 版面紧张时的备选，但可检索性差 |

本刊为中文单盲评审，通常可以直接给出公开仓库地址。若编辑部要求匿名，见第 4 节。

## 4. 匿名评审期的写法

若必须匿名，二选一：

- **匿名仓库**：把仓库导入 https://anonymous.4open.science ，得到形如
  `https://anonymous.4open.science/r/KG2-LLM-7C3A` 的链接，正文写
  "代码与数据见匿名仓库：\<链接\>（评审期有效，录用后转为公开仓库）"。
- **承诺公开**：正文写"代码与自建数据集将在论文录用后于 GitHub 公开"，并在投稿信中说明
  可应编辑要求提供。**不建议**只写"可向作者索取"——近三年该写法在多数期刊会被要求改为公开链接。

## 5. LaTeX 插入片段

```latex
% 放在 \subsubsection{数据集} 之后、或 \subsection{基准协议} 开头
本文的全部实验代码、自建数据集 DNUI-IKG、各数据集的最终超参配置与逐折结果文件
均已开源\footnote{\url{https://github.com/renzhs/KG2-LLM}}，
其中包含无需 GPU、约 10\,s 即可重建本文表~\ref{tab:dataset}～表~\ref{tab:kt-ours}
并逐行比对的复现脚本。
```

```latex
% 若期刊有独立的"数据可用性声明"小节
\section*{数据可用性声明}
本文代码与自建数据集 DNUI-IKG 公开于 \url{https://github.com/renzhs/KG2-LLM}（MIT / CC BY-NC 4.0）。
ASSISTments 与 Statics2011 的原始数据受各自许可约束，未随仓库分发，获取方式见仓库文档。
```

## 6. 上传与后续维护

1. **建仓库**（已完成）：GitHub 上的 `renzhs/KG2-LLM` 为 Public，未勾选自动生成 README/LICENSE。
2. **推送**（已完成）：本地已 `git init`、提交并推送 `main` 分支。后续更新流程：

   ```bash
   git add -A && git commit -m "..." && git push
   ```

   在新机器重配远端时：

   ```bash
   git remote add origin git@github.com:renzhs/KG2-LLM.git
   git branch -M main
   git push -u origin main
   ```
3. **补元数据**：仓库 About 里填一句话简介，Topics 建议
   `knowledge-graph` `knowledge-tracing` `llm` `education` `recommender-system` `rag`；
   有 DOI 需求可在 Zenodo 打 tag 归档（`CITATION.cff` 已就绪，Zenodo 会读取）。

> 提醒：**不要**把 `logs/`、`data/public/`、`data/processed/` 提交上去（`.gitignore` 已覆盖）；
> 也**不要**把任何 `DEEPSEEK_API_KEY` 写进仓库——`models/llm_ges.py` 已改为只从环境变量读取。
