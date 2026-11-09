# 复现说明：环境、耗时、随机性与已知差异

## 1. 实测环境

| 项 | 值 |
|---|---|
| OS | Ubuntu（Linux x86_64） |
| Python | 3.12.3 |
| PyTorch | 2.4.1+cu121 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| GPU | 单张 Tesla V100-PCIE-32GB |
| 并行度 | 1（单进程串行；见第 4 节） |

`requirements.txt` 只锁下限（`torch>=2.0`），未锁死补丁版本：本文所有结论都基于
5 折 / 3 种子的**配对统计**，对单次运行 ±0.001 量级的波动不敏感。

## 2. 实测耗时（单卡 V100，仅作量级参考）

| 步骤 | 命令 | 耗时 |
|---|---|---|
| 数据准备（DNUI-IKG） | `bash scripts/00_prepare_data.sh` | **1.5 s** |
| 冒烟（11 模型 × 8 epoch） | `bash scripts/01_kt_smoke.sh` | **171 s** |
| 资源推荐 + CSG 归因 | `bash scripts/02_downstream.sh rec` | **< 60 s** |
| 重建表 1–4 并 diff | `bash scripts/03_tables.sh` | **≈ 10 s** |
| 完整主表（调参 + 5 折 × 11 模型） | `bash scripts/04_full_bench.sh` | 数天量级 |

主表慢在两处：ASSIST12（2.5 M 交互）与 ASSIST15（68 万交互）的**链式调参与基线调参**，
其余数据集单次训练多在分钟级。`run_bench.sh` 支持中断续跑（已完成 stage 与逐折结果会被复用），
建议在 tmux/nohup 中运行。

## 3. 随机性

- `run_kt.py` 在每次训练前设 `torch.manual_seed(seed)`、`np.random.seed(seed)`、`random.seed(seed)`，
  默认 `--seed 42`；自建划分数据集另用 42/43/44 三个种子。
- **未启用** `torch.backends.cudnn.deterministic`：同卡同驱动下可复现到 1e-3 量级，
  跨 GPU 型号/驱动不保证逐位一致。论文的显著性判断基于配对检验，不受单次波动影响。
- 数据切分：公开基准沿用 AKT 官方 5 折 CSV（固定文件，无随机性）；自建划分按用户随机切分，
  随机种子 42 写死在 `prep_datasets.py`。
- 半合成序列：`numpy.random.default_rng(42)`，同一份 CSV + 同一脚本 → 逐位可复现
  （实测 2 147 条序列 / 117 865 次交互 / 正确率 0.6270，与论文表 1 一致）。

## 4. 为什么是单卡串行

单卡并发时显存吃紧会触发 `CUDA error: unknown error`，而 OOM 发生后**运行不会崩**，
只是某些模型没写进结果文件——形成"文件存在、矩阵残缺"的静默失效（实测 GKT 在 22 个
"数据集×折"中有 16 个受影响）。因此：

- `scripts/04_full_bench.sh` 默认 `PAR=1`；
- `run_bench.sh` 内置 `check_bench.py` / `check_basetune.py` / `check_maincfg.py` 三道闸门，
  矩阵不完整时**拒绝聚合出表**。要提速请调大 `PAR`，但不要绕过闸门。

## 5. LLM（GES）的复现

- 调用 `deepseek-chat`，`temperature=0`，带 **磁盘缓存**（`logs/ges_cache/`，键为请求 payload 的哈希）。
- 仓库**不包含** API 密钥；未设置 `DEEPSEEK_API_KEY` 且未命中缓存时，`--task ges` 会写明原因并退出。
- 论文中 GES 的数值由 `results/ges_dnui_syn.json` 存档；重跑可能得到措辞不同但结构相同的
  生成文本，指标（HR/ETR/SCR/PCR）稳定在千分位。
- 若要与论文完全一致又不想调用 API：直接读 `results/ges_dnui_syn.json`（含逐样本输出），
  不需要重跑。

## 6. 快照模式（`--frozen`）

`results/` 是**投稿版快照**，其文件 mtime 由 clone 顺序决定、不含信息。`make_paper_tables.py`
默认会用"多种子产物是否比主结果新"来判断派生结果是否陈旧；在快照上该判据会误报，
因此 `scripts/03_tables.sh` 传 `--frozen` 跳过它。

- 复核论文数值 → 用 `scripts/03_tables.sh`（即 `--frozen`）。
- 重跑实验后建表 → 用 `python experiments/tools/make_paper_tables.py`（不带 `--frozen`），
  让守卫拦截陈旧派生产物。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `缺少 data/processed/xxx.pkl` | 未做数据准备 | `bash scripts/00_prepare_data.sh`（公开基准加 `--public`） |
| `KeyError: 'assist15_f1.pkl'` 类报错 | 公开基准未下载或目录不符 | 见 `docs/EXPERIMENTS.md` 第 0 节的目录约定 |
| 重跑后 `results/` 里的数字变了 | 未加 `--tag`，覆盖了论文报告值 | `git checkout -- results/` 还原；下次加 `--tag repro` |
| `CUDA out of memory` | 并发过高或 batch 过大 | 降 `PAR`、降 `--batch`；补齐缺口后**让闸门通过**再聚合 |
| GES 报"未设置 DEEPSEEK_API_KEY" | 未配置凭证且未命中缓存 | `export DEEPSEEK_API_KEY=...`，或改用存档结果 |
| 表格 diff 全部不一致 | 换行符（CRLF/LF）差异 | 用 `diff --strip-trailing-cr`；快照已统一为 LF |

## 8. 与论文数值可能不一致的三处（按可能性排序）

1. **`--graph-scope` 用错**：KT 类实验用 `train`，推荐/路径用 `cut`，早期口径是 `all`；
   混用会让先修图把测试集信息带进来，指标虚高。
2. **未加载 `--model-cfg-file`**：论文的主表数字是"验证集选出的最优配置"下的结果，
   用默认超参重跑会系统性偏低 0.005~0.02。
3. **折数不齐**：只跑了部分折就聚合，会把某一侧的方差算小（`check_bench.py` 就是为此存在）。
