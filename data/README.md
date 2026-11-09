# data/

```
data/
├── dnui_ikg/     自建数据集 DNUI-IKG（脱敏 CSV，随仓库发布，CC BY-NC 4.0）
├── public/       公开基准（ASSIST09/15/17、Statics2011），受各自许可约束，不入库
└── processed/    统一格式后的 .pkl（由 experiments/data/prep_*.py 生成，不入库）
```

- **DNUI-IKG 的字段、隐私处理、半合成口径与校验和**：见 [`../docs/DATA_CARD.md`](../docs/DATA_CARD.md)
- **公开基准的获取方式**：见 [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) 第 0 节

快速生成 `.pkl`：

```bash
python experiments/data/prep_datasets.py --only dnui     # 自建数据集（离线可用）
python experiments/data/prep_bench.py --only assist09,assist15,assist17,statics11   # 需先下载公开数据
```
