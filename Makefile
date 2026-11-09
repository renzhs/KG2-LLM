# KG²-LLM —— 一键入口
# 用法: make setup | make data | make smoke | make tables | make bench
PY ?= python3
SHELL := /bin/bash

.PHONY: help setup data smoke tables bench downstream clean

help:
	@echo "make setup       安装依赖（pip install -r requirements.txt）"
	@echo "make data        生成 data/processed/*.pkl（自建数据集；公开基准需先下载）"
	@echo "make smoke       DNUI-IKG 上跑 8 epoch 的冒烟训练（CPU 亦可）"
	@echo "make tables      从 results/ 的已存结果重建表 1-4 并与快照 diff（无需 GPU）"
	@echo "make downstream  推荐 / 路径 / 重规划 / 消融（不含 GES）"
	@echo "make bench       完整主表：调参 + 5 折 x 11 模型 + 聚合（单卡约数天）"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) experiments/data/prep_datasets.py --only dnui

smoke:
	bash scripts/01_kt_smoke.sh

tables:
	bash scripts/03_tables.sh

downstream:
	bash scripts/02_downstream.sh

bench:
	bash scripts/04_full_bench.sh

clean:
	rm -rf data/processed logs
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
