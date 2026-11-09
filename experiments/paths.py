#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一路径定义。所有脚本通过本模块定位资源，避免各脚本对 ROOT 的基准不一致。

目录约定（本仓库）:
  <REPO>/data/public/          公开数据集（按 docs/DATA_CARD.md 自行下载，不入库）
  <REPO>/data/dnui_ikg/        自建 DNUI-IKG 的脱敏 CSV（随仓库发布）
  <REPO>/data/processed/       统一格式后的 .pkl（由 experiments/data/prep_*.py 生成）
  <REPO>/results/              实验结果 JSON（随仓库发布的是论文报告值）
  <REPO>/figs/                 论文插图
  <REPO>/logs/                 运行日志与 LLM 磁盘缓存（不入库）
"""
import os

EXPROOT = os.path.dirname(os.path.abspath(__file__))   # <REPO>/experiments
REPO = os.path.dirname(EXPROOT)                        # <REPO>

PUB = os.path.join(REPO, "data", "public")
DNUI = os.path.join(REPO, "data", "dnui_ikg")
PROCESSED = os.path.join(REPO, "data", "processed")
KG_CYPHER = os.path.join(EXPROOT, "kg", "cypher")
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "figs")
LOGS = os.path.join(REPO, "logs")

for _p in (PUB, PROCESSED, RESULTS, FIGS, LOGS):
    os.makedirs(_p, exist_ok=True)


def assert_layout(verbose=True):
    ok = True
    checks = [("REPO", REPO, True), ("PUB", PUB, False), ("DNUI", DNUI, True),
              ("PROCESSED", PROCESSED, True), ("RESULTS", RESULTS, True)]
    if verbose:
        for n, p, must in checks:
            e = os.path.exists(p)
            print(f"  {'OK ' if e else ('MISS' if must else 'skip')} {n:<10} {p}")
            if must and not e:
                ok = False
    return ok


if __name__ == "__main__":
    assert_layout()
