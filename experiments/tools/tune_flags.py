#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从超参搜索结果里提取最优配置，输出可直接拼到 run_kt.py 后面的命令行参数。

用法:
  python experiments/tools/tune_flags.py --dataset assist15_f1 --stage 4
  -> --seq-enc gru --layers 1 --graph-in（视选中配置而定）
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import RESULTS                        # noqa: E402

# 超参名 -> run_kt.py 的 CLI 旗标
FLAG = {
    "seq_enc": "--seq-enc", "layers": "--layers", "d": "--d", "dropout": "--dropout",
    "lr": "--lr", "steps": "--steps", "hops": "--hops", "topm": "--topm",
    "wprior": "--wprior",
}
BOOL_FLAG = {"gated_fuse": "--gated-fuse", "graph_in": "--graph-in"}


def load_best(ds, stage=None, stages=(5, 4, 3, 2, 1)):
    """按优先级读取搜索结果；stage 为 None 时依次尝试 stages。"""
    cand = [stage] if stage else list(stages)
    for st in cand:
        p = os.path.join(RESULTS, f"kt_tune_{ds}_stage{st}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("best"):
                keys = list(d["rows"][0].keys()) if d.get("rows") else []
                cfg = {k: d["best"][k] for k in keys if k in FLAG or k in BOOL_FLAG}
                return cfg, st, d["best"]
    return None, None, None


def to_flags(cfg):
    out = []
    for k, v in cfg.items():
        if k in BOOL_FLAG:
            if v:
                out.append(BOOL_FLAG[k])
        elif k in FLAG:
            out.append(f"{FLAG[k]} {v}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg, st, best = load_best(a.dataset, a.stage or None)
    if cfg is None:
        if not a.quiet:
            print(f"[tune_flags] 未找到 {a.dataset} 的搜索结果，返回空参数",
                  file=sys.stderr)
        print("")
        return
    if not a.quiet:
        val = best.get("val")
        print(f"[tune_flags] {a.dataset} 采用 stage{st} 最优配置 {cfg} "
              f"val={val:.4f}", file=sys.stderr)
    print(to_flags(cfg))


if __name__ == "__main__":
    main()
