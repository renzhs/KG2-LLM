#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多种子消融：DGCL-KT 各组件的贡献必须以 mean±std + 配对比较呈现。

背景
----
单种子消融显示各组件的 ΔAUC 只有 ±0.002，而 assist09 上 DGCL-KT 的逐种子标准差
本身就达 0.0014——**Δ 与噪声同量级，单种子结果不足以支撑任何结论**。
因此本脚本对每个变体跑 3 个随机种子，报告 mean±std，并相对"完整模型"做
按种子配对的比较（胜出种子数 + 平均 Δ）。

用法
----
  python experiments/run/run_abl_multi.py --dataset dnui_syn --seeds 42,43,44
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
sys.path.insert(0, HERE)
from paths import PROCESSED as PROC, RESULTS as RES                 # noqa: E402
from run_kt import (train_one, build_graph_arrays, build_scope_mask,  # noqa: E402
                    apply_tuned)

# (显示名, 消融开关) —— None 表示完整模型
VARIANTS = [
    ("完整模型", {}),
    ("w/o 图编码", {"ab_no_graph": True}),
    ("w/o 反向标定", {"ab_no_calib": True}),
    ("仅结构先验", {"ab_struct_only": True}),
    ("w/o CSG 门控", {"ab_no_scope": True}),
]


def final_cfg_path(ds):
    """阶段B（正式主表）所用的同一份模型配置。"""
    return os.path.join(RES, f"kt_model_cfg_{ds}.json")


def load_final_cfg(ds):
    """读阶段B 正式评测用的那份配置并返回 DGCL-KT 的一段（缺失返回 None）。

    为什么必须与主表同源
    --------------------
    阶段B 用 `run_kt.py --model-cfg-file results/kt_model_cfg_<ds>.json`，
    而本脚本早前走 `--use-tune`（读 kt_tune_<ds>_stage*.json）——**两个不同的配置源**。
    只要其中一处被更新（例：把序列编码器纳入验证集搜索），主表会重跑而消融**不会**，
    于是表里的「完整模型」悄悄偏离主表的 DGCL-KT。实测 ASSIST09 偏离达 0.0174，
    是本表所要测量的消融效应量（~0.0008）的 20 倍 —— 整张表的结论随之失效，
    而表面上看只是一次"正常"的运行。现改为直接复用主表的配置文件。
    """
    p = final_cfg_path(ds)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, OSError):
        return None
    return (d.get("DGCLKT") or d.get("DGCL-KT") or None)


def make_args(ds, epochs, batch, seed, ab, cfg_override=None):
    a = argparse.Namespace()
    a.dataset = ds
    a.models = ["DGCLKT"]
    a.epochs = epochs
    a.batch = batch
    a.lr = 2e-3
    a.patience = 5
    a.max_len = 200
    a.min_len = 3
    a.max_users = 0
    a.seed = seed
    a.tag = "abl"
    a.use_tune = True                       # 兜底：配置文件缺失时回退到搜索最优
    a.ablation = bool(ab)
    a.ab_no_graph = False
    a.ab_no_calib = False
    a.ab_no_scope = False
    a.ab_struct_only = False
    a = apply_tuned(a)
    cfg = cfg_override if cfg_override is not None else load_final_cfg(ds)
    if cfg:
        for k, v in cfg.items():            # 与阶段B 完全同源
            setattr(a, k, v)
    for k, v in ab.items():
        setattr(a, k, v)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()

    with open(os.path.join(PROC, f"{a.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    adj, acalib = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)
    seeds = [int(s) for s in a.seeds.split(",")]

    per = {}
    t00 = time.time()
    for name, ab in VARIANTS:
        per[name] = []
        for sd in seeds:
            t0 = time.time()
            try:
                r = train_one("DGCLKT", make_args(a.dataset, a.epochs, a.batch, sd, ab),
                              obj, adj, acalib, scope, s2i, sd)
                per[name].append(float(r["AUC"]))
                print(f"  {name:<14} seed={sd}  AUC={r['AUC']:.4f} ({time.time()-t0:.0f}s)")
            except Exception as e:                                  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  {name:<14} seed={sd}  FAILED: {e}")

    full = np.array(per["完整模型"], dtype=np.float64)
    cfg = load_final_cfg(a.dataset)
    out = dict(dataset=a.dataset, seeds=seeds, epochs=a.epochs,
               cfg_source=(final_cfg_path(a.dataset) if cfg else "kt_tune_<ds>_stage*.json"),
               summary={}, paired_vs_full={})
    for name, _ in VARIANTS:
        v = np.array(per[name], dtype=np.float64)
        out["summary"][name] = dict(
            mean=round(float(v.mean()), 4) if v.size else None,
            std=round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
            per_seed=[round(float(x), 4) for x in v])
        if name == "完整模型" or not v.size:
            continue
        n = min(len(full), len(v))
        d = v[:n] - full[:n]
        out["paired_vs_full"][name] = dict(
            n=n, mean_delta=round(float(d.mean()), 4),
            std_delta=round(float(d.std(ddof=1)), 4) if n > 1 else 0.0,
            better_seeds=int((d > 0).sum()), worse_seeds=int((d < 0).sum()))

    p = os.path.join(RES, f"kt_{a.dataset}_abl.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n===== {a.dataset} 多种子消融（{len(seeds)} seeds）=====")
    print(f"{'变体':<14}{'AUC(mean±std)':<20}{'Δ vs 完整':<12}{'更好/更差种子'}")
    for name, _ in VARIANTS:
        s = out["summary"][name]
        if s["mean"] is None:
            print(f"{name:<14}{'失败':<20}"); continue
        if name in out["paired_vs_full"]:
            pv = out["paired_vs_full"][name]
            print(f"{name:<14}{s['mean']:.4f}±{s['std']:<12.4f}"
                  f"{pv['mean_delta']:+.4f}    {pv['better_seeds']}/{pv['worse_seeds']}")
        else:
            print(f"{name:<14}{s['mean']:.4f}±{s['std']:<12.4f}{'—':<12}{'—'}")
    print("SAVED", p, f"({time.time()-t00:.0f}s)")

    # ---- 操作点自检：本表的「完整模型」必须等于主表的 DGCL-KT ----
    # 不等就说明配置不同源（例如主表已换用新搜索出的编码器，而本表仍在旧配置上跑），
    # 此时表内那些 ~0.001 级的 Δ 全部不可比。宁可显著失败，也不静默出表。
    bp = os.path.join(RES, f"kt_bench_{a.dataset}.json")
    if os.path.exists(bp) and out["summary"]["完整模型"]["mean"] is not None:
        try:
            with open(bp, encoding="utf-8") as f:
                main = (((json.load(f).get("summary") or {}).get("DGCLKT") or {})
                        .get("AUC") or {}).get("mean")
        except (ValueError, OSError):
            main = None
        if main is not None:
            dv = abs(out["summary"]["完整模型"]["mean"] - main)
            tag = "OK" if dv <= 0.005 else "!! 操作点漂移"
            print(f"[op-check] 完整模型={out['summary']['完整模型']['mean']:.4f}  "
                  f"主表 DGCL-KT={main:.4f}  |Δ|={dv:.4f}  {tag}")
            print(f"[op-check] 配置来源: {out['cfg_source']}")
            if dv > 0.005:
                print("[op-check] 该结果**不可**直接进论文：请确认 cfg 已同步后重跑。")


if __name__ == "__main__":
    main()
