#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏知识点分组评测：回答"图谱证据到底帮在哪里"。

动机
----
多种子消融显示，各图谱组件对**整体** AUC 的贡献只有 ±0.0003，
低于逐种子标准差——若只看总指标，会得出"图谱没用"的错误结论。
但图谱先验的作用本应集中在「交互稀疏」的知识点上：
高频技能的作答序列已足够，稀疏技能才需要借助先修结构共享统计强度。
本脚本按训练集中知识点的出现频次把测试交互分为
「稀疏组」（频次最低的 30%）与「稠密组」，分别计算 AUC。

用法
----
  python experiments/run/run_sparse_eval.py --dataset dnui_syn --seeds 42,43,44
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
sys.path.insert(0, HERE)
from paths import PROCESSED as PROC, RESULTS as RES                  # noqa: E402
from run_kt import (KTData, collate, forward_model, train_one,       # noqa: E402
                    build_graph_arrays, build_scope_mask, apply_tuned,
                    metrics, DEV)

VARIANTS = [
    ("完整模型", {}),
    ("w/o 图编码", {"ab_no_graph": True}),
    ("w/o 反向标定", {"ab_no_calib": True}),
    ("仅结构先验", {"ab_struct_only": True}),
    ("w/o CSG 门控", {"ab_no_scope": True}),
]
Q_SPARSE = 0.30        # 频次最低的 30% 知识点记为稀疏组
MAX_LEN, MIN_LEN = 200, 3   # 与主实验 run_kt.py 默认保持一致


def final_cfg_path(ds):
    """阶段B（正式主表）所用的同一份模型配置。"""
    return os.path.join(RES, f"kt_model_cfg_{ds}.json")


def load_final_cfg(ds):
    """读阶段B 正式评测用的那份配置并返回 DGCL-KT 的一段（缺失返回 None）。

    本脚本早前走 `--use-tune`（kt_tune_<ds>_stage*.json），而阶段B 用
    `--model-cfg-file kt_model_cfg_<ds>.json` —— **两个不同的配置源**。
    配置一旦更新，主表重跑而本表不重跑，「完整模型」就会悄悄偏离主表的 DGCL-KT
    （实测 ASSIST09 偏 0.0174），表内 ~0.001 级的 Δ 随之全部不可比。现改为同源。
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


def make_args(ds, epochs, batch, seed, ab):
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
    a.tag = "sparse"
    a.use_tune = True
    a.return_model = True
    a.ablation = bool(ab)
    a.ab_no_graph = a.ab_no_calib = a.ab_no_scope = a.ab_struct_only = False
    a = apply_tuned(a)
    cfg = load_final_cfg(ds)
    if cfg:
        for k, v in cfg.items():            # 与阶段B 完全同源
            setattr(a, k, v)
    for k, v in ab.items():
        setattr(a, k, v)
    return a


def collect(model, loader, K):
    r"""收集 (label, pred, skill) 三元组，供分组计算 AUC。

    必须与 run_kt.py 的 loader 契约一致：`collate` 产出
    `(sk, ir, tg, qq, mask, sid)` **六元组**，且 `forward_model` 要显式接收题目 id
    `qq`（AKT 的 Rasch 难度、SAINT/SimpleKT 的题目嵌入依赖它）。
    早前此处按**五元组**解包且漏传 `qq`，一进循环就
    `ValueError: too many values to unpack (expected 5)` —— 稀疏表因此整表跑不出来。
    """
    model.eval()
    ys, ps, ks = [], [], []
    with torch.no_grad():
        for sk, ir, tg, qq, mask, sid in loader:
            sk_d = sk.to(DEV); ir_d = ir.to(DEV); tg_d = tg.to(DEV)
            qq_d = qq.to(DEV); mask_d = mask.to(DEV); sid_d = sid.to(DEV)
            out = forward_model(model, sk_d, ir_d, qq_d, sid_d, K)
            yh = out.gather(2, sk_d.unsqueeze(-1)).squeeze(-1)
            ys.append(tg_d[mask_d].cpu().numpy())
            ps.append(yh[mask_d].cpu().numpy())
            ks.append(sk_d[mask_d].cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(ks))


def auc_of(y, p):
    if y.size < 10 or len(np.unique(y)) < 2:
        return None
    return float(metrics(y, p)["AUC"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()

    with open(os.path.join(PROC, f"{a.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    K = obj["n_skill"]
    adj, acalib = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)

    # ---- 按训练集频次划分知识点 ----
    cnt = np.zeros(K, dtype=np.int64)
    for s in obj["train"]:
        for k in s["skill"]:
            cnt[int(k)] += 1
    nz = cnt[cnt > 0]
    thr = float(np.percentile(nz, Q_SPARSE * 100)) if nz.size else 0.0
    sparse_skills = set(int(i) for i in np.where((cnt > 0) & (cnt <= thr))[0])
    print(f"[{a.dataset}] K={K} 稀疏阈值(30%分位)={thr:.0f} 次，"
          f"稀疏知识点 {len(sparse_skills)} 个")

    te = KTData(obj["test"], max_len=MAX_LEN, min_len=MIN_LEN, scope=scope, s2i=s2i)
    tl = DataLoader(te, batch_size=a.batch * 2, shuffle=False,
                    collate_fn=collate, num_workers=2)

    seeds = [int(s) for s in a.seeds.split(",")]
    out = dict(dataset=a.dataset, seeds=seeds, q_sparse=Q_SPARSE,
               n_sparse_skills=len(sparse_skills), sparse_thr=thr, groups={})
    t00 = time.time()
    for name, ab in VARIANTS:
        acc = {"all": [], "sparse": [], "dense": []}
        for sd in seeds:
            args = make_args(a.dataset, a.epochs, a.batch, sd, ab)
            r = train_one("DGCLKT", args, obj, adj, acalib, scope, s2i, sd)
            model = r["_model"]
            y, p, k = collect(model, tl, K)
            m_sparse = np.array([int(x) in sparse_skills for x in k])
            acc["all"].append(auc_of(y, p))
            acc["sparse"].append(auc_of(y[m_sparse], p[m_sparse]))
            acc["dense"].append(auc_of(y[~m_sparse], p[~m_sparse]))
            print(f"  {name:<14} seed={sd}  all={acc['all'][-1]:.4f} "
                  f"sparse={acc['sparse'][-1]} dense={acc['dense'][-1]}")
        out["groups"][name] = {}
        for g in ("all", "sparse", "dense"):
            v = np.array([x for x in acc[g] if x is not None], dtype=np.float64)
            out["groups"][name][g] = dict(
                mean=round(float(v.mean()), 4) if v.size else None,
                std=round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
                per_seed=[round(float(x), 4) for x in v])
    # 相对完整模型的 Δ（稀疏组）
    base = out["groups"]["完整模型"]
    for name, _ in VARIANTS:
        if name == "完整模型":
            continue
        g = out["groups"][name]
        for key in ("all", "sparse", "dense"):
            if g[key]["mean"] is not None and base[key]["mean"] is not None:
                g[key]["delta_vs_full"] = round(g[key]["mean"] - base[key]["mean"], 4)

    p = os.path.join(RES, f"kt_{a.dataset}_sparse.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n===== {a.dataset} 稀疏/稠密知识点分组 AUC =====")
    print(f"{'变体':<14}{'整体':<18}{'稀疏组':<18}{'稠密组':<18}")
    for name, _ in VARIANTS:
        g = out["groups"][name]
        def cell(k):
            v = g[k]
            if v["mean"] is None:
                return "--"
            s = f"{v['mean']:.4f}±{v['std']:.4f}"
            if "delta_vs_full" in v:
                s += f"({v['delta_vs_full']:+.4f})"
            return s
        print(f"{name:<14}{cell('all'):<18}{cell('sparse'):<18}{cell('dense'):<18}")
    print("SAVED", p, f"({time.time()-t00:.0f}s)")

    # ---- 操作点自检：本表的「完整模型」必须等于主表的 DGCL-KT ----
    bp = os.path.join(RES, f"kt_bench_{a.dataset}.json")
    if os.path.exists(bp) and out["groups"]["完整模型"]["all"]["mean"] is not None:
        try:
            with open(bp, encoding="utf-8") as f:
                main = (((json.load(f).get("summary") or {}).get("DGCLKT") or {})
                        .get("AUC") or {}).get("mean")
        except (ValueError, OSError):
            main = None
        if main is not None:
            dv = abs(out["groups"]["完整模型"]["all"]["mean"] - main)
            print(f"[op-check] 完整模型={out['groups']['完整模型']['all']['mean']:.4f}  "
                  f"主表 DGCL-KT={main:.4f}  |Δ|={dv:.4f}  "
                  f"{'OK' if dv <= 0.005 else '!! 操作点漂移，不可直接进论文'}")


if __name__ == "__main__":
    main()
