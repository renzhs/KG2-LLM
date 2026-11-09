#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编码器对照：在**最终配置的操作点上**重做（固定其余一切，只换编码器）。

要解决的问题
------------
4.7.1 节的编码器对照表原先取自"序列通路搜索阶段"（stage1）的产物：那时容量、
图证据结构、先验强度还停在 stage1 的默认取值。而论文主表用的是四段链式搜索
（序列通路 → 容量 → 图证据结构 → 先验强度）之后的**最终配置**。
两者不在同一操作点上，于是出现自相矛盾：
对照表里最好的编码器只有 0.7491，而主表同一数据集（ASSIST09 自建划分）是 0.7656。
审稿人会直接问："你的消融为什么要在一个更差的模型上做？"

做法
----
以最终配置（results/kt_model_cfg_<ds>.json 的 DGCLKT 段）作为基线操作点，
**只改编码器三要素** seq_enc / layers / graph_in，其余（d、dropout、lr、
hops、steps、topm、wprior、gated_fuse）与最终模型完全一致，
每个变体跑 N 个随机种子，报 mean±std，并与最终选择做逐种子配对比较。

这样"对照表"与"主表"在同一操作点上可比，且能直接读出：
选中的编码器到底比次优好多少、比 Transformer 好多少。

用法
----
  python experiments/run/run_enc_ablation.py --datasets dnui_syn,assist09 --seeds 42,43,44
  python experiments/run/run_enc_ablation.py --datasets dnui_syn --seeds 42 --epochs 20   # 冒烟
"""
import argparse
import json
import os
import pickle
import re
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

# (显示名, 覆盖项)；覆盖项只涉及编码器三要素
VARIANTS = [
    ("Transformer(2层)",   {"seq_enc": "transformer", "layers": 2, "graph_in": False}),
    ("Transformer(3层)",   {"seq_enc": "transformer", "layers": 3, "graph_in": False}),
    ("LSTM(1层)",          {"seq_enc": "lstm", "layers": 1, "graph_in": False}),
    ("GRU(1层)",           {"seq_enc": "gru", "layers": 1, "graph_in": False}),
    ("GRU(1层)+图平滑输入", {"seq_enc": "gru", "layers": 1, "graph_in": True}),
    ("GRU(2层)",           {"seq_enc": "gru", "layers": 2, "graph_in": False}),
]

# 最终配置里参与"固定"的键（与编码器无关的全部超参）
FIXED_KEYS = ["d", "dropout", "lr", "steps", "hops", "topm", "wprior", "gated_fuse"]


def load_final_cfg(ds):
    """读最终配置（results/kt_model_cfg_<ds>.json 的 DGCLKT 段）。

    A 组（官方 5 折）的配置文件名带 _f1 后缀（调参只在 fold 1 上做），
    因此先试原名，再试去掉末尾折号后的 _f1 名。
    """
    cands = [f"kt_model_cfg_{ds}.json"]
    base = re.sub(r"_f\d+$", "", ds)
    if base != ds:
        cands.append(f"kt_model_cfg_{ds}.json")          # 已带 _f1，同名即可
    cands.append(f"kt_model_cfg_{base}_f1.json")
    seen = set()
    for name in cands:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(RES, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        cfg = d.get("DGCLKT")
        if cfg:
            return cfg, name
    return None, None


def make_args(ds, cfg, epochs, batch, seed):
    a = argparse.Namespace()
    a.dataset = ds
    a.models = ["DGCLKT"]
    a.epochs = epochs
    a.batch = batch
    a.lr = float(cfg.get("lr", 2e-3))
    a.patience = 10
    a.max_len = 200
    a.min_len = 3
    a.max_users = 0
    a.seed = seed
    a.tag = "encabl"
    a.ablation = False
    a.ab_no_graph = False
    a.ab_no_calib = False
    a.ab_no_scope = False
    a.ab_struct_only = False
    a.no_chunk = False
    a.model_cfg = {}
    a.lr_map = {}
    # 第一步：借 apply_tuned 把"搜索阶段会注入的全部键"补齐（它只做载入，不选参），
    #         避免手工列举 FIXED_KEYS 时漏掉某个键导致 train_one 取到意外默认值。
    a.use_tune = True
    a = apply_tuned(a, verbose=False)
    a.use_tune = False                    # 之后不再让它按 stage 优先级覆盖最终配置
    # 第二步：以**最终配置**为权威操作点覆盖；随后变体再覆盖编码器三要素。
    for k in FIXED_KEYS + ["seq_enc", "layers", "graph_in"]:
        if k in cfg:
            setattr(a, k, cfg[k])
    return a


def run_one_dataset(ds, seeds, epochs, batch, smoke):
    cfg, cfgname = load_final_cfg(ds)
    if cfg is None:
        print(f"[{ds}] 未找到最终配置 results/kt_model_cfg_*.json，跳过")
        return None
    print(f"[{ds}] 最终配置来自 {cfgname}: {json.dumps(cfg, ensure_ascii=False)}")

    with open(os.path.join(PROC, f"{ds}.pkl"), "rb") as f:
        obj = pickle.load(f)
    adj, acalib = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)

    per, val = {}, {}
    t00 = time.time()
    for name, ov in VARIANTS:
        per[name], val[name] = [], []
        for sd in seeds:
            if smoke:
                print(f"  [smoke] {name} seed={sd} 只做参数装配校验")
                continue
            t0 = time.time()
            a = make_args(ds, cfg, epochs, batch, sd)
            for k, v in ov.items():
                setattr(a, k, v)
            try:
                r = train_one("DGCLKT", a, obj, adj, acalib, scope, s2i, sd)
                per[name].append(float(r["AUC"]))
                if r.get("best_valid_auc") is not None:
                    val[name].append(float(r["best_valid_auc"]))
                print(f"  {name:<18} seed={sd}  AUC={r['AUC']:.4f} "
                      f"val={r.get('best_valid_auc')} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:                                      # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  {name:<18} seed={sd}  FAILED: {e}", flush=True)

    out = dict(dataset=ds, seeds=seeds, epochs=epochs, batch=batch,
               final_cfg=cfg, final_cfg_file=os.path.basename(cfgname or ""),
               variants={k: dict(seq_enc=v["seq_enc"], layers=v["layers"],
                                 graph_in=v["graph_in"]) for k, v in VARIANTS},
               summary={}, paired_vs_ref={})
    ref_vals = None
    for name, _ in VARIANTS:
        v = np.array(per[name], dtype=np.float64)
        vv = np.array(val[name], dtype=np.float64) if val[name] else np.array([])
        out["summary"][name] = dict(
            mean=round(float(v.mean()), 4) if v.size else None,
            std=round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
            per_seed=[round(float(x), 4) for x in v],
            val_mean=round(float(vv.mean()), 4) if vv.size else None,
            val_per_seed=[round(float(x), 4) for x in vv])
    # 参照组 = 最终配置实际选中的那个变体
    sel = f"{'GRU' if cfg.get('seq_enc') == 'gru' else 'LSTM' if cfg.get('seq_enc') == 'lstm' else 'Transformer'}"
    sel += f"({cfg.get('layers')}层)" + ("+图平滑输入" if cfg.get("graph_in") else "")
    out["selected_variant_guess"] = sel
    ref_vals = np.array(per.get(sel, []), dtype=np.float64)
    for name, _ in VARIANTS:
        v = np.array(per[name], dtype=np.float64)
        if v.size and ref_vals.size and name != sel:
            n = min(len(v), len(ref_vals))
            d = v[:n] - ref_vals[:n]
            out["paired_vs_ref"][name] = dict(
                n=n, mean_delta=round(float(d.mean()), 4),
                better_seeds=int((d > 0).sum()), worse_seeds=int((d < 0).sum()))

    p = os.path.join(RES, f"kt_{ds}_encabl.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n===== {ds} 编码器对照（{len(seeds)} seeds，最终配置操作点）=====")
    print(f"（最终配置选中的编码器 = {sel}）")
    print(f"{'编码器':<20}{'AUC(mean±std)':<20}{'val':<10}{'Δ vs 选中'}")
    for name, _ in VARIANTS:
        s = out["summary"][name]
        if s["mean"] is None:
            print(f"{name:<20}{'失败':<20}")
            continue
        d = out["paired_vs_ref"].get(name, {}).get("mean_delta")
        mark = " ← 最终选择" if name == sel else ("  " + f"{d:+.4f}" if d is not None else "")
        vm = f"{s['val_mean']:.4f}" if s.get("val_mean") is not None else "--"
        print(f"{name:<20}{s['mean']:.4f}±{s['std']:<13.4f}{vm:<10}{mark}")
    print("SAVED", p, f"({time.time() - t00:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dnui_syn,assist09")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--smoke", action="store_true", help="只校验参数装配，不真正训练")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    for ds in [x.strip() for x in a.datasets.split(",") if x.strip()]:
        run_one_dataset(ds, seeds, a.epochs, a.batch, a.smoke)


if __name__ == "__main__":
    main()
