#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本文方法 DGCL-KT 的分阶段链式调参 + 与基线配置合并。

为什么需要
----------
公平比较的前提是"我方与对手有同等的调参预算"。基线侧已经由
`tune_baselines.py` 搜了 **学习率 + 容量**（DKT hidden、DKVMN memory、
Transformer 的 d/层数 等）。而 run_kt_tune.py 的单个 stage 是**互相独立**的
（例如 stage4 只搜序列通路、不搜 lr），直接用 stage4 的最优配置跑主实验，
DGCL-KT 实际仍在用默认 lr=2e-3、d=128 —— 反而成了"我方不调、对手精调"。

本脚本按固定顺序把各 stage 串起来，每步把上一步选出的最优配置作为下一步的
`--base`（未搜到的键沿用 base），最终得到一份**联合最优**配置：

    stage 4  序列通路   seq_enc × layers × graph_in
    stage 5  容量       d × dropout × lr          @ base=4
    stage 1  图证据结构  gated_fuse × steps × hops × topm @ base=4+5
    stage 3  结构先验    wprior × topm             @ base=4+5+1

选择依据始终是**验证集最优 AUC**，测试集只记录。结果落到：
    results/kt_tune_<ds>_stage<N>.json           （各 stage 明细，可续跑）
    results/kt_maincfg_<ds>.json                 （DGCLKT 联合配置）
    results/kt_model_cfg_<ds>.json               （基线配置 + DGCLKT 配置合并，喂给
                                                  run_kt.py --model-cfg-file）

用法
----
    python experiments/tools/tune_main_chain.py --dataset assist17_f1
    python experiments/tools/tune_main_chain.py --dataset assist17_f1 --stages 4,5
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXPROOT, "run"))
sys.path.insert(0, EXPROOT)

from paths import PROCESSED as PROC, RESULTS                      # noqa: E402
from run_kt import train_one, build_graph_arrays, build_scope_mask  # noqa: E402
from run_kt_tune import STAGES, build_args                        # noqa: E402

CHAIN = [4, 5, 1, 3]


def _grid_fp(grid):
    """网格指纹：网格内容一变，指纹就变，据此让旧 stage 产物自动失效。

    为什么必须有：`run_stage` 的复用判据原先只看 `base` 与生效训练集规模。
    一旦我们**扩大了搜索网格**（例如把 stage5 的学习率从 2 点扩到与基线一致的
    5 点），旧产物虽然 base 相同、却是用**窄网格**搜出来的，会被错误复用 ——
    等于修复不生效。加了指纹后，扩网格即自动重跑该 stage 及其下游。
    """
    import hashlib
    blob = json.dumps(grid, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:12]


def run_stage(ds, st, base, epochs, batch, seed, obj, adj, ac, scope, s2i, force=False,
              tmu=0, patience=10):
    path = os.path.join(RESULTS, f"kt_tune_{ds}_stage{st}.json")
    grid = STAGES[st]
    # 本 stage 网格中出现过的**配置键**（不含 val/test/acc 等指标键）
    cfg_keys = set()
    # "生效的训练集规模"：上限高于训练集大小时该上限不生效，等价于全量。
    # 旧产物没有该字段 —— 它们一律是在"全量训练集"下跑的，故默认取 n_train。
    n_train = len(obj["train"])
    eff = min(tmu, n_train) if tmu > 0 else n_train
    for _cfg in grid:
        cfg_keys |= set(_cfg.keys())
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("best") and d.get("base") == base and \
                d.get("grid_fp") == _grid_fp(grid) and \
                int(d.get("eff_train_users", n_train) or 0) == eff:
            print(f"[chain] stage{st} 已有匹配 base 的结果，复用")
            # 只返回配置键，**绝不**把 best 行里的 val/test/acc/epochs_run/sec
            # 带进 base：否则下一步 `dict(**full, val=...)` 会因关键字重复而
            # 抛 TypeError。该缺陷只在"复用已有 stage 结果"（即断点续跑）时
            # 触发，全新计算时 grid 路径返回的 sel 是干净的 —— 因此首次运行
            # 掩盖了它，续跑时 5 个数据集的链式调参静默失败。
            sel = d.get("sel")
            if not sel:
                sel = {k: d["best"][k] for k in cfg_keys if k in d["best"]}
            return dict(sel), d["best"]
    rows = []
    print(f"[chain] --- stage{st} on {ds} (base={base}) 网格 {len(grid)} "
          f"fp={_grid_fp(grid)} patience={patience} ---")
    for i, cfg in enumerate(grid, 1):
        full = {**base, **cfg}
        t0 = time.time()
        try:
            r = train_one("DGCLKT", build_args(ds, full, epochs, batch, seed, tmu, patience),
                          obj, adj, ac, scope, s2i, seed)
        except Exception as e:                                      # noqa: BLE001
            print(f"    [{i}/{len(grid)}] {cfg} → 失败 {type(e).__name__}: {e}")
            continue
        row = dict(**full, val=r.get("best_valid_auc"), test=r["AUC"], acc=r["ACC"],
                   epochs_run=r.get("epochs_run"), sec=round(time.time() - t0, 1))
        rows.append(row)
        v = row["val"] if row["val"] is not None else float("nan")
        print(f"    [{i}/{len(grid)}] {cfg} → val={v:.4f} test={row['test']:.4f} "
              f"({row['sec']:.0f}s)")
    if not rows:
        print(f"[chain] stage{st} 无有效试验，跳过")
        return {}, None
    best = max(rows, key=lambda r: (r["val"] if r["val"] is not None else -1))
    sel = {k: best[k] for k in grid[0].keys()}
    print(f"[chain] stage{st} 最优 {sel}  val={best['val']:.4f}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=ds, stage=st, base=base, epochs=epochs, seed=seed,
                       train_max_users=int(tmu or 0), eff_train_users=eff,
                       grid_fp=_grid_fp(grid), patience=int(patience),
                       n_trials=len(rows), rows=rows, best=best, sel=sel), f,
                  ensure_ascii=False, indent=1)
    return dict(sel), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stages", default=",".join(map(str, CHAIN)))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10,
                    help="早停耐心值（须与基线侧 tune_baselines 一致，默认 10）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="忽略已有 stage 结果重跑")
    ap.add_argument("--train-max-users", type=int, default=0,
                    help="搜索阶段训练集用户上限（0=全量）。只截断训练集，验证/测试保持全量；"
                         "用于控制超大数据集（ASSIST2012 等）的链式搜索开销。")
    ap.add_argument("--merge-only", action="store_true", dest="merge_only",
                    help="只重建合并配置（用于基线调参完成后再合并一次）")
    a = ap.parse_args()
    if a.merge_only:
        main_path = os.path.join(RESULTS, f"kt_maincfg_{a.dataset}.json")
        base = {}
        if os.path.exists(main_path):
            with open(main_path, encoding="utf-8") as f:
                base = json.load(f).get("DGCLKT") or {}
        cfg = {}
        bt = os.path.join(RESULTS, f"kt_basetune_{a.dataset}.json")
        if os.path.exists(bt):
            with open(bt, encoding="utf-8") as f:
                b = json.load(f)
            for m, v in (b.get("best") or {}).items():
                cfg[m] = dict(v.get("cfg") or {}, lr=v["lr"])
        cfg["DGCLKT"] = base
        comb = os.path.join(RESULTS, f"kt_model_cfg_{a.dataset}.json")
        with open(comb, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        print(f"[chain] merged -> {comb}  ({len(cfg)} models: {list(cfg)})")
        return
    stages = [int(x) for x in a.stages.split(",") if x.strip()]

    with open(os.path.join(PROC, f"{a.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    adj, ac = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)
    print(f"[chain] dataset={a.dataset} K={obj['n_skill']} q={obj['n_question']} "
          f"CSG={'on' if scope is not None else 'off'} epochs={a.epochs} "
          f"train_max_users={a.train_max_users}")

    base, best_all = {}, None
    for st in stages:
        sel, best = run_stage(a.dataset, st, base, a.epochs, a.batch, a.seed,
                              obj, adj, ac, scope, s2i, a.force, a.train_max_users,
                              a.patience)
        if sel:
            # 双保险：显式剔除指标键，确保 base 只含**配置**（见 run_stage 注释）
            sel = {k: v for k, v in sel.items()
                   if k not in ("val", "test", "acc", "epochs_run", "sec")}
            base.update(sel)
            best_all = best
            print(f"[chain] 累计配置: {base}")

    main_path = os.path.join(RESULTS, f"kt_maincfg_{a.dataset}.json")
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump({"DGCLKT": base}, f, ensure_ascii=False, indent=1)
    print(f"[chain] saved {main_path}  -> DGCLKT {base}")

    # 与基线配置合并，供 run_kt.py --model-cfg-file 使用
    bt = os.path.join(RESULTS, f"kt_basetune_{a.dataset}.json")
    cfg = {}
    if os.path.exists(bt):
        with open(bt, encoding="utf-8") as f:
            b = json.load(f)
        for m, v in (b.get("best") or {}).items():
            cfg[m] = dict(v.get("cfg") or {}, lr=v["lr"])
    cfg["DGCLKT"] = base
    comb = os.path.join(RESULTS, f"kt_model_cfg_{a.dataset}.json")
    with open(comb, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print(f"[chain] saved {comb}  (含 {len(cfg)} 个模型: {list(cfg)})")

    if best_all:
        print(f"[chain] DGCLKT 最终 val={best_all['val']:.4f} test={best_all['test']:.4f}")


if __name__ == "__main__":
    main()
