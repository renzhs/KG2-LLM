#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多种子知识追踪实验：用多次随机初始化报告 mean±std，避免"单次跑分"不可复现的质疑。

输出 results/kt_<dataset>_multi.json:
  { "dataset":..., "seeds":[...], "models":[...],
    "per_seed_detail": {model: {seed: {AUC,ACC,RMSE,train_sec}}},
    "summary": {model: {"AUC": {"mean":..,"std":..,"per_seed":[...]}, ...}},
    "complete": {model: {seed: bool}},          # 逐 (模型,种子) 完成矩阵
    "all_complete": bool,                        # 任一缺口即为 False
    "win_count": {model: 该模型取得最优 AUC 的种子次数} }

配对比较：本文模型 vs 每个基线（按种子配对），便于报告胜出种子数。

设计要点（均为此前踩过的坑）
--------------------------
1. **必须支持 --model-cfg-file**：多种子跑分必须与主表（tab:kt_bench）用同一套
   "验证集选出的超参"。旧版没有这个参数，基线会静默退回全局默认 --lr，
   于是"多种子表"与"主表"口径不一致，同一模型两个数字对不上。
2. **逐 (模型,种子) 增量落盘 + 续跑**：单个训练可能数小时（GKT 尤甚），
   旧版只在全部跑完才写盘，中途中断即全丢。
3. **禁止静默的部分聚合**：旧版对失败/跳过的 (模型,种子) 只 print 一行，
   随后用**剩下的更少种子**算 mean±std —— 与 GKT 在主表里"只剩 1 折却写成
   5 折均值"是同一类缺陷。现在显式写 complete 矩阵与 all_complete，
   并在输出末尾列出缺口。
"""
import os, sys, json, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
sys.path.insert(0, HERE)
from paths import PROCESSED as PROC, RESULTS as RES          # noqa: E402
from run_kt import (train_one, build_graph_arrays, build_scope_mask,  # noqa: E402
                    apply_tuned, ALL_MODELS)
import pickle                                                  # noqa: E402

METRICS = ("AUC", "ACC", "RMSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default="", dest="seed_list",
                    help="显式指定种子序列（逗号分隔），覆盖 --seeds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-users", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--ab-no-graph", action="store_true")
    ap.add_argument("--ab-no-calib", action="store_true")
    ap.add_argument("--ab-no-scope", action="store_true")
    ap.add_argument("--ab-struct-only", action="store_true")
    ap.add_argument("--tag", default="multi")
    ap.add_argument("--model-cfg-file", default="", dest="model_cfg_file",
                    help="每模型完整配置（与 run_kt.py 同义）。多种子跑分**必须**与"
                         "主表用同一套验证集选出的超参，否则两表口径不一致。")
    ap.add_argument("--no-reuse", action="store_true", dest="no_reuse",
                    help="关闭逐 (模型,种子) 续跑（默认开启）")
    ap.add_argument("--use-tune", action="store_true",
                    help="采用验证集搜索出的最优配置训练 DGCL-KT")
    args = apply_tuned(ap.parse_args())
    # 与 run_kt.py 保持同一套配置解析（否则基线会用默认超参）
    args.lr_map = {}
    args.model_cfg = {}
    if args.model_cfg_file and os.path.exists(args.model_cfg_file):
        with open(args.model_cfg_file, encoding="utf-8") as f:
            raw = json.load(f)
        args.model_cfg = {k: v for k, v in raw.items() if isinstance(v, dict)}
        print(f"[model-cfg] 载入 {len(args.model_cfg)} 个模型的独立配置: "
              f"{ {k: v for k, v in list(args.model_cfg.items())[:4]} } ...")

    p = os.path.join(PROC, f"{args.dataset}.pkl")
    if not os.path.exists(p):
        print("missing", p); return 1
    with open(p, "rb") as f:
        obj = pickle.load(f)
    adj, adj_calib = build_graph_arrays(obj)
    scope, s2i = build_scope_mask(obj)
    print(f"dataset={args.dataset} skill={obj['n_skill']} "
          f"scope={'yes' if scope is not None else 'no'} "
          f"calib_nnz={int((adj_calib > 0).sum())}")

    models = args.models
    if models == ["all"]:
        models = list(ALL_MODELS)
    seeds = ([int(s) for s in args.seed_list.split(",") if s.strip()]
             if args.seed_list.strip() else [42 + i for i in range(args.seeds)])

    p_out = os.path.join(RES, f"kt_{args.dataset}_{args.tag}.json")
    # ---- 逐 (模型,种子) 续跑：先继承已有明细 ----
    detail = {}
    if os.path.exists(p_out) and not args.no_reuse:
        try:
            with open(p_out, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old.get("per_seed_detail"), dict):
                detail = old["per_seed_detail"]
                print(f"[resume] 已有 {sum(len(v) for v in detail.values())} 条 "
                      f"(模型,种子) 记录")
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] 旧产物无法解析（将重算）: {e}")

    def save():
        summary, complete = {}, {}
        for name in models:
            summary[name], complete[name] = {}, {}
            for k in METRICS:
                vals = [detail.get(name, {}).get(sd, {}).get(k) for sd in seeds]
                v = np.array([x for x in vals if x is not None], dtype=np.float64)
                summary[name][k] = dict(
                    mean=round(float(v.mean()), 4) if v.size else None,
                    std=round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
                    n=int(v.size), per_seed=[round(float(x), 4) for x in v])
            for sd in seeds:
                complete[name][str(sd)] = (sd in (detail.get(name) or {})
                                           and "AUC" in detail[name][sd])
        n_exp = len(models) * len(seeds)
        n_got = sum(1 for m in models for sd in seeds if complete[m][str(sd)])
        win = {m: 0 for m in models}
        n_full = sum(1 for sd in seeds
                     if all(complete[m][str(sd)] for m in models))
        for sd in seeds:
            aucs = {m: (detail.get(m, {}).get(sd) or {}).get("AUC") for m in models}
            aucs = {m: v for m, v in aucs.items() if v is not None}
            if len(aucs) == len(models):
                best = max(aucs.values())
                for m, v in aucs.items():
                    if abs(v - best) < 1e-9:
                        win[m] += 1
        ours = "DGCLKT" if "DGCLKT" in models else models[0]
        paired = {}
        for m in models:
            if m == ours:
                continue
            a = [detail.get(ours, {}).get(sd, {}).get("AUC") for sd in seeds]
            b = [detail.get(m, {}).get(sd, {}).get("AUC") for sd in seeds]
            d = np.array([x - y for x, y in zip(a, b)
                          if x is not None and y is not None], dtype=np.float64)
            if not d.size:
                continue
            paired[m] = dict(n=int(d.size), wins=int((d > 0).sum()),
                             losses=int((d < 0).sum()),
                             mean_delta=round(float(d.mean()), 4),
                             std_delta=round(float(d.std(ddof=1)), 4) if d.size > 1 else 0.0)
        out = dict(dataset=args.dataset, seeds=seeds, models=models,
                   model_cfg_file=args.model_cfg_file,
                   args=dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
                             patience=args.patience, max_len=args.max_len,
                             ablation=args.ablation, patience_used=args.patience),
                   per_seed_detail=detail, summary=summary, complete=complete,
                   n_done=n_got, n_expected=n_exp, all_complete=(n_got == n_exp),
                   n_full_seeds=n_full, best_seed_wins=win,
                   paired_vs_ours=paired, ours=ours,
                   wall_sec=round(time.time() - t00, 1))
        with open(p_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        return out

    t00 = time.time()
    for si, sd in enumerate(seeds):
        print(f"\n===== seed {sd} ({si+1}/{len(seeds)}) =====", flush=True)
        for name in models:
            if (not args.no_reuse and sd in (detail.get(name) or {})
                    and "AUC" in detail[name][sd]):
                print(f"  {name:<8} [reuse] AUC={detail[name][sd]['AUC']}", flush=True)
                continue
            t0 = time.time()
            try:
                r = train_one(name, args, obj, adj, adj_calib, scope, s2i, sd)
                detail.setdefault(name, {})[sd] = {
                    k: (round(float(r[k]), 4) if r.get(k) is not None else None)
                    for k in METRICS} | dict(train_sec=r.get("train_sec"))
                print(f"  {name:<8} AUC={r.get('AUC'):.4f} ACC={r.get('ACC'):.4f} "
                      f"RMSE={r.get('RMSE'):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  {name:<8} FAILED: {e}", flush=True)
            save()                                   # 每次训练后立即落盘

    out = save()
    n_ok = out["n_full_seeds"]
    print(f"\n===== {args.dataset} 多种子汇总 (完整种子数={n_ok}/{len(seeds)}) =====")
    print(f"{'模型':<12}{'AUC':<20}{'ACC':<20}{'RMSE':<20}{'n':<4}")
    for m in models:
        s = out["summary"][m]
        print(f"{m:<12}"
              f"{str(s['AUC']['mean']) + '±' + str(s['AUC']['std']):<20}"
              f"{str(s['ACC']['mean']) + '±' + str(s['ACC']['std']):<20}"
              f"{str(s['RMSE']['mean']) + '±' + str(s['RMSE']['std']):<20}"
              f"{s['AUC']['n']:<4}")
    gaps = [(m, sd) for m in models for sd in seeds
            if not out["complete"][m][str(sd)]]
    if gaps:
        print(f"\n[WARN] {len(gaps)} 个 (模型,种子) 未完成（明细表已被标记不完整，"
              f"不得直接写入论文）:")
        for m, sd in gaps:
            print(f"    {m:<10} seed={sd}")
    print("最优次数:", out["best_seed_wins"])
    print("配对比较(本文 vs 基线):", json.dumps(out["paired_vs_ours"], ensure_ascii=False))
    print("SAVED", p_out)
    print(f"总耗时 {out['wall_sec']}s")
    return 0 if not gaps else 1


if __name__ == "__main__":
    sys.exit(main())
