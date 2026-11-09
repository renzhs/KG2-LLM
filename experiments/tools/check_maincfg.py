#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸门：校验每个数据集的 DGCL-KT 配置**确实来自本轮链式调参**（而非陈旧残留）。

为什么必须有
------------
`run_bench.sh` 的阶段A3 原本只检查 `results/kt_maincfg_<ds>.json` **是否存在**。
但 `tune_main_chain.py` 只在**链跑完**时才写这个文件 —— 于是"链中途失败/被中断"
时，**上一轮遗留的 kt_maincfg 依然存在**，闸门放行，阶段B 就会用一份过时配置
跑本文方法。这类失败完全静默（文件在、内容旧），而且恰好在"扩了搜索网格、
必须重跑"的时候最危险。

判据（三重，全部可离线检验，无需加载数据集）
--------------------------------------------
1. 链上的每个 stage 产物都存在，且其 `grid_fp` == 当前代码里该 stage 网格的指纹
   —— 挡住"网格已扩大、产物还是窄网格搜出来的"；
2. 各 stage 的 `base` 严格按链式累积（stage4 无 base；stage5 的 base == stage4 的 sel；
   stage1 的 base == stage4+5；stage3 的 base == stage4+5+1）—— 挡住"某步被跳过/复用错"；
3. `kt_maincfg[<ds>][DGCLKT]` 恰好等于各 stage `sel` 的累积合并 —— 挡住"文件没被本轮重写"。

用法
----
    python experiments/tools/check_maincfg.py                       # 检查全部 7 个数据集
    python experiments/tools/check_maincfg.py --datasets assist09,assist12
退出码 0 = 全部通过；1 = 有不通过项（调用方应拒绝进入阶段B）。
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXPROOT, "run"))
sys.path.insert(0, EXPROOT)

from paths import RESULTS                                            # noqa: E402
from run_kt_tune import STAGES                                       # noqa: E402

DEFAULT_DS = ["assist17_f1", "statics11_f1", "assist15_f1", "assist09_akt_f1",
              "dnui_syn", "assist09", "assist12"]
METRIC_KEYS = ("val", "test", "acc", "epochs_run", "sec")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_one(ds, chain, _grid_fp, verbose=True):
    """返回 (ok, [问题描述...])。

    注意报告纪律：一旦某个 stage 的**网格指纹**不符，链就在那里断了 —— 从它开始的
    所有 `base` 继承比较都失去意义（期望值是拿"上一 stage 的 sel"算的，而那个 sel
    本身就不可信）。故指纹一失败就停止继承校验，避免刷出一大片级联噪声，
    让真正的原因（"这个 stage 用了旧网格"）被埋掉。
    """
    problems = []
    main_path = os.path.join(RESULTS, f"kt_maincfg_{ds}.json")
    if not os.path.exists(main_path):
        return False, [f"{ds}: 缺少 kt_maincfg_{ds}.json"]

    base, chain_ok, need_rerun = {}, True, []
    for st in chain:
        p = os.path.join(RESULTS, f"kt_tune_{ds}_stage{st}.json")
        if not os.path.exists(p):
            problems.append(f"{ds}: 缺少 stage{st} 产物")
            chain_ok = False
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        want_fp = _grid_fp(STAGES[st])
        fp_ok = d.get("grid_fp") == want_fp
        if not fp_ok:
            problems.append(
                f"{ds}: stage{st} 网格指纹不符（产物 {d.get('grid_fp')} ≠ 当前 {want_fp}）"
                f" —— 该 stage 是用旧网格搜的，必须重跑")
            need_rerun.append(st)
            chain_ok = False
        # 只在"链一直完好"时才做继承比较（否则期望值本身不可信）
        if chain_ok and d.get("base") != base:
            problems.append(f"{ds}: stage{st} 的 base 与链式累积不一致"
                            f"（产物 {d.get('base')} ≠ 期望 {base}）")
            need_rerun.append(st)
            chain_ok = False
        if fp_ok:
            sel = {k: v for k, v in (d.get("sel") or {}).items() if k not in METRIC_KEYS}
            base.update(sel)

    if not chain_ok:
        if need_rerun:
            problems.append(f"{ds}: 需重跑的 stage = {sorted(set(need_rerun))}；"
                            f"链一断，后续 stage 的继承校验自动跳过（避免级联噪声）")
        return False, problems

    main_cfg = main_cfg_get(main_path)
    if main_cfg != base:
        diff_k = sorted({*main_cfg, *base})
        detail = ", ".join(f"{k}: {main_cfg.get(k)!r}≠{base.get(k)!r}"
                           for k in diff_k if main_cfg.get(k) != base.get(k))
        problems.append(f"{ds}: kt_maincfg 内容与各 stage 的累积配置不一致（{detail}）"
                        f" —— 文件很可能不是本轮重写的")
        return False, problems

    if verbose:
        print(f"  [OK] {ds:<18} lr={base.get('lr')} d={base.get('d')} "
              f"seq_enc={base.get('seq_enc')} 配置键={len(base)}")
    return True, []


def main_cfg_get(main_path):
    with open(main_path, encoding="utf-8") as f:
        return (json.load(f).get("DGCLKT") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DS))
    ap.add_argument("--chain", default="4,5,1,3")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()

    tm = _load(os.path.join(EXPROOT, "tools", "tune_main_chain.py"), "_tm_gate")
    chain = [int(x) for x in a.chain.split(",") if x.strip()]
    dss = [x for x in a.datasets.split(",") if x.strip()]

    bad = []
    print(f"[gate] 校验 {len(dss)} 个数据集的 DGCL-KT 配置（链 {chain}）")
    for ds in dss:
        ok, probs = check_one(ds, chain, tm._grid_fp, verbose=not a.quiet)
        if not ok:
            bad.append((ds, probs))
    if bad:
        print("[gate] 不通过：")
        for ds, probs in bad:
            for p in probs:
                print(f"    - {p}")
        print("[gate] 拒绝进入阶段B：否则本文方法会用到过时/窄网格的配置，"
              "得到一份无法辩护的对比结果。请重跑对应数据集的链式调参。")
        return 1
    print(f"[gate] 全部通过（{len(dss)} 个数据集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
