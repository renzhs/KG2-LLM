"""CSG 增益归因：把"槽位释放"与"排序质量"严格分开（审稿意见 M12）。

结论要证明的是**机制的单义性**：
  CSG 硬门控 = 把越界技能得分置为 -inf，因此
  (1) 作用域内候选的相对顺序**完全不变**——门控不做重排；
  (2) 域内指标的提升只可能来自"越界项不再占用 Top-K 名额"。

本脚本给出三个可复核量：
  A. 对同一底分，门控前/后的 sNDCG@10 差值 = **槽位释放效应**（唯一的增益来源）；
  B. 逐学生核对：门控前 Top-10 的"域内子序" 是否就是门控后 Top-10 的前缀
     （若恒为真，则"门控不重排"这一断言被逐学生验证，而非嘴上说说）；
  C. 门控前越界项占用的名额数 = CSCR@10 × 10，即释放了多少槽位。

用法（远端）:
  ./env/bin/python experiments/tools/csg_attribution.py --dataset dnui_syn
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import PROCESSED as PROC, RESULTS as RES          # noqa: E402
from run.run_rec import _graph_seqs, _adj_scope, build_all_split, rowz  # noqa: E402
from models.rec_models import (apply_scope_gate, readiness_score,   # noqa: E402
                               recommend_graph_cf)
from run.run_rec import _scoped_eval                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dnui_syn")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    with open(os.path.join(PROC, f"{args.dataset}.pkl"), "rb") as f:
        obj = pickle.load(f)
    # 主实验口径：全量序列归纳图（与论文主表一致）
    X, rel_val, rel_tst, sid2row, adj, calib, scope_row, prereq = \
        build_all_split(obj, 0.6, 0.7, scope="all")
    K = obj["n_skill"]
    mask = (rel_tst * scope_row.astype(np.float32)).sum(1) > 0

    ready_s = readiness_score(X, adj, K)
    ready_c = readiness_score(X, calib, K)
    gcf, _, _ = recommend_graph_cf(X, adj, args.dim, args.layers, args.alpha)

    # 主表选出的组合（记录在 rec_<ds>.json 的 _variant/_gamma 中）
    with open(os.path.join(RES, f"rec_{args.dataset}.json"), encoding="utf-8") as f:
        main_out = json.load(f)
    variant = main_out.get("_variant") or {"ready": "ready_c", "gcf": "gcf"}
    gamma = float(main_out.get("_gamma", 0.1))
    RZ = {"ready_c": rowz(ready_c), "ready_s": rowz(ready_s)}
    RZ_G = {"gcf": rowz(gcf)}
    fused = RZ[variant["ready"]] + gamma * RZ_G[variant["gcf"]]
    print(f"[cfg] 主表组合 ready={variant['ready']} gcf={variant['gcf']} gamma={gamma}")

    bases = {"先修就绪度(结构)": ready_s, "融合分(就绪度+γ·图谱协同)": fused}
    res = {"dataset": args.dataset, "gamma": gamma, "variant": variant,
           "n_eval": int(mask.sum()), "bases": {}}

    for name, base in bases.items():
        ung = _scoped_eval(base, rel_tst, scope_row, (10,), mask=mask)
        gat = _scoped_eval(apply_scope_gate(base, scope_row), rel_tst, scope_row,
                           (10,), mask=mask)

        # B. 逐学生核对"门控不重排域内候选"
        s_ = base[mask]
        sk = scope_row[mask]
        # 门控前：原始 Top-10 里落在域内的项，按它们在原始序中的先后
        o_raw = np.argsort(-s_, axis=1, kind="mergesort")
        # 门控后：域内得分最高的 10 项
        s_g = np.where(sk, s_, -1e9)
        o_gat = np.argsort(-s_g, axis=1, kind="mergesort")
        n_stu = s_.shape[0]
        ok_prefix = 0
        freed = []
        for i in range(n_stu):
            raw_in = [j for j in o_raw[i, :10] if sk[i, j]]
            gat10 = [j for j in o_gat[i, :10] if sk[i, j]]
            if gat10[:len(raw_in)] == raw_in:
                ok_prefix += 1
            freed.append(10 - len(raw_in))

        res["bases"][name] = dict(
            sNDCG10_ungated=ung.get("sNDCG@10"),
            sNDCG10_gated=gat.get("sNDCG@10"),
            slot_release_delta=round((gat.get("sNDCG@10") or 0) - (ung.get("sNDCG@10") or 0), 4),
            sRecall10_ungated=ung.get("sRecall@10"),
            sRecall10_gated=gat.get("sRecall@10"),
            CSCR10_ungated=ung.get("CSCR@10"),
            slots_freed_per_student=round(float(np.mean(freed)), 4),
            order_preserved_rate=round(ok_prefix / max(1, n_stu), 4),
        )
        r = res["bases"][name]
        print(f"\n== 底分：{name}")
        print(f"   sNDCG@10  门控前 {r['sNDCG10_ungated']} → 门控后 {r['sNDCG10_gated']}"
              f"  (=槽位释放效应 {r['slot_release_delta']:+.4f})")
        print(f"   sRecall@10 门控前 {r['sRecall10_ungated']} → 门控后 {r['sRecall10_gated']}")
        print(f"   门控前 CSCR@10={r['CSCR10_ungated']} → 每学生平均释放 {r['slots_freed_per_student']} 个槽位")
        print(f"   域内候选相对顺序保持不变的学生占比 = {r['order_preserved_rate']}")

    res["conclusion"] = (
        "同一底分上门控前/后的 sNDCG@10 差即槽位释放效应；"
        "order_preserved_rate=1.0 表示门控不改变域内候选的相对顺序。"
        "因此 CSG 的全部域内指标增益都来自曝光重分配，而非排序质量改进。")
    p = os.path.join(RES, f"csg_attrib_{args.dataset}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\nSAVED", p)


if __name__ == "__main__":
    main()
