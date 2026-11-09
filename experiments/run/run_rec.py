#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐 / 路径 实验主程序。

--task rec   学习资源推荐: 流行度 / SVD / KG-CF / KG-CF+CSG(本文) / 就绪度 / DGCL-Rec(本文)
             指标 Recall@K, NDCG@K, Prec@K, CSCR@K(越低越好)
--task path  学习路径推荐: 拓扑序 / 贪心就绪度 / 随机 / DGCL-Path(本文)
             指标 完备性 / 先修合规率 / 难度平滑度 / 预期增益 / 路径串味率
--task ges   LLM 推荐: Direct(无图谱证据, 无作用域) vs GES(图谱证据子图+CSG门控+输出校验)
             指标 HR(幻觉率) / ETR / SCR / NDCG, 并输出同名跨课程混淆的案例分析
--task mdrl  MDRL 触发式重规划 vs 周期性重规划: 重规划次数 / 路径质量 / token 与耗时

用法:
  python experiments/run/run_rec.py --dataset dnui_syn --task rec
  python experiments/run/run_rec.py --dataset dnui_syn --task ges --llm-students 60
"""
import os, sys, json, math, pickle, argparse, time, random
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
sys.path.insert(0, EXPROOT)
from paths import PROCESSED as PROC, RESULTS as RES   # noqa: E402
from models.rec_models import (build_interaction_matrix,                    # noqa: E402
                               build_interaction_matrix_split,
                               recommend_svd,
                               recommend_graph_cf, apply_scope_gate, readiness_score,
                               evaluate_ranking, cscr_at_k)
from models.llm_ges import (DeepSeek, build_evidence_subgraph,
                            build_evidence_subgraph_for, ges_recommend,      # noqa: E402
                            direct_recommend, validate, scope_compliance,
                            prereq_legality, extract_json)

KS = (5, 10, 20)


# --------------------------------------------------------------------------
def load(dataset):
    with open(os.path.join(PROC, f"{dataset}.pkl"), "rb") as f:
        return pickle.load(f)


def _graph_seqs(obj, scope, cut1):
    """按 `--graph-scope` 选择用于**归纳先修图**的序列集合。

    这是审稿意见 M10 的修复点。`induce_prereq` 由答题序列的"共现 + 位置序"
    归纳有向先修 DAG，此前一律传 `train + valid + test`——但

      · KT 任务是按**学生**划分的，测试学生的序列不该参与建图；
      · 推荐/路径任务是按**位置**三段切分的，[cut2,1) 段是评测真值，
        用它归纳先修边会给图谱类方法带来相对无图谱基线的信息优势。

    因此这里给出三种可核验的范围：`all` 为原口径（仅用于复现旧结果与做
    敏感性对照），`train` 为 KT 的正确范围，`cut` 为推荐/路径的正确范围。
    """
    if scope == "train":
        return list(obj.get("train") or [])
    if scope == "cut":
        out = []
        for s in (obj.get("train") or []) + (obj.get("valid") or []) + (obj.get("test") or []):
            sk = s["skill"]
            if len(sk) < 3:
                continue
            n = max(1, int(round(len(sk) * cut1)))
            out.append(dict(skill=np.asarray(sk[:n], dtype=np.int64)))
        return out
    return (list(obj.get("train") or []) + list(obj.get("valid") or [])
            + list(obj.get("test") or []))


def _adj_scope(obj, sid2row, K, scope="all", cut1=0.6):
    """构造结构邻接 adj、反向标定邻接 calib、作用域掩码 scope_row、先修表 prereq。

    公开数据集无先修标注时，用 models.kg_induct 归纳有向先修 DAG（共现+位置序），
    与 run_kt.build_graph_arrays 使用同一套逻辑，保证 KT 与推荐/路径任务图一致。
    `scope` 取 "all"/"train"/"cut"，语义见 `_graph_seqs`。
    """
    prereq = obj.get("prereq")
    # 只有"图靠归纳得到"的数据集才受 graph-scope 影响。
    # DNUI-IKG 系列的先修边来自课程教学大纲（且带学段—专业标注），
    # 若对它重新归纳就等于用合成/真实答题序列覆盖真实课程结构 —— 必须排除。
    from_curriculum = bool(obj.get("skill_scope")) or \
        str((obj.get("meta") or {}).get("source", "")).startswith("DNUI")
    if scope != "all" and not from_curriculum:
        from models.kg_induct import induce_prereq
        prereq, ne, _ = induce_prereq(_graph_seqs(obj, scope, cut1), K)
        print(f"[graph] graph-scope={scope} → 重归纳先修边 {ne} 条")
    elif scope != "all" and from_curriculum:
        print("[graph] 先修边来自教学大纲，graph-scope 不生效（保持原图）")
    elif not prereq:
        from models.kg_induct import induce_prereq
        seqs = obj["train"] + obj["valid"] + obj["test"]
        prereq, ne, _ = induce_prereq(seqs, K)
        print(f"[graph] 无先修标注 → 归纳先修边 {ne} 条（共现+位置序）")
    adj = np.zeros((K, K), dtype=np.float32)
    calib = np.zeros((K, K), dtype=np.float32)
    # 反向标定（LKG→CKG 方向）：用真实掌握数据估出的边权 w 调整先修强度。
    # 已排查（2026-09-21）：曾怀疑 w 量纲过小（DNUI-IKG 上 w∈[0,0.41]、均值 0.044）
    # 导致 1+tanh(w) 与结构边的 1.0 几乎无差别、标定形同虚设；实测把 w 按最大值归一化
    # 到 [1,2] 后推荐指标不变（sNDCG@10 0.2117→0.2116）。可见反向标定在推荐任务上的
    # 边际性是方法层面的，而非实现尺度问题——故保留原公式与 KT 实验口径保持一致。
    for u, lst in enumerate(prereq):
        for v, w in (lst or []):
            adj[u, v] = 1.0
            calib[u, v] = 1.0 + math.tanh(w)
    n_stu = max(1, max(sid2row.values()) + 1) if sid2row else 1
    scope_row = np.ones((n_stu, K), dtype=bool)
    ss = obj.get("skill_scope"); stu = obj.get("student_scope") or {}
    if ss and stu:
        meta = {int(k): (set(v["grades"]), set(v["majors"])) for k, v in ss.items()}
        for s, r in sid2row.items():
            v = stu.get(s)
            if v is None:
                continue
            g, mj = (v.get("grade"), v.get("major")) if isinstance(v, dict) else (v[0], v[1])
            for k, (gs, ms) in meta.items():
                scope_row[r, k] = bool((g in gs) and ((mj in ms) or (mj == "其他")))
    return adj, calib, scope_row, prereq


def build_all(obj, cutoff=0.7, scope="all", cut1=0.6):
    K = obj["n_skill"]
    seqs = obj["train"] + obj["valid"] + obj["test"]
    X, rel, sid2row = build_interaction_matrix(seqs, K, cutoff=cutoff)
    adj, calib, scope_row, prereq = _adj_scope(obj, sid2row, K, scope, cut1)
    return X, rel, sid2row, adj, calib, scope_row, prereq


def build_all_split(obj, cut1=0.6, cut2=0.7, scope="all"):
    """规范三段切分：X 用 [0,cut1)，验证真值 [cut1,cut2)，测试真值 [cut2,1)。"""
    K = obj["n_skill"]
    seqs = obj["train"] + obj["valid"] + obj["test"]
    X, rv, rt, sid2row = build_interaction_matrix_split(seqs, K, 3, cut1, cut2)
    adj, calib, scope_row, prereq = _adj_scope(obj, sid2row, K, scope, cut1)
    return X, rv, rt, sid2row, adj, calib, scope_row, prereq


def rowz(a):
    """行内 z 标准化：使同一学生各技能得分可比（top-K 排序必需）。"""
    m = a.mean(1, keepdims=True); s = a.std(1, keepdims=True) + 1e-9
    return (a - m) / s


def _per_student(score, rel, scope_row, k=10):
    """逐学生 NDCG@k（与 evaluate_ranking 完全同口径的 scoped 口径）。

    单独抽出来是为了做配对显著性检验：把「学生」当作独立重采样单位，
    只比较同一批学生上两个方法的差异，避免「均值差 0.003」这类噪声被当成结论。
    """
    n = score.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    kk = min(k, score.shape[1])
    order = np.argsort(-score, axis=1, kind="mergesort")[:, :kk]
    rows = np.arange(n)[:, None]
    disc = 1.0 / np.log2(np.arange(2, kk + 2))
    gains = rel[rows, order] * disc
    ideal = np.sort(rel, axis=1)[:, ::-1][:, :kk] * disc
    idcg = ideal.sum(1)
    return np.where(idcg > 0, gains.sum(1) / np.maximum(idcg, 1e-9), 0.0)


def _paired_boot(a, b, n_boot=2000, seed=20260921):
    """配对 bootstrap 检验 mean(a)-mean(b)。

    返回 (delta, ci_lo, ci_hi, p)。ci 为 95% 百分位区间；p 为双侧
    bootstrap 尾部概率（落在 0 对侧的分布质量 ×2）。ci_lo>0 即「a 显著优于 b」。
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = a.size
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    delta = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return delta, lo, hi, min(1.0, p)


def _scoped_eval(score, rel, scope_row, Ks, mask=None):
    """同时给出原始指标与作用域内(scoped)指标。

    scoped 的含义：真值只保留作用域内的正例，越界推荐既不计命中、又占用 Top-K 名额。
    这是约束感知排序的公平评测——在课程场景里"推荐越界知识点"本就是一种错误，
    而不是"与本次目标无关的中立选择"，因此不应像原始指标那样对它免罚。

    重要：评测对象限定在「至少有一个作用域内正例」的学生子集上。否则那些本来
    就没有作用域内正例的学生会被记为 0 命中，线性地把所有门控方法的分数压低，
    那是指标假象而非方法缺陷。原始指标与 scoped 指标使用同一子集以保证可比。
    """
    if mask is None:
        mask = (rel * scope_row.astype(np.float32)).sum(1) > 0
    if mask.sum() == 0:
        return {}
    sc_ = score[mask]; rl_ = rel[mask]; sk_ = scope_row[mask]
    out = evaluate_ranking(sc_, rl_, sk_, Ks)
    out.update({("s" + k): v for k, v in
                evaluate_ranking(sc_, rl_ * sk_.astype(np.float32), sk_, Ks).items()})
    out["n_eval"] = int(mask.sum())
    return out


# ==========================================================================
GAMMA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)


def _rec_core(X, rel_val, rel_tst, scope_row, adj, calib, args, verbose=True):
    """DGCL-Rec 全套对比，返回指标字典（含 _gamma / _gamma_sweep）。

    设计定型（由 experiments/tools/diag_rec2.py 在验证集上选出）:
      · 就绪度项用「反向标定邻接」——数据驱动地修正先修强度；
      · 协同项用「结构邻接」传播——标定边权在多层传播中会引入噪声，验证集得分更低；
      · γ（协同权重）在验证集上选，测试集只用于报告。

    单独抽出本函数是为了让「多变切分鲁棒性实验」(run_rec_robust.py) 复用同一套评测，
    避免两处实现漂移。
    """
    K = X.shape[1]
    ready_s = readiness_score(X, adj, K)          # 结构先修就绪度
    ready_c = readiness_score(X, calib, K)        # 反向标定先修就绪度
    gcf, _, _ = recommend_graph_cf(X, adj, args.dim, args.layers, args.alpha)
    gcf_c, _, _ = recommend_graph_cf(X, adj, args.dim, args.layers, args.alpha, calibrate=calib)

    # 评测子集：至少有 1 个作用域内正例的学生（原始与 scoped 指标共用）
    mask = (rel_tst * scope_row.astype(np.float32)).sum(1) > 0
    vmask = (rel_val * scope_row.astype(np.float32)).sum(1) > 0
    rel_val_s = rel_val * scope_row.astype(np.float32)

    # ---- 设计选择全部在验证集上完成：(就绪度项来源, 协同项来源, γ) 三元组 ----
    # 动机：标定边权在单层就绪度上一致变好，但多层传播会放大其噪声；因此「用结构图
    # 还是标定图做协同项」不应由人拍定，而应像 γ 一样在验证集上选。测试集只用于报告。
    RZ_READY = {"ready_c": rowz(ready_c), "ready_s": rowz(ready_s)}
    RZ_GCF = {"gcf": rowz(gcf), "gcf_c": rowz(gcf_c)}
    VARIANTS = (("ready_c", "gcf"), ("ready_c", "gcf_c"), ("ready_s", "gcf"))
    sweep, best = [], None
    for rk, gk in VARIANTS:
        for g in GAMMA_GRID:
            sc = apply_scope_gate(RZ_READY[rk] + g * RZ_GCF[gk], scope_row)
            m = evaluate_ranking(sc[vmask], rel_val_s[vmask], scope_row[vmask], (10,))
            sweep.append(dict(ready=rk, gcf=gk, gamma=g,
                              val_sNDCG10=m["NDCG@10"], val_sRecall10=m["Recall@10"]))
            if best is None or m["NDCG@10"] > best["val_sNDCG10"]:
                best = dict(ready=rk, gcf=gk, gamma=g, val_sNDCG10=m["NDCG@10"])
    best_g = best["gamma"]
    if verbose:
        print(f"验证集选择：就绪度项={best['ready']} 协同项={best['gcf']} γ*={best_g} "
              f"(val sNDCG@10={best['val_sNDCG10']:.4f})")
        for rk, gk in VARIANTS:
            row = [(s["gamma"], s["val_sNDCG10"]) for s in sweep
                   if s["ready"] == rk and s["gcf"] == gk]
            print(f"  {rk}+{gk}: " + " ".join(f"{g}:{v:.4f}" for g, v in row))

    E = lambda s_: _scoped_eval(s_, rel_tst, scope_row, KS, mask=mask)   # noqa: E731

    def _val_ndcg(rk):
        sc = apply_scope_gate(RZ_READY[rk] + best_g * RZ_GCF[best["gcf"]], scope_row)
        return float(evaluate_ranking(sc[vmask], rel_val_s[vmask], scope_row[vmask], (10,))["NDCG@10"])

    # 记录「反向标定」在验证集上的作用（诚实报告：若验证集也未提升，就不宣称其增益）
    _vc, _vs = _val_ndcg("ready_c"), _val_ndcg("ready_s")
    # 本文方法：验证集选出的 (就绪度项 ⊕ γ·协同项) → CSG 硬门控
    fused = RZ_READY[best["ready"]] + best_g * RZ_GCF[best["gcf"]]
    # 消融：相对验证集所选组合，各替换/去掉一个组件
    ab_calib = RZ_READY["ready_s"] + best_g * RZ_GCF[best["gcf"]]
    scores = {
        "Popularity": np.tile(X.sum(0)[None, :], (X.shape[0], 1)),
        "SVD(MF)": recommend_svd(X, args.dim),
        "KG-CF(无门控)": gcf,
        "KG-CF+反向标定": gcf_c,
        "先修就绪度(结构)": ready_s,
        "先修就绪度(反向标定)": ready_c,
        # CSG 增益的**归因隔离**对照（审稿意见 M12）：基分与"先修就绪度(结构)"
        # 完全相同、只多一道硬门控。二者作用域内的相对顺序一致，故差值只能来自
        # "越界项不再占用 Top-K 名额"这一槽位释放效应，可把"释放"与"重排"分开。
        "先修就绪度(结构)+CSG(仅释放槽位)": apply_scope_gate(ready_s, scope_row),
        "先修就绪度+CSG": apply_scope_gate(ready_c, scope_row),
        "DGCL-Rec(本文)": apply_scope_gate(fused, scope_row),
        "DGCL-Rec-无CSG(消融)": fused,
        "DGCL-Rec-无反向标定(消融)": apply_scope_gate(ab_calib, scope_row),
        "DGCL-Rec-无图谱协同(消融)": apply_scope_gate(RZ_READY[best["ready"]], scope_row),
    }
    out, per = {}, {}
    rel_tst_s = rel_tst[mask] * scope_row[mask].astype(np.float32)
    for k_, v_ in scores.items():
        out[k_] = E(v_)
        per[k_] = _per_student(v_[mask], rel_tst_s, scope_row[mask], 10)

    # 配对显著性检验：以本文方法为参照，报告 ΔNDCG@10 的 95% CI 与双侧 p。
    # 目的：当方法间只差小数点后两三位时，用「同一批学生上的配对差异」判定
    # 该差异是否只是抽样噪声，而不是直接拿均值大小下结论。
    base = per["DGCL-Rec(本文)"]
    sig = {}
    for k_, v_ in per.items():
        if k_ == "DGCL-Rec(本文)":
            continue
        d, lo, hi, p = _paired_boot(base, v_)
        sig[k_] = dict(delta=round(d, 4), ci=[round(lo, 4), round(hi, 4)],
                       p=round(p, 4), win=bool(lo > 0))
    out["_sig"] = sig
    out["_sig_ref"] = "DGCL-Rec(本文)"
    out["_sig_n"] = int(mask.sum())

    # ---- CSG 是否重排了域内候选？给出可复核的对照量（审稿意见 M12）----
    # 若"仅释放槽位"后域内 Top-10 的顺序与门控前**逐个学生完全相同**，
    # 则门控对域内指标的提升只能来自槽位释放，而非"重新排序"。
    _s_ = ready_s[mask]
    _sk = scope_row[mask]
    _o1 = np.argsort(-_s_, axis=1, kind="mergesort")
    _o2 = np.argsort(-np.where(_sk, _s_, -1e9), axis=1, kind="mergesort")
    _same = 0
    for _i in range(_s_.shape[0]):
        _a = [j for j in _o1[_i] if _sk[_i, j]][:10]
        _b = [j for j in _o2[_i] if _sk[_i, j]][:10]
        _same += int(_a == _b)
    out["_csg_inscope_top10_identical_rate"] = round(_same / max(1, _s_.shape[0]), 4)
    out["_csg_slot_release_delta"] = round(
        (out["先修就绪度(结构)+CSG(仅释放槽位)"] or {}).get("sNDCG@10", 0.0)
        - (out["先修就绪度(结构)"] or {}).get("sNDCG@10", 0.0), 4)
    if verbose:
        print(f"\n配对显著性（参照=DGCL-Rec(本文)，n={int(mask.sum())} 学生，指标=NDCG@10）:")
        for k_, v_ in sorted(sig.items(), key=lambda t: -t[1]["delta"]):
            mark = "胜 " if v_["win"] else ("负 " if v_["ci"][1] < 0 else "≈  ")
            print(f"  {mark}{k_:<28} Δ={v_['delta']:+.4f} "
                  f"CI[{v_['ci'][0]:+.4f},{v_['ci'][1]:+.4f}] p={v_['p']:.4f}")
    out["_gamma"] = best_g
    out["_gamma_sweep"] = sweep
    out["_variant"] = dict(ready=best["ready"], gcf=best["gcf"], gamma=best_g)
    out["_val_sNDCG10_calib"] = round(float(_vc), 4)
    out["_val_sNDCG10_struct"] = round(float(_vs), 4)
    return out


def task_rec(obj, args):
    X, rel_val, rel_tst, sid2row, adj, calib, scope_row, prereq = \
        build_all_split(obj, args.cut1, args.cut2,
                        scope=getattr(args, "graph_scope", "all"))
    K = obj["n_skill"]
    n_in = scope_row.sum(1)
    mask = (rel_tst * scope_row.astype(np.float32)).sum(1) > 0
    print(f"X={X.shape} 掌握率={X.mean():.3f} 验证正例/人={rel_val.sum(1).mean():.2f} "
          f"测试正例/人={rel_tst.sum(1).mean():.2f}")
    print(f"作用域内候选数: mean={n_in.mean():.1f} min={n_in.min()} "
          f">=10 的学生占比={float((n_in>=10).mean()):.3f} >=20 占比={float((n_in>=20).mean()):.3f}")
    print(f"真值中作用域内占比={float((rel_tst*scope_row).sum()/max(1,rel_tst.sum())):.4f}")
    print(f"评测学生子集={int(mask.sum())}/{X.shape[0]}（至少有 1 个作用域内正例）")
    t0 = time.time()
    out = _rec_core(X, rel_val, rel_tst, scope_row, adj, calib, args)
    out["_split"] = dict(cut1=args.cut1, cut2=args.cut2)
    out["_n_eval"] = int(mask.sum())
    out["_scope_avail"] = dict(mean=float(n_in.mean()), min=int(n_in.min()),
                               pct_ge10=float((n_in >= 10).mean()),
                               pct_ge20=float((n_in >= 20).mean()),
                               in_scope_share_of_rel=float(
                                   (rel_tst * scope_row).sum() / max(1, rel_tst.sum())))
    print(f"耗时 {time.time()-t0:.1f}s")

    cols = ["Recall@10", "NDCG@10", "sRecall@10", "sNDCG@10", "CSCR@10", "CSCR@20"]
    print(f"\n{'方法':<30}" + "".join(f"{c:<12}" for c in cols))
    for k, v in out.items():
        if k.startswith("_"):
            continue
        print(f"{k:<30}" + "".join(f"{v[c]:<12}" for c in cols))
    return out


# ==========================================================================
def _scope_vec(obj, sid, sid2row, scope_row, K):
    r = sid2row.get(sid)
    if r is None:
        return np.ones(K, dtype=bool)
    return scope_row[r]


def task_ges(obj, args):
    """LLM 推荐对比。为控制成本，只抽取 n 个学生。"""
    X, rel, sid2row, adj, calib, scope_row, prereq = build_all(
        obj, args.cutoff, scope=getattr(args, "graph_scope", "all"), cut1=args.cut1)
    K = obj["n_skill"]
    names = obj["skill_names"]
    ktmeta = []
    for i in range(K):
        ss = (obj.get("skill_scope") or {}).get(str(i)) or (obj.get("skill_scope") or {}).get(i) or {}
        ktmeta.append(dict(kt_id=i, name=names[i], course=ss.get("course"),
                           grades=ss.get("grades"), majors=ss.get("majors")))
    # 选出真值非空、**且至少有一个作用域内正例**的学生，以放大串味现象并可公平比较
    cand_rows = [r for r in range(X.shape[0])
                 if rel[r].sum() > 0 and ((rel[r] * scope_row[r]).sum() > 0)]
    random.Random(args.seed).shuffle(cand_rows)
    test_rows = cand_rows[:args.llm_students]
    print(f"参与 LLM 评测的学生数={len(test_rows)} (候选池={len(cand_rows)})")

    ds = DeepSeek(temperature=0.0, max_tokens=1500)
    res = defaultdict(lambda: defaultdict(list))
    cases = []

    def _desc(r):
        for s, rr in sid2row.items():
            if rr == r:
                v = (obj.get("student_scope") or {}).get(s)
                if isinstance(v, dict):
                    return f"年级={v['grade']} 专业={v['major']}"
                if v:
                    return f"年级={v[0]} 专业={v[1]}"
                break
        return "年级未知 专业未知"

    _SS = obj.get("skill_scope") or {}

    def _ss(k):
        return _SS.get(k) or _SS.get(str(k)) or {}

    def _kt_info(k):
        v = _ss(k)
        return dict(kt_id=int(k), name=names[int(k)], course=v.get("course"),
                    grades=list(v.get("grades") or []), majors=list(v.get("majors") or []))

    # Direct 基线的召回池：用**最强纯相关性信号**（反向标定先修就绪度）取 Top-N，
    # 但不过滤学段-专业作用域、也不提供任何图谱证据——正是"有相关性、无合规性"的现状系统。
    # 每次循环重算纯属浪费，提前算一次。
    gcf_all, _, _ = recommend_graph_cf(X, adj, 32, 2, 0.5)
    ready_all = readiness_score(X, calib, K)
    for i, r in enumerate(test_rows):
        mastered = {k: float(X[r, k]) for k in np.nonzero(X[r])[0]}
        cur = int(max(mastered, key=lambda k: X[r, k])) if mastered else 0
        sc = scope_row[r]
        # ---- 共享召回池：两法候选集完全一致，仅「作用域过滤+证据注入」不同，NDCG 才可比 ----
        recall = np.argsort(-ready_all[r])[:args.cand_size].tolist()
        recall_s = [k for k in recall if sc[k]]
        if len(recall_s) < args.topk:      # 作用域内候选不足时，从作用域内按就绪度补齐
            extra = [int(k) for k in np.argsort(-ready_all[r])
                     if sc[k] and int(k) not in recall_s][:args.cand_size]
            recall_s = recall_s + extra
        sub = build_evidence_subgraph_for(recall_s, mastered, cur, prereq, names)
        desc = _desc(r)
        # --- 基线 Direct：同一召回池，但不做作用域过滤、不给任何图谱证据 ---
        top_all = recall
        try:
            d_recs, d_info = direct_recommend(ds, top_all, names, desc, topk=args.topk)
            d_recs, d_st = validate(d_recs, top_all, prereq, names)
        except Exception as e:
            d_recs, d_st = [], dict(error=str(e)); print("  direct fail", e)
        # --- 本文 GES：作用域过滤后的候选 + 证据子图 + 输出校验 ---
        try:
            g_recs, g_info = ges_recommend(ds, sub, names, desc, topk=args.topk,
                                           scope_filter=sc)
            allowed = g_info["cands"]
            g_recs, g_st = validate(g_recs, allowed, prereq, names)
        except Exception as e:
            g_recs, g_st = [], dict(error=str(e)); print("  ges fail", e)

        for tag, recs, st in (("Direct", d_recs, d_st), ("GES", g_recs, g_st)):
            rl = rel[r]
            # 规范 NDCG@10：DCG / IDCG，且分子分母都只认「作用域内」命中
            # （越界项既不算命中，也不进理想排序——与 CSCR 的口径保持一致）
            n_pos = int((rl * scope_row[r]).sum())
            ide = min(args.topk, n_pos)
            idcg = sum(1.0 / math.log2(j + 2) for j in range(ide))
            dcg = 0.0
            for j, rr in enumerate(recs[:args.topk]):
                kk = int(rr["kt_id"])
                if 0 <= kk < K and rl[kk] > 0 and scope_row[r][kk]:
                    dcg += 1.0 / math.log2(j + 2)
            res[tag]["HR"].append(st.get("hallucination_rate", 0.0))
            res[tag]["ETR"].append(st.get("evidence_traceable_rate", 0.0))
            res[tag]["evidence_cited_rate"].append(st.get("evidence_cited_rate", 0.0))
            res[tag]["SCR"].append(scope_compliance(recs, scope_row[r]))
            res[tag]["PR"].append(prereq_legality(recs, mastered, prereq))
            res[tag]["NDCG@10"].append(dcg / idcg if idcg > 0 else 0.0)
            res[tag]["n"].append(len(recs))

        # --- 真实串味案例：Direct 推荐了作用域外的知识点（用于论文案例分析表）---
        _sc = scope_row[r]
        d_ids = [int(x["kt_id"]) for x in d_recs]
        g_ids = [int(x["kt_id"]) for x in g_recs]
        d_out = [k for k in d_ids if not _sc[k]]
        g_out = [k for k in g_ids if not _sc[k]]
        if d_out:
            cases.append(dict(student_row=r, student=desc,
                              n_direct=len(d_ids), n_direct_out_of_scope=len(d_out),
                              n_ges=len(g_ids), n_ges_out_of_scope=len(g_out),
                              direct_out_of_scope=[_kt_info(k) for k in d_out[:5]],
                              direct_ids=d_ids[:10], ges_ids=g_ids[:10]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_rows)}  done  budget={ds.budget.as_dict()}")

    # ======================================================================
    # 受控案例探查：同名跨课程知识点（C1 的 86/87/88 vs C3 的 188/189/190）
    # 目的不是统计平均，而是把「串味」这一机制显式暴露出来：
    #   候选池里同时放入「作用域内同名项 a」与「作用域外同名项 b」，
    #   Direct 只看到名称（无作用域信息、无图证据）→ 可能挑中 b；
    #   GES 拿到作用域约束与先修证据 → 应挑中 a。
    # ======================================================================
    probe = []
    # 从技能名里自动找出「同名但不同课程/不同作用域」的知识点对，
    # 不硬编码 id —— 预处理会把原始 kt_id 重映射为 0..K-1。
    _byname = defaultdict(list)
    for k, nm in enumerate(names):
        _byname[str(nm).strip()].append(k)
    dup_pairs = []
    for nm, ks in _byname.items():
        if len(ks) < 2:
            continue
        for ii in range(len(ks)):
            for jj in range(ii + 1, len(ks)):
                dup_pairs.append((ks[ii], ks[jj], nm))
    print(f"同名跨课程知识点对: {[(a, b, nm) for a, b, nm in dup_pairs][:8]}"
          f"（共 {len(dup_pairs)} 对）")
    for a, b, nm in dup_pairs[:8]:
        # 需要存在"作用域覆盖 a、不覆盖 b"的学生，才能构成串味场景
        rs = [rr for rr in range(X.shape[0])
              if scope_row[rr][a] and not scope_row[rr][b] and X[rr].sum() > 0]
        if not rs:
            continue
        r = rs[0]
        mastered = {k: float(X[r, k]) for k in np.nonzero(X[r])[0]}
        pool = [a, b] + [int(k) for k in np.argsort(-ready_all[r])[:max(0, args.cand_size - 2)]
                         if scope_row[r][k] and int(k) not in (a, b)]
        sub = build_evidence_subgraph_for(pool, mastered, a, prereq, names)
        desc = _desc(r)
        try:
            d_recs, _ = direct_recommend(ds, pool, names, desc, topk=args.topk)
            g_recs, _ = ges_recommend(ds, sub, names, desc, topk=args.topk,
                                      scope_filter=scope_row[r])
        except Exception as e:
            print("  probe fail", e); continue
        d_ids = [int(x.get("kt_id")) for x in d_recs
                 if str(x.get("kt_id", "")).lstrip("-").isdigit()][:args.topk]
        g_ids = [int(x.get("kt_id")) for x in g_recs
                 if str(x.get("kt_id", "")).lstrip("-").isdigit()][:args.topk]
        probe.append(dict(
            pair=[a, b], name=nm, student=r,
            in_scope=a, out_of_scope=b,
            direct_picked_out=(b in d_ids), ges_picked_out=(b in g_ids),
            direct_picked_in=(a in d_ids), ges_picked_in=(a in g_ids),
            direct_out=d_ids, ges_out=g_ids))
        if b in d_ids or b in g_ids:
            cases.append(probe[-1])
    probe_summary = dict(
        n=len(probe),
        direct_out_of_scope=int(sum(1 for p in probe if p["direct_picked_out"])),
        ges_out_of_scope=int(sum(1 for p in probe if p["ges_picked_out"])),
        direct_in_scope=int(sum(1 for p in probe if p["direct_picked_in"])),
        ges_in_scope=int(sum(1 for p in probe if p["ges_picked_in"])))

    out = {}
    for tag in res:
        out[tag] = {k: round(float(np.mean(v)), 4) for k, v in res[tag].items()}
    out["_budget"] = ds.budget.as_dict()
    out["_cases"] = sorted(cases, key=lambda c: -c["n_direct_out_of_scope"])[:12]
    out["_n_cases"] = len(cases)
    out["_probe"] = probe
    out["_probe_summary"] = probe_summary
    print("\n=== LLM 推荐对比 ===")
    for tag in ("Direct", "GES"):
        if tag in out:
            print(f"  {tag:<8}", {k: out[tag][k] for k in
                                  ("HR", "ETR", "SCR", "PR", "NDCG@10", "n")
                                  if k in out[tag]})
    print("  budget:", out["_budget"])
    print(f"  Direct 越界推荐的真实案例数: {out['_n_cases']} "
          f"(已存 {len(out['_cases'])} 条到 results)")
    print(f"  同名跨课程受控探查(3 对): {out['_probe_summary']}")
    return out


# ==========================================================================
def topo_order(prereq, K):
    indeg = [0] * K; adj = [[] for _ in range(K)]
    for u, lst in enumerate(prereq):
        for v, w in lst:
            adj[u].append(v); indeg[v] += 1
    from collections import deque
    dq = deque([i for i in range(K) if indeg[i] == 0]); order = []
    while dq:
        x = dq.popleft(); order.append(x)
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    return order + [i for i in range(K) if i not in set(order)]


def eval_path(path, preds, scope_row, X_row, skill_diff, K, target_raw, target_s):
    """路径质量指标。

    完备性        = |路径 ∩ 全部目标| / |全部目标|        （目标含作用域外，门控方法天然吃亏）
    完备性_s      = |路径 ∩ 作用域内目标| / |作用域内目标|  （约束感知的公平覆盖率）
    可执行覆盖率  = 「既在路径中、且其位于路径内的先修都排在它之前」的目标占比 ★核心指标
    先修合规率    = 路径涉及的先修对中「先修在前」的比例
    难度平滑度    = -mean(|相邻难度差|)，越接近 0 越平滑
    路径串味率    = 路径中作用域外知识点占比，越低越好

    为什么需要「可执行覆盖率」：单看完备性会偏向"撒点"式策略（随机撒 10 个点也能覆盖
    不少目标），但那些目标在路径里次序错乱、学生根本无法按此执行。可执行覆盖率同时
    要求「覆盖到」与「顺序合法」，才是路径推荐真正有意义的指标。
    """
    pos = {k: i for i, k in enumerate(path)}
    ok, tot = 0, 0
    for k in path:
        for p in preds[k]:
            if p in pos:
                tot += 1
                if pos[p] < pos[k]:
                    ok += 1
    covered = len(set(path) & target_raw) / max(1, len(target_raw))
    covered_s = len(set(path) & target_s) / max(1, len(target_s))
    good = 0
    for k in target_s:
        if k not in pos:                     # 未覆盖
            continue
        # 路径外的先修视为"学生已具备"，不构成障碍
        if all((p not in pos) or (pos[p] < pos[k]) for p in preds[k]):
            good += 1
    ec = good / max(1, len(target_s))
    diffs = [skill_diff[k] for k in path if k in skill_diff]
    smooth = float(np.mean(np.abs(np.diff(diffs)))) if len(diffs) > 1 else 0.0
    bad = sum(1 for k in path if not scope_row[k]) / max(1, len(path))
    # 没有任何先修关系时，合规率无定义（记为 None，聚合时跳过），不能报成 0
    comp = round(ok / tot, 4) if tot > 0 else None
    return dict(完备性=round(covered, 4), 完备性_s=round(covered_s, 4),
                可执行覆盖率=round(ec, 4), 先修合规率=comp,
                难度平滑度=round(-smooth, 4),
                路径串味率=round(bad, 4), 路径长度=len(path))


# --------------------------------------------------------------------------
# 学习路径：基线选点策略（固定，不参选）
def _sel_topo_scope(ctx):
    return [k for k in ctx["order_all"] if ctx["sc"][k]][:ctx["N"]]


def _sel_rand(ctx):
    rnd = random.Random(ctx["seed"] + ctx["r"])
    return rnd.sample(ctx["pool"], min(ctx["N"], len(ctx["pool"])))


def _sel_ready_ungated(ctx):
    return list(np.argsort(-ctx["ready_c"])[:ctx["N"]])


def _sel_gate_ready(ctx):
    return sorted(ctx["pool"], key=lambda k: -ctx["ready_c"][k])[:ctx["N"]]


# --- 可配置的选点族：family(选点口径) × ready_key(是否反向标定) × order_topo(是否拓扑定序) ---
PATH_FAMILIES = ("ready", "front_ready")
PATH_FAMILY_LABEL = {"ready": "就绪度选点", "front_ready": "前沿推进+就绪度"}
ABL_NO_CALIB = "DGCL-Path-无反向标定(消融)"
ABL_NO_ORDER = "DGCL-Path-无拓扑定序(消融)"


def make_path_selector(family, ready_key="ready_c", order_topo=True):
    """生成一个选点函数。

    family      : "ready" 只按就绪度选；"front_ready" 先按课程前沿距离、再按就绪度
    ready_key   : "ready_c" 用反向标定就绪度；"ready_s" 用结构就绪度（消融）
    order_topo  : True 按先修拓扑定序输出；False 按就绪度输出（消融）
    """
    def _f(ctx):
        pool = ctx["pool"]
        if family == "ready":
            picked = sorted(pool, key=lambda k: -ctx[ready_key][k])[:ctx["N"]]
        else:
            r0, rank = ctx["r0"], ctx["rank"]
            picked = sorted(pool, key=lambda k: (abs(rank[k] - r0),
                                                 -ctx[ready_key][k]))[:ctx["N"]]
        if order_topo:
            return sorted(picked, key=lambda k: ctx["rank"][k])
        return sorted(picked, key=lambda k: -ctx[ready_key][k])
    return _f


def build_path_methods(chosen_family=None):
    """返回 name -> selector。chosen_family 给出时，额外加入针对该族的两个消融。"""
    m = {
        "拓扑序(门控)": _sel_topo_scope,
        "作用域内随机": _sel_rand,
        "贪心就绪度(无门控)": _sel_ready_ungated,
        "贪心就绪度+CSG(无序)": _sel_gate_ready,
    }
    for fam in PATH_FAMILIES:
        m[PATH_FAMILY_LABEL[fam] + "+拓扑定序"] = make_path_selector(fam, "ready_c", True)
    if chosen_family is not None:
        m[ABL_NO_CALIB] = make_path_selector(chosen_family, "ready_s", True)
        m[ABL_NO_ORDER] = make_path_selector(chosen_family, "ready_c", False)
    return m


# 参与「验证集选型」的候选（基线不参选，否则可能把基线选成最终方法）
PATH_CANDIDATES = [PATH_FAMILY_LABEL[f] + "+拓扑定序" for f in PATH_FAMILIES]


def _path_eval(rows, X, rel_tgt, scope_row, preds, rank, order_all, ready_s, ready_c,
               skill_diff, N, seed, methods):
    """在给定真值段上评测 methods（name -> selector），返回 (每方法平均指标, 评测学生数)。"""
    agg = defaultdict(lambda: defaultdict(list))
    n_used = 0
    for r in rows:
        sc = scope_row[r]
        tgt_raw = set(np.nonzero(rel_tgt[r] > 0)[0].tolist())
        tgt_s = set(np.nonzero((rel_tgt[r] > 0) & sc)[0].tolist())
        if not tgt_s:                 # 该段没有作用域内目标 → 退回"作用域内未掌握"
            tgt_s = set(np.nonzero(sc & (X[r] <= 0))[0].tolist()[:8])
        if not tgt_raw:
            tgt_raw = set(tgt_s)
        if not tgt_s:
            continue                  # 无任何作用域内可推荐目标，排除以免指标假象
        n_used += 1
        pool = np.nonzero(sc)[0].tolist()
        mastered_in = [k for k in pool if X[r, k] > 0]
        ctx = dict(r=r, sc=sc, pool=pool, rank=rank, order_all=order_all, N=N, seed=seed,
                   r0=(max(rank[k] for k in mastered_in) + 1) if mastered_in else 0,
                   ready_s=ready_s[r], ready_c=ready_c[r])
        for name, fn in methods.items():
            p = fn(ctx)
            m = eval_path(p, preds, sc, X[r], skill_diff, len(rank), tgt_raw, tgt_s)
            for kk, vv in m.items():
                agg[name][kk].append(vv)
    out = {}
    for t, d in agg.items():
        dd = {}
        for k, v in d.items():
            vv = [x for x in v if x is not None]      # None = 该指标无定义（如无先修边）
            dd[k] = round(float(np.mean(vv)), 4) if vv else None
        out[t] = dd
    return out, n_used


def _fmt(v):
    return "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def task_path(obj, args):
    """学习路径推荐。

    评测协议（避免用测试集选型）:
      1) 训练段 [0,cut1) 构造掌握矩阵 X；
      2) 验证段 [cut1,cut2) 的目标用于「选点策略选型」；
      3) 测试段 [cut2,1) 的目标用于最终报告，基线/候选/消融同表列出。

    核心指标「可执行覆盖率」(EC)：目标知识点既被覆盖、又满足其位于路径内的先修
    都排在它之前。单看完备性会奖励"撒点"式策略（随机撒点也能覆盖不少目标），
    EC 同时要求"覆盖到"与"顺序合法"，才能反映路径是否真的可执行。

    本文 DGCL-Path = CSG 硬门控 + 课程前沿推进选点 + 反向标定就绪度微调 + 先修拓扑定序。
    """
    X, rel_val, rel_tst, sid2row, adj, calib, scope_row, prereq = \
        build_all_split(obj, args.cut1, args.cut2,
                        scope=getattr(args, "graph_scope", "all"))
    K = obj["n_skill"]
    skill_diff = {}
    for i, v in (obj.get("skill_scope") or {}).items():
        if v.get("difficulty") is not None:
            skill_diff[int(i)] = float(v["difficulty"])
    if not skill_diff:
        skill_diff = {i: i * 1.0 / K for i in range(K)}
    order_all = topo_order(prereq, K)
    rank = {k: i for i, k in enumerate(order_all)}
    # 反向先修表：preds[v] = v 的先修集合（prereq[u] 给的是 u 的后继）
    preds = [[] for _ in range(K)]
    for u, lst in enumerate(prereq):
        for v, w in lst:
            preds[v].append(u)
    ready_s = readiness_score(X, adj, K)
    ready_c = readiness_score(X, calib, K)

    N = args.path_len
    # 学生抽样：超过上限时随机抽样（固定种子可复现），避免只用"编号靠前"的学生带来偏差
    n_rows = min(args.path_students, X.shape[0])
    if n_rows < X.shape[0]:
        rows = sorted(random.Random(args.seed).sample(range(X.shape[0]), n_rows))
    else:
        rows = list(range(X.shape[0]))

    # ---- 1) 验证集选型：只在候选选点族中比较，平局优先更简单的口径 ----
    val_methods = build_path_methods(None)
    val_out, n_val = _path_eval(rows, X, rel_val, scope_row, preds, rank, order_all,
                                ready_s, ready_c, skill_diff, N, args.seed, val_methods)
    val_ecu = {c: val_out[c]["可执行覆盖率"] for c in PATH_CANDIDATES}
    chosen = max(PATH_CANDIDATES, key=lambda c: ((val_ecu[c] or 0.0),
                                                 -PATH_CANDIDATES.index(c)))
    fam = PATH_FAMILIES[PATH_CANDIDATES.index(chosen)]
    print(f"验证集选型(EC): {val_ecu} → 选定「{chosen}」(选点族={fam}, 验证学生数={n_val})")

    # ---- 2) 测试集报告：基线 + 候选 + 针对选定族的两个消融 ----
    methods = build_path_methods(fam)
    out, n_used = _path_eval(rows, X, rel_tst, scope_row, preds, rank, order_all,
                             ready_s, ready_c, skill_diff, N, args.seed, methods)
    out["_selected"] = chosen
    out["_selected_family"] = fam
    out["_val_ecu"] = val_ecu
    out["_n_eval"] = n_used
    out["_split"] = dict(cut1=args.cut1, cut2=args.cut2)
    cols = ["完备性", "完备性_s", "可执行覆盖率", "先修合规率", "难度平滑度",
            "路径串味率", "路径长度"]
    print(f"\n评测学生数={n_used}   选定方法「{chosen}」")
    print(f"{'方法':<28}" + "".join(f"{m:<12}" for m in cols))
    for t, d in out.items():
        if t.startswith("_"):
            continue
        mark = "  <-- 本文" if t == chosen else ""
        print(f"{t:<28}" + "".join(f"{_fmt(d[m]):<12}" for m in cols) + mark)
    return out


# ==========================================================================
def task_mdrl(obj, args):
    """MDRL: 掌握度阈值触发式重规划 vs 周期性重规划。
    模拟一学期 T 个时间片, 每片学生产生若干作答; 比较重规划次数与路径质量。"""
    X, rel, sid2row, adj, calib, scope_row, prereq = build_all(
        obj, args.cutoff, scope=getattr(args, "graph_scope", "all"), cut1=args.cut1)
    K = obj["n_skill"]
    order_all = topo_order(prereq, K)
    rank = {k: i for i, k in enumerate(order_all)}
    # eval_path 要求的是**反向**先修表 preds[v] = v 的先修集合（prereq 给的是 u 的后继）
    preds = [[] for _ in range(K)]
    for u, lst in enumerate(prereq):
        for v, _ in lst:
            preds[v].append(u)
    # 难度平滑度需要真实难度：沿用 task_path 的取法（缺标注时退化为均匀间隔）
    skill_diff = {}
    for i, v in (obj.get("skill_scope") or {}).items():
        if v.get("difficulty") is not None:
            skill_diff[int(i)] = float(v["difficulty"])
    if not skill_diff:
        skill_diff = {i: i * 1.0 / K for i in range(K)}
    n_stu = min(args.mdrl_students, X.shape[0])
    T = args.mdrl_steps
    # 只在"确有作用域内候选"的学生中取满 n_stu 名。
    # 旧实现按 X 的前 n_stu 行遍历、遇到空作用域就 continue，于是
    # "24 步 × 50 学生"这句话与实际被模拟的学生数并不相等
    # （DNUI-IKG-S 上只有 37/50），而"每步/每5步"的调用次数却是按 50 人**推算**的
    # ——分子是实测、分母是推算，这正是审稿意见所指的那类口径错位。
    # 现在三条策略的调用次数与质量全部来自同一批实测学生。
    _valid = [r for r in range(X.shape[0]) if scope_row[r].any()]
    if n_stu < len(_valid):
        _pick = np.random.default_rng(args.seed + 991).choice(
            np.asarray(_valid), size=n_stu, replace=False)
        rows_m = sorted(int(x) for x in _pick)
    else:
        rows_m = _valid
        n_stu = len(rows_m)
    STRATS = ["每步重规划", "每5步重规划", "MDRL阈值触发"]
    counts = {s: 0 for s in STRATS}
    # 路径质量逐时步累计（审稿意见 M5）：只报"调用次数降了多少"是不够的，
    # 必须同时给出"少调用之后路径质量掉没掉"。三条策略共享完全相同的交互流，
    # 唯一差别是"什么时候重算路径"，因此质量差可直接归因于重规划的时机。
    QUAL_KEYS = ("可执行覆盖率", "先修合规率", "难度平滑度")
    quals = {s: {q: [] for q in QUAL_KEYS} for s in STRATS}
    rng = np.random.default_rng(args.seed)
    for r in rows_m:
        sc = scope_row[r]
        sc_idx = np.nonzero(sc)[0]
        if len(sc_idx) == 0:
            continue
        mastery = X[r].copy()
        cur_val = -1
        path = {s: None for s in STRATS}
        for t in range(T):
            # 学生随机作答若干项, 掌握度按 0/1 演进（三条策略共用这一随机流）
            for _ in range(rng.integers(1, 4)):
                k = int(rng.choice(sc_idx))
                mastery[k] = 1.0 if rng.random() < 0.62 else mastery[k]
            ready = readiness_score(mastery[None, :], adj, K)[0]
            newval = float((mastery >= 0.6).sum())
            # 三条策略各自的触发判据
            fires = {
                "每步重规划": t > 0,
                "每5步重规划": t > 0 and t % 5 == 0,
                # MDRL: 掌握度越过阈值 (0.6) 的项数变化超过 delta 才重规划
                "MDRL阈值触发": (t == 0) or (abs(newval - cur_val) >= args.mdrl_delta),
            }
            for tag in STRATS:
                if fires[tag]:
                    if t > 0:
                        counts[tag] += 1
                    path[tag] = sorted(
                        sc_idx.tolist(),
                        key=lambda k: (-(ready[k] * 2.0) + rank[k] * 0.01))[:args.path_len]
            if fires["MDRL阈值触发"]:
                cur_val = newval
            # 逐时步记录三条策略**当前生效**路径的质量
            # 注意 eval_path 的签名要求同时给出 (target_raw, target_s)——
            # 旧版只传了 7 个位置参数（少 target_s），这一行因此从未真正跑通，
            # 也正是"只报调用次数、没有质量对照"的直接原因。
            tgt_s = set(np.nonzero(sc & (mastery <= 0))[0].tolist()[:10])
            tgt_raw = set(tgt_s)
            for tag in STRATS:
                m = eval_path(path[tag] or [], preds, sc, mastery, skill_diff,
                              len(rank), tgt_raw, tgt_s)
                for q in QUAL_KEYS:
                    if m[q] is not None:
                        quals[tag][q].append(m[q])
    quality = {s: {q: (round(float(np.mean(v)), 4) if v else None)
                   for q, v in quals[s].items()} for s in STRATS}
    out = dict(steps=T, students=n_stu, replan_counts=counts,
               path_quality=quality,
               reduction_vs_every_step=round(
                   1 - counts["MDRL阈值触发"] / max(1, counts["每步重规划"]), 4),
               reduction_vs_period5=round(
                   1 - counts["MDRL阈值触发"] / max(1, counts["每5步重规划"]), 4),
               note="重规划次数按 '触发次数 = 重规划次数' 计; MDRL 仅在掌握度增量越过"
                    f" delta={args.mdrl_delta} 时触发; 掌握度轨迹由合成过程产生"
                    "（作用域内随机作答、单次答对概率 0.62、掌握后不回退），"
                    "三策略共享同一交互流")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dnui_syn")
    ap.add_argument("--task", default="rec", choices=["rec", "path", "ges", "mdrl"])
    ap.add_argument("--cutoff", type=float, default=0.7)
    # 先修图的归纳数据范围（审稿意见 M10）：all=旧口径(含 test) / train=仅训练划分
    # / cut=仅每条序列前 cut1 比例（推荐与路径任务的正确范围）
    ap.add_argument("--graph-scope", default="all",
                    choices=["all", "train", "cut"],
                    help="推断先修图所用的数据范围；rec/path 建议 cut，KT 建议 train")
    # 推荐任务的三段切分（验证集用于选 γ，测试集用于报告）
    ap.add_argument("--cut1", type=float, default=0.6)
    ap.add_argument("--cut2", type=float, default=0.7)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cand-size", type=int, default=24)
    ap.add_argument("--llm-students", type=int, default=40)
    ap.add_argument("--path-len", type=int, default=10)
    ap.add_argument("--path-students", type=int, default=200)
    ap.add_argument("--mdrl-steps", type=int, default=20)
    ap.add_argument("--mdrl-students", type=int, default=30)
    ap.add_argument("--mdrl-delta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="",
                    help="结果文件名后缀（空=主实验口径）。用于把 M10 的 "
                         "graph-scope 敏感性对照写到独立文件，避免覆盖主结果。")
    args = ap.parse_args()

    obj = load(args.dataset)
    print(f"dataset={args.dataset} skill={obj['n_skill']} "
          f"train/val/test={len(obj['train'])}/{len(obj['valid'])}/{len(obj['test'])}")
    fn = dict(rec=task_rec, path=task_path, ges=task_ges, mdrl=task_mdrl)[args.task]
    out = fn(obj, args)
    tag = f"_{args.tag}" if args.tag else ""
    p = os.path.join(RES, f"{args.task}_{args.dataset}{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SAVED", p)


if __name__ == "__main__":
    sys.exit(main())
