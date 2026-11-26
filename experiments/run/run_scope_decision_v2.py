#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1.v2：标注不完整时的作用域判定（多挂载候选集版）

与 E1 的区别
------------
E1 假设「每题有唯一 kt，标注以概率 rho 丢失」，真值为 legal(s,q)=s 的作用域匹配 q.kt。
实测该假设与数据不符：467 个题目行聚合为 432 个题目实体，其中 32 个实体同时挂在
2~3 个知识点上（21 组跨课程）。86 与 188 的 6 道题有 5 道逐字相同。
也就是说「这道题属于哪个知识点」本来就不唯一，标注不完整是「多挂载中只记了一条」，
而非「唯一真值丢失」。

于是本实验把真值改为：
    K(q)      = 题目实体 q 的全部真实挂载（来自 resource_layer.json）
    legal(s,q)= ∃ k ∈ K(q): s.grade ∈ grade_scope(k) 且 s.major ∈ major_scope(k)

证据不完整有两种形态，都保留：
    (a) 结构性不完整：标注只记了 K(q) 中的一条，其余天然缺失（rho=0 时就存在）
    (b) 受控注入不完整：以概率 rho 抹掉该条标注，此时只能靠题干 bigram 反推 top-k 候选

对比方法
    M1 元数据硬过滤   仅用标注的那一条，作用域不匹配即拒；无标注直接拒
    M2 语义 top-1     无标注时取题干 bigram top-1
    M3 语义 + 阈值    置信度不足以放行
    M4 候选集可达性   在 K̂（标注 ∪ 语义 top-k）上取「任一作用域匹配」即放行 —— 本文
    M5 M4 + 挂载核验  对 K̂ 全不匹配者发起一次挂载查询（返回 K(q)），预算 B
    M6 上界           直接用 K(q)

指标
    CSCR@10  展示结果中对本人无任何合法挂载的比例（越界）
    FRR      合法且相关的候选中被拒的比例（误拒）
    EC       有效覆盖率
    verify   每生核验次数
    MMR      多挂载题上的判定正确率（判定「是否至少有一个合法挂载」）
"""
import os, csv, json, math, random, argparse
from collections import defaultdict

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data", "dnui_ikg")
LAYER = os.path.join(DATA, "resource_layer.json")
OUT = os.environ.get("SCOPE2_OUT", os.path.join(REPO, "results", "scope_decision_v2.json"))

K = 10
VERIFY_BUDGET = 3
TOPK = 3
TAU_COV = 0.5      # 先修可达度下限：低于此值的语义候选不参与判定
TAU_SEM = 0.45     # 语义置信度下限：无标注且无核验时，低于此值不放行
HOMONYM = {}       # 同名跨课知识点：name -> [kt_id]
EQUIV = {}         # 挂载等价关系：kt_id -> {kt_id}（同名 ∪ 跨课名称相似）


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load():
    kts = {}
    for r in read_csv("kts.csv"):
        kts[int(r["kt_id"])] = dict(
            name=r["kt_name"].strip(), course=r["course_id"],
            grades={int(x) for x in r["grade_scope"].split("|") if x.strip()},
            majors={x.strip() for x in r["major_scope"].split("|") if x.strip()},
            n_inter=int(r["n_interactions"] or 0))
    qid2kt = {r["q_id"]: int(r["kt_id"]) for r in read_csv("questions.csv")}
    stem = {r["q_id"]: (r["stem"] or "").strip() for r in read_csv("questions.csv")}
    stu = {int(r["student_id"]): dict(grade=int(r["grade"]), major=r["major"].strip())
           for r in read_csv("students.csv") if int(r["has_interaction"] or 0) == 1}
    hist = defaultdict(list)
    acc = defaultdict(lambda: [0, 0])          # (sid, kt) -> [答对数, 题数]
    for r in read_csv("interactions.csv"):
        sid = int(r["student_id"])
        if sid in stu:
            hist[sid].append(r["test_id"])
            a = acc[(sid, int(r["kt_id"]))]
            a[0] += int(r["correct"])
            a[1] += 1
    with open(LAYER, encoding="utf-8") as f:
        layer = json.load(f)
    # qid -> 该题目实体的全部挂载 kt
    qid2mounts = {}
    for it in layer["items"]:
        ks = [m["kt_id"] for m in it["mounts"]]
        for q in it["qids"]:
            qid2mounts[q] = ks
    closure = {int(k): set(v) for k, v in layer["closure"].items()}
    return kts, qid2kt, stem, stu, hist, qid2mounts, closure, acc


POLICY = os.environ.get("SCOPE_POLICY", "down")   # down=式(10)单侧不等式；eq=集合等号（旧口径）


def ok_scope(s, kt):
    """作用域判定，必须与论文式 (10) 同一政策。

    down（默认）：学生年级 ≥ 知识点最低适用学段，即允许向下兼容；
    eq          ：学生年级属于知识点学段集合（本数据单值，等价于要求年级相等）。
    """
    if POLICY == "down":
        ok_g = bool(kt["grades"]) and s["grade"] >= min(kt["grades"])
    else:
        ok_g = s["grade"] in kt["grades"]
    return ok_g and s["major"] in kt["majors"]


def reach(sid, k, closure, acc, theta=0.6):
    """先修可达度：闭包内已掌握（练过且正确率≥theta）的比例；闭包为空视为可达"""
    P = closure.get(k) or set()
    if not P:
        return 1.0
    ok = 0
    for p in P:
        c, n = acc.get((sid, p), (0, 0))
        if n > 0 and c / n >= theta:
            ok += 1
    return ok / len(P)


def legal(s, qid, kts, qid2mounts):
    """真值：至少存在一个合法挂载"""
    return any(ok_scope(s, kts[k]) for k in qid2mounts.get(qid, []))


# ---------------------------------------------------------------- 语义
def bigrams(text):
    t = "".join(ch for ch in text if ch.strip())
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}


def build_semantic(stem, qid2kt, tagged):
    docs = {q: bigrams(t) for q, t in stem.items()}
    df = {}
    for g in docs.values():
        for x in g:
            df[x] = df.get(x, 0) + 1
    N = len(docs)
    idf = {x: math.log((N + 1) / (c + 1)) + 1.0 for x, c in df.items()}
    vec = {}
    for q, g in docs.items():
        v = {x: idf[x] for x in g}
        n = math.sqrt(sum(w * w for w in v.values())) or 1.0
        vec[q] = {x: w / n for x, w in v.items()}
    centers = defaultdict(dict)
    for q, k in qid2kt.items():
        if not tagged.get(q, True):
            continue
        for x, w in vec[q].items():
            centers[k][x] = centers[k].get(x, 0.0) + w
    for k, v in centers.items():
        n = math.sqrt(sum(w * w for w in v.values())) or 1.0
        centers[k] = {x: w / n for x, w in v.items()}
    out = {}
    for q, v in vec.items():
        sc = []
        for k, c in centers.items():
            a, b = (v, c) if len(v) < len(c) else (c, v)
            s = sum(w * b.get(x, 0.0) for x, w in a.items())
            if s > 0:
                sc.append((s, k))
        sc.sort(key=lambda t: -t[0])
        out[q] = sc[:TOPK]
    return out


# ---------------------------------------------------------------- 评估集
def build_rows(hist, stu, cut1=0.6, cut2=0.8, min_len=20):
    rows = []
    for sid, seq in hist.items():
        if len(seq) < min_len:
            continue
        a, b = int(len(seq) * cut1), int(len(seq) * cut2)
        seen = set(seq[:a])
        rel = set(seq[b:]) - seen
        if len(rel) < 3:
            continue
        rows.append(dict(sid=sid, seen=seen, rel=rel,
                         grade=stu[sid]["grade"], major=stu[sid]["major"]))
    rows.sort(key=lambda r: r["sid"])
    return rows


def ranker(rows, qid2kt, kts):
    out = {}
    for s in rows:
        lst = []
        for q, k in qid2kt.items():
            if q in s["seen"]:
                continue
            kt = kts[k]
            sc = math.log1p(kt["n_inter"]) + (0.8 if s["grade"] in kt["grades"] else 0.0)
            lst.append((sc, q))
        lst.sort(key=lambda t: -t[0])
        out[s["sid"]] = [q for _, q in lst]
    return out


# ---------------------------------------------------------------- 门控
def cand_set(s, qid, tagged, sem, qid2kt, closure, acc, tau_cov=0.5):
    """K̂：标注项无条件保留；语义候选须过先修可达性筛（这是与全放行版本的关键区别）"""
    cs = []
    if tagged.get(qid, True):
        cs.append(qid2kt[qid])
    for _, k in (sem.get(qid) or []):
        if k in cs:
            continue
        if reach(s["sid"], k, closure, acc) >= tau_cov:
            cs.append(k)
    return cs


def gate(mode, s, qid, kts, qid2kt, tagged, sem, qid2mounts, st, closure, acc, tau=0.5):
    if mode == "meta":
        if not tagged.get(qid, True):
            return 0
        return 1 if ok_scope(s, kts[qid2kt[qid]]) else 0
    if mode in ("share", "share2"):
        # 挂载等价推断：标注不匹配时，若等价知识点可学，视为该题的另一条挂载
        # share  仅用同名关系
        # share2 同名 ∪ 跨课程名称相似（bigram Jaccard ≥ 0.4）
        if tagged.get(qid, True):
            k = qid2kt[qid]
            if ok_scope(s, kts[k]):
                return 1
            pool = HOMONYM.get(kts[k]["name"], []) if mode == "share" else EQUIV.get(k, ())
            for k2 in pool:
                if k2 != k and ok_scope(s, kts[k2]):
                    return 1
            return 0
        cands = sem.get(qid) or []
        if not cands:
            return 0
        k = cands[0][1]
        if ok_scope(s, kts[k]):
            return 1
        pool = HOMONYM.get(kts[k]["name"], []) if mode == "share" else EQUIV.get(k, ())
        for k2 in pool:
            if k2 != k and ok_scope(s, kts[k2]):
                return 1
        return 0
    if mode == "sem1":
        cs = cand_set(s, qid, tagged, sem, qid2kt, closure, acc)[:1]
        return 1 if cs and ok_scope(s, kts[cs[0]]) else 0
    if mode == "conf":
        cs = cand_set(s, qid, tagged, sem, qid2kt, closure, acc)
        if not cs:
            return 0
        p = 1.0 if ok_scope(s, kts[cs[0]]) else 0.0
        sc = sem.get(qid) or []
        if sc:
            p = max(p, sc[0][0])
        return 1 if p >= tau else 0
    if mode in ("ours", "ours_verify"):
        can_verify = (mode == "ours_verify")
        if tagged.get(qid, True):
            k = qid2kt[qid]
            if ok_scope(s, kts[k]):
                return 1
            # 标注不匹配不等于终点：该题可能另有挂载（标注不完整的真实形态）
            if can_verify and st["used"] < VERIFY_BUDGET and len(qid2mounts.get(qid, [])) > 1:
                st["used"] += 1
                return 1 if legal(s, qid, kts, qid2mounts) else 0
            return 0
        # 无标注：在语义 top-k 中取「先修可达且置信度最高」的一个候选，仍做单点判定
        cands = [(sc, k) for sc, k in (sem.get(qid) or [])
                 if reach(s["sid"], k, closure, acc) >= TAU_COV]
        if not cands:
            return 0
        sc, k = cands[0]
        if ok_scope(s, kts[k]):
            return 1 if sc >= TAU_SEM else 0
        if can_verify and st["used"] < VERIFY_BUDGET:
            st["used"] += 1
            return 1 if legal(s, qid, kts, qid2mounts) else 0
        return 0
    if mode == "union":
        # 同候选预算基线：在与本文完全相同的 3 个候选上「任一作用域匹配即放行」，
        # 不引入等价关系。用于证明等价挂载集的收益来自结构关系而非候选预算本身。
        cands = cand_set(s, qid, tagged, sem, qid2kt, closure, acc)[:VERIFY_BUDGET]
        return 1 if any(ok_scope(s, kts[k]) for k in cands) else 0
    if mode == "oracle":
        return 1 if legal(s, qid, kts, qid2mounts) else 0
    raise ValueError(mode)


def evaluate(rows, kts, qid2kt, tagged, sem, qid2mounts, mode, ranks, closure, acc, tau=0.5):
    """除展示层指标外，另在全候选池上统计门控混淆矩阵，避免排序器的学段加权掩盖门控质量"""
    n = 0; bad = 0; shown_ok = 0; shown_den = 0
    denied_pos = 0; pool_pos = 0; verify = 0
    mm_tot = 0; mm_ok = 0
    tp = fp = tn = fn = 0
    for s in rows:
        st = dict(used=0)
        allow = {}
        cand = ranks[s["sid"]]
        for qid in cand:
            allow[qid] = gate(mode, s, qid, kts, qid2kt, tagged, sem, qid2mounts, st, closure, acc, tau)
        verify += st["used"]

        pos = [q for q in s["rel"] if legal(s, q, kts, qid2mounts)]
        pool_pos += len(pos)
        denied_pos += sum(1 for q in pos if not allow.get(q, 0))

        # 全池混淆矩阵：门控决策 vs 真值
        for qid in s["rel"]:
            lg = legal(s, qid, kts, qid2mounts)
            al = bool(allow.get(qid, 0))
            if lg and al:
                tp += 1
            elif lg and not al:
                fn += 1
            elif not lg and al:
                fp += 1
            else:
                tn += 1

        shown = [q for q in cand if allow.get(q)][:K]
        lr = set(pos)
        shown_ok += sum(1 for q in shown if q in lr)
        shown_den += min(K, len(lr))
        bad += sum(1 for q in shown if not legal(s, q, kts, qid2mounts))
        n += 1

        for q in s["rel"]:
            if len(qid2mounts.get(q, [])) > 1:
                mm_tot += 1
                mm_ok += (bool(allow.get(q, 0)) == legal(s, q, kts, qid2mounts))

    return dict(
        n=n,
        CSCR=round(bad / (n * K), 4) if n else 0.0,
        FRR=round(denied_pos / pool_pos, 4) if pool_pos else 0.0,
        EC=round(shown_ok / shown_den, 4) if shown_den else 0.0,
        FAR=round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        GPrec=round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        GRec=round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        verify_per_student=round(verify / n, 2) if n else 0.0,
        pos_per_student=round(pool_pos / n, 2) if pool_pos else 0.0,
        MMR=round(mm_ok / mm_tot, 4) if mm_tot else 0.0,
        MM_n=mm_tot,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", default="0,0.1,0.2,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--students", type=int, default=400)
    ap.add_argument("--eq_th", type=float, default=0.4,
                    help="异名跨课挂载等价的名称 bigram Jaccard 阈值")
    args = ap.parse_args()

    kts, qid2kt, stem, stu, hist, qid2mounts, closure, acc = load()
    global HOMONYM, EQUIV
    _by = defaultdict(list)
    for k, v in kts.items():
        _by[v["name"]].append(k)
    HOMONYM = {n: ks for n, ks in _by.items() if len(ks) > 1}
    # 等价挂载关系：同名（跨课程）∪ 跨课程且名称 bigram Jaccard ≥ 阈值
    EQ_TH = args.eq_th
    _eq = defaultdict(set)
    _ks = sorted(kts)
    for i, a in enumerate(_ks):
        for b in _ks[i + 1:]:
            A, B = bigrams(kts[a]["name"]), bigrams(kts[b]["name"])
            j = len(A & B) / len(A | B) if (A | B) else 0.0
            same = kts[a]["name"] == kts[b]["name"]
            cross = kts[a]["course"] != kts[b]["course"]
            if same or (cross and j >= EQ_TH):
                _eq[a].add(b)
                _eq[b].add(a)
    EQUIV = dict(_eq)
    _cross = [(a, b) for a in EQUIV for b in EQUIV[a]
              if kts[a]["course"] != kts[b]["course"] and kts[a]["name"] != kts[b]["name"]]
    print("同名跨课组 %d: %s" % (len(HOMONYM), list(HOMONYM)[:3]))
    print("等价挂载关系 %d 条 kt 有邻居，其中异名跨课 %d 条: %s"
          % (len(EQUIV), len(_cross) // 2,
             [(a, b) for a, b in _cross][:4]))
    rows_all = build_rows(hist, stu)
    rng0 = random.Random(SEED)
    rng0.shuffle(rows_all)
    rows = rows_all[:args.students]
    ranks = ranker(rows, qid2kt, kts)
    n_multi = sum(1 for v in qid2mounts.values() if len(v) > 1)
    print("学生 %d / 题目 %d / 多挂载题 %d" % (len(rows), len(qid2kt), n_multi))

    rho_list = [float(x) for x in args.rho.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = ("meta", "share", "share2", "sem1", "conf", "union",
               "ours", "ours_verify", "oracle")
    res = {str(r): {m: [] for m in methods} for r in rho_list}

    for rho in rho_list:
        for seed in seeds:
            rng = random.Random(seed)
            tagged = {q: rng.random() >= rho for q in qid2kt}
            sem = build_semantic(stem, qid2kt, tagged)
            base = evaluate(rows, kts, qid2kt, tagged, sem, qid2mounts, "ours", ranks, closure, acc)
            best_tau, best_gap = 0.5, 9.9
            for tau in [i / 100 for i in range(5, 96, 5)]:
                r3 = evaluate(rows, kts, qid2kt, tagged, sem, qid2mounts, "conf", ranks, closure, acc, tau)
                gap = abs(r3["CSCR"] - base["CSCR"])
                if gap < best_gap:
                    best_tau, best_gap = tau, gap
            res[str(rho)]["conf"].append(
                evaluate(rows, kts, qid2kt, tagged, sem, qid2mounts, "conf", ranks, closure, acc, best_tau) | {"tau": best_tau})
            for m in ("meta", "share", "share2", "sem1", "union",
                      "ours", "ours_verify", "oracle"):
                res[str(rho)][m].append(
                    evaluate(rows, kts, qid2kt, tagged, sem, qid2mounts, m, ranks, closure, acc))
        print("rho=%.1f 完成" % rho)

    summ = {}
    for r, modes in res.items():
        summ[r] = {m: {k: round(sum(x[k] for x in lst) / len(lst), 4) for k in lst[0]}
                   for m, lst in modes.items()}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(K=K, topk=TOPK, verify_budget=VERIFY_BUDGET, n_students=len(rows),
                       seeds=seeds, rho_list=rho_list, summary=summ, raw=res),
                  f, ensure_ascii=False, indent=1)

    print("\nrho  方法        CSCR    FRR     FAR    GPrec  GRec   EC     verify  MMR")
    for r in sorted(summ, key=float):
        for m in methods:
            a = summ[r][m]
            print("%-4s %-11s %.4f  %.4f  %.4f  %.4f %.4f %.4f  %5.2f  %.4f"
                  % (r, m, a["CSCR"], a["FRR"], a["FAR"], a["GPrec"], a["GRec"],
                     a["EC"], a["verify_per_student"], a["MMR"]))
        print()
    print("写入", OUT)


if __name__ == "__main__":
    main()
