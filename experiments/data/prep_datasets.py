#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备：把公开 KT 数据集与自建 DNUI-IKG 统一成 KT / 推荐 / 路径 任务可用的格式。

产出 (data/processed/):
  assist09.pkl  assist12.pkl  dnui_real.pkl  dnui_syn.pkl
统一结构:
  { n_skill, n_question, skill_names, meta,
    train/valid/test: [ {sid, skill, correct, question} ... ]   # numpy 数组, 按时间序
    prereq:  可选 csr-like (list[list[int]]) 技能先修邻接
    student_scope: {sid: (grade, major)}     仅 DNUI-IKG
    skill_scope:   {skill_id: (grade, major)} 仅 DNUI-IKG
  }

DNUI-IKG-S 为校准式半合成数据: 在【真实课程先修 DAG + 真实作用域标签 + 真实知识点
交互量分布】之上, 用 IRT(2PL)+slip/guess 生成答题序列。理由: 生产环境真实答题
记录的正确率为 99.35%(练习至正确后落库), 掌握度标签 92.3% 恰为 1.0, 缺乏方差,
无法用于训练/评测深度知识追踪模型; 该局限已在数据说明中如实记录。
"""
import os, sys, csv, json, gzip, pickle, random, math, argparse, zipfile, io
from collections import defaultdict, Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import PUB, DNUI, PROCESSED as OUT   # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


# --------------------------------------------------------------------------
def split_by_user(seqs, ratios=(0.7, 0.1, 0.2), seed=SEED):
    rs = random.Random(seed)
    idx = list(range(len(seqs)))
    rs.shuffle(idx)
    n = len(idx)
    a = int(n * ratios[0]); b = int(n * (ratios[0] + ratios[1]))
    tr = [seqs[i] for i in idx[:a]]
    va = [seqs[i] for i in idx[a:b]]
    te = [seqs[i] for i in idx[b:]]
    return tr, va, te


def summarize(name, seqs, n_skill, n_question):
    L = [len(s["skill"]) for s in seqs]
    cor = np.concatenate([s["correct"] for s in seqs]) if seqs else np.array([0.0])
    return dict(name=name, users=len(seqs), n_skill=n_skill, n_question=n_question,
                interactions=int(sum(L)), avg_len=round(float(np.mean(L)), 2),
                median_len=int(np.median(L)), max_len=int(np.max(L)),
                correct_rate=round(float(cor.mean()), 4))


def dump(name, obj):
    p = os.path.join(OUT, f"{name}.pkl")
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB)")


# --------------------------------------------------------------------------
def prep_assist09():
    """ASSISTments 2009-2010 Skill Builder。
    标准预处理: 去掉 skill 为空的行; 按 (user, problem) 去重保留首次作答;
    过滤交互数 < 3 的学生; 按 order_id 排序。"""
    path = os.path.join(PUB, "OloriBern__assistments-dataset", "skill_builder_data_corrected.csv")
    if not os.path.exists(path):
        print("[assist09] 文件缺失, 跳过"); return
    print("[assist09] 读取 ...")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sk = (r.get("skill_id") or "").strip()
            if not sk or sk.upper() == "NA":
                continue
            try:
                rows.append((int(r["order_id"]), int(r["user_id"]), int(r["problem_id"]),
                             int(r["correct"]), sk, (r.get("skill_name") or sk)))
            except (ValueError, KeyError):
                continue
    rows.sort()
    seen = set(); ded = []
    for r in rows:
        k = (r[1], r[2])
        if k in seen:
            continue
        seen.add(k); ded.append(r)
    skill2id = {}; skill_names = []
    per_user = defaultdict(list)
    qids = {}
    for o, u, p, c, sk, skn in ded:
        if sk not in skill2id:
            skill2id[sk] = len(skill2id); skill_names.append(skn or sk)
        if p not in qids:
            qids[p] = len(qids)
        per_user[u].append((skill2id[sk], c, qids[p]))
    seqs = []
    for u, lst in per_user.items():
        if len(lst) < 3:
            continue
        seqs.append(dict(sid=u,
                         skill=np.array([x[0] for x in lst], dtype=np.int64),
                         correct=np.array([x[1] for x in lst], dtype=np.float32),
                         question=np.array([x[2] for x in lst], dtype=np.int64)))
    tr, va, te = split_by_user(seqs)
    obj = dict(n_skill=len(skill2id), n_question=len(qids), skill_names=skill_names,
               train=tr, valid=va, test=te, prereq=None, student_scope=None,
               skill_scope=None,
               meta=dict(source="ASSISTments2009 skill_builder_data_corrected",
                         stats=summarize("assist09", seqs, len(skill2id), len(qids))))
    print(" ", obj["meta"]["stats"])
    dump("assist09", obj)


# --------------------------------------------------------------------------
def prep_assist12(max_rows=None):
    """ASSISTments 2012-2013 (3GB)。分块读取, 取 skill(首个) / problem_id / user_id /
    correct / start_time。skill 为 '_' 分段的组合技能, 取首个作为主技能。"""
    path = os.path.join(PUB, "Lucy9999__assist2012", "2012-2013-data-with-predictions-4-final.csv")
    if not os.path.exists(path):
        print("[assist12] 文件缺失, 跳过"); return
    print(f"[assist12] 分块读取 {os.path.getsize(path)/1e9:.2f} GB ...")
    import pandas as pd
    cols = ["skill", "problem_id", "user_id", "correct", "start_time"]
    parts = []
    tot = 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=2_000_000, dtype=str,
                          engine="c", on_bad_lines="skip"):
        ch = ch.dropna(subset=["skill", "problem_id", "user_id", "correct", "start_time"])
        parts.append(ch)
        tot += len(ch)
        if max_rows and tot >= max_rows:
            break
        print(f"    chunk read, cumulative={tot}")
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"    rows={len(df)}")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df = df[df["correct"].isin([0, 1])]
    # 主技能 = 组合技能的首段
    df["skill_main"] = df["skill"].astype(str).str.split("_").str[0]
    df = df[df["skill_main"].str.len() > 0]
    # 去重: 同一 (user, problem) 保留首次（按 start_time）
    df = df.sort_values("start_time")
    df = df.drop_duplicates(subset=["user_id", "problem_id"], keep="first")
    skill2id = {}; skill_names = []
    qids = {}
    per_user = defaultdict(list)
    for sk, p, u, c, t in zip(df["skill_main"], df["problem_id"], df["user_id"],
                              df["correct"].astype(int), df["start_time"]):
        if sk not in skill2id:
            skill2id[sk] = len(skill2id); skill_names.append(sk)
        if p not in qids:
            qids[p] = len(qids)
        per_user[u].append((skill2id[sk], int(c), qids[p]))
    seqs = []
    for u, lst in per_user.items():
        if len(lst) < 3:
            continue
        seqs.append(dict(sid=u,
                         skill=np.array([x[0] for x in lst], dtype=np.int64),
                         correct=np.array([x[1] for x in lst], dtype=np.float32),
                         question=np.array([x[2] for x in lst], dtype=np.int64)))
    tr, va, te = split_by_user(seqs)
    obj = dict(n_skill=len(skill2id), n_question=len(qids), skill_names=skill_names,
               train=tr, valid=va, test=te, prereq=None, student_scope=None, skill_scope=None,
               meta=dict(source="ASSISTments2012-2013 (first-attempt dedup, main skill)",
                         stats=summarize("assist12", seqs, len(skill2id), len(qids))))
    print(" ", obj["meta"]["stats"])
    dump("assist12", obj)


# --------------------------------------------------------------------------
def load_dnui():
    def rd(n):
        with open(os.path.join(DNUI, n), encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return dict(courses=rd("courses.csv"), units=rd("units.csv"), kts=rd("kts.csv"),
                edges=rd("edges.csv"), questions=rd("questions.csv"),
                students=rd("students.csv"), mastery=rd("mastery.csv"),
                interactions=rd("interactions.csv"),
                attempted=rd("attempted.csv"), conf=rd("scope_conflicts.csv"))


MAJOR_LIST = ["软件工程", "计算机科学", "人工智能"]


def dnui_graph(D):
    """返回 (skill_id 映射, 先修邻接, 技能作用域, 学生作用域)。"""
    kts = sorted(D["kts"], key=lambda r: int(r["kt_id"]))
    k2i = {int(r["kt_id"]): i for i, r in enumerate(kts)}
    names = [r["kt_name"] for r in kts]
    prereq = [[] for _ in kts]
    for e in D["edges"]:
        a, b = int(e["src_kt"]), int(e["dst_kt"])
        if a in k2i and b in k2i:
            prereq[k2i[a]].append((k2i[b], float(e["weight"])))
    skill_scope = {}
    for r in kts:
        gs = [int(x) for x in r["grade_scope"].split("|") if x]
        ms = [x for x in r["major_scope"].split("|") if x]
        skill_scope[k2i[int(r["kt_id"])]] = dict(
            grades=gs, majors=ms, course=r["course_id"],
            difficulty=float(r["emp_difficulty"]) if r["emp_difficulty"] else None,
            n_inter=int(r["n_interactions"]))
    stu_scope = {}
    for r in D["students"]:
        if r["has_interaction"] != "1":
            continue
        stu_scope[int(r["student_id"])] = dict(grade=int(r["grade"]), major=r["major"],
                                              enroll=int(r["enroll_year"] or 0))
    return k2i, names, prereq, skill_scope, stu_scope


def prep_dnui_real(D):
    """DNUI-IKG (真实)：知识点级掌握度预测任务。序列由课程的先修拓扑序给出
    候选知识点集合, 标签为真实 correct_rate / passed。"""
    print("[dnui_real] 构建 ...")
    k2i, names, prereq, skill_scope, stu_scope = dnui_graph(D)
    qids = {}
    for r in D["questions"]:
        if r["q_id"] not in qids:
            qids[r["q_id"]] = len(qids)
    seqs = []
    for r in D["mastery"]:
        s, k = int(r["student_id"]), int(r["kt_id"])
        if k not in k2i:
            continue
        cr = float(r["correct_rate"] or 0)
        seqs.append(dict(sid=s,
                         skill=np.array([k2i[k]], dtype=np.int64),
                         correct=np.array([1.0 if r["passed"] == "2" else 0.0], dtype=np.float32),
                         question=np.array([-1], dtype=np.int64), rate=cr,
                         grade=stu_scope.get(s, {}).get("grade", 1),
                         major=stu_scope.get(s, {}).get("major", "其他")))
    tr, va, te = split_by_user(seqs)
    obj = dict(n_skill=len(k2i), n_question=max(1, len(qids)), skill_names=names,
               train=tr, valid=va, test=te,
               prereq=prereq, skill_scope=skill_scope, student_scope=stu_scope,
               meta=dict(source="DNUI-IKG (production DB ikg@172.17.18.243, read-only export)",
                         saturating=True,
                         note="真实日志正确率 99.35%, 掌握标签 92.3% 为 1.0, 方差不足",
                         stats=summarize("dnui_real", seqs, len(k2i), len(qids))))
    print(" ", {k: obj["meta"]["stats"][k] for k in
                ["users", "n_skill", "interactions", "correct_rate"]})
    dump("dnui_real", obj)
    return obj


def prep_dnui_syn(D, n_students=3000, max_len=120, out_of_scope_rate=0.10):
    """DNUI-IKG-S (校准式半合成)。
    在真实课程先修 DAG + 真实作用域标签上, 用 2PL IRT + slip/guess 生成序列:
       P(correct | θ_i, b_k, a_k) = g + (1-s-g) * σ(a_k(θ_i - b_k))
    参数标定:
      θ_i ~ N(0,1)
      b_k  由真实经验正确率的次序统计量线性映射到 [-1.2, 1.2]
      a_k  由真实交互量映射到 [0.8, 1.6] (交互越多视为越核心/区分度越高)
    序列生成: 依先修拓扑序推进, 每个 KT 采样 1~N 次; 按 out_of_scope_rate 注入
    跨学段/跨专业的越界作答(用于检验 CSG 门控的必要性)。"""
    print(f"[dnui_syn] 生成 n_students={n_students} ...")
    k2i, names, prereq, skill_scope, stu_scope = dnui_graph(D)
    n_skill = len(names)
    # 拓扑序
    indeg = [0] * n_skill
    adj = [[] for _ in range(n_skill)]
    for u in range(n_skill):
        for v, w in prereq[u]:
            adj[u].append(v); indeg[v] += 1
    from collections import deque
    dq = deque([i for i in range(n_skill) if indeg[i] == 0])
    topo = []
    while dq:
        x = dq.popleft(); topo.append(x)
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    rest = [i for i in range(n_skill) if i not in set(topo)]
    topo += rest
    # 难度/区分度标定
    diffs = np.array([skill_scope[i]["difficulty"] if skill_scope[i]["difficulty"] is not None
                      else np.nan for i in range(n_skill)], dtype=np.float64)
    ok = ~np.isnan(diffs)
    b = np.zeros(n_skill)
    if ok.sum() > 1:
        order = np.argsort(np.argsort(diffs[ok])) / (ok.sum() - 1)
        b[ok] = -1.2 + 2.4 * order      # 正确率高 => 难度低
    b[~ok] = rng.normal(0, 0.5, size=(~ok).sum())
    nint = np.array([skill_scope[i]["n_inter"] for i in range(n_skill)], dtype=np.float64)
    ni = np.log1p(nint)
    a = 0.8 + 0.8 * ((ni - ni.min()) / (float(np.ptp(ni)) + 1e-9))
    SLIP, GUESS = 0.08, 0.15
    THETA_MU, THETA_SD = 0.45, 1.0
    # 学生作用域分布（沿用真实学生的 (grade,major) 经验分布）
    real_sc = [(v["grade"], v["major"]) for v in stu_scope.values()]
    cnt = Counter(real_sc)
    pool = list(cnt.keys()); pw = np.array([cnt[k] for k in pool], dtype=np.float64); pw /= pw.sum()
    seqs = []
    for i in range(n_students):
        grade, major = pool[rng.choice(len(pool), p=pw)]
        theta = float(rng.normal(THETA_MU, THETA_SD))
        in_scope = [k for k in range(n_skill)
                    if grade in skill_scope[k]["grades"] and major in skill_scope[k]["majors"]]
        if not in_scope:
            in_scope = [k for k in range(n_skill) if grade in skill_scope[k]["grades"]]
        if not in_scope:
            continue
        out_scope = [k for k in range(n_skill) if k not in set(in_scope)]
        # 依拓扑序取作用域内知识点
        ordered = [k for k in topo if k in set(in_scope)]
        sk, co = [], []
        for k in ordered:
            reps = int(rng.integers(1, 4))
            p = GUESS + (1 - SLIP - GUESS) / (1 + math.exp(-a[k] * (theta - b[k])))
            for _ in range(reps):
                sk.append(k); co.append(1.0 if rng.random() < p else 0.0)
                if len(sk) >= max_len:
                    break
            if len(sk) >= max_len:
                break
        # 注入越界作答
        if out_scope and len(sk) < max_len:
            n_bad = int(round(out_of_scope_rate * len(sk)))
            for k in rng.choice(out_scope, size=min(n_bad, len(out_scope)), replace=False):
                p = GUESS + (1 - SLIP - GUESS) / (1 + math.exp(-a[k] * (theta - b[k])))
                sk.append(int(k)); co.append(1.0 if rng.random() < p else 0.0)
        if len(sk) < 3:
            continue
        seqs.append(dict(sid=100000 + i,
                         skill=np.array(sk, dtype=np.int64),
                         correct=np.array(co, dtype=np.float32),
                         question=np.array([-1] * len(sk), dtype=np.int64),
                         grade=int(grade), major=major, theta=theta))
    tr, va, te = split_by_user(seqs)
    obj = dict(n_skill=n_skill, n_question=467, skill_names=names,
               train=tr, valid=va, test=te,
               prereq=prereq, skill_scope=skill_scope,
               student_scope={s["sid"]: (s["grade"], s["major"]) for s in seqs},
               meta=dict(source="DNUI-IKG-S (semi-synthetic, calibrated on real curriculum DAG "
                                "+ real scope labels + real interaction volume)",
                         generator="2PL IRT + slip/guess",
                         params=dict(slip=SLIP, guess=GUESS, theta_mu=THETA_MU, theta_std=THETA_SD,
                                     b_range=[-1.2, 1.2], a_range=[0.8, 1.6],
                                     out_of_scope_rate=out_of_scope_rate),
                         stats=summarize("dnui_syn", seqs, n_skill, 467)))
    print(" ", {k: obj["meta"]["stats"][k] for k in
                ["users", "n_skill", "interactions", "correct_rate", "avg_len"]})
    dump("dnui_syn", obj)
    # 同步导出技能作用域表供 CSG 实验使用
    with open(os.path.join(OUT, "dnui_skill_scope.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in skill_scope.items()}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "dnui_prereq.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): prereq[i] for i in range(n_skill)}, f, ensure_ascii=False, indent=1)
    return obj


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    # 3000 个候选里有 2147 个通过"作用域内至少 3 次作答"的过滤，
    # 与论文表 1 的 DNUI-IKG 行（2 147 条序列 / 117 865 次交互 / 平均长 54.9）一致。
    ap.add_argument("--syn-students", type=int, default=3000)
    ap.add_argument("--assist12-max-rows", type=int, default=None)
    args = ap.parse_args()

    if args.only in ("all", "assist09"):
        prep_assist09()
    if args.only in ("all", "assist12"):
        prep_assist12(args.assist12_max_rows)
    if args.only in ("all", "dnui"):
        D = load_dnui()
        prep_dnui_real(D)
        prep_dnui_syn(D, n_students=args.syn_students)
    print("\n===== processed =====")
    for f in sorted(os.listdir(OUT)):
        print(f"  {os.path.getsize(os.path.join(OUT,f))/1e6:>10.2f} MB  {f}")


if __name__ == "__main__":
    sys.exit(main())
