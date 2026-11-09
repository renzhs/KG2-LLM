#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 AKT 官方 5 折划分（CSV）转换为本项目统一的 .pkl schema。

输入: data/public/AKT_repo/data/{repo_dir}/{prefix}_{train,valid,test}{k}.csv  (k=1..5)
输出: experiments/data/processed/{name}_f{k}.pkl

AKT 数据格式（**无表头**，按学生分组，每组 3 行或 4 行）：
  DATA     (3 行)：  id[,true_student_id] / skill_seq / correct_seq
  PID_DATA (4 行)：  id[,true_student_id] / problem_seq / skill_seq / correct_seq
  依据 arghosh/AKT 的 load_data.py（DATA / PID_DATA 两个类）。
  · assist2009_pid、assist2017_pid：4 行（题目与技能分开）
  · assist2015、statics2011：3 行（只有技能/题目同一列）
本脚本用「行首计数自增」自动判别分组大小，不依赖硬编码。

统一 schema（与 prep_datasets.py 一致，故 run_kt.py / run_rec.py 无需改动）:
  { n_skill, n_question, skill_names, fold, meta,
    train/valid/test: [ {sid, skill, correct, question} ... ],
    prereq: list[list[(dst_skill, weight)]]   # 公开数据集无先修图，由 kg_induct 归纳
    student_scope / skill_scope: None          # 公开数据集无学段-专业标注
  }

为什么沿用 AKT 划分：AKT(2020)、DIMKT(KDD2022)、SimpleKT(ACL2023) 等论文均在
该仓库发布的 5 折划分上报告 AUC，沿用同一划分才能与已发表数字直接对比。
"""
import argparse
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
from paths import PROCESSED as OUT                       # noqa: E402
from models.kg_induct import induce_prereq, dag_report    # noqa: E402

AKT_ROOT = os.path.join(os.path.dirname(EXP), "data", "public", "AKT_repo", "data")

# 本地数据集名 -> (仓库内目录, 文件名前缀)；statics 目录下前缀是 "statics"
DATASETS = {
    "assist09_akt": ("assist2009_pid", "assist2009_pid"),
    "assist15": ("assist2015", "assist2015"),
    "assist17": ("assist2017_pid", "assist2017_pid"),
    "statics11": ("statics2011", "statics"),
}
FOLDS = (1, 2, 3, 4, 5)

IDX_RE = re.compile(r"^(\d+)(,\d+)?$")


def _split_ints(line):
    out = []
    for tok in line.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            out.append(int(float(tok)))
        except ValueError:
            pass
    return out


def _is_binary_line(line):
    """判断一行是否「只由 0/1 组成」——用于识别 correct 行。"""
    vals = _split_ints(line)
    if not vals:
        return False
    return all(v in (0, 1) for v in vals)


def _detect_group_size(raw):
    """自动判别每个学生占 3 行（DATA）还是 4 行（PID_DATA）。

    判据：correct 行一定是 0/1 序列。对候选 gsize∈{3,4}，检查所有
    index%gsize==gsize-1 的行是否 0/1；同时 id 行字段数应 ≤2。
    取得分更高者。这比硬编码更稳（statics 是 3 行、assist*_pid 是 4 行）。
    """
    n = len(raw)
    best, best_score = 3, -1.0
    for g in (3, 4):
        if n < g:
            continue
        idx = list(range(g - 1, n, g))
        if not idx:
            continue
        score = sum(1 for i in idx if _is_binary_line(raw[i])) / len(idx)
        # id 行字段数惩罚（正确格式下 id 行只有 1~2 个字段）
        ididx = list(range(0, n, g))
        ok_id = sum(1 for i in ididx if len(_split_ints(raw[i])) <= 2) / max(1, len(ididx))
        score = score * 0.8 + ok_id * 0.2
        # 无法整除时轻微惩罚
        if n % g:
            score -= 0.1
        if score > best_score:
            best, best_score = g, score
    return best, best_score


def parse_akt_file(path):
    """解析 AKT 分组格式，返回 ({sid: (skill_ids, problem_ids, corrects)}, info)。

    groupsize=4 (PID_DATA): id / problem / skill / correct
    groupsize=3 (DATA)    : id / skill(=question) / correct
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = [ln.strip() for ln in f if ln.strip() != ""]
    gsize, score = _detect_group_size(raw)
    n_full = len(raw) // gsize
    per = {}
    for j in range(n_full):
        g = raw[j * gsize:(j + 1) * gsize]
        if len(g) != gsize:
            break
        # sid 必须唯一：部分数据集（如 statics2011）首行 id 会重复，故追加组序号
        sid = f"{os.path.basename(path)}#{j}#{g[0]}"
        if gsize == 4:                      # PID_DATA: id / problem / skill / correct
            prob, skill, corr = _split_ints(g[1]), _split_ints(g[2]), _split_ints(g[3])
        else:                               # DATA:     id / skill / correct
            skill, corr = _split_ints(g[1]), _split_ints(g[2])
            prob = skill
        n = min(len(skill), len(prob), len(corr))
        if n == 0:
            continue
        per[sid] = (skill[:n], prob[:n], corr[:n])
    return per, dict(group_size=gsize, detect_score=round(score, 3),
                     n_groups=n_full, n_dropped=len(raw) - n_full * gsize)


def load_bench(name, repo_dir, prefix, folds, min_len):
    d = os.path.join(AKT_ROOT, repo_dir)
    allfolds, info = {}, {}
    for k in folds:
        per = {}
        for split in ("train", "valid", "test"):
            fn = os.path.join(d, f"{prefix}_{split}{k}.csv")
            if not os.path.exists(fn):
                print(f"  [!] 缺 {fn}")
                continue
            rows, meta = parse_akt_file(fn)
            per[split] = rows
            if split == "train":
                info = meta
        allfolds[k] = per
    return allfolds, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--folds", default="1,2,3,4,5")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x]

    if args.only == "check":
        for name, (rd, pre) in DATASETS.items():
            p = os.path.join(AKT_ROOT, rd, f"{pre}_train1.csv")
            if not os.path.exists(p):
                print(f"  [!] 缺失 {p}")
                continue
            per, info = parse_akt_file(p)
            totsk = set()
            totq = set()
            for sk, pr, _ in per.values():
                totsk |= set(sk)
                totq |= set(pr)
            print(f"  {name:<14} {info}  student={len(per)} "
                  f"skill={len(totsk)} question={len(totq)}")
        return

    targets = DATASETS if args.only == "all" else {args.only: DATASETS[args.only]}
    for name, (repo_dir, prefix) in targets.items():
        allfolds, info = load_bench(name, repo_dir, prefix, folds, args.min_len)
        if not allfolds:
            continue
        # 全局词表（跨折一致，保证不同折的技能/题目编号可比）
        skill2id, prob2id, skill_names = {}, {}, []
        for per in allfolds.values():
            for split in ("train", "valid", "test"):
                for sk, pr, _ in per.get(split, {}).values():
                    for s in sk:
                        if s not in skill2id:
                            skill2id[s] = len(skill2id)
                            skill_names.append(f"K{s}")
                    for q in pr:
                        if q not in prob2id:
                            prob2id[q] = len(prob2id)
        K, Q = len(skill2id), max(1, len(prob2id))
        print(f"\n[{name}] 目录={repo_dir} 组大小={info.get('group_size')} "
              f"n_groups={info.get('n_groups')} 异常组={info.get('n_bad')}")
        print(f"         skill={K} question={Q}")

        # 先修图归纳（公开数据集无先修标注）；induce_prereq 返回 (prereq, n_edges, pos)
        # 与 run_kt.py 的默认行为保持一致：用 train+valid+test 的序列归纳结构图
        all_seqs = []
        for k in folds:
            for split in ("train", "valid", "test"):
                for sid, (sk, pr, co) in allfolds.get(k, {}).get(split, {}).items():
                    if len(sk) >= args.min_len:
                        all_seqs.append(dict(
                            skill=np.array([skill2id[s] for s in sk], dtype=np.int64)))
        prereq, ne, _pos = induce_prereq(all_seqs, K)
        print(f"         归纳先修图: {dag_report(prereq, K)}")

        for k in folds:
            per = allfolds.get(k)
            if not per:
                continue
            out = {"n_skill": K, "n_question": Q, "skill_names": skill_names,
                   "fold": k, "prereq": prereq,
                   "student_scope": None, "skill_scope": None}
            stats = []
            for split in ("train", "valid", "test"):
                seqs = []
                for sid, (sk, pr, co) in per.get(split, {}).items():
                    if len(sk) < args.min_len:
                        continue
                    seqs.append(dict(
                        sid=sid,
                        skill=np.array([skill2id[s] for s in sk], dtype=np.int64),
                        correct=np.array([float(c) for c in co], dtype=np.float32),
                        question=np.array([prob2id[q] for q in pr], dtype=np.int64)))
                out[split] = seqs
                stats.append((split, len(seqs), int(sum(len(s["skill"]) for s in seqs))))
            alls = out["train"] + out["valid"] + out["test"]
            L = [len(s["skill"]) for s in alls]
            cor = np.concatenate([s["correct"] for s in alls]) if alls else np.array([0.0])
            out["meta"] = dict(
                source=f"AKT benchmark split ({repo_dir}), fold {k}",
                split_protocol="AKT official 5-fold (arghosh/AKT, arXiv:2007.12324)",
                stats=dict(name=f"{name}_f{k}", users=len(alls), n_skill=K, n_question=Q,
                           interactions=int(sum(L)),
                           avg_len=round(float(np.mean(L)), 2),
                           median_len=int(np.median(L)), max_len=int(np.max(L)),
                           correct_rate=round(float(cor.mean()), 4),
                           prereq_edges=int(sum(len(x) for x in prereq))))
            p = os.path.join(OUT, f"{name}_f{k}.pkl")
            with open(p, "wb") as f:
                pickle.dump(out, f, protocol=4)
            print(f"         fold{k}: " +
                  " / ".join(f"{sp}={n}({i})" for sp, n, i in stats) +
                  f"  正确率={cor.mean():.3f}  -> {os.path.basename(p)}")

    print("\n===== processed 目录 =====")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pkl"):
            print(f"  {os.path.getsize(os.path.join(OUT, f))/1e6:>9.2f} MB  {f}")


if __name__ == "__main__":
    sys.exit(main())
