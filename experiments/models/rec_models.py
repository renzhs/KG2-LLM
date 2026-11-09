#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习资源推荐 / 学习路径推荐 与 跨作用域串味率(CSCR)评测。

核心思想: 现有基于知识图谱的推荐方法在「技能表示空间」里做打分, 对
学段—专业作用域没有任何约束, 因而会把同名/近义但不同课程、不同学段的知识点
推荐给不匹配的学生(知识串味)。本文提出 CSG 硬门控与 CSCR 指标。

CSCR@K (Cross-Scope Contamination Rate) 定义:
    CSCR@K = (1/K) * |{ i in TopK : scope(i) ∩ scope(学生) = ∅ }|
其中 scope(i) = (grade_scope, major_scope), 学生作用域由其年级与专业给出。
CSCR 越低越好; 完全合规应为 0。
"""
import math
import numpy as np


# --------------------------------------------------------------------------
def scope_of_student(mask_row):
    return np.asarray(mask_row, dtype=bool)


def cscr_at_k(topk, scope_mask_row):
    """topk: (n, K) 0/1 或索引矩阵；scope_mask_row: (n, K) bool。
    返回逐学生的串味率。"""
    out = []
    for i in range(topk.shape[0]):
        idx = np.nonzero(topk[i])[0]
        if len(idx) == 0:
            out.append(0.0); continue
        bad = (~scope_mask_row[i][idx]).sum()
        out.append(bad / len(idx))
    return np.array(out, dtype=np.float64)


def evaluate_ranking(score, rel, scope_row=None, Ks=(5, 10, 20)):
    """score: (n,K) 越大越相关; rel: (n,K) 0/1 真值; scope_row: (n,K) bool 可用。
    返回 Recall@K / NDCG@K / Prec@K / CSCR@K。"""
    n, K = score.shape
    if scope_row is None:
        scope_row = np.ones_like(rel, dtype=bool)
    order = np.argsort(-score, axis=1, kind="mergesort")
    res = {}
    for k in Ks:
        rec = np.zeros((n, K), dtype=bool)
        rows = np.arange(n)[:, None]
        rec[rows, order[:, :k]] = True
        hit = (rec & (rel > 0)).sum(1)
        denom = np.maximum((rel > 0).sum(1), 1)
        recall = hit / denom
        # NDCG
        gains = rel[rows, order[:, :k]] / np.log2(np.arange(2, k + 2))
        ideal = np.sort(rel, axis=1)[:, ::-1][:, :k] / np.log2(np.arange(2, k + 2))
        idcg = ideal.sum(1)
        ndcg = np.where(idcg > 0, gains.sum(1) / np.maximum(idcg, 1e-9), 0.0)
        prec = hit / k
        # CSCR: TopK 中越界项占比
        bad = (~scope_row[rows, order[:, :k]]).sum(1)
        cscr = bad / k
        res[f"Recall@{k}"] = round(float(recall.mean()), 4)
        res[f"NDCG@{k}"] = round(float(ndcg.mean()), 4)
        res[f"Prec@{k}"] = round(float(prec.mean()), 4)
        res[f"CSCR@{k}"] = round(float(cscr.mean()), 4)
        res[f"CSCR_cnt@{k}"] = int(bad.sum())
    return res


# --------------------------------------------------------------------------
def build_interaction_matrix(seqs, n_skill, min_len=3, cutoff=0.7):
    """由序列构造 (学生 x 技能) 二元矩阵。
    训练段 = 前 cutoff 比例; 真值 = 后段中出现且未被掌握的技能。"""
    X, rel, sid2row, _ = _build_split(seqs, n_skill, min_len, cutoff, cutoff)
    return X, rel, sid2row


def build_interaction_matrix_split(seqs, n_skill, min_len=3, cut1=0.6, cut2=0.7):
    """三段切分，用于「验证集选超参 + 测试集报告」的规范流程。

    训练段 = [0, cut1) → 构造 X（掌握矩阵，取该段内多次作答的多数票）
    验证真值 = [cut1, cut2) 段出现且训练段未掌握的技能
    测试真值 = [cut2, 1)   段出现且训练段未掌握的技能

    返回 (X, rel_val, rel_test, sid2row)
    """
    if cut2 <= cut1:
        raise ValueError("cut2 必须大于 cut1")
    return _build_split(seqs, n_skill, min_len, cut1, cut2)


def _build_split(seqs, n_skill, min_len, cut1, cut2):
    """通用切分实现。cut1==cut2 时退化为「训练段 / 单一真值段」两段切分。"""
    rows, cols, vals = [], [], []
    sid2row = {}
    rv_r, rv_c, rt_r, rt_c = [], [], [], []
    for s in seqs:
        sk, co = s["skill"], s["correct"]
        n = len(sk)
        if n < min_len:
            continue
        sid = s.get("sid")
        if sid not in sid2row:
            sid2row[sid] = len(sid2row)
        r = sid2row[sid]
        i1 = max(1, int(round(n * cut1)))
        i2 = max(i1, int(round(n * cut2)))
        # 训练段：同一技能多次作答取多数票（>=0.5 视为已掌握）
        seen = {}
        for a, b in zip(sk[:i1], co[:i1]):
            seen.setdefault(int(a), []).append(float(b))
        for a, lst in seen.items():
            rows.append(r); cols.append(a); vals.append(1.0 if np.mean(lst) >= 0.5 else 0.0)
        mid = set(int(x) for x in sk[i1:i2])
        tail = set(int(x) for x in sk[i2:])
        for a in mid:
            rv_r.append(r); rv_c.append(a)
        for a in tail:
            rt_r.append(r); rt_c.append(a)

    n_users = len(sid2row)
    X = np.zeros((n_users, n_skill), dtype=np.float32)
    if rows:
        X[rows, cols] = vals
    unmastered = (X <= 0).astype(np.float32)

    def _rel(rr, cc):
        m = np.zeros((n_users, n_skill), dtype=np.float32)
        if rr:
            m[rr, cc] = 1.0
        return m * unmastered            # 已掌握的技能不必再推荐

    rel_val = _rel(rv_r, rv_c)
    rel_test = _rel(rt_r, rt_c)
    if cut1 == cut2:
        # 两段模式：把验证位置当作唯一真值返回
        return X, rel_test, sid2row, None
    return X, rel_val, rel_test, sid2row


# --------------------------------------------------------------------------
def svd_embed(X, dim=32, seed=42):
    """截断 SVD 隐语义分解（等价于隐因子模型, 快速且可复现）。"""
    Xc = X - X.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    n, m = Xc.shape
    d = min(dim, min(n, m) - 1)
    P = rng.normal(0, 1 / math.sqrt(m), size=(m, d)).astype(np.float32)
    for _ in range(12):                      # 幂迭代提高稳定性
        Z = Xc @ P
        Q, _ = np.linalg.qr(Z)
        Y = Xc.T @ Q
        P, _ = np.linalg.qr(Y)
    B = P.T @ Xc.T                           # [d, n]
    U = B.T                                  # 用户嵌入
    V = P                                    # 技能嵌入
    return U.astype(np.float32), V.astype(np.float32)


def recommend_popularity(X, V=None):
    pop = X.sum(0)
    return np.tile(pop[None, :], (X.shape[0], 1))


def recommend_svd(X, dim=32):
    U, V = svd_embed(X, dim)
    return U @ V.T


def graph_propagate(E, adj, layers=2, alpha=0.5):
    """在归一化先修邻接上做多层传播: E <- (1-a)E + a * Â E。"""
    n = adj.shape[0]
    A = np.asarray(adj, dtype=np.float32).copy()
    A = A + np.eye(n, dtype=np.float32)
    deg = A.sum(1)
    dinv = np.power(deg + 1e-9, -0.5)
    An = (dinv[:, None] * A) * dinv[None, :]
    out = E.copy()
    for _ in range(layers):
        out = (1 - alpha) * out + alpha * (An @ out)
    return out


def recommend_graph_cf(X, adj, dim=32, layers=2, alpha=0.5, calibrate=None):
    """KG 协同过滤：SVD 技能嵌入在先修图上传播后再打分（无作用域约束）。"""
    U, V = svd_embed(X, dim)
    A = adj if calibrate is None else calibrate
    Vg = graph_propagate(V, A, layers, alpha)
    return U @ Vg.T, U, Vg


def apply_scope_gate(score, scope_row, hard=True):
    """CSG 硬门控: 越界技能得分置为 -inf；soft 时给固定惩罚。"""
    s = score.astype(np.float64).copy()
    if hard:
        s[~scope_row] = -1e9
    else:
        s[~scope_row] -= 1.0
    return s


def readiness_score(X, adj, n_skill, alpha=1.0):
    """先修就绪度: 技能的父节点掌握度均值 - 该技能自身掌握度。
    数值越高表示"前置已具备且尚未掌握", 越应优先推荐。"""
    n, _ = X.shape
    A = np.asarray(adj, dtype=np.float32)
    par = A.T.copy()                       # par[v,u]=1 表示 u 是 v 的先修
    psum = par.sum(1)
    psum[psum == 0] = 1
    ready = (X @ par.T) / psum[None, :]     # 父节点掌握比例
    return ready - X                         # 减去自身掌握
