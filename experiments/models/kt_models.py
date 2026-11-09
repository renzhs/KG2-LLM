#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识追踪模型库。

基线: BKT / DKT / DKVMN / SAKT / AKT / GKT
本文方法: DGCL-KT  (Dual-Graph Collaborative Learning - Knowledge Tracing)

DGCL-KT 三处与既有方法本质不同:
  (1) 双图谱双向耦合
      正向 CKG->LKG: 在学生状态更新中对「先修图上的多跳技能邻域」做加权消息传递;
      反向 LKG->CKG: 先修邻接按真实共观测掌握度重标定 Â = D^-1/2 (A ⊙ W_calib) D^-1/2,
                    其中 w_calib = 1 + tanh(w_emp), w_emp 为共观测学生 correct_rate 相关系数。
  (2) CSG 作用域门控
      学生 (grade, major) 与技能 (grade_scope, major_scope) 的交集构成硬门控掩码,
      在注意力/图传播中屏蔽跨学段跨专业技能, 从结构上消除"知识串味"。
  (3) MDRL 阈值触发
      掌握度跨阈时触发重规划(在 run/run_path.py 中评测).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 通用组件
# ==========================================================================
class PosEncoder(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def gcn_norm(adj, add_self=True):
    """对称归一化 Â = D^-1/2 (A+I) D^-1/2 。
    dense=True 时返回稠密张量（K 很小，稠密更省事且可广播到 batch）。"""
    n = adj.shape[0]
    a = adj.copy()
    if add_self:
        a = a + np.eye(n, dtype=a.dtype)
    deg = a.sum(1)
    d_inv = np.power(deg + 1e-9, -0.5)
    D = np.diag(d_inv)
    an = (D @ a @ D).astype(np.float32)
    idx = np.nonzero(an)
    vals = an[idx]
    return torch.sparse_coo_tensor(
        torch.tensor(np.vstack(idx), dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


def hop_neighborhood(adj, hops=3):
    """无向 h 跳可达矩阵（含自身）。用于构造图谱证据子图掩码。"""
    n = adj.shape[0]
    A = (np.asarray(adj) > 0).astype(bool)
    A = A | A.T | np.eye(n, dtype=bool)
    reach = A.copy()
    cur = A.copy()
    for _ in range(hops - 1):
        cur = (cur @ A) > 0
        reach |= cur
    return reach


def gcn_norm_dense(adj, add_self=True):
    n = adj.shape[0]
    a = np.asarray(adj, dtype=np.float32).copy()
    if add_self:
        a = a + np.eye(n, dtype=a.dtype)
    deg = a.sum(1)
    d_inv = np.power(deg + 1e-9, -0.5)
    an = (d_inv[:, None] * a) * d_inv[None, :]
    return torch.tensor(an, dtype=torch.float32)


# ==========================================================================
# 1. BKT (Bayesian Knowledge Tracing) —— 逐技能 HMM, 用 EM/网格搜索拟合
# ==========================================================================
class BKT:
    """四参数 BKT (L0, T, S, G) 逐技能前向-后向 EM。

    状态转移:  P(k_{t+1}=1|k_t=0)=T,  P(k_{t+1}=1|k_t=1)=1
    观测似然:  P(o=1|k=0)=G,          P(o=1|k=1)=1-S
    """
    def __init__(self, n_skill, n_iter=60, tol=1e-5):
        self.K, self.n_iter, self.tol = n_skill, n_iter, tol
        self.L0 = np.full(n_skill, 0.3); self.T = np.full(n_skill, 0.2)
        self.S = np.full(n_skill, 0.1); self.G = np.full(n_skill, 0.2)

    # ---- EM ----
    def _em_one(self, obs_list, l0, t_, s_, g_):
        """obs_list: list[np.ndarray] 同一技能的多个学生作答序列。

        转移矩阵取**标准 BKT**（只学习、不遗忘）：
            P(k_{t+1}=1|k_t=0) = T,  P(k_{t+1}=1|k_t=1) = 1
        早期实现用了对称"翻转"矩阵 [[1-T,T],[T,1-T]]（允许掌握度下降），
        那既非 BKT 也不符合"练习单调提升掌握"的教学假设；前后向当时虽自洽，
        但参数语义已偏，导致基线被系统性削弱。此处前向/后向/充分统计量
        全部按标准 BKT 重写，保持 EM 自洽。
        """
        # 预处理：把观测转成 float 数组，并预先算好"答错/答对"的布尔掩码
        # （掩码在 EM 迭代间不变，放循环外可省下大量重复比较）。
        seqs = []
        for r in obs_list:
            if len(r) == 0:
                continue
            rr = np.asarray(r, dtype=np.float64)
            seqs.append((rr, rr <= 0.5, rr > 0.5))
        if not seqs:
            return l0, t_, s_, g_
        m = float(len(seqs))

        # EM 主循环：与逐元素实现**数学等价**（标准 BKT 前向-后向 + 充分统计量），
        # 但把"每个时刻一次 Python 函数调用"改成"每个序列一次 numpy 向量运算"，
        # 并加入标准的收敛早停（参数最大变化 < tol）。
        # 动机：ASSIST2012 训练集有 1.9 万学生、175 万次作答，旧实现 60 次 EM ×
        # 逐元素闭包调用需数分钟；这既拖慢全量对比实验，也不改变解本身。
        for _ in range(self.n_iter):
            n_l0 = n_t_num = n_t_den = n_s_num = n_s_den = n_g_num = n_g_den = 0.0
            for rr, wrong, right in seqs:
                n = rr.shape[0]
                # 观测似然（向量化）：P(o|k=0)、P(o|k=1)
                e0 = np.where(right, g_, 1.0 - g_)
                e1 = np.where(right, 1.0 - s_, s_)
                e0l = e0.tolist(); e1l = e1.tolist()
                # ---- 前向 ----
                fwd = np.empty((n + 1, 2))
                a0 = 1.0 - l0; a1 = l0
                fwd[0, 0] = a0; fwd[0, 1] = a1
                for i in range(n):
                    a0n = a0 * (1.0 - t_)                 # 未掌握且未学会
                    a1n = a0 * t_ + a1                    # 学会 或 原本已掌握
                    a0 = a0n * e0l[i]
                    a1 = a1n * e1l[i]
                    fwd[i + 1, 0] = a0; fwd[i + 1, 1] = a1
                Z = a0 + a1
                if Z <= 1e-300:
                    continue
                # ---- 后向 ----
                bwd = np.empty((n + 1, 2))
                bwd[n, 0] = 1.0; bwd[n, 1] = 1.0
                b0 = 1.0; b1 = 1.0
                for i in range(n - 1, -1, -1):
                    b0n = (1.0 - t_) * e0l[i] * b0 + t_ * e1l[i] * b1
                    b1n = e1l[i] * b1
                    b0 = b0n; b1 = b1n
                    bwd[i, 0] = b0; bwd[i, 1] = b1
                # ---- 后验 γ 与充分统计量 ----
                gam = fwd * bwd
                gam = gam / np.maximum(gam.sum(1, keepdims=True), 1e-300)
                n_l0 += float(gam[0, 1])
                g1v = gam[:n, 1]; g0v = gam[:n, 0]
                n_t_num += float((fwd[:n, 0] * t_ * e1 * bwd[1:n + 1, 1]).sum()) / Z
                n_t_den += float(g0v.sum())
                n_s_den += float(g1v.sum())
                n_g_den += float(g0v.sum())
                n_s_num += float(g1v[wrong].sum())
                n_g_num += float(g0v[right].sum())

            l0_new = float(np.clip(n_l0 / max(m, 1.0), 0.01, 0.95))
            t_new, s_new, g_new = t_, s_, g_
            if n_t_den > 1e-6:
                t_new = float(np.clip(n_t_num / n_t_den, 0.01, 0.95))
            if n_s_den > 1e-6:
                s_new = float(np.clip(n_s_num / n_s_den, 0.01, 0.45))
            if n_g_den > 1e-6:
                g_new = float(np.clip(n_g_num / n_g_den, 0.01, 0.45))
            delta = max(abs(l0_new - l0), abs(t_new - t_),
                        abs(s_new - s_), abs(g_new - g_))
            l0, t_, s_, g_ = l0_new, t_new, s_new, g_new
            if delta < self.tol:
                break
        return l0, t_, s_, g_

    def fit(self, seqs, min_obs=20, max_obs=0):
        from collections import defaultdict
        # ---- 观测总量上限（大训练集专用的耗时控制）----
        # EM 的每轮开销随「(技能,学生) 观测对」数量线性增长：ASSIST2012 训练集
        # 有 1.9 万学生、177 万次作答，按技能切分后约 177 万个短序列，
        # 60 轮 EM 需要约 1 亿次序列级运算（实测单次 fit > 10 分钟）。
        # 而 BKT 只有 4K 个**技能级共享**参数，用数百名学生的作答即可充分估计；
        # 故当观测总量超预算时按**等间隔**抽样降采样（等间隔而非取前缀，
        # 避免受序列文件顺序影响）。仅对超大数据集生效，其余数据集不受影响。
        if max_obs:
            tot = sum(len(s["correct"]) for s in seqs)
            if tot > max_obs:
                step = int(np.ceil(tot / float(max_obs)))
                seqs = seqs[::step]
        per = defaultdict(lambda: defaultdict(list))
        # ---- 热启动：用训练集的每技能经验正确率初始化 L0 ----
        # 动机：EM 对 L0 的更新是 P(k_0=1|观测) 的后验均值，会向初始先验收缩；
        # 在 Statics11 这类"每题每学生至多作答一次"的数据集上，每技能的观测
        # 只有一次/学生、T 与 L0 不可分辨，EM 会收敛到一个近乎均匀的退化解
        # （实测 L0 的跨技能标准差仅 0.045），预测退化为常数（AUC≈0.50）。
        # 而题目难度是真实存在的信号（仅用"训练集题目答对率"预测测试集可得
        # AUC≈0.79），故把经验正确率作为 L0 的热启动，让 EM 在此基础上精修。
        cnt = np.zeros(self.K); hit = np.zeros(self.K)
        for s in seqs:
            for a, b in zip(s["skill"], s["correct"]):
                a = int(a); cnt[a] += 1.0; hit[a] += float(b)
        gm = hit.sum() / max(cnt.sum(), 1.0)
        for k in range(self.K):
            if cnt[k] > 0:
                # 拉普拉斯平滑，避免 0/1 极端值被 clip 截断
                self.L0[k] = float(np.clip((hit[k] + gm) / (cnt[k] + 1.0), 0.05, 0.95))
            else:
                self.L0[k] = float(np.clip(gm, 0.05, 0.95))
        for s in seqs:
            byk = defaultdict(list)
            for a, b in zip(s["skill"], s["correct"]):
                byk[int(a)].append(float(b))
            for k, v in byk.items():
                if len(v) >= 2:
                    per[k][s.get("sid")].append(np.array(v))
        for k, bystu in per.items():
            obs_list = [np.concatenate(v) if len(v) > 1 else v[0] for v in bystu.values()]
            if sum(len(x) for x in obs_list) < min_obs:
                continue
            self.L0[k], self.T[k], self.S[k], self.G[k] = self._em_one(
                obs_list, self.L0[k], self.T[k], self.S[k], self.G[k])

    def predict_seq(self, seqs):
        """对每个序列做逐步递推, 返回与序列等长的预测概率数组列表。

        **每个技能维护独立的状态**，且状态在整个学生序列上延续
        （不是"技能一旦切换就重置"）。这是 BKT 用于知识追踪的标准做法：
        同一技能的多次作答按时间顺序构成该技能 HMM 的观测序列，
        中间夹杂其它技能并不打断其状态演化。
        早期实现在 `k != prev_k` 时把 alpha 重置为初始分布，导致
        「技能每次切换就丢失历史」—— 在 Statics11（平均序列长 568、
        相邻同技能占比 0.1%）这类数据集上，预测退化为"按技能取常数"。
        """
        out = []
        for s in seqs:
            sk = s["skill"]; co = s["correct"]
            n = len(sk)
            pr = np.zeros(n, dtype=np.float32)
            state = {}          # skill -> np.array([P(k=0), P(k=1)])
            for i in range(n):
                k = int(sk[i])
                alpha = state.get(k)
                if alpha is None:
                    l0 = self.L0[k]
                    alpha = np.array([1 - l0, l0], dtype=np.float64)
                alpha = alpha / max(alpha.sum(), 1e-300)
                pr[i] = alpha[0] * self.G[k] + alpha[1] * (1 - self.S[k])
                # ---- 状态转移（标准 BKT：只学习不遗忘）----
                t_, s_, g_ = self.T[k], self.S[k], self.G[k]
                a0 = alpha[0] * (1 - t_)
                a1 = alpha[0] * t_ + alpha[1]
                # ---- 用本步观测更新 ----
                o = co[i]
                e0 = g_ if o == 1 else (1 - g_)
                e1 = (1 - s_) if o == 1 else s_
                state[k] = np.array([a0 * e0, a1 * e1])
            out.append(pr)
        return out


class BKTPlus(BKT):
    """BKT+：标准 BKT 上叠加**个体化初始掌握度**（Yudelson et al., 2013）。

    这也是 AKT(2020) 等论文表格中 ``BKT+'' 基线的口径，故单列一行以便与文献对照。

    动机：标准 BKT 的 L0 只依赖技能（在 Statics2011 上技能即题目），无法表达
    "同一题目对不同学生难度不同"；而该数据集每个学生每题至多作答一次，
    逐学生序列内技能只出现一次，递推状态无从发挥，预测退化为"按技能取常数"
    （实测 AUC≈0.60，而仅用题目难度的先验就有 0.79）。

    BKT+ 引入一个**在线估计**的学生能力项 θ_s：
        L0_eff(k, s) = σ( σ⁻¹(L0_k) + γ·θ_s )
        θ_s = clip( logit( (已作答中正确数 + 1) / (已作答数 + 2) ), -1.5, 1.5 )
    θ_s 只使用**当前时刻之前**的作答估计（因果），并与技能先验相乘性融合，
    等价于把 BKT 从"仅题目难度"扩展到 2PL-IRT 式 的"题目难度 × 学生能力"。
    γ 在**验证集**上从 {0, 0.5, 1.0} 中选（γ=0 退化为标准 BKT），不涉及测试集。
    """
    def __init__(self, n_skill, n_iter=60, tol=1e-5, gamma=1.0):
        super().__init__(n_skill, n_iter, tol)
        self.gamma = float(gamma)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def select_gamma(self, valid_seqs, grid=(0.0, 0.5, 1.0)):
        """在验证集上选 γ（选择依据为验证集 AUC，测试集不参与）。"""
        best_g, best_auc = self.gamma, -1.0
        for g in grid:
            self.gamma = float(g)
            seqs = [dict(skill=s["skill"], correct=s["correct"]) for s in valid_seqs]
            preds = self.predict_seq(seqs)
            ys, ps = [], []
            for s, p in zip(seqs, preds):
                pr = np.concatenate([[0.5], p[:-1]])
                ys.append(s["correct"]); ps.append(pr)
            y = np.concatenate(ys); pv = np.concatenate(ps)
            pos, neg = pv[y == 1], pv[y == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            from run_kt import _rankdata_avg
            r = _rankdata_avg(pv)
            auc = float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0)
                        / (len(pos) * len(neg)))
            if auc > best_auc:
                best_auc, best_g = auc, float(g)
        self.gamma = best_g
        return best_g, best_auc

    def predict_seq(self, seqs):
        out = []
        for s in seqs:
            sk = s["skill"]; co = s["correct"]
            n = len(sk)
            pr = np.zeros(n, dtype=np.float32)
            state = {}
            cum_n = 0.0; cum_c = 0.0
            for i in range(n):
                k = int(sk[i])
                l0 = self.L0[k]
                theta = 0.0
                if self.gamma > 0 and cum_n > 0:
                    theta = float(np.clip(
                        self._logit((cum_c + 1.0) / (cum_n + 2.0)), -1.5, 1.5))
                l0e = float(1.0 / (1.0 + math.exp(-(self._logit(l0) + self.gamma * theta))))
                alpha = state.get(k)
                if alpha is None:
                    alpha = np.array([1 - l0e, l0e], dtype=np.float64)
                alpha = alpha / max(alpha.sum(), 1e-300)
                pr[i] = alpha[0] * self.G[k] + alpha[1] * (1 - self.S[k])
                t_, s_, g_ = self.T[k], self.S[k], self.G[k]
                a0 = alpha[0] * (1 - t_)
                a1 = alpha[0] * t_ + alpha[1]
                o = co[i]
                e0 = g_ if o == 1 else (1 - g_)
                e1 = (1 - s_) if o == 1 else s_
                state[k] = np.array([a0 * e0, a1 * e1])
                # θ_s 的在线统计：只用当前时刻之前的作答
                cum_n += 1.0
                cum_c += float(o)
            out.append(pr)
        return out


# ==========================================================================
# 2. DKT —— LSTM + skill one-hot
# ==========================================================================
class DKT(nn.Module):
    def __init__(self, n_skill, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.K = n_skill
        self.emb = nn.Embedding(2 * n_skill, 64)
        self.lstm = nn.LSTM(64, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_skill)

    def forward(self, skill, correct):
        x = skill + self.K * correct.long()              # 融合编码
        e = self.emb(x)
        h, _ = self.lstm(e)
        return torch.sigmoid(self.out(self.drop(h)))         # [B,T,K]


# ==========================================================================
# 3. DKVMN —— 键值记忆网络
# ==========================================================================
class DKVMN(nn.Module):
    def __init__(self, n_skill, mem_size=64, key_dim=64, val_dim=64):
        super().__init__()
        self.K, self.M, self.dk, self.dv = n_skill, mem_size, key_dim, val_dim
        self.k_emb = nn.Embedding(n_skill, key_dim)          # 技能 -> key
        self.v_emb = nn.Embedding(n_skill, val_dim)          # 技能 -> value 分量
        self.r_emb = nn.Embedding(2, val_dim)                # 作答结果 -> value 分量
        self.Mk = nn.Parameter(torch.randn(mem_size, key_dim) * 0.1)
        self.Mv0 = nn.Parameter(torch.randn(mem_size, val_dim) * 0.1)
        self.erase = nn.Linear(val_dim, val_dim)
        self.add = nn.Linear(val_dim, val_dim)
        self.fc = nn.Linear(val_dim + key_dim, 128)
        self.out = nn.Linear(128, n_skill)

    def forward(self, skill, correct):
        B, T = skill.shape
        Mv = self.Mv0.unsqueeze(0).repeat(B, 1, 1)
        q = self.k_emb(skill)                                             # [B,T,dk]
        r = self.v_emb(skill) + self.r_emb(correct.long().clamp(0, 1))    # [B,T,dv]
        outs = []
        for t in range(T):
            w = F.softmax(q[:, t] @ self.Mk.t() / math.sqrt(self.dk), -1)   # [B,M]
            read = (w.unsqueeze(-1) * Mv).sum(1)                            # [B,dv]
            f = torch.tanh(self.fc(torch.cat([read, q[:, t]], -1)))
            outs.append(torch.sigmoid(self.out(f)))                          # [B,K]
            e = torch.sigmoid(self.erase(r[:, t]))
            a = torch.tanh(self.add(r[:, t]))
            Mv = Mv * (1 - w.unsqueeze(-1) * e.unsqueeze(1)) + w.unsqueeze(-1) * a.unsqueeze(1)
        return torch.stack(outs, 1)


# ==========================================================================
# 4. SAKT —— 自注意力知识追踪
# ==========================================================================
class SAKT(nn.Module):
    def __init__(self, n_skill, d=128, heads=8, layers=2, dropout=0.2, max_len=256):
        super().__init__()
        self.K = n_skill
        self.emb = nn.Embedding(2 * n_skill, d)
        self.pos = PosEncoder(d, max_len)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q = nn.Embedding(n_skill, d)
        self.out = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, skill, correct):
        x = self.pos(self.emb(skill + self.K * correct.long()))
        # 因果掩码
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.enc(x, mask=mask)
        q = self.q(skill)                                  # 查询当前技能
        logit = self.out(self.drop(h * q)).squeeze(-1)      # [B,T]
        return torch.sigmoid(logit).unsqueeze(-1).expand(-1, -1, self.K)


# ==========================================================================
# 5. AKT —— 上下文感知注意力 + Rasch 难度嵌入
# ==========================================================================
class AKT(nn.Module):
    def __init__(self, n_skill, n_question, d=128, heads=8, layers=1, dropout=0.2,
                 max_len=256):
        super().__init__()
        self.K, self.Q = n_skill, max(n_question, 1)
        self.skill_emb = nn.Embedding(n_skill, d)
        self.diff_emb = nn.Embedding(self.Q, d)
        self.ans_emb = nn.Embedding(2, d)
        self.pos = PosEncoder(d, max_len)
        self.proj_q = nn.Linear(d, d); self.proj_k = nn.Linear(d, d); self.proj_v = nn.Linear(d, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.fc1 = nn.Linear(2 * d, d)
        self.fc2 = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, skill, correct, question=None):
        B, T = skill.shape
        if question is None or (question < 0).all():
            question = torch.zeros_like(skill)
        question = question.clamp(min=0)
        s = self.skill_emb(skill)
        c = self.ans_emb(correct.long().clamp(0, 1))
        d = self.diff_emb(question)
        x = self.pos(s + c + d)
        Tmask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.enc(x, mask=Tmask)
        qf = torch.cat([h, d], -1)                     # 融合上下文与难度
        z = torch.tanh(self.fc1(self.drop(qf)))
        return torch.sigmoid(self.fc2(z)).squeeze(-1).unsqueeze(-1).expand(-1, -1, self.K)


# ==========================================================================
# 6. GKT —— 图知识追踪（技能图为先验，无学生作用域）
# ==========================================================================
class GKT(nn.Module):
    def __init__(self, n_skill, hidden=64, steps=3, dropout=0.2, adj=None):
        super().__init__()
        self.K, self.h, self.steps = n_skill, hidden, steps
        if adj is None:
            adj = np.eye(n_skill, dtype=np.float32)
        A = gcn_norm_dense(np.asarray(adj, dtype=np.float32))
        # 稀疏化：先修图通常很稀疏（每行几个非零），但 Â 按稠密存会让每步图传播
        # 变成 K²·h 的稠密乘。Statics11 的 K=1223、h=64 时单步约 0.77 GFLOP，
        # 乘上 T=200 与批数后单折需数小时。稀疏化后按 nnz 计算，实测提速 1~2 个量级。
        # ------------------------------------------------------------------
        # 图传播一律走**稠密 einsum**。同为 Â 的两种表示，数学等价，但实测
        # torch.sparse.mm 的稀疏调度开销远大于它省下的 FLOPs：
        #   K=100/B=64/T=200  ：稀疏 1850ms vs 稠密 337ms（5.5×），显存同为 4.32GB
        #   K=1223/B=8/T=200  ：稀疏 1976ms vs 稠密 370ms（5.3×），显存同为 6.82GB
        # 即 K 小或大、密度低或高，稠密路径都更快，故不再按密度分派。
        # 仍注册 A_sp 供诊断脚本（diag_gkt*.py）复用。
        # ------------------------------------------------------------------
        self.register_buffer("A_dense", A)
        self.register_buffer("A_sp", A.to_sparse())
        self._use_sparse = False
        self.emb = nn.Embedding(2 * n_skill, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, n_skill)
        self.drop = nn.Dropout(dropout)

    def _prop(self, hn, B, x_t):
        # 稠密图传播：见 __init__ 注释 —— 恒快于 torch.sparse.mm（5.3~5.5×），数学等价。
        return torch.einsum("ij,bjd->bid", self.A_dense, hn)

    def forward(self, skill, correct):
        B, T = skill.shape
        H = torch.zeros(B, self.K, self.h, device=skill.device)
        outs = []
        x = self.emb(skill + self.K * correct.long())
        for t in range(T):
            flat = H.reshape(B * self.K, self.h)
            hn = self.gru(x[:, t].repeat_interleave(self.K, 0), flat).reshape(B, self.K, self.h)
            for _ in range(self.steps):
                hn = self._prop(hn, B, x[:, t])
            H = torch.tanh(hn)
            # 只取「当前时刻所问技能」的输出，避免构造 (B,K,K) 大张量。
            # 原写法 self.out(H).diagonal(...) 会先算出 B×K×K（K=1223 时每步 383MB，
            # 200 步连同反向图 > 70GB，必然 OOM）；而 BCE 只用到 gather 出的对角线元素，
            # 故改为按当前技能索引取权重，等价且显存降至 MB 级。
            idx = skill[:, t]
            wk = self.out.weight[idx]                                  # (B,h)
            hk = self.drop(H.gather(
                1, idx.view(B, 1, 1).expand(-1, -1, self.h)).squeeze(1))   # (B,h)
            y = torch.sigmoid((wk * hk).sum(-1) + self.out.bias[idx])
            outs.append(y)
        ys = torch.stack(outs, 1)                                       # (B,T)
        # 还原成 (B,T,K) 接口（非对角位置为 0，评测只 gather 对角，结果等价）
        return torch.zeros(B, T, self.K, device=skill.device).scatter(
            2, skill.unsqueeze(-1), ys.unsqueeze(-1))


# ==========================================================================
# 7. 近年强基线：DKT+ / SAINT / SimpleKT
#    统一复现：同一数据管线、同一评测口径（全部作答位置 AUC/ACC/RMSE），
#    目的是让"本文方法 vs 近年 SOTA"的对比在同一协议下可直接比对。
# ==========================================================================
class DKTPlus(nn.Module):
    """DKT+ (Yeung & Yeung, 2018)。

    在 DKT 之上加入三项正则，缓解"预测跳变"与"同技能重复作答不一致"：
        L = BCE + lam_r*L_r + lam_w1*L_w1 + lam_w2*L_w2
      L_r : 用 ReLU(W_r h_t) 重建下一时刻输入 (技能, 作答) 的 one-hot
      L_w1: 同技能连续作答且上一次正确 -> 隐状态应保持相似（惩罚 -log sigmoid(h_t·h_{t+1})）
      L_w2: 同技能连续作答且上一次错误 -> 隐状态应变化（惩罚 -log sigmoid(-h_t·h_{t+1})）
    正则项经 self.aux_loss 交由通用训练循环累加，不侵入其它模型实现。
    """
    def __init__(self, n_skill, hidden=128, dropout=0.2,
                 lam_r=0.1, lam_w1=0.1, lam_w2=0.01):
        super().__init__()
        self.K = n_skill
        self.emb = nn.Embedding(2 * n_skill, 64)
        self.lstm = nn.LSTM(64, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_skill)
        self.rec = nn.Linear(hidden, 2 * n_skill)
        self.lam = (lam_r, lam_w1, lam_w2)
        self.aux_loss = None

    def forward(self, skill, correct):
        x = skill + self.K * correct.long()
        h, _ = self.lstm(self.emb(x))
        y = torch.sigmoid(self.out(self.drop(h)))
        _B, T, _d = h.shape
        zero = h.sum() * 0.0
        if T > 1:
            r = F.relu(self.rec(h[:, :-1]))
            nxt = (skill[:, 1:] + self.K * correct[:, 1:].long()).clamp(0, 2 * self.K - 1)
            tgt = torch.zeros_like(r).scatter_(2, nxt.unsqueeze(-1), 1.0)
            l_r = ((r - tgt) ** 2).sum(-1).mean()
            dot = (F.normalize(h[:, :-1], dim=-1) * F.normalize(h[:, 1:], dim=-1)).sum(-1)
            same = skill[:, 1:] == skill[:, :-1]
            ok = correct[:, :-1] > 0.5
            m1 = same & ok
            m2 = same & (~ok)
            l1 = (-F.logsigmoid(dot))[m1].mean() if bool(m1.any()) else zero
            l2 = (-F.logsigmoid(-dot))[m2].mean() if bool(m2.any()) else zero
            self.aux_loss = self.lam[0] * l_r + self.lam[1] * l1 + self.lam[2] * l2
        else:
            self.aux_loss = zero
        return y


class SAINT(nn.Module):
    """SAINT (Choi et al., 2020)：Transformer 编码器-解码器。

    ⚠ **因果性（关键）**：本实现默认 `enc_causal=True`，即编码器也加因果掩码。

    为什么必须加：ASSISTments2015/2017 这类"技能练习集"数据里，题目 id 呈
    **连续 run**（同一题反复作答直到答对），而 run 内最后一次作答几乎必为正确。
    若编码器不设掩码（原版 SAINT 在"整段序列给定、只预测末位"的场景下如此），
    解码器在位置 t 通过跨注意力就能读到 `q[t+1]`，从而判断"当前位置是否为 run
    的末次尝试"，把 AUC 从 ~0.75 抬到 **0.97**（实测 ASSIST2017 fold1 =
    0.9747±0.0018，验证集亦然，故与训练泄漏无关，纯属**未来信息泄漏**）。
    这与逐位置（per-position）知识追踪协议的"仅可用历史"前提直接冲突，
    也会让对比失去意义，故默认关闭双向可见性。
    设 `enc_causal=False` 可复现该泄漏现象，仅用于消融说明。
    """
    def __init__(self, n_skill, n_question=1, d=128, heads=8, enc_layers=1, dec_layers=1,
                 dropout=0.2, max_len=256, enc_causal=True, strict_causal=True):
        super().__init__()
        self.K, self.Q = n_skill, max(n_question, 1)
        self.enc_causal = bool(enc_causal)
        self.strict_causal = bool(strict_causal)
        self.q_emb = nn.Embedding(self.Q, d)
        self.s_emb = nn.Embedding(n_skill, d)
        self.a_emb = nn.Embedding(2, d)
        self.pos = PosEncoder(d, max_len)
        el = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                        batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        dl = nn.TransformerDecoderLayer(d, heads, d * 4, dropout,
                                        batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.out = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, skill, correct, question=None):
        B, T = skill.shape
        if question is None or bool((question < 0).all()):
            question = torch.zeros_like(skill)
        question = question.clamp(min=0)
        tmask = torch.triu(torch.ones(T, T, device=skill.device, dtype=torch.bool), 1)
        xe = self.pos(self.q_emb(question) + self.s_emb(skill))
        he = self.enc(xe, mask=tmask) if self.enc_causal else self.enc(xe)
        xd = self.pos(self.q_emb(question) + self.s_emb(skill)
                      + self.a_emb(correct.long().clamp(0, 1)))
        # ⚠ memory_mask 与 tgt_mask 必须**同时**给：
        #   · tgt_mask 只限制解码器自身的自注意力；
        #   · 但 `he[t']` 即使来自因果编码器，其**该位置自身**的 token 信息
        #     （q[t']、s[t']，即未来题目/技能 id）依然完整地编码在 he[t'] 里。
        #   若不给 memory_mask，位置 t 的跨注意力仍可读 he[t+1..T-1]，
        #   于是模型能知道"下一个题目是否与当前相同"，从而判断当前是否为
        #   练习 run 的末次尝试。ASSIST2017 的末次尝试答对率高达 0.87~0.94
        #   （run 长 2~5），前几次仅 0.25~0.46 —— 仅凭"是否为末次"即可把
        #   AUC 从 ~0.79 抬到 0.93+（实测 0.9364）。这是**未来信息泄漏**，
        #   与逐位置 KT 协议"只用历史"的前提冲突，故默认严格因果。
        mmask = tmask if self.strict_causal else None
        hd = self.dec(xd, he, tgt_mask=tmask, memory_mask=mmask)
        return torch.sigmoid(self.out(self.drop(hd))).squeeze(-1).unsqueeze(-1).expand(
            -1, -1, self.K)


class SimpleKT(nn.Module):
    """SimpleKT (Liu et al., ACL 2023) 的题目中心简化复现。

    要点：以"题目"而非"技能"为中心建模；技能表示经 Rasch 难度调制；
    因果自注意力后，输出头融合学生状态与当前题目嵌入做预测。
    """
    def __init__(self, n_skill, n_question=1, d=128, heads=8, layers=1, dropout=0.2,
                 max_len=256, use_rasch=True):
        super().__init__()
        self.K, self.Q = n_skill, max(n_question, 1)
        self.q_emb = nn.Embedding(self.Q, d)
        self.s_emb = nn.Embedding(n_skill, d)
        self.a_emb = nn.Embedding(2, d)
        self.d_emb = nn.Embedding(self.Q, 1)
        self.use_rasch = bool(use_rasch)
        self.pos = PosEncoder(d, max_len)
        el = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                        batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)
        self.fc1 = nn.Linear(2 * d, d)
        self.fc2 = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, skill, correct, question=None):
        B, T = skill.shape
        if question is None or bool((question < 0).all()):
            question = torch.zeros_like(skill)
        question = question.clamp(min=0)
        s = self.s_emb(skill)
        if self.use_rasch:
            s = s * torch.sigmoid(self.d_emb(question))   # Rasch 难度调制
        x = self.pos(self.q_emb(question) + s + self.a_emb(correct.long().clamp(0, 1)))
        tmask = torch.triu(torch.ones(T, T, device=skill.device, dtype=torch.bool), 1)
        h = self.enc(x, mask=tmask)
        z = torch.tanh(self.fc1(self.drop(torch.cat([h, self.q_emb(question)], -1))))
        return torch.sigmoid(self.fc2(z)).squeeze(-1).unsqueeze(-1).expand(-1, -1, self.K)


# ==========================================================================
# 8. DGCL-KT —— 本文方法
# ==========================================================================
class DGCLKT(nn.Module):
    """
    双图谱协同知识追踪。

    参数
    ----
    n_skill      : 技能数 K
    n_question   : 题目数（用于 Rasch 难度）
    adj          : 先修邻接 (K,K) float; 若为 None 则退化为无图版本(消融)
    adj_calib    : 反向标定后的邻接, 若为 None 则使用 adj (消融: 去掉反向耦合)
    scope_mask   : (S,K) bool, 学生 s 对技能 k 是否在作用域内; None 表示不使用 CSG 门控
    """
    def __init__(self, n_skill, n_question=1, d=128, heads=8, layers=1, steps=2,
                 dropout=0.2, max_len=256, adj=None, adj_calib=None, scope_mask=None,
                 use_graph=True, use_calib=True, use_scope=True,
                 hops=None, gated_fuse=False, gate_init=-2.0,
                 topm=0, wprior=0.0, graph_in=False, seq_enc="transformer"):
        super().__init__()
        self.K, self.Q = n_skill, max(n_question, 1)
        self.d, self.steps = d, steps
        self.use_graph, self.use_calib, self.use_scope = use_graph, use_calib, use_scope
        # ---- 图证据融合方式 ----
        # False: 拼接融合   —— fc1 吃 [h; s; g]（3d）
        # True : 残差门控融合 —— fc1 吃 [h; s]（2d），图证据经可学习标量 alpha 注入。
        # 动机：在无真实先修标注、只能用归纳图的数据集上（如 ASSIST09/12），拼接融合会把
        # 图噪声硬塞进学生表示；而残差门控的 alpha 初值经 sigmoid 后 ≈0.12，使模型可从
        # "近似无图"起步、由数据自行决定图证据开启程度，从而对低质量先修图更鲁棒。
        self.gated_fuse = bool(gated_fuse)
        self.hops = int(hops) if hops else steps + 1
        # ---- 证据检索的稀疏化与结构先验 ----
        # topm  : 每个时刻只保留注意力得分最高的 m 个邻域技能（0=不稀疏）。
        #         动机：公开数据集技能空间大（ASSIST09 K=149、ASSIST12 K=123），
        #         在整片邻域上做 softmax 会把注意力稀释成近似均匀分布，
        #         图证据退化为全图均值，反而不如不用图。
        # wprior: 在注意力 logits 上叠加先修强度的对数先验 β·log P(v|u)，
        #         引导证据检索偏向先修关系更强的知识点。
        self.topm = int(topm) if topm else 0
        self.wprior = float(wprior or 0.0)
        # graph_in: 用图卷积平滑后的技能表示 E 作为序列通路的输入嵌入。
        # 动机：长尾数据集（ASSIST09 中位数序列长仅 23）中大量技能交互稀疏，
        # 其独立嵌入训练不充分；用先修图做一次平滑相当于在技能间共享统计强度。
        self.graph_in = bool(graph_in)
        # seq_enc: 序列通路编码器。公开数据集上单层 Transformer 的表达力并不占优
        # （ASSIST09 上 AKT/本文均约 0.732，而 LSTM 结构的 DKT 达 0.7409），
        # 因此把编码器类型也交给验证集选择，而非写死。
        self.seq_enc = str(seq_enc or "transformer").lower()
        if self.seq_enc not in ("transformer", "gru", "lstm"):
            raise ValueError(f"未知 seq_enc={self.seq_enc}")

        self.skill_emb = nn.Embedding(n_skill, d)
        self.diff_emb = nn.Embedding(self.Q, d)
        self.ans_emb = nn.Embedding(2, d)
        self.pos = PosEncoder(d, max_len)
        self.proj_q = nn.Linear(d, d); self.proj_k = nn.Linear(d, d); self.proj_v = nn.Linear(d, d)
        if self.seq_enc == "transformer":
            enc = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True,
                                             norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
        elif self.seq_enc == "gru":
            self.enc = nn.GRU(d, d, num_layers=max(layers, 1), batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        else:  # lstm
            self.enc = nn.LSTM(d, d, num_layers=max(layers, 1), batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        # ---- CKG 正向: 先修图上的图卷积 (结构编码) ----
        self.gcn_w = nn.ModuleList([nn.Linear(d, d) for _ in range(steps)])
        # ---- 作用域受限的图谱注意力检索 (query=学生状态+当前技能, key/value=CKG 技能表示) ----
        self.q_proj = nn.Linear(2 * d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)          # 自门控: 控制图证据注入强度
        self.fc1 = nn.Linear((2 if self.gated_fuse else 3) * d, d)
        if self.gated_fuse:
            self.alpha = nn.Parameter(torch.tensor(float(gate_init)))
        self.fc2 = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

        if adj is None:
            adj = np.eye(n_skill, dtype=np.float32)
        self.register_buffer("A_struct", gcn_norm_dense(np.asarray(adj, dtype=np.float32)))
        ac = adj_calib if (adj_calib is not None and use_calib) else adj
        self.register_buffer("A_calib", gcn_norm_dense(np.asarray(ac, dtype=np.float32)))

        if self.wprior:
            _A = np.asarray(ac if (adj_calib is not None and use_calib) else adj,
                            dtype=np.float32)
            _P = _A / (_A.sum(1, keepdims=True) + 1e-9)
            self.register_buffer("logP",
                                 torch.tensor(np.log(_P + 1e-6), dtype=torch.float32))
        else:
            self.logP = None

        if scope_mask is not None and use_scope:
            self.register_buffer("scope", torch.tensor(np.asarray(scope_mask), dtype=torch.bool))
        else:
            self.scope = None

        # ---- 图谱证据子图: 以无向可达性计算 h 跳邻域 (含自身) ----
        if use_graph:
            Nb = hop_neighborhood(np.asarray(adj, dtype=np.float32), hops=self.hops)
            self.register_buffer("neigh", torch.tensor(Nb, dtype=torch.bool))
        else:
            self.neigh = None

    def _graph_encode(self, base):
        """在标定加权的先修图上做 steps 层消息传递, 得到技能结构表示 E [K,d]。"""
        H = base                                          # [K,d]
        for i, lin in enumerate(self.gcn_w):
            msg = torch.einsum("ij,jd->id", self.A_calib, H)
            H = torch.tanh(lin(H + msg))
        return H

    def forward(self, skill, correct, question=None, scope_id=None):
        B, T = skill.shape
        if question is None or (question < 0).all():
            question = torch.zeros_like(skill)
        question = question.clamp(min=0)

        # ---- CKG 技能结构编码 (含反向耦合标定边权) ----
        E = self._graph_encode(self.skill_emb.weight) if self.use_graph else self.skill_emb.weight
        s = E[skill] if self.graph_in else self.skill_emb(skill)   # [B,T,d]
        c = self.ans_emb(correct.long().clamp(0, 1))
        q = self.diff_emb(question)
        # 循环结构自带时序归纳偏置，不再叠加位置编码；Transformer 需要显式位置信息
        x = self.pos(s + c + q) if self.seq_enc == "transformer" else (s + c + q)

        if self.seq_enc == "transformer":
            Tmask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            h = self.enc(x, mask=Tmask)                   # LKG: 学生认知状态序列 [B,T,d]
        else:
            h = self.enc(x)[0]

        # ---- CSG 作用域门控: 构造 [B,K] 的可用性掩码 ----
        allow = None
        if self.scope is not None and scope_id is not None:
            n_scope = self.scope.size(0)
            valid = (scope_id >= 0) & (scope_id < n_scope)
            m = self.scope[scope_id.clamp(0, n_scope - 1)]        # [B,K] bool
            allow = m | (~valid).unsqueeze(-1)                    # 无画像学生不做门控

        # ---- 图谱证据子图掩码: 只允许检索当前技能的多跳邻域 ----
        if self.neigh is not None:
            nb = self.neigh[skill]                                # [B,T,K]
        else:
            nb = torch.ones(B, T, self.K, dtype=torch.bool, device=skill.device)

        # ---- 作用域受限的图谱注意力检索 ----
        qq = self.q_proj(torch.cat([h, s], -1))                    # [B,T,d]
        kk = self.k_proj(E).unsqueeze(0).expand(B, -1, -1)         # [B,K,d]
        vv = self.v_proj(E).unsqueeze(0).expand(B, -1, -1)
        att = torch.einsum("btd,bkd->btk", qq, kk) / math.sqrt(self.d)
        if self.logP is not None:                                 # 结构先验偏置
            att = att + self.wprior * self.logP[skill]
        valid_k = nb & (allow.unsqueeze(1) if allow is not None else True)
        any_valid = valid_k.any(-1, keepdim=True)
        att = att.masked_fill(~(valid_k | (~any_valid)), float("-inf"))
        if 0 < self.topm < self.K:                                # 邻域稀疏化
            thr = torch.topk(att, self.topm, dim=-1).values[..., -1:]
            att = att.masked_fill(att < thr, float("-inf"))
        att = F.softmax(att, dim=-1)
        g = torch.einsum("btk,bkd->btd", att, vv)                  # 图证据 [B,T,d]
        g = g * torch.sigmoid(self.gate(qq))                       # 自门控

        z = torch.tanh(self.fc1(self.drop(torch.cat([h, s], -1)))
                       + torch.sigmoid(self.alpha) * g) if self.gated_fuse else \
            torch.tanh(self.fc1(self.drop(torch.cat([h, s, g], -1))))
        logit = self.fc2(z).squeeze(-1)                            # [B,T]
        return torch.sigmoid(logit).unsqueeze(-1).expand(-1, -1, self.K)


# ==========================================================================
MODEL_ZOO = dict(BKT=None, DKT=DKT, DKVMN=DKVMN, SAKT=SAKT, AKT=AKT, GKT=GKT, DGCLKT=DGCLKT)
