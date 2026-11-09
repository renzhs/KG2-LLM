#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识追踪实验主程序。

用法:
  python experiments/run/run_kt.py --dataset dnui_syn --models DKT AKT DGCLKT
  python experiments/run/run_kt.py --dataset assist09 --models all --epochs 30
  python experiments/run/run_kt.py --dataset dnui_syn --ablation

指标: AUC / ACC / RMSE （全部作答位置）
"""
import os, sys, json, time, math, gc, pickle, argparse, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.dirname(HERE)
REPO = os.path.dirname(EXPROOT)
sys.path.insert(0, EXPROOT)

from paths import PROCESSED as PROC, RESULTS as RES   # noqa: E402
from models.kt_models import (BKT, BKTPlus, DKT, DKTPlus, DKVMN, SAKT, AKT, GKT,
                              SAINT, SimpleKT, DGCLKT)  # noqa: E402

# 全模型顺序（单一事实来源：run_kt_multi.py 也引用它，避免两处列表漂移）
ALL_MODELS = ["BKT", "BKT+", "DKT", "DKT+", "DKVMN", "SAKT", "AKT", "GKT",
              "SAINT", "SimpleKT", "DGCLKT"]

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 超参搜索结果里不属于配置的字段（指标列），载入最优配置时要剔除
TUNE_META = ("val", "test", "acc", "epochs_run", "sec")


def apply_tuned(args, verbose=True):
    """若指定 --use-tune：用 results/kt_tune_<ds>_stage*.json 中**验证集**最优配置覆盖 args。

    配置本身由 run_kt_tune.py 按验证集 AUC 选出；本函数只做"载入"，
    不让测试集参与任何选择，保证"测试集只用于报告"的实验纪律。
    """
    if not getattr(args, "use_tune", False):
        return args
    for st in (5, 4, 3, 2, 1):
        p = os.path.join(RES, f"kt_tune_{args.dataset}_stage{st}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        cfg = {k: v for k, v in (d.get("best") or {}).items() if k not in TUNE_META}
        if not cfg:
            continue
        for k, v in cfg.items():
            setattr(args, k, v)
        if verbose:
            print(f"[use-tune] {args.dataset}: 采用验证集最优配置 {cfg}（stage{st}）")
        return args
    if verbose:
        print(f"[use-tune] {args.dataset}: 未找到搜索结果，沿用默认配置")
    return args


# --------------------------------------------------------------------------
class KTData(Dataset):
    """把序列转成 (skill, 右移correct) -> target 的标准 KT 形式。
    sid 会通过 s2i 映射为作用域掩码的行号（无画像则返回 -1）。

    长序列处理：**切块(chunk)而非截断(truncate)**。
    AKT 官方 loader 对 len>seqlen 的序列做 n_split 分段（见 arghosh/AKT
    load_data.py），这样评测覆盖**全部**交互，且不同模型的评测样本量一致。
    早期实现用 sk[:max_len] 硬截断，会在 Statics11/ASSIST17 这类长序列数据集上
    丢弃大量测试交互（Statics11 平均序列长 568，截断后仅剩 ~35%），
    使 AUC 不可比、也无法与已发表数字对齐。
    """
    def __init__(self, seqs, max_len=200, min_len=3, scope=None, s2i=None, chunk=True):
        self.items = []
        self.s2i = s2i or {}
        for s in seqs:
            sk, co = s["skill"], s["correct"]
            if len(sk) < min_len:
                continue
            qq = s.get("question")
            if qq is None:
                qq = np.full(len(sk), -1, dtype=np.int64)
            if chunk and len(sk) > max_len:
                for st in range(0, len(sk), max_len):
                    a, b = st, min(st + max_len, len(sk))
                    if b <= a:
                        continue
                    self.items.append((sk[a:b], co[a:b], qq[a:b], s.get("sid")))
            else:
                self.items.append((sk[:max_len], co[:max_len], qq[:max_len], s.get("sid")))
        self.scope = scope

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sk, co, qq, sid = self.items[i]
        inp_r = np.concatenate([[0.0], co[:-1]]).astype(np.float32)   # 右移一位
        row = self.s2i.get(sid, -1) if (self.scope is not None and sid is not None) else -1
        # 题目 id 一并返回：AKT/DIMKT/SimpleKT 等基线依赖 Rasch 难度/题目级嵌入，
        # 早期实现未传递该列，导致这些基线实际退化为"只有技能级"，对它们不公平。
        return (torch.from_numpy(sk.astype(np.int64)),
                torch.from_numpy(inp_r),
                torch.from_numpy(co.astype(np.float32)),
                torch.from_numpy(np.asarray(qq, dtype=np.int64)),
                torch.tensor(row, dtype=torch.long))


def collate(batch):
    T = max(x[0].size(0) for x in batch)
    B = len(batch)
    sk = torch.zeros(B, T, dtype=torch.long)
    qq = torch.full((B, T), -1, dtype=torch.long)
    ir = torch.zeros(B, T); tg = torch.zeros(B, T); mask = torch.zeros(B, T, dtype=torch.bool)
    sid = torch.zeros(B, dtype=torch.long)
    for i, (s_, r_, t_, q_, d_) in enumerate(batch):
        n = s_.size(0)
        sk[i, :n] = s_; ir[i, :n] = r_; tg[i, :n] = t_
        qq[i, :n] = q_; mask[i, :n] = True
        sid[i] = d_
    return sk, ir, tg, qq, mask, sid


# --------------------------------------------------------------------------
def build_graph_arrays(obj, scope="all"):
    """从数据集构造 (adj, adj_calib)。

    先修来源：
      · DNUI-IKG 等：pkl 里带教学大纲给出的真实 PREREQUISITE_OF（含数据标定的边权 w）；
      · 公开数据集（ASSIST09/12 等）：**没有**先修标注，用 models.kg_induct 的
        「相邻共现 + 平均相对位置」归纳一个有向无环先修图（方向由标量位置序决定，
        必然无环），边权 w = 位置差 ∈ [0,1]。该图同时供路径任务计算先修合规性。

    `scope` 只对"归纳型"图生效（教学大纲型图不受影响），用于审稿意见 M10 的
    泄漏敏感性对照：
      · "all"（默认，主实验口径）—— 用 train+valid+test 全量序列归纳，
        与既有 KT 文献对基准图的处理一致，且**对全部对比方法施加同一张图**，
        故不改变模型间的相对比较；
      · "train" —— 只用训练学生的序列归纳，彻底排除测试学生的行为信息。
    """
    K = obj["n_skill"]
    pre = obj.get("prereq")
    # `obj["prereq"]` 对公开基准**非空**：它是预处理阶段用"共现+位置序"归纳后
    # 落盘的结果（见 experiments/data/prep_bench.py），并非课程标注。
    # 因此不能用"pre 是否为空"来判断图是否可重归纳，否则 --graph-scope 会静默失效
    # （实测：adj nnz 恒为 256，与 all 口径完全相同）。判据必须与 run_rec._adj_scope
    # 一致——只有带学段—专业标注 / 来源为 DNUI 的数据集才是教学大纲图。
    from_curriculum = bool(obj.get("skill_scope")) or \
        str((obj.get("meta") or {}).get("source", "")).startswith("DNUI")
    if scope == "train" and not from_curriculum:
        from models.kg_induct import induce_prereq
        n_old = sum(len(v or []) for v in (obj.get("prereq") or []))
        pre, ne, _ = induce_prereq(list(obj["train"]), K)
        print(f"[graph] graph-scope=train → 只用训练学生重归纳先修边 {ne} 条"
              f"（原落盘图 {n_old} 条，已弃用）")
    elif scope == "train":
        print("[graph] 先修边来自教学大纲，graph-scope 不生效（保持原图）")
    elif not pre:
        from models.kg_induct import induce_prereq
        seqs = obj["train"] + obj["valid"] + obj["test"]
        pre, ne, _ = induce_prereq(seqs, K)
        print(f"[graph] 无先修标注 → 归纳先修边 {ne} 条（共现+位置序）")
    adj = np.zeros((K, K), dtype=np.float32)
    adj_calib = np.zeros((K, K), dtype=np.float32)
    for u, vs in enumerate(pre):
        for v, w in (vs or []):
            adj[u, v] = 1.0
            # 反向标定: w_calib = 1 + tanh(w)，w 为数据标定/位置差得到的先修强度
            adj_calib[u, v] = 1.0 + math.tanh(w)
    return adj, adj_calib


def build_scope_mask(obj):
    """(S,K) 作用域掩码 + student_id -> row 映射。
    student_scope 的值支持两种格式: (grade, major) 元组 或 dict(grade=, major=)。"""
    ss = obj.get("skill_scope")
    stu = obj.get("student_scope") or {}
    if not ss or not stu:
        return None, None
    ids = sorted(stu.keys())
    s2i = {s: i for i, s in enumerate(ids)}
    K = obj["n_skill"]
    M = np.zeros((len(ids), K), dtype=bool)
    # 预取技能作用域
    meta = {int(k): (set(v["grades"]), set(v["majors"])) for k, v in ss.items()}
    for s in ids:
        v = stu[s]
        if isinstance(v, dict):
            g, mj = v.get("grade"), v.get("major")
        else:
            g, mj = v[0], v[1]
        for k, (gs, ms) in meta.items():
            M[s2i[s], k] = bool((g in gs) and ((mj in ms) or (mj == "其他")))
    return M, s2i


# --------------------------------------------------------------------------
def _rankdata_avg(a):
    """平均秩（并列值取秩的平均），Mann-Whitney U 公式要求如此。

    早期实现用 `argsort(kind="mergesort")` 直接赋 1..N 的秩，遇到并列值会把
    "先出现"的样本排在前面 —— 而正样本被 np.concatenate 放在数组前段，
    于是**并列时正样本系统性获得更小的秩**，AUC 被系统性压低。
    极端情况：常数预测（全部并列）本应得 AUC=0.5，旧实现给出 AUC=0.0；
    BKT 在 Statics11 上预测值高度集中（99.9% 的块长为 1，预测近似按技能取常数），
    旧实现因此报出 AUC=0.03 的假象。
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        # 该并列组的平均秩（秩从 1 开始）
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    acc = float(((p >= 0.5).astype(int) == y.astype(int)).mean())
    # AUC (秩和法，并列值取平均秩)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        ranks = _rankdata_avg(p)
        rp = ranks[y == 1].sum()
        n1, n0 = len(pos), len(neg)
        auc = float((rp - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return dict(AUC=round(auc, 4), ACC=round(acc, 4), RMSE=round(rmse, 4),
                n=int(len(y)), base_rate=round(float(y.mean()), 4))


@torch.no_grad()
def eval_torch(model, loader, K, collect=False):
    model.eval()
    ys, ps = [], []
    for sk, ir, tg, qq, mask, sid in loader:
        sk = sk.to(DEV); ir = ir.to(DEV); tg = tg.to(DEV); mask = mask.to(DEV)
        sid_d = sid.to(DEV); qq_d = qq.to(DEV)
        out = forward_model(model, sk, ir, qq_d, sid_d, K)
        # 取每个位置真实技能对应的预测
        idx = sk.unsqueeze(-1)
        yh = out.gather(2, idx).squeeze(-1)
        ys.append(tg[mask].cpu().numpy()); ps.append(yh[mask].cpu().numpy())
    return metrics(np.concatenate(ys), np.concatenate(ps))


def forward_model(model, sk, ir, qq, sid, K):
    """统一前向分发。qq 为题目 id（无题目列的数据集为 -1）。

    需要题目信息的模型（AKT 的 Rasch 难度、SAINT/SimpleKT 的题目嵌入）显式接收 qq；
    其余模型只吃 (skill, response)。
    """
    if isinstance(model, DGCLKT):
        return model(sk, ir, qq, sid)
    if isinstance(model, (AKT, SAINT, SimpleKT)):
        return model(sk, ir, qq)
    return model(sk, ir)


def predict_bkt(model, seqs):
    ys, ps = [], []
    seqs = [dict(skill=s["skill"], correct=s["correct"]) for s in seqs]
    preds = model.predict_seq(seqs)
    for s, p in zip(seqs, preds):
        # 右移: 用 x_0..x_{t-1} 预测 r_t -> 用上一位置的概率
        pr = np.concatenate([[0.5], p[:-1]])
        ys.append(s["correct"]); ps.append(pr)
    return metrics(np.concatenate(ys), np.concatenate(ps))


# --------------------------------------------------------------------------
def effective_batch(name, K, batch):
    """按技能空间规模自适应下调 batch，避免大 K 数据集 OOM。

    显存随 K 线性（图/输出层）或 B·K 线性（GKT 的逐技能 GRU）增长：
      · Statics11 K=1223 时，batch=64 会让 GKT 的 B·K=78k 行 GRU 反向图吃掉数十 GB；
      · DGCLKT 的图注意力同样按 K 铺开。
    故对 K>256 的数据集统一下调 4 倍，GKT 再额外下调 2 倍（其 B·K 复杂度最高）。
    该下调只影响优化批大小，不改变数据与评测协议。
    """
    bs = batch
    if K > 256:
        bs = max(4, bs // 4)
    if K > 256 and name == "GKT":
        bs = max(2, bs // 2)
    return bs


def train_one(name, args, obj, adj, adj_calib, scope, s2i, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    K, Q = obj["n_skill"], obj["n_question"]
    # 每模型独立配置（公平比较协议）：
    #   `--model-cfg-file` 给出 {model: {lr, d, layers, hidden, ...}}，由
    #   tools/tune_baselines.py 在**验证集**上选出；未给出的键回落到全局 args。
    #   基线各自在自己的 lr 与容量上取最优，避免"我方精调、对手不调"的不对等比较。
    mcfg = dict((getattr(args, "model_cfg", None) or {}).get(name) or {})
    _lrm = getattr(args, "lr_map", None) or {}
    lr = float(mcfg.get("lr", _lrm.get(name, args.lr)))
    _chunk = not getattr(args, "no_chunk", False)
    # 超参搜索开销控制：`train_max_users` 只截断**训练集**（前 N 名用户），
    # 验证集/测试集保持全量 —— 保证"选参依据"与"最终评测"都不被稀释。
    # 仅由调参脚本（tune_baselines / tune_main_chain）设置；阶段B 评测时恒为 0（全量）。
    _tru = int(getattr(args, "train_max_users", 0) or 0)
    _tr_seqs = obj["train"][:_tru] if _tru > 0 else obj["train"]
    tr = KTData(_tr_seqs, args.max_len, args.min_len, scope, s2i, chunk=_chunk)
    va = KTData(obj["valid"], args.max_len, args.min_len, scope, s2i, chunk=_chunk)
    te = KTData(obj["test"], args.max_len, args.min_len, scope, s2i, chunk=_chunk)
    if args.max_users:
        tr.items = tr.items[:args.max_users]
        va.items = va.items[:max(1, args.max_users // 10)]
        te.items = te.items[:max(1, args.max_users // 5)]
    # 每模型批大小：model-cfg 里的 `batch` 可覆盖全局值。用于"逐时刻循环实现"的
    # DKVMN/GKT —— 它们每个时刻都要发起十余个小内核，成本由**内核启动次数**支配，
    # batch=64 时 GPU 利用率极低（单 epoch 与数据量几乎无关）。放大批大小使前向
    # 次数按比例减少，是纯粹的开销优化，不改变模型容量。缺省仍走 effective_batch
    # 的自适应下调（大 K 时防 OOM）。
    bs = int(mcfg.get("batch", 0) or 0) or effective_batch(name, K, args.batch)

    def make():
        # 容量超参统一从 args 取（getattr 兜底），使 tools/tune_baselines.py 的
        # 第二阶段"容量搜索"能把 d/hidden/layers/mem 加进搜索空间。
        # 未显式给出时等于各模型论文/参考实现的常用默认值。
        _g = lambda a, dflt: mcfg.get(a, getattr(args, a, dflt))   # noqa: E731
        _d = int(_g("d", 128)); _ly = int(_g("layers", 1)); _dr = float(_g("dropout", 0.2))
        _hid = int(_g("hidden", 128))
        if name == "DKT":
            return DKT(K, hidden=_hid, layers=max(1, _ly), dropout=_dr).to(DEV)
        if name == "DKT+":
            return DKTPlus(K, hidden=_hid, dropout=_dr).to(DEV)
        if name == "DKVMN":
            return DKVMN(K, mem_size=int(_g("mem_size", 64)),
                         key_dim=int(_g("key_dim", 64)),
                         val_dim=int(_g("val_dim", 64))).to(DEV)
        if name == "SAKT":
            return SAKT(K, d=_d, layers=max(1, int(_g("layers", 2))), dropout=_dr).to(DEV)
        if name == "AKT":
            return AKT(K, Q, d=_d, layers=_ly, dropout=_dr).to(DEV)
        if name == "GKT":
            return GKT(K, hidden=int(_g("hidden", 64)), steps=int(_g("steps", 3)),
                       dropout=_dr, adj=adj).to(DEV)
        if name == "SAINT":
            return SAINT(K, Q, d=_d, enc_layers=_ly, dec_layers=_ly,
                         dropout=_dr).to(DEV)
        if name == "SimpleKT":
            return SimpleKT(K, Q, d=_d, layers=_ly, dropout=_dr).to(DEV)
        if name == "DGCLKT":
            opt = dict(adj=adj, adj_calib=adj_calib, scope_mask=scope,
                       d=_g("d", 128), layers=_g("layers", 1), steps=_g("steps", 2),
                       dropout=_g("dropout", 0.2), gated_fuse=bool(_g("gated_fuse", False)),
                       hops=(_g("hops", None) or None),
                       topm=_g("topm", 0), wprior=_g("wprior", 0.0),
                       graph_in=bool(_g("graph_in", False)),
                       seq_enc=_g("seq_enc", "transformer"),
                       use_graph=not args.ab_no_graph, use_calib=not args.ab_no_calib,
                       use_scope=not args.ab_no_scope)
            if args.ab_struct_only:
                opt["adj_calib"] = adj
            if args.ab_no_scope:
                opt["scope_mask"] = None
            return DGCLKT(K, Q, **opt).to(DEV)
        raise KeyError(name)

    if name in ("BKT", "BKT+"):
        m = BKT(K) if name == "BKT" else BKTPlus(K)
        t0 = time.time()
        # max_obs：EM 耗时随观测总量线性增长；超大数据集（ASSIST2012）按等间隔
        # 抽样把参与拟合的观测限制在 40 万以内。BKT/BKT+ 只有 4K 个技能级共享
        # 参数，该预算下已充分收敛；其余数据集观测总量低于该预算，不受影响。
        m.fit(obj["train"][:20000], max_obs=400000)
        r_gamma = None
        if name == "BKT+":
            # γ 在**验证集**上选（测试集只记录），与其它基线"调参只用验证集"的纪律一致
            r_gamma = m.select_gamma(obj["valid"])
            print(f"    [BKT+] 验证集选 γ={r_gamma[0]} (val AUC={r_gamma[1]:.4f})")
        r = predict_bkt(m, obj["test"] if not args.max_users else obj["test"][:args.max_users // 5])
        r["train_sec"] = round(time.time() - t0, 1)
        r["params"] = 4 * K
        if r_gamma:
            r["gamma"] = r_gamma[0]; r["best_valid_auc"] = round(r_gamma[1], 4)
        return r

    def _fit(bs):
        """按给定批大小完整训练 + 评测一次。

        抽成函数是为了支持 **OOM 自适应重试**：阶段B 多进程共享单卡时，
        显存竞争会让"本来不超显存"的模型（实测 GKT 的 B·K 行 GRU 反向图最吃紧）
        偶发 OOM —— 崩溃会被上层 except 吞掉并只记一行 error，最终静默地少一个基线。
        这里改为"降批重试"，把偶发竞争转成可恢复事件。
        """
        tl = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=len(tr) > bs)
        vl = DataLoader(va, batch_size=bs * 2, shuffle=False, collate_fn=collate, num_workers=2)
        tl2 = DataLoader(te, batch_size=bs * 2, shuffle=False, collate_fn=collate, num_workers=2)
        model = make()
        nparam = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
        bce = nn.BCELoss(reduction="none")
        best, best_state, patience = -1, None, 0
        t0 = time.time(); hist = []
        for ep in range(1, args.epochs + 1):
            model.train(); tot, nb = 0.0, 0
            for sk, ir, tg, qq, mask, sid in tl:
                sk = sk.to(DEV); ir = ir.to(DEV); tg = tg.to(DEV); mask = mask.to(DEV)
                out = forward_model(model, sk, ir, qq.to(DEV), sid.to(DEV), K)
                yh = out.gather(2, sk.unsqueeze(-1)).squeeze(-1)
                loss = (bce(yh, tg) * mask).sum() / mask.sum().clamp(min=1)
                # 支持带辅助正则项的基线（如 DKT+ 的重建/单调性/一致性损失）
                _aux = getattr(model, "aux_loss", None)
                if _aux is not None:
                    loss = loss + _aux
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); tot += loss.item(); nb += 1
            vm = eval_torch(model, vl, K)
            hist.append(dict(epoch=ep, loss=round(tot / max(nb, 1), 4), valid=vm))
            cur = vm["AUC"] if not math.isnan(vm["AUC"]) else vm["ACC"]
            sched.step(cur)
            if cur > best:
                best, patience = cur, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        r = eval_torch(model, tl2, K)
        r["epochs_run"] = len(hist); r["train_sec"] = round(time.time() - t0, 1)
        r["params"] = nparam; r["history"] = hist[-6:]
        # 供超参搜索使用：验证集最优 AUC（测试集只用于最终报告，不参与选择）
        r["best_valid_auc"] = round(float(best), 4)
        # 供分组评测脚本复用训练好的模型（默认关闭，避免污染 JSON 序列化）
        if getattr(args, "return_model", False):
            r["_model"] = model
        return r

    def _is_oom(e):
        return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

    tries = max(0, int(getattr(args, "oom_retries", 3) or 0))
    backoff = float(getattr(args, "oom_backoff", 15) or 0)
    used = []
    for t in range(tries + 1):
        try:
            r = _fit(bs)
            if used:
                r["oom_downgrades"] = used          # 记录降批历史，便于核查协议影响
            r["batch_used"] = bs
            return r
        except Exception as e:                      # noqa: BLE001
            if not _is_oom(e) or t >= tries or bs <= 2:
                raise
            used.append(bs)
            del e
            gc.collect()
            if DEV == "cuda":
                torch.cuda.empty_cache()
            nxt = max(2, bs // 2)
            print(f"   [OOM-retry {t + 1}/{tries}] {name}: batch {bs} -> {nxt}，"
                  f"等待 {backoff:.0f}s 让同卡进程释放显存", flush=True)
            bs = nxt
            time.sleep(backoff)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--no-chunk", action="store_true",
                    help="长序列改为硬截断（默认按 AKT 协议切块，覆盖全部交互）")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-users", type=int, default=0)
    ap.add_argument("--train-max-users", type=int, default=0,
                    help="只截断训练集到前 N 名用户（0=全量）；用于调参阶段控制开销，"
                         "验证/测试集不受影响。阶段B 评测请保持为 0。")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    # ---- DGCL-KT 结构/容量超参（原为硬编码；暴露出来以便在验证集上做选择）----
    ap.add_argument("--d", type=int, default=128, help="DGCLKT 隐层维度")
    ap.add_argument("--layers", type=int, default=1, help="Transformer 层数")
    ap.add_argument("--steps", type=int, default=2, help="图卷积层数")
    ap.add_argument("--hidden", type=int, default=128, help="DKT/DKT+ 隐层维度")
    ap.add_argument("--mem-size", type=int, default=64, dest="mem_size", help="DKVMN 记忆槽数")
    ap.add_argument("--key-dim", type=int, default=64, dest="key_dim", help="DKVMN key 维度")
    ap.add_argument("--val-dim", type=int, default=64, dest="val_dim", help="DKVMN value 维度")
    ap.add_argument("--hops", type=int, default=0, help="图证据邻域跳数(0 表示 steps+1)")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gated-fuse", action="store_true", help="图证据残差门控融合")
    ap.add_argument("--topm", type=int, default=0, help="图证据邻域稀疏化：只保留 top-m 邻域")
    ap.add_argument("--wprior", type=float, default=0.0, help="注意力结构先验强度 β")
    ap.add_argument("--graph-in", action="store_true", help="序列输入使用图平滑后的技能嵌入")
    ap.add_argument("--seq-enc", default="transformer", choices=("transformer", "gru", "lstm"),
                    help="DGCLKT 序列通路编码器类型")
    ap.add_argument("--use-tune", action="store_true",
                    help="自动载入 results/kt_tune_<ds>_stage*.json 中验证集最优配置")
    ap.add_argument("--lr-map", default="",
                    help='每模型学习率 JSON，如 \'{"AKT":0.001,"DKT":0.002}\'；'
                         "未列出的模型用 --lr。由 tools/baseline_lr_map.py 生成")
    ap.add_argument("--model-cfg-file", default="", dest="model_cfg_file",
                    help="每模型完整配置 JSON 文件：{model: {lr,d,layers,hidden,...}}。"
                         "由 tools/tune_baselines.py 两阶段（lr → 容量）搜索产出，"
                         "键值自动并入该模型的全局参数。")
    ap.add_argument("--ablation", action="store_true")
    # ---- 容错与续跑（阶段B 多进程共享单卡时必需）----
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries",
                    help="CUDA OOM 时降批重试的次数（默认 3，即 64→32→16→8）。"
                         "多进程共享单卡时显存竞争会让 GKT 这类大户偶发 OOM，"
                         "旧行为是直接抛异常被吞掉、静默少一个基线。")
    ap.add_argument("--oom-backoff", type=float, default=15.0, dest="oom_backoff",
                    help="每次降批前的等待秒数，给同卡进程释放显存的时间")
    ap.add_argument("--no-reuse", action="store_true", dest="no_reuse",
                    help="关闭模型级续跑（默认开启：输出文件里已成功的模型直接复用，"
                         "带 error 的模型自动重算），使「只补个别模型」的续跑成为可能")
    ap.add_argument("--redo-models", nargs="*", default=[], dest="redo_models",
                    help="强制重算这些模型（即使输出里已有成功结果）。用于"
                         "tools/extend_lr_search.py 改过某基线的 lr/容量后，"
                         "让该模型在新配置下重跑，而其余模型保持复用。")
    # 消融开关
    ap.add_argument("--ab-no-graph", action="store_true", help="E2: 去掉 CKG 图编码")
    ap.add_argument("--ab-no-calib", action="store_true", help="E3: 去掉反向耦合标定")
    ap.add_argument("--ab-no-scope", action="store_true", help="E4: 去掉 CSG 作用域门控")
    ap.add_argument("--ab-struct-only", action="store_true", help="E5: 仅结构先验, 无经验边权")
    ap.add_argument("--graph-scope", default="all", choices=("all", "train"),
                    help="归纳先修图的数据范围：all=全量序列（主实验口径），"
                         "train=仅训练学生（M10 泄漏敏感性对照）。"
                         "对教学大纲型图（DNUI-IKG）不生效。")
    args = apply_tuned(ap.parse_args())
    # 归一化成 dict，便于 train_one 直接查表；empty 表示全部走全局 --lr
    args.lr_map = json.loads(args.lr_map) if args.lr_map.strip() else {}
    args.model_cfg = {}
    if args.model_cfg_file and os.path.exists(args.model_cfg_file):
        with open(args.model_cfg_file, encoding="utf-8") as f:
            raw = json.load(f)
        args.model_cfg = {k: v for k, v in raw.items() if isinstance(v, dict)}
        print(f"[model-cfg] 载入 {len(args.model_cfg)} 个模型的独立配置: "
              f"{ {k: v for k, v in list(args.model_cfg.items())[:4]} } ...")

    p = os.path.join(PROC, f"{args.dataset}.pkl")
    if not os.path.exists(p):
        print("missing", p); return
    with open(p, "rb") as f:
        obj = pickle.load(f)
    print(f"dataset={args.dataset} skill={obj['n_skill']} q={obj['n_question']} "
          f"train/val/test={len(obj['train'])}/{len(obj['valid'])}/{len(obj['test'])}")
    print("meta:", json.dumps(obj.get("meta", {}), ensure_ascii=False)[:400])

    adj, adj_calib = build_graph_arrays(obj, scope=getattr(args, "graph_scope", "all"))
    scope, s2i = build_scope_mask(obj)
    print(f"adj nnz={int((adj>0).sum())}  calib nnz={int((adj_calib>0).sum())}  "
          f"scope={'yes' if scope is not None else 'no'}"
          f"{' shape='+str(scope.shape) if scope is not None else ''}")

    models = args.models
    if models == ["all"]:
        models = list(ALL_MODELS)
    out = dict(dataset=args.dataset, args=vars(args), results={},
               data_meta={k: v for k, v in obj.get("meta", {}).items()
                          if k != "stats"} | dict(stats=obj.get("meta", {}).get("stats")))
    tag = args.tag or ("abl" if args.ablation else "main")
    outfile = os.path.join(RES, f"kt_{args.dataset}_{tag}.json")
    # ---- 模型级续跑 --------------------------------------------------------
    # 必须**继承**已有 results 再改写，否则「只补跑 GKT」会把这个折里其余 10 个
    # 模型的结果全部丢掉（旧实现每次从空 dict 开始）。
    # 复用判据是「有 AUC 且无 error」：失败过的模型自动重算，成功的复用。
    prev = {}
    if os.path.exists(outfile) and not getattr(args, "no_reuse", False):
        try:
            with open(outfile, encoding="utf-8") as f:
                _p = json.load(f)
            if isinstance(_p.get("results"), dict):
                prev = _p["results"]
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] 旧产物无法解析（将整体重算）: {e}")
    out["results"] = dict(prev)
    if prev:
        nok = sum(1 for v in prev.values()
                  if isinstance(v, dict) and "AUC" in v and "error" not in v)
        nbad = sum(1 for v in prev.values()
                   if isinstance(v, dict) and "error" in v)
        print(f"[resume] 已有 {len(prev)} 条记录：可复用 {nok}，失败待重算 {nbad}")
    for name in models:
        old = prev.get(name)
        if name in (getattr(args, "redo_models", None) or []):
            print(f"\n>>> {name} on {args.dataset} ... [redo：配置已更新，强制重算]")
        elif (not getattr(args, "no_reuse", False) and isinstance(old, dict)
                and "AUC" in old and "error" not in old):
            print(f"\n>>> {name} on {args.dataset} ... [reuse] AUC={old.get('AUC')}")
            continue
        else:
            print(f"\n>>> {name} on {args.dataset} ...", flush=True)
        try:
            r = train_one(name, args, obj, adj, adj_calib, scope, s2i, args.seed)
            print("   ", {k: r[k] for k in ("AUC", "ACC", "RMSE", "n", "params") if k in r})
            out["results"][name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            out["results"][name] = dict(error=str(e)[:400])
        # 增量落盘
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    missing = [m for m in out["results"]
               if isinstance(out["results"][m], dict) and "AUC" not in out["results"][m]]
    if missing:
        print(f"\n[WARN] 以下模型**未成功**（结果将被聚合脚本判为缺失，请续跑）: {missing}")
    print("\nSAVED", outfile)


if __name__ == "__main__":
    sys.exit(main())
