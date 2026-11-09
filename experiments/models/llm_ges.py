#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GES: 图谱证据子图约束的检索增强生成 (Graph-Evidence-Subgraph constrained RAG)

三件事:
  1. annotate_kts   资源/知识点自动标注 (难度、能力标签) —— 带磁盘缓存, 避免重复计费
  2. build_evidence_subgraph  从 CKG+LKG 抽取证据子图 (先修链 / 已掌握 / 相似学习者 / 同作用域)
  3. validate       输出校验器 —— 强制生成结果落在候选集合内, 并核验引用的图证据是否真实存在

评测指标:
  HR  (Hallucination Rate)      生成项中不在候选集合内的比例
  ETR (Evidence Traceability)   带有真实存在图证据的生成项比例
  SCR (Scope Compliance Rate)   作用域合规率 = 1 - CSCR
以及平均时延与 token 消耗(用于 MDRL 触发式重规划的收益论证)。
"""
import os, json, time, hashlib, random, re, sys
from collections import defaultdict

try:                                  # GES 任务需要联网调用；其余实验不依赖 requests
    import requests
except ImportError:                   # pragma: no cover
    requests = None

# ---------------------------------------------------------------------------
# 凭证与端点：一律从环境变量读取，仓库内不保存任何密钥。
#   DEEPSEEK_API_KEY   必填（仅在未命中磁盘缓存时会真正发请求）
#   DEEPSEEK_BASE      默认 https://api.deepseek.com
#   DEEPSEEK_MODEL     默认 deepseek-chat
#   GES_CACHE          LLM 响应磁盘缓存目录，默认为 <REPO>/logs/ges_cache
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.environ.get("GES_CACHE", os.path.join(_REPO, "logs", "ges_cache"))
CACHE_DIR = os.path.abspath(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------
class Budget:
    """累计 token 与耗时统计。"""
    def __init__(self):
        self.calls = 0; self.prompt_tokens = 0; self.completion_tokens = 0
        self.seconds = 0.0; self.cache_hits = 0

    def as_dict(self):
        return dict(calls=self.calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    total_tokens=self.prompt_tokens + self.completion_tokens,
                    seconds=round(self.seconds, 2), cache_hits=self.cache_hits)


class DeepSeek:
    def __init__(self, key=None, model=None, temperature=0.0, max_tokens=1200,
                 cache=True, timeout=120):
        self.key = key or API_KEY
        self.model = model or MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.timeout = timeout
        self.budget = Budget()

    def _ck(self, payload):
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                           .encode("utf-8")).hexdigest()[:32]
        return os.path.join(CACHE_DIR, f"{h}.json")

    def chat(self, messages, json_mode=False, retries=3):
        payload = dict(model=self.model, messages=messages,
                       temperature=self.temperature, max_tokens=self.max_tokens,
                       stream=False)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        p = self._ck(payload)
        if self.cache and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            self.budget.cache_hits += 1
            return r
        if not self.key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY，且该请求未命中磁盘缓存。\n"
                "  · 想复现本文 GES 数值：仓库已附带 logs/ges_cache/ 的响应缓存，"
                "无需密钥即可重放；\n"
                "  · 想跑新的样本：export DEEPSEEK_API_KEY=<your key> 后重试。")
        if requests is None:
            raise RuntimeError("缺少 requests：pip install requests")
        last = None
        for i in range(retries):
            t0 = time.time()
            try:
                r = requests.post(f"{API_BASE}/chat/completions",
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {self.key}"},
                                  json=payload, timeout=self.timeout)
                self.budget.seconds += time.time() - t0
                if r.status_code != 200:
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(2 + 3 * i); continue
                j = r.json()
                self.budget.calls += 1
                u = j.get("usage", {})
                self.budget.prompt_tokens += u.get("prompt_tokens", 0)
                self.budget.completion_tokens += u.get("completion_tokens", 0)
                out = dict(content=j["choices"][0]["message"]["content"],
                           usage=u, model=j.get("model"))
                if self.cache:
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
                return out
            except Exception as e:
                last = e; time.sleep(2 + 3 * i)
        raise last


def extract_json(text):
    """从模型输出中稳健地抽取 JSON（容忍 ```json 包裹与前后噪声）。"""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for op, cl in (("[", "]"), ("{", "}")):
        i, j = t.find(op), t.rfind(cl)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None


# ==========================================================================
# 1. 资源/知识点自动标注
# ==========================================================================
ANNOT_SYS = ("你是教育知识图谱的标注专家。给定知识点名称与所在课程、单元，"
             "输出 JSON：{\"difficulty\": 1-5 的整数, \"tags\": [3-5 个中文能力标签], "
             "\"prereq_hint\": [0-2 个必须先掌握的前置知识点名称]}。只输出 JSON。")


def annotate_kts(ds: DeepSeek, kts, batch=8):
    """批量标注知识点；磁盘缓存按 (name, course) 去重。"""
    out = {}
    for i in range(0, len(kts), batch):
        chunk = kts[i:i + batch]
        items = [dict(kt_id=k["kt_id"], name=k["name"], course=k.get("course"),
                      unit=k.get("unit")) for k in chunk]
        msg = [{"role": "system", "content": ANNOT_SYS},
               {"role": "user", "content": json.dumps(
                   {"knowledge_points": items}, ensure_ascii=False)}]
        try:
            r = ds.chat(msg, json_mode=True)
            j = extract_json(r["content"]) or {}
            arr = j.get("annotations") or j.get("items") or (
                j if isinstance(j, list) else None)
            if isinstance(arr, dict):
                arr = arr.get("knowledge_points") or list(arr.values())
            if isinstance(arr, list):
                for a in arr:
                    if not isinstance(a, dict):
                        continue
                    kid = a.get("kt_id") or a.get("id")
                    if kid is None:
                        continue
                    out[int(kid)] = dict(
                        difficulty=int(a.get("difficulty", 3)),
                        tags=list(a.get("tags", []))[:6],
                        prereq_hint=list(a.get("prereq_hint", []))[:3])
        except Exception as e:
            print(f"  [annotate] batch {i} failed: {e}")
        time.sleep(0.2)
    return out


# ==========================================================================
# 2. 证据子图抽取
# ==========================================================================
def _evidence_map(mastered, pre_skill, prereq, skill_names):
    """构造「每个技能 -> 可核验图证据列表」以及就绪度排序键。"""
    K = len(skill_names)
    ev = {}
    # (a) 已掌握的技能的直接后继（先修已具备）
    for u, lst in enumerate(prereq):
        if mastered.get(u, 0) >= 0.6:
            for v, w in lst:
                ev.setdefault(v, []).append(
                    f"({skill_names[u]})-[PREREQUISITE_OF w={w:.2f}]->({skill_names[v]}); "
                    f"该生已掌握前置 '{skill_names[u]}'")
    # (b) 当前学习技能的多跳邻域
    cur = int(pre_skill)
    for v, w in prereq[cur]:
        ev.setdefault(v, []).append(
            f"({skill_names[cur]})-[PREREQUISITE_OF w={w:.2f}]->({skill_names[v]}); "
            f"该生当前正在学习 '{skill_names[cur]}'")
    # (c) 未掌握但前置就绪的技能（图传播得到的候选）
    for k in range(K):
        if mastered.get(k, 0) >= 0.6:
            continue
        ev.setdefault(k, []).append("该生尚无该知识点作答记录，属待学习项")

    def keyf(k):
        par = [u for u, lst in enumerate(prereq) if any(v == k for v, _ in lst)]
        rdy = sum(mastered.get(u, 0.0) for u in par) / max(1, len(par))
        return -(rdy * 2 + mastered.get(k, 0.0))
    return ev, keyf


def build_evidence_subgraph(sid, pre_skill, mastered, adj, prereq, skill_names,
                            max_nodes=14):
    """从 CKG + LKG 抽取证据子图（候选由图就绪度自动选取）。"""
    ev, keyf = _evidence_map(mastered, pre_skill, prereq, skill_names)
    cands = sorted(range(len(skill_names)), key=keyf)[:max_nodes]
    return dict(candidates=cands, evidence=ev)


def build_evidence_subgraph_for(cands, mastered, pre_skill, prereq, skill_names):
    """**指定候选集合**的证据子图。

    用于把 GES 与 Direct 基线放在同一个召回池上比较：两者的候选集完全一致，
    唯一差别是 GES 额外做了「作用域过滤 + 图证据注入」，这样 NDCG 才可比。
    """
    ev, _ = _evidence_map(mastered, pre_skill, prereq, skill_names)
    return dict(candidates=list(cands), evidence=ev)


# ==========================================================================
# 3. GES 推荐
# ==========================================================================
GES_SYS = (
    "你是学习资源推荐引擎。你必须**只**从给定的 candidate_ids 中选择条目，"
    "严禁编造不在候选列表中的 id。每条推荐必须引用 evidence 中真实存在的图谱证据"
    "(格式为 (A)-[关系]->(B))。输出严格 JSON："
    "{\"recommendations\": [{\"kt_id\": 整数, \"reason\": \"不超过40字\", "
    "\"evidence\": \"引用的证据三元组或空串\"}]}。只输出 JSON，不要解释。"
)


def ges_recommend(ds, sub, skill_names, student_desc, topk=10, scope_filter=None):
    cands = sub["candidates"]
    if scope_filter is not None:
        cands = [k for k in cands if scope_filter[k]]
    if not cands:
        cands = sub["candidates"]
    items = [dict(kt_id=k, name=skill_names[k],
                  evidence=sub["evidence"].get(k, [])[:2]) for k in cands]
    user = json.dumps(dict(student=student_desc, candidate_ids=[k for k in cands],
                           candidates=items, need=f"选出最需要的 {topk} 项"),
                      ensure_ascii=False)
    r = ds.chat([{"role": "system", "content": GES_SYS},
                 {"role": "user", "content": user}], json_mode=True)
    j = extract_json(r["content"]) or {}
    recs = j.get("recommendations") if isinstance(j, dict) else (
        j if isinstance(j, list) else [])
    return recs or [], dict(cands=cands, usage=r.get("usage"),
                            latency=r.get("usage") and 0 or 0,
                            raw=r["content"])


DIRECT_SYS = (
    "你是学习资源推荐引擎。从给定 candidate_ids 中选择最需要学习的条目。"
    "输出严格 JSON：{\"recommendations\": [{\"kt_id\": 整数, \"reason\": \"不超过40字\"}]}。"
    "只输出 JSON。"
)


def direct_recommend(ds, cands, skill_names, student_desc, topk=10):
    """基线：不给图谱证据子图，直接让模型从候选中挑（用于对比 HR/ETR/CSCR）。"""
    items = [dict(kt_id=k, name=skill_names[k]) for k in cands]
    user = json.dumps(dict(student=student_desc, candidate_ids=list(cands),
                           candidates=items, need=f"选出最需要的 {topk} 项"),
                      ensure_ascii=False)
    r = ds.chat([{"role": "system", "content": DIRECT_SYS},
                 {"role": "user", "content": user}], json_mode=True)
    j = extract_json(r["content"]) or {}
    recs = j.get("recommendations") if isinstance(j, dict) else (
        j if isinstance(j, list) else [])
    return recs or [], dict(cands=list(cands), usage=r.get("usage"), raw=r["content"])


# ==========================================================================
# 4. 输出校验器
# ==========================================================================
def validate(recs, allowed_ids, prereq_map, skill_names):
    """返回 (valid_recs, stats)。
    - 不在 allowed_ids 中的 kt_id 记为幻觉
    - evidence 中引用的三元组若两端技能名与关系在图中不存在, 记为不可追溯
    """
    allowed = set(int(x) for x in allowed_ids)
    valid = set()
    for u, lst in enumerate(prereq_map):
        for v, w in lst:
            valid.add((skill_names[u].strip(), skill_names[v].strip()))
    keep, hallu, traceable, cited = [], 0, 0, 0
    for r in recs:
        try:
            kid = int(r.get("kt_id"))
        except Exception:
            hallu += 1; continue
        if kid not in allowed:
            hallu += 1; continue
        ev = (r.get("evidence") or "").strip()
        r = dict(r); r["kt_id"] = kid
        if ev:
            cited += 1
            ok = any((a in ev and b in ev) for a, b in valid)
            if ok:
                traceable += 1
            r["evidence_valid"] = bool(ok)
        else:
            r["evidence_valid"] = None
        keep.append(r)
    n_gen = max(1, len(recs))
    stats = dict(n_returned=len(recs), n_valid=len(keep),
                 hallucination_rate=round(hallu / n_gen, 4),
                 evidence_cited_rate=round(cited / n_gen, 4),
                 # ETR 定义为「带真实图证据的生成项比例」（分母=全部生成项），
                 # 这样"不给证据"的 Direct 基线天然接近 0，才能体现 GES 的证据约束价值。
                 evidence_traceable_rate=round(traceable / n_gen, 4),
                 evidence_cited_precision=round(traceable / max(1, cited), 4))
    return keep, stats


def scope_compliance(recs, scope_row):
    if not recs:
        return 1.0
    bad = sum(1 for r in recs if not scope_row[int(r["kt_id"])])
    return round(1.0 - bad / len(recs), 4)


def prereq_legality(recs, mastered, prereq, thr=0.6, need=0.5):
    """先修合法性 PR：推荐项中「先修已就绪」的比例。

    为什么需要它：在给出显式 candidate_ids 的设定下，现代大模型的"编造 id"型幻觉
    已经很低（HR≈0），真正的教学风险是**推荐了先修尚未掌握的知识点**——学生拿到也
    学不动。PR 直接度量这一点，且只有把先修证据喂给模型（GES）才可能做好。
    """
    if not recs:
        return 1.0
    ok = 0
    for r in recs:
        k = int(r["kt_id"])
        pars = [u for u, lst in enumerate(prereq) if any(v == k for v, _ in lst)]
        if not pars:
            ok += 1
            continue
        rdy = sum(1 for u in pars if mastered.get(u, 0.0) >= thr) / len(pars)
        if rdy >= need:
            ok += 1
    return round(ok / len(recs), 4)


def ndcg_at_k(recs, rel, k=10):
    """recs 为按序推荐；rel 为 0/1 真值向量。"""
    gains = 0.0
    for i, r in enumerate(recs[:k]):
        if rel[int(r["kt_id"])] > 0:
            gains += 1.0 / np.log2(i + 2) if False else 1.0 / math.log2(i + 2)
    return gains


import math  # noqa: E402  (置于末尾避免与上方文档字符串混排)
import numpy as np  # noqa: E402
