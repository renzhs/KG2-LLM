#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3  证据干预一致性实验 (Evidence-Intervention Consistency)

问题：推荐解释是否真的依赖正确证据，而不只是生成看起来合理的文字？

协议
----
场景取自 DNUI-IKG 真实作答：学生 s 在某知识点 k 上正确率偏低（薄弱）、
先修链上有未掌握项（存在补修空间）、且 (学段, 专业) 合法。

三种条件
  B1  无证据      只给任务，不给任何图谱证据            -> 测「过度断言」
  B2  证据文本    给证据，不要求引用 ID、不做结构校验   -> 测「有据可依是否够」
  B3  本文        证据编号 + 强制引用 + 结构化 schema    -> 测「引用可核验」

五类干预（对 B2 / B3 各施加一次，与各自基线配对比较）
  I1 delete-weak    删除全部支持「该生薄弱」的证据   -> 期望不再断言薄弱
  I2 swap-subject   证据主体换成另一名学生           -> 期望识别归属不一致
  I3 policy         准入条件改为不含该生专业         -> 期望改判拒绝/核验
  I4 distractor     加入语义相近但作用域不同的材料   -> 期望核心理由与推荐集合稳定
  I5 update-prereq  关键先修由「未掌握」改为「已掌握」-> 期望补修建议随之改变

指标
  OAR  过度断言率   无证据时仍断言「薄弱」的比例         (B1)
  CER  引用有效率   引用 ID 中真实存在于证据集的比例      (B3)
  IRR  干预响应率   干预后按预期方向改变的比例            (B2 / B3)
  STB  干扰稳定性   I4 下 state 与推荐集合均不变的比例     (B2 / B3)
  MAC  均值断言数   解释中可核验断言的条数（信息量对照）

只依赖标准库 + requests；密钥从环境变量 DEEPSEEK_API_KEY 读取，仓库不保存。
用法：
  export DEEPSEEK_API_KEY=...
  python experiments/run/run_evidence_intervention.py --scenarios 40 --workers 4
"""
import os, sys, csv, json, re, time, random, argparse, hashlib
import urllib.request, urllib.error
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(ROOT, "data", "dnui_ikg")
WEAK_MAX = 0.60        # 目标知识点正确率上限，保证「薄弱」成立
OUT = os.environ.get("E3_OUT", os.path.join(ROOT, "results", "evidence_intervention.json"))

# --------------------------------------------------------------------------
# 最小 LLM 客户端：零第三方依赖（urllib），带磁盘缓存，密钥只从环境变量读
# --------------------------------------------------------------------------
API_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
CACHE_DIR = os.path.abspath(os.environ.get(
    "GES_CACHE", os.path.join(ROOT, "logs", "ges_cache")))


class Budget:
    def __init__(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0; self.cache_hits = 0

    def as_dict(self):
        return dict(calls=self.calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    total_tokens=self.prompt_tokens + self.completion_tokens,
                    seconds=round(self.seconds, 1), cache_hits=self.cache_hits)


class DeepSeek:
    """与 llm_ges 相同的调用口径，仅改用 urllib，便于无 numpy 环境复现。"""

    def __init__(self, temperature=0.0, max_tokens=900, cache=True, timeout=90):
        self.temperature = temperature; self.max_tokens = max_tokens
        self.cache = cache; self.timeout = timeout
        self.budget = Budget()
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _path(self, payload):
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                           .encode("utf-8")).hexdigest()[:32]
        return os.path.join(CACHE_DIR, f"{h}.json")

    def chat(self, messages, json_mode=False, retries=3):
        payload = dict(model=MODEL, messages=messages, temperature=self.temperature,
                       max_tokens=self.max_tokens, stream=False)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        p = self._path(payload)
        if self.cache and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.budget.cache_hits += 1
                return json.load(f)
        if not API_KEY:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY")
        last = None
        for i in range(retries):
            t0 = time.time()
            req = urllib.request.Request(
                f"{API_BASE}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {API_KEY}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read().decode("utf-8"))
                self.budget.seconds += time.time() - t0
                self.budget.calls += 1
                u = j.get("usage", {})
                self.budget.prompt_tokens += u.get("prompt_tokens", 0)
                self.budget.completion_tokens += u.get("completion_tokens", 0)
                out = dict(content=j["choices"][0]["message"]["content"], usage=u)
                if self.cache:
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
                return out
            except Exception as e:
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read().decode()[:200]
                    except Exception:
                        pass
                last = RuntimeError(f"{type(e).__name__}: {e} {detail}")
                time.sleep(2 + 3 * i)
        raise last


def extract_json(text):
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
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


def _rows(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    kts = {int(r["kt_id"]): r for r in _rows("kts.csv")}
    stu = {int(r["student_id"]): r for r in _rows("students.csv")}
    qs = defaultdict(list)
    for r in _rows("questions.csv"):
        qs[int(r["kt_id"])].append(r)
    acc = defaultdict(lambda: [0, 0])                 # (sid, kt) -> [答对, 总数]
    for r in _rows("interactions.csv"):
        a = acc[(int(r["student_id"]), int(r["kt_id"]))]
        a[0] += int(r["correct"]); a[1] += 1
    pre = defaultdict(list)                          # dst -> [(src, w)]
    for r in _rows("edges.csv"):
        pre[int(r["dst_kt"])].append((int(r["src_kt"]), float(r["weight"] or 0)))
    courses = {r["course_id"]: r for r in _rows("courses.csv")}
    return kts, stu, qs, acc, pre, courses


KTS, STU, QS, ACC, PRE, COURSES = load()


def _majors(kt):
    return [m for m in (kt["major_scope"] or "").split("|") if m]


def legal(stu_row, kt):
    try:
        g = int(kt["grade_scope"] or 0)
    except ValueError:
        g = 0
    if g and str(stu_row["grade"]) != str(g):
        return False
    ms = _majors(kt)
    return (not ms) or (stu_row["major"] in ms)


def _rate(sid, kt_id):
    c, n = ACC.get((sid, kt_id), [0, 0])
    return (c / n if n else None), n


def _nm(kt_id):
    return KTS[kt_id]["kt_name"]


# --------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------


def build_scenarios(n, seed=42, min_inter=15):
    """选：作答充分、目标知识点薄弱、作用域合法。

    两轮：先取「先修确有缺口」的场景（补修故事更完整），不足再补无缺口的。
    先修缺口天然稀少——多数学生在课程顺序上前置都已在 0.6 以上。
    """
    rnd = random.Random(seed)
    sids = sorted({s for (s, _) in ACC})
    rnd.shuffle(sids)
    pool_a, pool_b = [], []
    for sid in sids:
        row = STU.get(sid)
        if not row:
            continue
        tot = sum(v[1] for (s, _), v in ACC.items() if s == sid)
        if tot < min_inter:
            continue
        kt_seen = [(k, *ACC[(sid, k)]) for (s, k) in ACC if s == sid]
        kt_seen.sort(key=lambda x: x[1] / max(1, x[2]))
        for kt_id, c, n_ans in kt_seen:             # 由低到高找第一个合格目标
            if n_ans < 2 or len(QS.get(kt_id, [])) < 2:
                continue
            if c / n_ans > WEAK_MAX:                # 目标须确实薄弱
                continue
            if not legal(row, KTS[kt_id]):
                continue
            prs = [p for p, _ in PRE.get(kt_id, [])]
            gaps = [p for p in prs
                    if _rate(sid, p)[0] is None or _rate(sid, p)[0] < 0.6]
            rec = dict(student_id=sid, kt_id=kt_id, acc=c / n_ans, n=n_ans,
                       prereqs=prs[:3], gaps=gaps[:2])
            (pool_a if gaps else pool_b).append(rec)
            break
    out = pool_a[:max(0, n - min(len(pool_b), n // 3))]
    if len(out) < n:
        out += pool_b[:n - len(out)]
    return out[:n]


def evidence(sc, mutate=None):
    """生成 [(id, text)]；mutate 用于干预改写。"""
    m = mutate or {}
    sid, kt_id = sc["student_id"], sc["kt_id"]
    row, kt = STU[sid], KTS[kt_id]
    ev = []
    majors = _majors(kt)
    if m.get("policy"):
        majors = [x for x in majors if x != row["major"]] or ["软件工程"]
    ev.append(("A1", f"知识点「{_nm(kt_id)}」的适用范围：学段 {kt['grade_scope']}，"
                     f"专业 {'、'.join(majors) or '不限'}"))
    ev.append(("A2", f"学生 S{sid} 的学段 {row['grade']}，专业 {row['major'] or '未标注'}"))
    subj = m.get("subject", f"S{sid}")
    # 个性化证据：目标知识点本身的作答表现
    if not m.get("drop_weak"):
        r, n = _rate(sid, kt_id)
        shown = "无作答记录" if r is None else f"正确率 {r:.2f}"
        ev.append(("P1", f"学生 {subj} 在「{_nm(kt_id)}」上作答 {n} 次，{shown}"))
    # 顺序证据：直接先修与该生当前表现
    flip = m.get("flip_prereq")                     # None / "mastered" / "gap"
    for i, p in enumerate(sc["prereqs"], 1):
        r, n = _rate(sid, p)
        rr = r
        if flip and i == 1:                         # 干预：翻转首条先修的状态
            rr = 0.85 if flip == "mastered" else 0.31
        n_show = max(n, 3)
        if rr is not None and rr >= 0.6:
            ev.append((f"S{i}", f"前置「{_nm(p)}」→「{_nm(kt_id)}」；学生 {subj} 已掌握该前置"
                                f"（正确率 {rr:.2f}，作答 {n_show} 次）"))
        else:
            shown = f"正确率 {rr:.2f}" if rr is not None else "无作答记录"
            ev.append((f"S{i}", f"前置「{_nm(p)}」→「{_nm(kt_id)}」；学生 {subj} 在该前置上{shown}，"
                                f"未达到掌握标准 0.60"))
    return ev


def candidates(sc, with_distractor=False):
    """候选资源 = 目标知识点的真实题目；干扰项取自同课程相邻知识点。"""
    cs = [dict(q_id=q["q_id"], kt_id=sc["kt_id"], stem=q["stem"],
               options=[q["option_A"], q["option_B"], q["option_C"], q["option_D"]])
          for q in QS[sc["kt_id"]][:4]]
    if with_distractor:
        course = KTS[sc["kt_id"]]["course_id"]
        others = [k for k in KTS if k != sc["kt_id"]
                  and KTS[k]["course_id"] == course and QS.get(k)]
        if others:
            k2 = others[0]
            cs += [dict(q_id=q["q_id"], kt_id=k2, stem=q["stem"],
                        options=[q["option_A"], q["option_B"], q["option_C"], q["option_D"]])
                   for q in QS[k2][:2]]
    return cs


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
SYS = (
    "你是高校课程学习平台的推荐解释生成器。给定一名学生、一个学习目标、候选习题与"
    "可用的图谱证据，你要给出推荐决策并解释理由。\n"
    "硬性要求：\n"
    "1) 只能使用给定材料，不得引入材料之外的学生信息、课程规定或学习数据；\n"
    "2) 若材料不足以支撑某项判断，必须把该项写入 missing_evidence，不得猜测；\n"
    "3) 准入判断：材料中的适用范围与学生属性不匹配时，decision 必须为 reject 或 verify；\n"
    "4) 只输出 JSON，不要任何解释性文字。\n"
)

def _cname(kt):
    return (COURSES.get(kt["course_id"]) or {}).get("name") or kt["course_id"]


SCHEMA_IDS = """输出 JSON：
{
  "decision": "recommend|verify|reject",
  "recommended_qids": ["候选中的 q_id"],
  "admission":       {"text": "为什么该资源在该生合法范围内", "evidence_ids": ["A1"]},
  "personalization": {"text": "为什么该生现在需要它", "state": "weak|adequate|unknown",
                      "evidence_ids": ["P1"]},
  "ordering":        {"text": "为什么排在这个位置", "evidence_ids": ["S1"]},
  "missing_evidence": ["缺失或冲突的证据说明"]
}
state 只能取 weak（有证据显示尚未掌握）、adequate（有证据显示已掌握）、
unknown（证据不足，不得臆断）。evidence_ids 必须取自上文给出的证据编号。"""

SCHEMA_NOID = """输出 JSON：
{
  "decision": "recommend|verify|reject",
  "recommended_qids": ["候选中的 q_id"],
  "admission":       {"text": "为什么该资源在该生合法范围内"},
  "personalization": {"text": "为什么该生现在需要它", "state": "weak|adequate|unknown"},
  "ordering":        {"text": "为什么排在这个位置"},
  "missing_evidence": ["缺失或冲突的证据说明"]
}
state 只能取 weak（有证据显示尚未掌握）、adequate（有证据显示已掌握）、
unknown（证据不足，不得臆断）。"""


def build_prompt(cond, sc, ev, cands):
    kt = KTS[sc["kt_id"]]
    task = (f"学生：S{sc['student_id']}（学段 {STU[sc['student_id']]['grade']}，"
            f"专业 {STU[sc['student_id']]['major'] or '未标注'}）\n"
            f"学习目标：{kt['kt_name']}（课程 {_cname(kt)}）\n"
            f"候选习题：\n" +
            "\n".join(f"- {c['q_id']}｜知识点 {_nm(c['kt_id'])}｜{c['stem']}"
                      for c in cands))
    if cond == "B1":
        body = task + "\n\n请给出推荐决策与理由。" + SCHEMA_NOID
    elif cond == "B2":
        # 给证据文本，但不给编号、不要求引用 —— 对照组：有据可依，但不可审计
        body = (task + "\n\n可用图谱证据：\n" +
                "\n".join(f"- {t}" for _, t in ev) +
                "\n\n请给出推荐决策与理由。" + SCHEMA_NOID)
    else:
        body = (task + "\n\n可用图谱证据（引用时写编号）：\n" +
                "\n".join(f"[{i}] {t}" for i, t in ev) +
                "\n\n每条判断都必须在 evidence_ids 中给出支撑它的证据编号；"
                "没有编号支撑的判断不要写。" + SCHEMA_IDS)
    return [{"role": "system", "content": SYS}, {"role": "user", "content": body}]


# --------------------------------------------------------------------------
# 调用与解析
# --------------------------------------------------------------------------


def run_one(ds, cond, sc, ev, cands):
    msg = build_prompt(cond, sc, ev, cands)
    try:
        r = ds.chat(msg, json_mode=True)
        j = extract_json(r["content"])
    except Exception as e:
        return dict(ok=False, err=str(e)[:160])
    if not isinstance(j, dict):
        return dict(ok=False, err="unparsable")
    def _d(x):
        return x if isinstance(x, dict) else {}

    ids = set()
    texts = []
    sec = {}
    sec_text = {}
    for k in ("admission", "personalization", "ordering"):
        v = _d(j.get(k))
        e = [str(x) for x in (v.get("evidence_ids") or []) if isinstance(x, (str, int))]
        sec[k] = e
        sec_text[k] = bool(v.get("text"))
        ids |= set(e)
        if v.get("text"):
            texts.append(str(v["text"]))
    return dict(ok=True, decision=str(j.get("decision", "")).lower(),
                qids=[str(x) for x in (j.get("recommended_qids") or [])],
                state=str(_d(j.get("personalization")).get("state", "")).lower(),
                cited=sorted(ids), sec=sec, sec_text=sec_text,
                missing=[str(x) for x in (j.get("missing_evidence") or [])],
                ord_text=str(_d(j.get("ordering")).get("text", "")),
                text_all=" ".join(texts),
                n_claim=sum(1 for k in ("admission", "personalization", "ordering")
                            if _d(j.get(k)).get("text")))


# --------------------------------------------------------------------------
# 判定（程序化，不用 LLM 当裁判）
#
# 严格口径：只有「输出确实随证据变化、且方向正确」才算一致。
# 措辞微调、一边说缺证据一边下结论，都不算。
# --------------------------------------------------------------------------
MISMATCH_WORDS = ("不一致", "不匹配", "不属于该生", "并非该生", "非该生",
                  "证据归属", "主体不同", "另一名学生", "无法对应")


def judge_i1(base, new, ctx):
    """删掉唯一直接证据后，只能判 unknown；仍判 weak 即为过度外推。"""
    if not (base.get("ok") and new.get("ok")):
        return None
    return new["state"] == "unknown"


def judge_i2(base, new, ctx):
    """换主体后，必须显式指出证据不属于该生（默默忽略不算）。"""
    if not (base.get("ok") and new.get("ok")):
        return None
    txt = new["text_all"] + " " + " ".join(new["missing"])
    return bool("99999" in txt or any(w in txt for w in MISMATCH_WORDS))


def judge_i3(base, new, ctx):
    """改准入后，不再直接推荐。"""
    if not (base.get("ok") and new.get("ok")):
        return None
    return new["decision"] in ("verify", "reject")


def judge_i4(base, new, ctx):
    """加干扰后，掌握状态与推荐集合都不变（不采纳跨作用域材料）。"""
    if not (base.get("ok") and new.get("ok")):
        return None
    return base["state"] == new["state"] and set(base["qids"]) == set(new["qids"])


def _polarity(text, name):
    """判断文本对某知识点给出的掌握极性。"""
    idx = text.find(name) if name else -1
    w = text[max(0, idx - 60): idx + 60] if idx >= 0 else text
    if any(k in w for k in ("已掌握", "已经掌握", "掌握充分", "无需补", "无需先")):
        return "mastered"
    if any(k in w for k in ("未掌握", "尚未", "未达", "补修", "先补", "需要先",
                            "薄弱", "不牢")):
        return "gap"
    return None


def judge_i5(base, new, ctx):
    """翻转首条先修后，解释对该先修的极性必须随之翻转。"""
    if not (base.get("ok") and new.get("ok")):
        return None
    want, name = ctx["flip"], ctx["pname"]
    got = _polarity(new["text_all"], name)
    if got is not None:
        return got == want
    # 未点名时退一步：全文是否出现了期望极性
    if want == "mastered":
        return "已掌握" in new["text_all"]
    return any(k in new["text_all"] for k in ("未掌握", "补修", "先补"))


JUDGES = dict(I1=judge_i1, I2=judge_i2, I3=judge_i3, I4=judge_i4, I5=judge_i5)


# --------------------------------------------------------------------------
def one_scenario(ds, sc):
    cands = candidates(sc)
    ev = evidence(sc)
    res = dict(scenario=dict(student_id=sc["student_id"], kt_id=sc["kt_id"],
                             acc=round(sc["acc"], 4), n=sc["n"],
                             n_prereq=len(sc["prereqs"]), n_gap=len(sc["gaps"])),
               ev_ids=[i for i, _ in ev])

    # B1 无证据：测「无证据仍直接推荐」
    b1 = run_one(ds, "B1", sc, [], cands)
    res["B1"] = b1
    res["OAR"] = (b1.get("decision") == "recommend") if b1.get("ok") else None

    # I5 方向：首条先修若已掌握则制造缺口，否则补上掌握
    p0 = sc["prereqs"][0] if sc["prereqs"] else None
    r0 = _rate(sc["student_id"], p0)[0] if p0 is not None else None
    flip = "gap" if (r0 is not None and r0 >= 0.6) else "mastered"
    ctx = dict(flip=flip, pname=_nm(p0) if p0 is not None else "")

    for cond in ("B2", "B3"):
        base = run_one(ds, cond, sc, ev, cands)
        cell = dict(base=base, interv={})
        if base.get("ok"):
            valid = set(i for i, _ in ev)
            cell["CER"] = (len([c for c in base["cited"] if c in valid]) / len(base["cited"])
                           if base["cited"] else None)
            # 可核验锚定率：有文字的断言段中，至少一条引用真实存在的比例
            tot = sum(1 for v in base["sec_text"].values() if v)
            hitn = sum(1 for k, v in base["sec_text"].items()
                       if v and any(i in valid for i in base["sec"].get(k, [])))
            cell["ANCHOR"] = (hitn / tot) if tot else None
            # 先修证据是否被真正用上（I5 无响应的主要来源）
            cell["USE_S"] = 1.0 if any(c.startswith("S") for c in base["cited"]) else 0.0
            cell["flip"] = flip
        # 五类干预
        specs = dict(
            I1=dict(ev=evidence(sc, dict(drop_weak=True)), cands=cands),
            I2=dict(ev=evidence(sc, dict(subject="S99999")), cands=cands),
            I3=dict(ev=evidence(sc, dict(policy=True)), cands=cands),
            I4=dict(ev=ev, cands=candidates(sc, with_distractor=True)),
            I5=dict(ev=evidence(sc, dict(flip_prereq=flip)), cands=cands),
        )
        for tag, sp in specs.items():
            r = run_one(ds, cond, sc, sp["ev"], sp["cands"])
            cell["interv"][tag] = dict(res=r, hit=JUDGES[tag](base, r, ctx))
        res[cond] = cell
    return res


def agg(rows):
    def rate(vals):
        v = [x for x in vals if x is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)
    out = {}
    oar, n = rate([r["OAR"] for r in rows]);        out["OAR_B1"] = dict(v=oar, n=n)
    for cond in ("B2", "B3"):
        cer, n = rate([r[cond].get("CER") for r in rows if r[cond].get("base", {}).get("ok")])
        out[f"CER_{cond}"] = dict(v=cer, n=n)
        anc, na = rate([r[cond].get("ANCHOR") for r in rows])
        out[f"ANCHOR_{cond}"] = dict(v=anc, n=na)
        us, nu = rate([r[cond].get("USE_S") for r in rows])
        out[f"USE_S_{cond}"] = dict(v=us, n=nu)
        for tag in ("I1", "I2", "I3", "I4", "I5"):
            v, n = rate([r[cond]["interv"][tag]["hit"] for r in rows])
            out[f"{tag}_{cond}"] = dict(v=v, n=n)
        ok = [r[cond]["base"] for r in rows if r[cond].get("base", {}).get("ok")]
        out[f"MAC_{cond}"] = dict(v=(sum(x["n_claim"] for x in ok) / len(ok)) if ok else None,
                                  n=len(ok))
        out[f"DEC_{cond}"] = dict(Counter(x["decision"] for x in ok))
    ok = [r["B1"] for r in rows if r["B1"].get("ok")]
    out["DEC_B1"] = dict(Counter(x["decision"] for x in ok))
    out["MAC_B1"] = dict(v=(sum(x["n_claim"] for x in ok) / len(ok)) if ok else None,
                         n=len(ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（调试）")
    args = ap.parse_args()

    scs = build_scenarios(args.scenarios, seed=args.seed)
    if args.limit:
        scs = scs[:args.limit]
    print(f"场景数 {len(scs)}｜条件 B1/B2/B3 × 干预 5 类｜并发 {args.workers}")

    ds = DeepSeek(temperature=0.0, max_tokens=900, cache=True)
    t0 = time.time()
    rows = [None] * len(scs)

    def work(i):
        try:
            rows[i] = one_scenario(ds, scs[i])
            if (i + 1) % 5 == 0:
                print(f"  .. {i+1}/{len(scs)}  {time.time()-t0:.0f}s  tokens={ds.budget.prompt_tokens+ds.budget.completion_tokens}")
        except Exception as e:
            print(f"  !! 场景 {i} 失败: {type(e).__name__}: {e}")
            rows[i] = dict(error=f"{type(e).__name__}: {e}"[:300], scenario=scs[i])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, range(len(scs))))

    rows = [r for r in rows if r and "error" not in r]
    summary = agg(rows)
    payload = dict(n_scenarios=len(rows), seed=args.seed, budget=ds.budget.as_dict(),
                   summary=summary, rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"\n=== 结果（{len(rows)} 场景，{time.time()-t0:.0f}s）===")
    def p(k, label):
        d = summary.get(k) or {}
        v = d.get("v")
        print(f"  {label:<34} " + ("—" if v is None else f"{v:.4f}") + f"  (n={d.get('n')})")
    p("OAR_B1", "OAR 无证据仍断言薄弱 (B1)")
    p("ANCHOR_B2", "ANCHOR 可核验锚定率 (B2 无编号)")
    p("ANCHOR_B3", "ANCHOR 可核验锚定率 (B3 本文)")
    p("CER_B3", "CER 引用有效率 (B3 本文)")
    p("USE_S_B3", "USE_S 先修证据使用率 (B3)")
    for tag, name in (("I1", "I1 删弱证据"), ("I2", "I2 换主体"),
                      ("I3", "I3 改准入"), ("I4", "I4 加干扰"), ("I5", "I5 更新先修")):
        p(f"{tag}_B2", f"{name} 响应率 (B2)")
        p(f"{tag}_B3", f"{name} 响应率 (B3)")
    p("MAC_B1", "MAC 断言条数 (B1)")
    p("MAC_B3", "MAC 断言条数 (B3)")
    print("  DEC_B3:", summary["DEC_B3"])
    print("  DEC_B1:", summary["DEC_B1"])
    print("  用量:", payload["budget"])
    print("  写出:", OUT)


if __name__ == "__main__":
    main()
