#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复核论文下游各部分（资源推荐、路径推荐、重规划、GES、门控鲁棒性）报告值。

与 scripts/03_tables.sh 的分工：
  03_tables.sh 覆盖知识追踪侧（表 1-4，逐折结果 → LaTeX 表 → 与快照 diff）；
  本脚本覆盖下游侧（表 6-10），直接从 results/ 的主口径 JSON 读数，
  与论文中印出的数值逐项比对。不需要 GPU、不需要重跑实验。

用法：
  python experiments/tools/check_downstream.py            # 全部检查
  python experiments/tools/check_downstream.py --only rec # 只查某一节
退出码 0 表示全部一致，1 表示存在不一致（便于 CI 或投稿前自检）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "results")
TOL = 5e-5          # 论文保留 4 位小数，允许末位舍入差


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


FAILS = []
COUNT = [0]


def chk(section, label, got, want):
    COUNT[0] += 1
    if got is None:
        FAILS.append((section, label, None, want))
        print("  [缺]   %-46s 期望 %s" % (label, want))
        return
    ok = abs(float(got) - float(want)) <= TOL
    if not ok:
        FAILS.append((section, label, got, want))
    print("  [%s] %-46s %8.4f  (论文 %.4f)" % ("OK " if ok else "差异", label, got, want))


# ---------------------------------------------------------------- 表 6
# 列序与论文一致：sRecall@10, sNDCG@10, Recall@10, NDCG@10, CSCR@10, CSCR@20
TAB6 = {
    "Popularity":                       [0.0516, 0.0393, 0.0365, 0.0350, 0.0815, 0.1210],
    "SVD(MF)":                          [0.0151, 0.0099, 0.0236, 0.0171, 0.1883, 0.2295],
    "KG-CF(无门控)":                     [0.0105, 0.0072, 0.0147, 0.0115, 0.1974, 0.2278],
    "KG-CF+反向标定":                    [0.0096, 0.0065, 0.0138, 0.0107, 0.1957, 0.2245],
    "先修就绪度(结构)":                   [0.2141, 0.1583, 0.1516, 0.1416, 0.0557, 0.0453],
    "先修就绪度(反向标定)":                [0.2158, 0.1616, 0.1529, 0.1447, 0.0557, 0.0453],
    "先修就绪度(结构)+CSG(仅释放槽位)":     [0.2259, 0.1710, 0.1562, 0.1485, 0.0000, 0.0064],
    "先修就绪度+CSG":                    [0.2277, 0.1743, 0.1575, 0.1516, 0.0000, 0.0064],
    "DGCL-Rec-无CSG(消融)":             [0.1532, 0.1298, 0.1176, 0.1238, 0.1755, 0.1970],
    "DGCL-Rec-无图谱协同(消融)":          [0.2277, 0.1743, 0.1575, 0.1516, 0.0000, 0.0064],
    "DGCL-Rec-无反向标定(消融)":          [0.1941, 0.1624, 0.1314, 0.1414, 0.0000, 0.0064],
    "DGCL-Rec(本文)":                   [0.1940, 0.1588, 0.1314, 0.1379, 0.0000, 0.0064],
}
TAB6_COLS = ["sRecall@10", "sNDCG@10", "Recall@10", "NDCG@10", "CSCR@10", "CSCR@20"]

# ---------------------------------------------------------------- 表 7
TAB7 = {
    "Popularity":            [0.2835, 0.1683],
    "SVD(MF)":               [0.1716, 0.0868],
    "KG-CF(无门控)":          [0.1400, 0.0746],
    "KG-CF+反向标定":         [0.1407, 0.0748],
    "先修就绪度(结构)":        [0.2743, 0.1894],
    "先修就绪度(反向标定)":     [0.2708, 0.1822],
    "先修就绪度+CSG":         [0.2708, 0.1822],
    "DGCL-Rec(本文)":        [0.2752, 0.1861],
}
TAB7_COLS = ["Recall@10", "NDCG@10"]


def sec_rec():
    print("\n== 表 6：DNUI-IKG 资源推荐 ==")
    d = load("rec_dnui_syn.json")
    if d is None:
        print("  [缺] rec_dnui_syn.json"); FAILS.append(("rec", "file", None, 1)); return
    for m, exp in TAB6.items():
        for c, v in zip(TAB6_COLS, exp):
            chk("表6", "%s · %s" % (m, c), (d.get(m) or {}).get(c), v)

    print("\n== 表 7：ASSIST09 资源推荐 ==")
    e = load("rec_assist09.json")
    if e is None:
        print("  [缺] rec_assist09.json"); FAILS.append(("rec", "file", None, 1)); return
    for m, exp in TAB7.items():
        for c, v in zip(TAB7_COLS, exp):
            chk("表7", "%s · %s" % (m, c), (e.get(m) or {}).get(c), v)


# ---------------------------------------------------------------- 表 9
TAB9A = {   # 方法 -> [(DNUI EC, DNUI 顺序), (A09 EC, A09 顺序), (A12 EC, A12 顺序)]
    "拓扑序(门控)":                        [(0.0833, 1.0000), (0.0657, None), (0.0242, None)],
    "作用域内随机":                        [(0.0302, 0.5186), (0.0125, 0.4979), (0.0071, 0.4758)],
    "贪心就绪度(无门控)":                   [(0.0397, 0.7113), (0.0351, 0.6127), (0.0133, 0.6288)],
    "贪心就绪度+CSG(无序)":                 [(0.0637, 0.9213), (0.0400, 0.5505), (0.0189, 0.6378)],
    "就绪度选点+拓扑定序":                  [(0.0656, 1.0000), (0.0801, 1.0000), (0.0226, 1.0000)],
    "前沿推进+就绪度+拓扑定序":              [(0.2439, 1.0000), (0.0190, 1.0000), (0.0025, 1.0000)],
    "DGCL-Path-无反向标定(消融)":           [(0.2439, 1.0000), (0.0799, 1.0000), (0.0238, 1.0000)],
    "DGCL-Path-无拓扑定序(消融)":           [(0.1731, 0.5372), (0.0400, 0.5505), (0.0189, 0.6378)],
}
TAB9B = {   # 名称 -> (次数, EC, 顺序合规率)
    "每步重规划":       (1150, 0.3815, 0.8958),
    "每5步重规划":      (200, 0.3103, 0.8921),
    "MDRL阈值触发":     (326, 0.3883, 0.8947),
}


def _pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def sec_path():
    print("\n== 表 9(a)：学习路径质量 ==")
    files = [("dnui_syn", "path_dnui_syn.json"), ("assist09", "path_assist09.json"),
             ("assist12", "path_assist12.json")]
    for i, (ds, fn) in enumerate(files):
        d = load(fn)
        if d is None:
            print("  [缺]", fn); FAILS.append(("path", fn, None, 1)); continue
        for m, exp in TAB9A.items():
            key = _pick(d, m)
            ec, sq = exp[i]
            chk("表9a", "%s · %s · EC" % (ds, m), (key or {}).get("可执行覆盖率"), ec)
            if sq is not None:
                chk("表9a", "%s · %s · 顺序合规率" % (ds, m),
                    (key or {}).get("顺序合规率"), sq)

    print("\n== 表 9(b)：动态重规划 ==")
    d = load("mdrl_dnui_syn.json")
    if d is None:
        print("  [缺] mdrl_dnui_syn.json"); FAILS.append(("mdrl", "file", None, 1)); return
    rc, pq = d.get("replan_counts", {}), d.get("path_quality", {})
    for name, (cnt, ec, sq) in TAB9B.items():
        key = _pick(rc, name, name.replace("每5步", "每5步"), name + "阈值触发")
        if name == "MDRL阈值触发":
            key = _pick(rc, "MDRL阈值触发", "MDRL")
        chk("表9b", "%s · 调用次数" % name, key, cnt)
        q = _pick(pq, name) or _pick(pq, "MDRL阈值触发" if name.startswith("MDRL") else name)
        if q:
            chk("表9b", "%s · EC" % name, q.get("可执行覆盖率"), ec)
            chk("表9b", "%s · 顺序合规率" % name, q.get("顺序合规率"), sq)


# ---------------------------------------------------------------- 表 10
def sec_ges():
    print("\n== 表 10：检索增强生成 ==")
    d = load("ges_dnui_syn.json")
    if d is None:
        print("  [缺] ges_dnui_syn.json"); FAILS.append(("ges", "file", None, 1)); return
    exp = {
        "Direct": dict(HR=0.0, ETR=0.0, SCR=0.8775, PR=0.6275, **{"NDCG@10": 0.1907},
                       empty=0.0, fill=1.0),
        # 2×2 匹配对照：门控与证据两个因素各自单独作用
        "GateOnly": dict(HR=0.0, ETR=0.0, SCR=1.0000, PR=0.6450, **{"NDCG@10": 0.2314},
                         empty=0.0, fill=1.0),
        "EvidOnly": dict(HR=0.0, ETR=0.7650, SCR=0.8700, PR=0.7925, **{"NDCG@10": 0.1893},
                         empty=0.0, fill=1.0),
        "GES": dict(HR=0.0, ETR=0.7300, SCR=1.0000, PR=0.7600, **{"NDCG@10": 0.1501},
                    empty=0.0, fill=1.0),
    }
    for name, e in exp.items():
        row = _pick(d, name)
        if row is None:
            print("  [缺] 方法行", name); FAILS.append(("ges", name, None, 1)); continue
        for k, v in e.items():
            chk("表10", "%s · %s" % (name, k), row.get(k), v)
        # 结构化核验的三个分步指标应与合计一致（论文 3.6 节的分步诊断）
        if name in ("GES", "EvidOnly"):
            for k in ("ETR_parse", "ETR_dir", "ETR_sup"):
                chk("表10", "%s · %s" % (name, k), row.get(k), e["ETR"])


# ---------------------------------------------------------------- 表 8
TAB8 = {    # rho -> [硬门控FRR, 仅同名FRR, 等价挂载集FRR, 语义top-1 FRR, 同预算FAR, 等价FAR, MMR]
    "0.0": [0.0002, 0.0001, 0.0000, 0.0002, 0.1562, 0.0017, 1.0000],
    "0.1": [0.1044, 0.0010, 0.0009, 0.0010, 0.1589, 0.0072, 1.0000],
    "0.2": [0.1996, 0.0012, 0.0012, 0.0013, 0.1616, 0.0110, 0.9998],
    "0.3": [0.3174, 0.0037, 0.0036, 0.0034, 0.1636, 0.0354, 0.9995],
    "0.5": [0.5123, 0.0053, 0.0050, 0.0041, 0.1845, 0.0441, 0.9998],
    "0.7": [0.7184, 0.0039, 0.0037, 0.0056, 0.3046, 0.0728, 0.9990],
}
TAB8_COLS = ["硬门控 FRR", "仅同名 FRR", "等价挂载集 FRR", "语义 top-1 FRR",
             "同预算联合 FAR", "等价挂载集 FAR", "MMR"]


def sec_scope():
    print("\n== 表 8：标注不完整下的门控判定 ==")
    d = load("scope_decision.json")
    if d is None:
        print("  [缺] scope_decision.json"); FAILS.append(("scope", "file", None, 1)); return
    summ = d.get("summary", {})
    for rho, exp in TAB8.items():
        row = summ.get(rho) or summ.get(str(float(rho)))
        if row is None:
            print("  [缺] rho=%s" % rho); FAILS.append(("scope", rho, None, 1)); continue
        got = [
            (row.get("meta") or {}).get("FRR"),      # 硬门控 = 式 (10) 原式，无标注即拒
            (row.get("share") or {}).get("FRR"),     # 仅同名等价
            (row.get("share2") or {}).get("FRR"),    # 等价挂载集（本文）
            (row.get("sem1") or {}).get("FRR"),      # 语义 top-1
            (row.get("union") or {}).get("FAR"),     # 同候选预算联合放行
            (row.get("share2") or {}).get("FAR"),
            (row.get("share2") or {}).get("MMR"),
        ]
        for c, g, v in zip(TAB8_COLS, got, exp):
            chk("表8", "rho=%s · %s" % (rho, c), g, v)


# ---------------------------------------------------------------- 正文引述
def sec_text():
    print("\n== 正文关键结论 ==")
    # EC 相对随机策略的倍数：0.2439 / 0.0302 = 8.08
    d = load("path_dnui_syn.json") or {}
    ours = (d.get("前沿推进+就绪度+拓扑定序") or {}).get("可执行覆盖率")
    rnd = (d.get("作用域内随机") or {}).get("可执行覆盖率")
    if ours and rnd:
        chk("正文", "EC 相对随机策略的倍数", round(ours / rnd, 2), 8.08)
    # 重规划降幅：(1150-326)/1150 = 71.65% → 71.7
    m = load("mdrl_dnui_syn.json") or {}
    rc = m.get("replan_counts", {})
    if rc.get("每步重规划") and rc.get("MDRL阈值触发"):
        chk("正文", "MDRL 相对每步的降幅(%)",
            round((rc["每步重规划"] - rc["MDRL阈值触发"]) / rc["每步重规划"] * 100, 1), 71.7)
        chk("正文", "MDRL 相对固定 5 步的高出(%)",
            round((rc["MDRL阈值触发"] - rc["每5步重规划"]) / rc["每5步重规划"] * 100, 1), 63.0)
    # 表 6 中本文与最强基线的差：0.1743 - 0.1588 = 0.0155
    r = load("rec_dnui_syn.json") or {}
    a = (r.get("先修就绪度+CSG") or {}).get("sNDCG@10")
    b = (r.get("DGCL-Rec(本文)") or {}).get("sNDCG@10")
    if a and b:
        chk("正文", "去除图谱协同带来的 sNDCG 增量", round(a - b, 4), 0.0155)


# ---------------------------------------------------------------- 表 14
TAB14 = {   # dataset key -> (ours_all, ours_train, base_all, base_train, gap_all, gap_train)
    "assist09_akt": (0.8184, 0.8188, 0.8087, 0.8087, 0.0097, 0.0102),
    "assist15": (0.7306, 0.7304, 0.7278, 0.7279, 0.0028, 0.0026),
    "assist17": (0.8013, 0.8002, 0.7315, 0.7315, 0.0698, 0.0687),
    "statics11": (0.8252, 0.8249, 0.8194, 0.8194, 0.0058, 0.0055),
}


def sec_ktgraph():
    """表 14：知识追踪侧「全量建图 vs 仅训练学生建图」的复核。"""
    print("\n== 表 14：知识追踪侧的建图范围 ==")
    d = load("kt_gstrain_pair.json")
    if d is None:
        print("  [缺] kt_gstrain_pair.json"); FAILS.append(("ktgraph", "file", None, 1)); return
    for key, (oa, ot, ba, bt, ga, gt) in TAB14.items():
        row = d.get(key)
        if row is None:
            print("  [缺] 数据集", key); FAILS.append(("ktgraph", key, None, 1)); continue
        m, b = row["method"], row["best_baseline_row"]
        chk("表14", "%s · 本文全量" % key, m["main_mean"], oa)
        chk("表14", "%s · 本文仅训练" % key, m["gstrain_mean"], ot)
        chk("表14", "%s · 基线全量" % key, b["main_mean"], ba)
        chk("表14", "%s · 基线仅训练" % key, b["gstrain_mean"], bt)
        chk("表14", "%s · 领先全量" % key, row["gap_main"], ga)
        chk("表14", "%s · 领先仅训练" % key, row["gap_gstrain"], gt)


SECTIONS = {"rec": sec_rec, "path": sec_path, "ges": sec_ges,
            "scope": sec_scope, "ktgraph": sec_ktgraph, "text": sec_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔：" + ",".join(SECTIONS))
    args = ap.parse_args()
    picks = [x.strip() for x in args.only.split(",") if x.strip()] or list(SECTIONS)
    print("仓库根目录:", REPO)
    for p in picks:
        SECTIONS[p]()
    print("\n===== 共检查 %d 项，不一致 %d 项 =====" % (COUNT[0], len(FAILS)))
    if FAILS:
        for sec, label, got, want in FAILS:
            print("  %-6s %-46s 实得 %s / 论文 %s" % (sec, label, got, want))
        return 1
    print("全部与论文报告值一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
