#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HF 镜像下载公开 KT 数据集到 data/public/。"""
import os, sys, json, urllib.request, urllib.error, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import PUB as _PUB   # noqa: E402

BASE = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
ROOT = os.environ.get("KT_PUBLIC_DIR", _PUB)
os.makedirs(ROOT, exist_ok=True)

# (dataset_id, [files...] or None=all small files)
TARGETS = [
    ("OloriBern/assistments-dataset", ["skill_builder_data_corrected.csv"]),
    ("Atomi/ASSISTments2009", ["data/train-00000-of-00001.parquet"]),
    ("badranx/ASSISTments09", None),
    ("Lucy9999/assist2009", None),
    ("Lucy9999/assist2012", None),
    ("Lucy9999/EdNet-KT1", None),
    ("Unggi/ednet5000_raw_data", None),
    ("Abdelrahman234013/assist2009-akt", None),
]

def list_files(ds):
    url = f"{BASE}/api/datasets/{ds}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        info = json.load(r)
    return [s["rfilename"] for s in (info.get("siblings") or [])]

def dl(ds, fn, out):
    url = f"{BASE}/datasets/{ds}/resolve/main/{fn}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out + ".part"
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); n += len(b)
    os.replace(tmp, out)
    return n

for ds, files in TARGETS:
    tag = ds.replace("/", "__")
    d = os.path.join(ROOT, tag)
    os.makedirs(d, exist_ok=True)
    try:
        allf = list_files(ds)
    except Exception as e:
        print(f"[SKIP] {ds}: {e}"); continue
    if files is None:
        # 只取非目录、非 .gitattributes 的文件
        files = [f for f in allf if not f.endswith(".gitattributes")]
    print(f"\n### {ds}  ({len(files)} files)")
    for fn in files:
        if fn.endswith("/"): continue
        out = os.path.join(d, os.path.basename(fn))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"   skip(exists) {fn}"); continue
        try:
            t0 = time.time()
            n = dl(ds, fn, out)
            print(f"   OK {n/1e6:>8.2f} MB  {time.time()-t0:>6.1f}s  {fn}")
        except Exception as e:
            print(f"   FAIL {fn}: {type(e).__name__} {e}")

print("\n===== SUMMARY =====")
for r, _, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(r, f)
        print(f"{os.path.getsize(p)/1e6:>10.2f} MB  {p.replace(ROOT+'/', '')}")
