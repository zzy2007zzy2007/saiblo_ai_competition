"""全局去重判断:如果删掉某个 junk 目录,它的内容在整个备份里还找得到吗?

对每个 junk 文件,按 (文件名, 大小) 去全备份(7 个版本)里找:
  A. 在"非 junk 位置"找得到   -> 删掉 junk 不丢内容(安全)
  B. 只在"别的 junk 目录"里也有一份 -> 删这一个安全,但别全删
  C. 哪都找不到                -> 删了就真没了(要人工看)
"""
import os
import re
import sys
from collections import defaultdict

ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3
JUNK_RE = re.compile(r"没用的|- 副本| - copy| copy |旧版本对战", re.I)


def main():
    ver_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))

    nonjunk = {}          # (basename, size) -> sample path
    junk_files = []       # (path, size, version, junk_root)

    for v in ver_dirs:
        vp = os.path.join(ROOT, v)
        for dp, dn, fn in os.walk(vp):
            rel = os.path.relpath(dp, vp)
            parts = [] if rel == "." else rel.split(os.sep)
            idx = next((i for i, p in enumerate(parts) if JUNK_RE.search(p)), None)
            for f in fn:
                full = os.path.join(dp, f)
                try:
                    s = os.stat(full).st_size
                except OSError:
                    continue
                if idx is None:
                    nonjunk.setdefault((f, s), full)
                else:
                    junk_files.append((full, s, v, os.sep.join(parts[: idx + 1])))
        print(f"  {v} scanned", flush=True)

    print(f"\n非 junk 文件种类 {len(nonjunk)};junk 文件 {len(junk_files)} 个", flush=True)

    junk_index = defaultdict(int)
    for p, s, v, root in junk_files:
        junk_index[(os.path.basename(p), s)] += 1

    agg = defaultdict(lambda: {"total": 0, "n": 0, "A": 0, "B": 0, "C": 0, "nA": 0, "nB": 0, "nC": 0})
    unique_examples = defaultdict(list)

    for p, s, v, root in junk_files:
        key = (os.path.basename(p), s)
        a = agg[root]
        a["total"] += s
        a["n"] += 1
        if key in nonjunk:
            a["A"] += s
            a["nA"] += 1
        elif junk_index[key] > 1:
            a["B"] += s
            a["nB"] += 1
        else:
            a["C"] += s
            a["nC"] += 1
            if len(unique_examples[root]) < 6:
                unique_examples[root].append((s, os.path.relpath(p, ROOT)))

    print("\n" + "=" * 118)
    print("A = 别处(非 junk)也有同样内容,删 junk 安全")
    print("B = 只在别的 junk 目录里有,删这一份安全、但别全删")
    print("C = 全备份里独一份,删了就没了")
    print("=" * 118)
    print(f"{'junk 目录':<44} {'合计':>9} {'A安全':>9} {'B别处junk':>10} {'C独有':>9}")
    print("-" * 118)
    for root, a in sorted(agg.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{root[:43]:<44} {a['total']/G:9.2f} {a['A']/G:9.2f} {a['B']/G:10.2f} {a['C']/G:9.2f}")

    tot = {k: sum(a[k] for a in agg.values()) for k in ("total", "A", "B", "C")}
    print("-" * 118)
    print(f"{'合计':<44} {tot['total']/G:9.2f} {tot['A']/G:9.2f} {tot['B']/G:10.2f} {tot['C']/G:9.2f}")

    print("\n=== C 类(独有)最多的 junk 目录,抽样看看是什么 ===")
    for root, a in sorted(agg.items(), key=lambda kv: -kv[1]["C"])[:6]:
        if a["C"] == 0:
            continue
        print(f"\n--- {root}  独有 {a['C']/G:.2f} GiB / {a['nC']} 文件 ---")
        for s, rel in unique_examples[root]:
            print(f"    {s/1024**2:9.1f} MiB  {rel[:100]}")


if __name__ == "__main__":
    main()
