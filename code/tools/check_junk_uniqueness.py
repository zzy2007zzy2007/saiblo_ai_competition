"""判断 '没用的' 到底是垃圾,还是"整个项目的一份拷贝"。

方法:把 '没用的' 里的每个文件,拿去和它所在版本的"项目根目录"(排除没用的* 自身)
按 (相对路径, 大小) 比对。
  命中 => 这一份是重复的
  未命中 => 这一份是独有的(删掉就真没了)
"""
import os
import sys
from collections import defaultdict

ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3

# (版本目录, junk 相对路径)
CASES = [
    ("智能体大赛备份4", r"2026智能体大赛\没用的"),
    ("智能体大赛备份5", r"2026智能体大赛\没用的"),
    ("智能体大赛备份5", r"2026智能体大赛\没用的_2"),
    ("智能体大赛备份3", r"2026智能体大赛\ai_neural_network_new\没用的旧模型"),
]


def snapshot(root, skip_prefixes=()):
    out = {}
    for dp, dn, fn in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if any(os.sep.join(parts[: len(p.split(os.sep))]) == p for p in skip_prefixes if parts):
            dn[:] = []
            continue
        for f in fn:
            full = os.path.join(dp, f)
            rel = os.path.normpath(os.path.join(rel_dir, f)) if rel_dir != "." else f
            try:
                out[rel] = os.stat(full).st_size
            except OSError:
                pass
    return out


def main():
    for ver, junk_rel in CASES:
        vp = os.path.join(ROOT, ver)
        junk = os.path.join(vp, junk_rel)
        if not os.path.isdir(junk):
            print(f"(不存在) {ver}\\{junk_rel}\n")
            continue

        # junk 的第一层名,用来在"其他位置"找同名目录
        J = snapshot(junk)
        j_bytes = sum(J.values())

        # 整个版本里,junk 之外的所有文件
        all_files = {}
        for dp, dn, fn in os.walk(vp):
            for f in fn:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, vp)
                if rel.startswith(junk_rel + os.sep):
                    continue
                try:
                    all_files[rel] = os.stat(full).st_size
                except OSError:
                    pass

        # 按 basename+size 建索引(因为相对路径的层级不同)
        idx = defaultdict(list)
        for rel, s in all_files.items():
            idx[(os.path.basename(rel), s)].append(rel)

        matched_bytes = 0
        matched = 0
        uniq = []
        for rel, s in J.items():
            key = (os.path.basename(rel), s)
            if key in idx:
                matched += 1
                matched_bytes += s
            else:
                uniq.append((s, rel))

        uniq.sort(reverse=True)
        print("=" * 100)
        print(f"### {ver}\\{junk_rel}")
        print(f"    自身 {j_bytes/G:.2f} GiB / {len(J)} 文件")
        print(f"    在 junk 之外找到同名同大小的: {matched} 个文件, {matched_bytes/G:.2f} GiB "
              f"({matched_bytes/max(j_bytes,1)*100:.1f}% 字节)")
        print(f"    独有的(删了就没): {len(uniq)} 个文件, {sum(s for s,_ in uniq)/G:.2f} GiB")
        print(f"    独有文件 Top 10:")
        for s, rel in uniq[:10]:
            print(f"      {s/1024**2:9.1f} MiB  {rel[:95]}")
        print()


if __name__ == "__main__":
    main()
