"""对 '没用的 / 副本' 目录做抽样取证:看结构 + 读几个文件的真实内容。

用法:
    python code/tools/inspect_junk.py --tree          # 只看结构(深度2)
    python code/tools/inspect_junk.py --read          # 抽几个文本文件读内容
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3
MB = 1024 ** 2

TARGETS = [
    r"智能体大赛备份4\2026智能体大赛\没用的",
    r"智能体大赛备份5\2026智能体大赛\没用的_2",
    r"智能体大赛备份5\2026智能体大赛\没用的",
    r"智能体大赛备份3\2026智能体大赛\ai_neural_network_new\没用的旧模型",
    r"智能体大赛备份3\2026智能体大赛\download\旧版本对战",
    r"智能体大赛备份2\2026智能体大赛\ai_neural_network_new - 副本 (2)",
]

TEXT_EXT = {".md", ".txt", ".py", ".log", ".json", ".cfg", ".ini", ".yml", ".yaml", ".sh", ".html"}


def dsize(p):
    t = 0
    n = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try:
                t += os.stat(os.path.join(dp, f)).st_size
                n += 1
            except OSError:
                pass
    return t, n


def read_text(path, max_lines=14, max_bytes=200_000):
    """尽力解码,GBK / UTF-8 都试。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
    except OSError as e:
        return f"<读取失败: {e}>"
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return "<无法解码>"
    lines = txt.splitlines()[:max_lines]
    return "\n".join(lines)


def tree(path, depth=2, max_children=14):
    print(f"\n{'='*100}")
    print(f"### {os.path.relpath(path, ROOT)}")
    size, n = dsize(path)
    print(f"    合计 {size/G:.2f} GiB / {n} 个文件")
    print(f"{'='*100}")

    def rec(p, lvl):
        if lvl > depth:
            return
        try:
            entries = sorted(os.scandir(p), key=lambda e: -(
                e.stat().st_size if e.is_file() else 0))
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        # 目录按内部体量排序
        dir_info = []
        for e in dirs:
            s, c = dsize(e.path)
            dir_info.append((s, c, e))
        dir_info.sort(key=lambda x: -x[0])
        for s, c, e in dir_info[:max_children]:
            print(f"{'  '*lvl}[DIR ] {s/G:8.2f} GiB n={c:<6} {e.name[:80]}")
            rec(e.path, lvl + 1)
        if len(dir_info) > max_children:
            print(f"{'  '*lvl}       ... 另有 {len(dir_info)-max_children} 个子目录")
        for e in files[:max_children]:
            try:
                s = e.stat().st_size
            except OSError:
                s = 0
            print(f"{'  '*lvl}[FILE] {s/MB:8.2f} MiB         {e.name[:80]}")
        if len(files) > max_children:
            print(f"{'  '*lvl}       ... 另有 {len(files)-max_children} 个文件")

    rec(path, 1)


def sample_read(path, keep=24):
    """在目录里挑小文本文件读内容;同时统计文件类型。"""
    ext_agg = defaultdict(lambda: [0, 0])
    picks = []
    for dp, dn, fn in os.walk(path):
        for f in fn:
            full = os.path.join(dp, f)
            e = os.path.splitext(f)[1].lower() or "(noext)"
            try:
                s = os.stat(full).st_size
            except OSError:
                continue
            ext_agg[e][0] += s
            ext_agg[e][1] += 1
            if e in TEXT_EXT and s < 300_000 and len(picks) < keep:
                picks.append((full, s, e))
        if len(picks) >= keep and len(ext_agg) > 5:
            # 已有足够样本,但仍想统计完整类型,所以只跳过取样
            pass

    print(f"\n{'='*100}")
    print(f"### 文件类型分布: {os.path.relpath(path, ROOT)}")
    print(f"{'='*100}")
    for e, (s, c) in sorted(ext_agg.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {s/G:8.2f} GiB  n={c:<7} {e}")

    # 去重后挑不同目录的代表,避免全是同一个 run 的同类文件
    seen_dir = set()
    uniq = []
    for full, s, e in picks:
        d = os.path.dirname(full)
        if d in seen_dir:
            continue
        seen_dir.add(d)
        uniq.append((full, s, e))
    print(f"\n{'='*100}")
    print(f"### 内容抽样: {os.path.relpath(path, ROOT)}")
    print(f"{'='*100}")
    for full, s, e in uniq[:10]:
        print(f"\n--- {os.path.relpath(full, ROOT)}  ({s/1024:.1f} KB) ---")
        print(read_text(full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--only", type=int, default=None, help="只处理 TARGETS 里的第 N 个(从0开始)")
    args = ap.parse_args()

    targets = TARGETS
    if args.only is not None:
        targets = [TARGETS[args.only]]

    for t in targets:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            print(f"  (不存在) {t}")
            continue
        if args.tree:
            tree(p)
        if args.read:
            sample_read(p)


if __name__ == "__main__":
    main()
