"""对比同一备份内 "原目录 vs 副本目录" 的重合度(仅用路径+大小,不读内容)。"""
import os
import sys
from collections import defaultdict

BASE = r"F:\智能体大赛备份_旧版\智能体大赛备份8\2026智能体大赛\Ant-Game"

PAIRS = [
    ("training_pipeline", "training_pipeline - 副本"),
    ("training_pipeline", "training_pipeline - 副本 (2)"),
    ("training_pipeline", "training_pipeline - 副本 (3)"),
    ("training_pipeline", "mcts_training_pipeline"),
]


def snapshot(root):
    """返回 {相对路径: 大小}"""
    out = {}
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                out[os.path.relpath(fp, root)] = os.stat(fp).st_size
            except OSError:
                pass
    return out


def main():
    cache = {}
    G = 1024 ** 3
    for a, b in PAIRS:
        for name in (a, b):
            if name not in cache:
                p = os.path.join(BASE, name)
                cache[name] = snapshot(p)
                tot = sum(cache[name].values())
                print(f"  scan {name}: {len(cache[name])} files, {tot/G:.2f} GiB", flush=True)

    print()
    for a, b in PAIRS:
        A, B = cache[a], cache[b]
        if not A or not B:
            continue
        b_bytes = sum(B.values())
        same = 0          # 相对路径相同 且 大小相同
        same_bytes = 0
        only_in_b = b_bytes
        for rel, s in B.items():
            if A.get(rel) == s:
                same += 1
                same_bytes += s
                only_in_b -= s
        print(f"=== {a}  vs  {b} ===")
        print(f"  {b}: {len(B)} files, {b_bytes/G:.3f} GiB")
        print(f"  其中与原目录同名同大小: {same} files, {same_bytes/G:.3f} GiB "
              f"({same_bytes/max(b_bytes,1)*100:.1f}%)")
        print(f"  仅存在于副本: {len(B)-same} files, {only_in_b/G:.3f} GiB")
        print(f"  => 这份副本里 {same_bytes/G:.3f} GiB 是纯浪费")
        print()


if __name__ == "__main__":
    main()
