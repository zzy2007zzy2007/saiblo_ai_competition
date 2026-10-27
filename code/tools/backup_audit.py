"""备份目录审计:把 1.8 TiB 拆成"该留的 / 该删的 / 该搬走的"。

分类规则(按路径片段 + 文件名 + 扩展名):
  junk_dir   路径里出现 没用的 / 副本 / copy / 旧 / old / backup / 备份 的目录
  temp_file  文件名以 temp_ 开头(训练中间产物)
  ckpt       模型权重 .npz .pth .pt .pkl
  dataset    数据 .json .zip .npy .pkl(>10MB)
  small      其余(源码/文档/日志/元数据)

输出 backup_audit_result.json + 控制台表格。
"""
import os
import json
import re
from collections import defaultdict

ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3

JUNK_RE = re.compile(r"没用的|副本|- copy|copy |_copy|旧版|old_|_old|backup|备份\s*\d", re.I)
CKPT_EXT = {".npz", ".pth", ".pt", ".pkl", ".ckpt", ".bin"}
DATA_EXT = {".json", ".zip", ".npy", ".npz", ".csv", ".jsonl", ".7z", ".tar", ".gz"}


def classify(rel_parts, filename, size):
    for p in rel_parts:
        if JUNK_RE.search(p):
            return "junk_dir", p
    if filename.lower().startswith("temp_"):
        return "temp_file", None
    ext = os.path.splitext(filename)[1].lower()
    if ext in CKPT_EXT:
        return "ckpt", None
    if ext in DATA_EXT and size > 10 * 1024 * 1024:
        return "dataset", None
    return "small", None


def main():
    cat_bytes = defaultdict(int)
    cat_files = defaultdict(int)
    per_ver = defaultdict(lambda: defaultdict(int))
    junk_roots = defaultdict(int)
    temp_roots = defaultdict(int)
    ver_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    total = 0
    n_files = 0

    for v in ver_dirs:
        vp = os.path.join(ROOT, v)
        short = v.replace("智能体大赛备份", "v")
        for dirpath, dirnames, filenames in os.walk(vp):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    size = os.stat(fp).st_size
                except OSError:
                    continue
                total += size
                n_files += 1
                parts = os.path.relpath(dirpath, vp).split(os.sep)
                rel_parts = [p for p in parts if p not in (".", "")]
                cat, jroot = classify(rel_parts, fn, size)
                cat_bytes[cat] += size
                cat_files[cat] += 1
                per_ver[short][cat] += size
                if cat == "junk_dir":
                    # 找到最浅的 junk 目录作为根
                    idx = next(i for i, p in enumerate(rel_parts) if JUNK_RE.search(p))
                    junk_roots[os.sep.join(rel_parts[: idx + 1])] += size
                elif cat == "temp_file":
                    temp_roots[os.sep.join(rel_parts[:3]) or "."] += size
        print(f"  {v} done", flush=True)

    print(f"\n总占用 {total/G:.2f} GiB / {n_files} files\n")
    print("== 分类 ==")
    for c in sorted(cat_bytes, key=lambda k: -cat_bytes[k]):
        print(f"  {c:<10} {cat_bytes[c]/G:9.2f} GiB  {cat_bytes[c]/total*100:5.1f}%  n={cat_files[c]}")

    print("\n== 各版本构成 (GiB) ==")
    cats = [c for c in sorted(cat_bytes, key=lambda k: -cat_bytes[k])]
    print("  ver " + "".join(f"{c:>12}" for c in cats) + f"{'合计':>12}")
    for v in sorted(per_ver):
        row = per_ver[v]
        print(f"  {v:<4}" + "".join(f"{row[c]/G:12.1f}" for c in cats) + f"{sum(row.values())/G:12.1f}")

    print("\n== '没用的/副本' 类目录 Top 20 ==")
    for d, b in sorted(junk_roots.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {b/G:8.2f} GiB  {d[:110]}")

    print("\n== temp_* 中间产物所在目录 Top 15 ==")
    for d, b in sorted(temp_roots.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {b/G:8.2f} GiB  {d[:110]}")

    json.dump(
        {
            "total": total,
            "n_files": n_files,
            "cat_bytes": dict(cat_bytes),
            "cat_files": dict(cat_files),
            "per_ver": {k: dict(v) for k, v in per_ver.items()},
            "junk_roots": dict(sorted(junk_roots.items(), key=lambda kv: -kv[1])),
            "temp_roots": dict(sorted(temp_roots.items(), key=lambda kv: -kv[1])),
        },
        open("backup_audit_result.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print("\n写成 backup_audit_result.json", flush=True)


if __name__ == "__main__":
    main()
