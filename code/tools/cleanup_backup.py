"""备份目录清理工具 —— **默认只列出,不删除**。

用法:
    python code/tools/cleanup_backup.py              # 只列出(默认)
    python code/tools/cleanup_backup.py --tier1      # 列出第一档候选
    python code/tools/cleanup_backup.py --tier1 --apply

安全设计:
  * 只允许在 BACKUP_ROOT 下面操作;根目录 / 版本根目录永远不删
  * 删除前先统计并打印,必须显式 --apply 才动手
  * 每删一项都写进 manifest(deleted_manifest_<时间>.tsv),事后可查
  * --apply 需要二次确认(输入 DELETE)

绝不碰的东西:
  * 任何 .py / .md / .txt / .json 源码与文档(除非它们正好在 junk 目录里)
    实际上:junk 目录内部会被整目录删掉,这里不做逐文件区分
"""
import argparse
import os
import shutil
import sys
import time

BACKUP_ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3

# ---- 第一档:名字本身就说"没用/是副本"的目录(按名字匹配,较深层的会随父目录一起删) ----
TIER1_DIR_KEYWORDS = [
    "没用的",
    "没用的旧模型",
    "旧版本对战",
    " - 副本",
    " - copy",
]

# ---- 第一档:训练中间产物文件 ----
TIER1_FILE_PREFIX = ["temp_"]

# 永远不会删除的路径(防手滑)
PROTECTED = {"", "2026智能体大赛"}


def dir_size(path):
    total = 0
    n = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.stat(os.path.join(dp, f)).st_size
                n += 1
            except OSError:
                pass
    return total, n


def collect_tier1():
    """返回 [(path, size, nfiles, reason)]。已做"父目录优先"去重。"""
    hits = []
    for v in sorted(os.listdir(BACKUP_ROOT)):
        vp = os.path.join(BACKUP_ROOT, v)
        if not os.path.isdir(vp):
            continue
        # 1) junk 目录:自顶向下扫,命中就整棵跳过(避免重复统计)
        for dirpath, dirnames, filenames in os.walk(vp):
            keep = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                rel = os.path.relpath(full, vp)
                if rel in PROTECTED or any(rel.startswith(p + os.sep) for p in PROTECTED):
                    keep.append(d)
                    continue
                if any(k in d for k in TIER1_DIR_KEYWORDS):
                    size, n = dir_size(full)
                    hits.append((full, size, n, f"junk目录:{d}"))
                    continue  # 不再往里面走
                keep.append(d)
            dirnames[:] = keep

        # 2) temp_* 文件
        for dirpath, dirnames, filenames in os.walk(vp):
            for f in filenames:
                if any(f.startswith(p) for p in TIER1_FILE_PREFIX):
                    full = os.path.join(dirpath, f)
                    try:
                        s = os.stat(full).st_size
                    except OSError:
                        continue
                    hits.append((full, s, 1, f"中间产物:{f}"))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", action="store_true", help="第一档:junk 目录 + temp_* 文件")
    ap.add_argument("--apply", action="store_true", help="真的删除(默认只列出)")
    args = ap.parse_args()

    if not args.tier1:
        print(__doc__)
        return

    if not os.path.isdir(BACKUP_ROOT):
        print(f"找不到 {BACKUP_ROOT}")
        return 1

    hits = collect_tier1()
    hits.sort(key=lambda x: -x[1])

    total = sum(h[1] for h in hits)
    nf = sum(h[2] for h in hits)

    print(f"=== 第一档候选:{len(hits)} 项,{total/G:.2f} GiB,{nf} 个文件 ===")
    print(f"{'-- 前 40 项 --':^100}")
    for p, s, n, why in hits[:40]:
        print(f"  {s/G:8.2f} GiB  n={n:<6} [{why}]  {os.path.relpath(p, BACKUP_ROOT)[:95]}")
    if len(hits) > 40:
        print(f"  ... 其余 {len(hits)-40} 项")

    if not args.apply:
        print(f"\n[只列出] 没有删除任何东西。确认后加 --apply 执行。")
        return 0

    ans = input(f"\n将永久删除 {total/G:.2f} GiB / {nf} 个文件。输入 DELETE 确认: ").strip()
    if ans != "DELETE":
        print("已取消。")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"..", "..", f"deleted_manifest_{stamp}.tsv")
    manifest = os.path.abspath(manifest)
    freed = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write("size_bytes\ttype\tpath\n")
        for p, s, n, why in hits:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=False)
                else:
                    os.remove(p)
                freed += s
                mf.write(f"{s}\t{why}\t{p}\n")
                mf.flush()
            except OSError as e:
                print(f"  !! 跳过 {p}: {e}")

    print(f"\n完成:释放 {freed/G:.2f} GiB。清单:{manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
