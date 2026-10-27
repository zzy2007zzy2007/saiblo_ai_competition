"""备份目录清理工具 —— **默认只列出,不删除**。

用法:
    python code/tools/cleanup_backup.py            # 列出全部候选(分级)
    python code/tools/cleanup_backup.py --apply    # 只删第 1、2 档

分级依据来自 docs/备份空间审计与git方案.md 第 6.2 节的全局唯一性检查:

  第 1 档  temp_*           266.20 GiB  GA 每代的临时 genome,训练完即无意义。
                                        且 备份5/6/7 各存一份,删了还有。
  第 2 档  *- 副本* / copy    ~290 GiB   按「文件名+大小」在全备份 7 个版本里比对,
                                        这些目录里 95%~100% 的字节在别处也存在。
  需人工  没用的 / 没用的_2 / 旧版本对战
                                        ⚠️ 里面混着全备份独一份的东西
                                        (dataset.pkl 15.3GB、good_versions 提交包、
                                         ga_nn_training_refactored 重构版框架、
                                         Trae聊天记录19.md 等),**工具绝不自动删**。

安全设计:
  * 只允许在 BACKUP_ROOT 下面操作;版本根目录永远不删
  * 必须显式 --apply,且要输入 DELETE 二次确认
  * 每删一项都写进 manifest(deleted_manifest_<时间>.tsv)
"""
import argparse
import os
import shutil
import sys
import time

BACKUP_ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3

# 第 1 档:训练中间产物文件
TIER1_FILE_PREFIX = ["temp_"]

# 第 2 档:经全局比对确认"别处也有"的副本类目录
# 注意 '没用的旧模型' 必须排在 MANUAL 判定之前 —— 它名字里含 '没用的',
# 但全局比对显示 C 类(独一份)= 0,是干净可删的。
TIER2_DIR_KEYWORDS = ["没用的旧模型", " - 副本", " - copy", " copy"]

# 需人工判断 —— 永不自动删。这里用**精确目录名**匹配,避免子串误伤。
MANUAL_DIR_NAMES = {"没用的", "没用的_2"}
MANUAL_DIR_KEYWORDS = ["旧版本对战"]

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


def collect():
    """返回 tiers = {1:[...], 2:[...], manual:[...]},元素 (path, size, nfiles, reason)。"""
    tiers = {1: [], 2: [], "manual": []}
    for v in sorted(os.listdir(BACKUP_ROOT)):
        vp = os.path.join(BACKUP_ROOT, v)
        if not os.path.isdir(vp):
            continue

        # --- 目录:自顶向下,命中就整棵跳过 ---
        for dirpath, dirnames, filenames in os.walk(vp):
            keep = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                rel = os.path.relpath(full, vp)
                if rel in PROTECTED:
                    keep.append(d)
                    continue
                # MANUAL 优先(精确名或关键字)
                if d in MANUAL_DIR_NAMES or any(k in d for k in MANUAL_DIR_KEYWORDS):
                    s, n = dir_size(full)
                    tiers["manual"].append((full, s, n, f"需人工:{d}"))
                    continue
                if any(k in d for k in TIER2_DIR_KEYWORDS):
                    s, n = dir_size(full)
                    tiers[2].append((full, s, n, f"副本目录:{d}"))
                    continue
                keep.append(d)
            dirnames[:] = keep

        # --- 文件:temp_* ---
        for dirpath, dirnames, filenames in os.walk(vp):
            for f in filenames:
                if any(f.startswith(p) for p in TIER1_FILE_PREFIX):
                    full = os.path.join(dirpath, f)
                    try:
                        s = os.stat(full).st_size
                    except OSError:
                        continue
                    tiers[1].append((full, s, 1, f"中间产物:{f}"))
    return tiers


def show(name, items, note=""):
    if not items:
        return 0
    b = sum(i[1] for i in items)
    n = sum(i[2] for i in items)
    print(f"\n-- {name}:{len(items)} 项,{b/G:.2f} GiB,{n} 个文件 {note}")
    for p, s, c, why in sorted(items, key=lambda x: -x[1])[:20]:
        print(f"   {s/G:8.2f} GiB  n={c:<6} {os.path.relpath(p, BACKUP_ROOT)[:92]}")
    if len(items) > 20:
        print(f"   ... 其余 {len(items)-20} 项")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的删除第 1、2 档(默认只列出)")
    args = ap.parse_args()

    if not os.path.isdir(BACKUP_ROOT):
        print(f"找不到 {BACKUP_ROOT}")
        return 1

    tiers = collect()
    t1 = show("第1档 temp_* 中间产物", tiers[1], "(可删)")
    t2 = show("第2档 副本类目录", tiers[2], "(按全局比对 95%~100% 别处也有,可删)")
    tm = show("需人工 没用的/没用的_2/旧版本对战", tiers["manual"], "(⚠️ 工具不会动)")

    auto = t1 + t2
    print(f"\n{'='*100}")
    print(f"可自动清理合计: {auto/G:.2f} GiB")
    print(f"需人工判断合计: {tm/G:.2f} GiB")
    print(f"{'='*100}")
    if tm:
        print("⚠️  '没用的' 里混着全备份独一份的内容(dataset.pkl 15.3GB、good_versions 提交包、")
        print("    ga_nn_training_refactored 重构版框架、Trae聊天记录19.md 等),")
        print("    请先跑 inspect_junk.py 看清楚再手动处理,不要整目录删。")

    if not args.apply:
        print("\n[只列出] 没有删除任何东西。确认后加 --apply 执行。")
        return 0

    ans = input(f"\n将永久删除 {auto/G:.2f} GiB(仅第 1、2 档)。输入 DELETE 确认: ").strip()
    if ans != "DELETE":
        print("已取消。")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     f"deleted_manifest_{stamp}.tsv"))
    freed = 0
    print()
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write("size_bytes\ttype\tpath\n")
        for label, items in (("T1", tiers[1]), ("T2", tiers[2])):
            for p, s, n, why in items:
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                    freed += s
                    mf.write(f"{s}\t{label}|{why}\t{p}\n")
                    mf.flush()
                except OSError as e:
                    print(f"  !! 跳过 {p}: {e}")

    print(f"完成:释放 {freed/G:.2f} GiB。清单:{manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
