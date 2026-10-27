"""元数据判重扫描(针对 USB 机械盘优化,尽量少读内容)。

核心思路:
  文件大小不同的两个文件,内容必然不同。
  => "大小唯一"的文件 100% 是独一无二的内容,不可能被去重。
  => 重复上界 = 总字节 - 大小唯一文件的字节。

  然后再只对"最值钱"的同大小分组(Top-K)读 128KB 头部做指纹确认,
  这部分覆盖了绝大部分体积,读取量很小。

输出:总量 / 扩展名分布 / 副本目录 / 重复上界 / 已确认重复量。
"""
import os
import sys
import hashlib
import json
from collections import defaultdict

ROOT = r"F:\智能体大赛备份_旧版"
HEAD = 64 * 1024
TOP_K = 400

COPY_HINTS = ("副本", " copy", "copy ", "_copy", "copy2", "(2)", "(3)", "备份")


def fingerprint(path, size):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        if size <= HEAD * 2:
            h.update(f.read())
        else:
            h.update(f.read(HEAD))
            f.seek(-HEAD, os.SEEK_END)
            h.update(f.read(HEAD))
    return h.hexdigest()


def main():
    files = []
    ver_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    for v in ver_dirs:
        vp = os.path.join(ROOT, v)
        n = 0
        for dirpath, dirnames, filenames in os.walk(vp):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                files.append((fp, st.st_size, v))
                n += 1
        print(f"  {v}: {n} files", flush=True)

    total = sum(s for _, s, _ in files)
    G = 1024 ** 3
    print(f"\n== 总量 ==\nfiles={len(files)}  total={total/G:.2f} GiB ({total/G/1024:.2f} TiB)", flush=True)

    # ---- 扩展名分布 ----
    ext_bytes = defaultdict(int)
    ext_cnt = defaultdict(int)
    for fp, s, _ in files:
        e = os.path.splitext(fp)[1].lower() or "(no ext)"
        ext_bytes[e] += s
        ext_cnt[e] += 1
    print("\n== 扩展名占用 (Top 20) ==")
    for e, b in sorted(ext_bytes.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {b/G:9.2f} GiB  {b/total*100:5.1f}%  n={ext_cnt[e]:>7}  {e}")

    # ---- 大小唯一 => 必然唯一内容 ----
    by_size = defaultdict(list)
    for fp, s, v in files:
        by_size[s].append((fp, v))
    uniq_bytes = sum(s for s, lst in by_size.items() if len(lst) == 1)
    uniq_files = sum(1 for lst in by_size.values() if len(lst) == 1)
    dup_label_bytes = total - uniq_bytes
    print(f"\n== 判重上界 ==")
    print(f"  大小唯一的文件: n={uniq_files}  {uniq_bytes/G:.2f} GiB  <-- 必然无法去重")
    print(f"  出现在同大小分组里的字节: {dup_label_bytes/G:.2f} GiB  <-- 重复上界")
    print(f"  即:最多能省 {dup_label_bytes/G:.2f} GiB,最少省 0")

    # ---- 对最值钱的分组做指纹确认 ----
    cand = [(s, lst) for s, lst in by_size.items() if len(lst) > 1]
    cand.sort(key=lambda kv: -kv[0] * (len(kv[1]) - 1))
    confirmed_saving = 0
    covered_bytes = 0
    groups_out = []
    budget_bytes = 150 * 1024 * 1024  # 最多读 150MB 内容
    read_done = 0
    for s, lst in cand:
        if read_done > budget_bytes:
            break
        if len(groups_out) >= TOP_K:
            break
        groups = defaultdict(list)
        for fp, v in lst:
            try:
                groups[fingerprint(fp, s)].append(fp)
                read_done += min(s, HEAD * 2)
            except OSError:
                continue
        covered_bytes += s * len(lst)
        for fp_hash, paths in groups.items():
            if len(paths) > 1:
                save = s * (len(paths) - 1)
                confirmed_saving += save
                groups_out.append((s, len(paths), save, paths[0]))
        print(f"    done size={s/1024**2:.1f}MiB n={len(lst)} read={read_done/1024**2:.0f}MB", flush=True)

    groups_out.sort(key=lambda x: -x[2])
    print(f"\n== 指纹确认 ==")
    print(f"  覆盖分组字节 {covered_bytes/G:.2f} GiB")
    print(f"  已确认的重复浪费 {confirmed_saving/G:.2f} GiB")
    print("\n== 最大的重复组 (Top 25) ==")
    for s, c, save, sample in groups_out[:25]:
        rel = os.path.relpath(sample, ROOT)
        print(f"  {s/1024**2:9.1f} MiB x{c:<4} 省 {save/G:7.2f} GiB   {rel[:110]}")

    # ---- 副本目录 ----
    copy_bytes = 0
    copy_dirs = []
    seen = set()
    for fp, s, v in files:
        rel = os.path.relpath(fp, ROOT)
        parts = rel.split(os.sep)[2:]  # 去掉 备份N/2026智能体大赛
        for i, part in enumerate(parts[:-1]):
            if any(h in part for h in COPY_HINTS):
                d = os.sep.join(parts[: i + 1])
                key = (v, d)
                if key not in seen:
                    seen.add(key)
                    copy_dirs.append((v, d))
                copy_bytes += s
                break
    print(f"\n== 名字像'副本/备份/copy'的目录 ==")
    print(f"  这些目录合计占用 {copy_bytes/G:.2f} GiB (含其唯一内容)")
    agg = defaultdict(int)
    for fp, s, v in files:
        rel = os.path.relpath(fp, ROOT)
        parts = rel.split(os.sep)[2:]
        for i, part in enumerate(parts[:-1]):
            if any(h in part for h in COPY_HINTS):
                agg[os.sep.join(parts[: i + 1])] += s
                break
    for d, b in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    {b/G:8.2f} GiB  {d[:110]}")

    json.dump(
        {
            "total": total,
            "n_files": len(files),
            "versions": ver_dirs,
            "unique_by_size_bytes": uniq_bytes,
            "dup_upper_bound_bytes": dup_label_bytes,
            "confirmed_saving_bytes": confirmed_saving,
            "covered_bytes": covered_bytes,
            "copy_like_bytes": copy_bytes,
            "ext": {e: [ext_bytes[e], ext_cnt[e]] for e in ext_bytes},
            "groups": [{"size": s, "n": c, "save": sv, "sample": os.path.relpath(p, ROOT)} for s, c, sv, p in groups_out],
        },
        open("dedupe_scan2_result.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print("\n写成 dedupe_scan2_result.json", flush=True)


if __name__ == "__main__":
    main()
