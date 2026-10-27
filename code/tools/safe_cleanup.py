"""第 1、2 档清理:先逐项验证"复制源是否真的存在",只删通过验证的。

对每个待删目录 D:
  1) 找它的"原件" O = 去掉 ' - 副本' / ' - copy' / ' copy N' 后缀后的同名目录
  2) 逐文件比对 (相对路径, 大小):D 的每个文件在 O 里是否同名同大小  -> coverage
  3) 抽 3 个文件,读头尾各 256KB 做指纹,确认内容真的一样
  4) 只有 coverage >= 99.5% 且抽样内容全一致 才删

对没有原件的目录(如 '没用的旧模型'):
  改用"同一版本内、该目录之外是否有同名同大小的文件",要求 100% 命中。

第 1 档 temp_* 直接删(训练临时产物,名称本身即约定)。

默认 dry-run。--apply 才真删,并写 manifest。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

BACKUP_ROOT = r"F:\智能体大赛备份_旧版"
G = 1024 ** 3
HEAD = 256 * 1024

SUFFIX_RE = re.compile(r"\s*(-\s*)?(副本|copy)(\s*\(\d+\))?(\s*[\d_]+)?\s*$", re.I)
TIER2_KEYWORDS = ["没用的旧模型", "- 副本", " - copy", " copy"]
MANUAL_DIR_NAMES = {"没用的", "没用的_2"}
MANUAL_KEYWORDS = ["旧版本对战"]
TEMP_PREFIX = "temp_"

PROTECTED_NAMES = {"2026智能体大赛"}


def fp(path, size):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        if size <= HEAD * 2:
            h.update(f.read())
        else:
            h.update(f.read(HEAD))
            f.seek(-HEAD, os.SEEK_END)
            h.update(f.read(HEAD))
    return h.hexdigest()


def strip_suffix(name):
    m = SUFFIX_RE.search(name)
    if not m:
        return None
    base = name[: m.start()].rstrip()
    return base or None


def list_files(root):
    """relpath -> size"""
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            full = os.path.join(dp, f)
            try:
                out[os.path.relpath(full, root)] = os.stat(full).st_size
            except OSError:
                pass
    return out


def dir_bytes(root):
    t = 0
    for dp, dn, fn in os.walk(root):
        for f in fn:
            try:
                t += os.stat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return t


def is_manual(name):
    return name in MANUAL_DIR_NAMES or any(k in name for k in MANUAL_KEYWORDS)


def collect_dirs():
    """返回 (tier2_dirs, manual_dirs),每项 (path, version_root, name)"""
    t2, man = [], []
    for v in sorted(os.listdir(BACKUP_ROOT)):
        vp = os.path.join(BACKUP_ROOT, v)
        if not os.path.isdir(vp):
            continue
        for dirpath, dirnames, filenames in os.walk(vp):
            keep = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if d in PROTECTED_NAMES:
                    keep.append(d)
                    continue
                if is_manual(d):
                    man.append((full, vp, d))
                    continue
                if any(k in d for k in TIER2_KEYWORDS):
                    t2.append((full, vp, d))
                    continue
                keep.append(d)
            dirnames[:] = keep
    return t2, man


def verify_dir(path, vroot, name):
    """返回 dict: ok, coverage, why, bytes, nfiles, sample_ok, original"""
    D = list_files(path)
    total = sum(D.values())
    res = {"path": path, "bytes": total, "nfiles": len(D), "original": None,
           "coverage": 0.0, "sample_ok": None, "ok": False, "why": ""}

    base = strip_suffix(name)
    orig = os.path.join(os.path.dirname(path), base) if base else None
    if orig and os.path.isdir(orig):
        O = list_files(orig)
        matched = 0
        pairs = []
        for rel, s in D.items():
            if O.get(rel) == s:
                matched += s
                pairs.append((rel, s, os.path.join(path, rel), os.path.join(orig, rel)))
        res["coverage"] = matched / total if total else 1.0
        res["original"] = orig
        # 抽样校验内容
        step = max(1, len(pairs) // 3)
        samples = pairs[::step][:3]
        ok = True
        for rel, s, pa, pb in samples:
            try:
                if fp(pa, s) != fp(pb, s):
                    ok = False
                    break
            except OSError:
                ok = False
                break
        res["sample_ok"] = ok if samples else None
        res["ok"] = res["coverage"] >= 0.995 and (ok if samples else True)
        res["why"] = f"原件 {base!r}: 逐文件匹配 {res['coverage']*100:.1f}%,抽样 {len(samples)} 个 {'一致' if ok else '不一致'}"
    else:
        # 无原件:同一版本内、本目录之外是否有同名同大小文件
        idx = defaultdict(int)
        for dp, dn, fn in os.walk(vroot):
            if os.path.abspath(dp).startswith(os.path.abspath(path) + os.sep) or os.path.abspath(dp) == os.path.abspath(path):
                dn[:] = []
                continue
            for f in fn:
                try:
                    idx[(f, os.stat(os.path.join(dp, f)).st_size)] += 1
                except OSError:
                    pass
        matched = sum(s for rel, s in D.items() if (os.path.basename(rel), s) in idx)
        res["coverage"] = matched / total if total else 1.0
        # 抽样:找一个能对上号的对照文件比内容
        samples = []
        for rel, s in D.items():
            if (os.path.basename(rel), s) in idx and len(samples) < 3:
                samples.append((rel, s))
        ok = True
        matched_n = 0
        for dp, dn, fn in os.walk(vroot):
            for f in fn:
                full = os.path.join(dp, f)
                try:
                    st = os.stat(full).st_size
                except OSError:
                    continue
                if full.startswith(path + os.sep):
                    continue
                for rel, s in list(samples):
                    if f == os.path.basename(rel) and st == s:
                        try:
                            if fp(full, s) == fp(os.path.join(path, rel), s):
                                matched_n += 1
                                samples.remove((rel, s))
                        except OSError:
                            pass
                if not samples:
                    break
            if not samples:
                break
        res["sample_ok"] = ok
        res["ok"] = res["coverage"] >= 0.999
        res["why"] = f"无原件:同版本内同名同大小命中 {res['coverage']*100:.1f}%"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认(仅在已经 dry-run 过的前提下用)")
    ap.add_argument("--skip-temp", action="store_true", help="只处理第2档,不删 temp_*")
    ap.add_argument("--skip-copies", action="store_true", help="只删 temp_*,不处理副本目录")
    args = ap.parse_args()

    t2, man = collect_dirs()
    if args.skip_copies:
        print(f"[跳过第2档] 需人工(不动){len(man)} 个\n")
        t2 = []
    else:
        print(f"待验证的第 2 档目录 {len(t2)} 个;需人工(不动){len(man)} 个\n", flush=True)

    passed, failed = [], []
    for path, vroot, name in sorted(t2, key=lambda x: -dir_bytes(x[0])):
        r = verify_dir(path, vroot, name)
        r["rel"] = os.path.relpath(path, BACKUP_ROOT)
        (passed if r["ok"] else failed).append(r)
        print(f"  [{'PASS' if r['ok'] else 'HOLD'}] {r['bytes']/G:7.2f} GiB  {r['rel'][:78]}")
        print(f"          {r['why']}", flush=True)

    pb = sum(r["bytes"] for r in passed)
    fb = sum(r["bytes"] for r in failed)
    print(f"\n第2档:通过 {len(passed)} 项 {pb/G:.2f} GiB / 保留 {len(failed)} 项 {fb/G:.2f} GiB")

    # 第1档 temp_*
    temp_files = []
    if not args.skip_temp:
        for v in sorted(os.listdir(BACKUP_ROOT)):
            vp = os.path.join(BACKUP_ROOT, v)
            if not os.path.isdir(vp):
                continue
            for dirpath, dirnames, filenames in os.walk(vp):
                dirnames[:] = [d for d in dirnames if not is_manual(d)]
                # 别碰虚拟环境:venv 里有 pip 自己的 temp_dir.py 之类
                low = dirpath.lower()
                if "venv" in low or "site-packages" in low:
                    dirnames[:] = []
                    continue
                for f in filenames:
                    if f.startswith(TEMP_PREFIX):
                        full = os.path.join(dirpath, f)
                        try:
                            temp_files.append((full, os.stat(full).st_size))
                        except OSError:
                            pass
    tb = sum(s for _, s in temp_files)
    print(f"第1档:temp_* {len(temp_files)} 个文件 {tb/G:.2f} GiB")
    if temp_files:
        pat = defaultdict(int)
        pb2 = defaultdict(int)
        for p, s in temp_files:
            b = os.path.basename(p)
            m = re.match(r"^(temp_[a-z]+)", b)
            k = m.group(1) if m else b
            pat[k] += 1
            pb2[k] += s
        print("  文件类型分布:")
        for k in sorted(pb2, key=lambda x: -pb2[x]):
            print(f"    {pb2[k]/G:8.2f} GiB  n={pat[k]:<5} {k}*")

    total = 0 if args.skip_copies else pb
    total += 0 if args.skip_temp else tb
    print(f"\n{'='*96}\n可安全删除合计: {total/G:.2f} GiB\n{'='*96}")

    if not args.apply:
        print("\n[dry-run] 没有删除任何东西。加 --apply 执行。")
        return 0

    if total <= 0:
        print("没有可删的。")
        return 0

    if not args.yes:
        ans = input(f"\n将永久删除 {total/G:.2f} GiB,输入 DELETE 确认: ").strip()
        if ans != "DELETE":
            print("已取消。")
            return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", f"deleted_manifest_{stamp}.tsv"))
    freed = 0
    errors = []
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write("size_bytes\tkind\tpath\n")
        if not args.skip_copies:
            for r in passed:
                p = r["path"]
                assert p.startswith(BACKUP_ROOT + os.sep), p
                assert os.path.basename(p) not in PROTECTED_NAMES, p
                try:
                    shutil.rmtree(p)
                    freed += r["bytes"]
                    mf.write(f"{r['bytes']}\tT2|{r['why']}\t{p}\n")
                    mf.flush()
                    print(f"  删 {r['rel']}  ({r['bytes']/G:.2f} GiB)")
                except OSError as e:
                    errors.append((p, str(e)))
                    print(f"  !! 失败 {p}: {e}")
        if not args.skip_temp:
            for p, s in temp_files:
                assert p.startswith(BACKUP_ROOT + os.sep), p
                try:
                    os.remove(p)
                    freed += s
                    mf.write(f"{s}\tT1|temp_*\t{p}\n")
                    if freed % (10 * 1024 ** 3) < s:
                        mf.flush()
                        print(f"  ... 已释放 {freed/G:.1f} GiB")
                except OSError as e:
                    errors.append((p, str(e)))

    print(f"\n完成:释放 {freed/G:.2f} GiB。失败 {len(errors)} 项。")
    print(f"清单: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
