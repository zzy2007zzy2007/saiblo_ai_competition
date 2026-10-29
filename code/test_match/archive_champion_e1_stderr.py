#!/usr/bin/env python
"""把 E1（我们 vs 冠军）各局的 stderr **按局归档**到 `match_results/champion_e1/stderr/`。

为什么要单独一个工具（而不是把逻辑塞在驱动里）：
  * `match_via_sdk.py` 的 stderr 落点固定是 `match_results/ai?_{label}_seed{N}[_<suffix>].stderr.log`，
    **不按 tag 分目录** ⇒ 同 seed 的不同批次/不同座位会互相覆盖（历史踩过两次）。
    驱动侧已经用 `MVS_STDERR_SUFFIX=1` + `DSH_PROGRESS_SUFFIX=_<TAG>` 唯一化；
    本工具负责把这些"带后缀的临时名"搬成**历史规范名**（去掉 `_<TAG>`），并按局处置：
      - 冠军那份 **gzip**（实测 ~0.2MB/回合 ⇒ 一局 100–150MB；`ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里无效，
        见 docs/us_vs_champion_20261028.md §1.2）；
      - 我们那份**原样**复制（`_batch_table.py` / `analyze_inject_vs_runnerup.py` 直接读明文）。
  * 幂等：目标已存在且大小一致就跳过（可以反复跑，可以边跑边归档）。
  * **只读源目录、不删任何源文件**（归档不是清理；原始 stderr 留在原地由人决定）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/archive_champion_e1_stderr.py \
      --dst match_results/champion_e1/stderr \
      --src match_results \
      --src "match_results/us_vs_external/_stale/US_VS_CHAMPION_BASE_20261028_234244" \
      --tag US_VS_CHAMPION_BASE
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUR_LABEL = "az_bridge_ai"
CHAMP_HINTS = ("champion", "magica")


def _norm_name(name: str, tag: str) -> str:
    """去掉驱动加的后缀：`ai1_champion_seed9_US_VS_CHAMPION_BASE.stderr.log` → `ai1_champion_seed9.stderr.log`。"""
    if tag:
        name = name.replace(f"_{tag}", "")
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="US_VS_CHAMPION_BASE")
    ap.add_argument("--dst", default="match_results/champion_e1/stderr")
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-age-min", type=float, default=0.0,
                    help="只归档**修改时间早于** N 分钟前的文件（>0 ⇒ 跳过正在跑的对局，可边跑边归档）")
    args = ap.parse_args()

    dst = REPO / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    srcs = [REPO / s for s in (args.src or ["match_results"])]
    import time as _t
    cutoff = (_t.time() - args.max_age_min * 60.0) if args.max_age_min > 0 else None

    seen: dict[str, Path] = {}
    for src in srcs:
        if not src.is_dir():
            print(f"[arch] ⚠️ 源目录不存在，跳过: {src}")
            continue
        # 只认本批的命名（带 tag 后缀），避免把别的实验的同名文件卷进来
        for f in sorted(src.glob(f"ai?_*_seed*_{args.tag}.stderr.log")):
            if cutoff is not None and f.stat().st_mtime > cutoff:
                continue                      # 可能还在写 ⇒ 这轮不碰（下一轮再来）
            name = _norm_name(f.name, args.tag)
            # 同一规范名在多个源里出现时：取**更大**的那个（更完整的对局），并在日志里说明
            if name in seen:
                old = seen[name]
                if f.stat().st_size > old.stat().st_size:
                    print(f"[arch] {name}: 用更大的源 {f}（{f.stat().st_size} > {old.stat().st_size}）")
                    seen[name] = f
                continue
            seen[name] = f

    n_new = n_skip = 0
    for name, src in sorted(seen.items()):
        is_champ = any(h in name for h in CHAMP_HINTS)
        out = dst / (name + ".gz") if is_champ else dst / name
        if is_champ:
            # 冠军那份用"源文件比 .gz 新 ⇒ 重压"而不是"存在即跳过"：
            # 边跑边归档时会先存到一份**还没写完**的 gz，收尾时必须能覆盖（否则归档里是半局）。
            fresh = (not out.exists()) or (src.stat().st_mtime > out.stat().st_mtime)
        else:
            fresh = (not out.exists()) or (out.stat().st_size != src.stat().st_size)
        if not fresh:
            n_skip += 1
            continue
        mb = src.stat().st_size / 1048576
        if args.dry:
            print(f"[arch] (dry) {name} <- {src}  {mb:.1f} MB  -> {out.name}")
            continue
        if is_champ:
            with open(src, "rb") as fi, gzip.open(out, "wb", compresslevel=1) as fo:
                shutil.copyfileobj(fi, fo, 1024 * 1024)
            print(f"[arch] 冠军 gzip {name}: {mb:.1f} MB -> {out.stat().st_size / 1048576:.1f} MB")
        else:
            shutil.copy2(src, out)
            print(f"[arch] 我方原样 {name}: {mb:.2f} MB")
        n_new += 1

    print(f"[arch] 归档完成 -> {dst}（新增 {n_new}，跳过 {n_skip}，共 {len(list(dst.iterdir()))} 个文件）")
    # 缺口清单（哪几局缺哪一侧的 stderr）——报告里要如实写
    miss = []
    for seat in (0, 1):
        for seed in range(7, 15):
            if not (dst / f"ai{seat}_{OUR_LABEL}_seed{seed}.stderr.log").exists():
                miss.append(f"缺我方 ai{seat} seed{seed}")
            if not (dst / f"ai{seat}_champion_seed{seed}.stderr.log.gz").exists():
                miss.append(f"缺冠军 ai{seat} seed{seed}")
    if miss:
        print(f"[arch] ⚠️ 缺口 {len(miss)} 项：")
        for m in miss:
            print(f"       - {m}")
    else:
        print("[arch] ✅ 8 个 seed × 2 座位的两侧 stderr 全齐")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
