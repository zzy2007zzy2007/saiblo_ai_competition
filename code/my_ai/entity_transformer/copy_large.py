#!/usr/bin/env python3
"""大文件分块拷贝 + 实时进度日志 + 完成后抽样校验。

为什么需要：之前用 robocopy 拷 board.npy（86.5 GB）时过早中断，
留下一个**大小正确但内容是零填充**的文件，导致后续训练全在损坏数据上跑。

用法：
  python code/my_ai/entity_transformer/copy_large.py \
      --src F:/.../board.npy --dst D:/.../board.npy \
      --log D:/.../copy_board.log
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

CHUNK = 1 << 30  # 1 GiB


def append(log: Path | None, msg: str) -> None:
    print(msg, flush=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def sample_hash(path: Path, size: int, n: int = 8, blk: int = 1 << 20) -> str:
    """对文件头部、尾部、以及中间 n 个位置各取 1 MB 做 sha256（用于快速一致性校验）。"""
    h = hashlib.sha256()
    offsets = [0, max(0, size - blk)]
    step = max(1, size // (n + 1))
    offsets += [min(max(0, step * (i + 1)), max(0, size - blk)) for i in range(n)]
    with path.open("rb") as f:
        for off in sorted(set(offsets)):
            f.seek(off)
            h.update(f.read(blk))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    log = Path(a.log) if a.log else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")

    size = src.stat().st_size
    append(log, f"[copy] {src} -> {dst}  ({size/1e9:.1f} GB)")
    t0 = time.time()
    done = 0
    with src.open("rb") as fi, dst.open("wb") as fo:
        while True:
            buf = fi.read(CHUNK)
            if not buf:
                break
            fo.write(buf)
            done += len(buf)
            el = time.time() - t0
            append(log, f"  {done/1e9:6.1f}/{size/1e9:.1f} GB  "
                        f"{done/1e6/max(el,1e-9):6.0f} MB/s  elapsed={el/60:5.1f} min")
    fo_size = dst.stat().st_size
    append(log, f"[copy] done size={fo_size/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min")
    if fo_size != size:
        append(log, f"[copy] SIZE MISMATCH {fo_size} != {size}")
        return 1
    hs = sample_hash(src, size)
    hd = sample_hash(dst, size)
    ok = hs == hd
    append(log, f"[verify] src={hs[:16]} dst={hd[:16]} equal={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
