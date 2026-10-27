#!/usr/bin/env python3
"""批量离线审计：挑最近 N 个**已完成**的 progress trace 重放，统计非法操作。

（`audit_replay_illegal.py` 的批量入口封装；挑样本时排除本次实验自己造的 trace。）

用法：
    python -u code/test_match/audit_replay_batch.py --n 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# 排除"本次实验自己造的" trace（故障注入 / 快速验证局），否则会把人为非法 op 当成历史发现
EXCLUDE = ("illegal_", "audit_", "_smoke", "seed7501", "seed7301", "seed7302", "seed7401")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="training_history/audit_replay_batch.log")
    args = ap.parse_args()
    traces = []
    for p in sorted((REPO / "training_history/progress").rglob("az_progress_*.txt"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        if any(x in str(p) for x in EXCLUDE):
            continue
        if p.stat().st_size == 0:
            continue
        try:
            if not p.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1].startswith("terminal"):
                continue
        except Exception:  # noqa: BLE001
            continue
        traces.append(p)
        if len(traces) >= args.n:
            break
    print(f"挑出 {len(traces)} 个已完成 trace（排除 {'/'.join(EXCLUDE)}）", flush=True)
    cmd = [sys.executable, "-u", str(REPO / "code/test_match/audit_replay_illegal.py")]
    for t in traces:
        cmd += ["--trace", str(t)]
    out = REPO / args.out
    with out.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode
    print(f"exit={rc}  log={out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
