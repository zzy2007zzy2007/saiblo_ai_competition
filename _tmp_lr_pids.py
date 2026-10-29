#!/usr/bin/env python
"""进程普查（只读）：某个 PID 是否还活着 + 它有多少存活后代。

用途：验证看门狗"**只杀自己那一局**"——跑之前/跑之后各查一次**别人那批真实对局**的主 PID
（`US_VS_CHAMPION_BASE` 的 ladder 进程），证明一个都没少。绝不 kill，只读进程表。

用法：python _tmp_lr_pids.py 58044 --label US_VS_CHAMPION_BASE
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def load_ladder():
    spec = importlib.util.spec_from_file_location("_lad_for_pids", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lad_for_pids"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    args = sys.argv[1:]
    label = ""
    if "--label" in args:
        i = args.index("--label")
        label = args[i + 1]
        del args[i:i + 2]
    pids = [int(x) for x in args]
    lad = load_ladder()
    table = lad._process_table()
    alive = {pid for pid, _ppid in table}
    print(f"[census] python 进程总数(表内): {len(table)}" + (f"   ({label})" if label else ""))
    for pid in pids:
        kids = [p for p in lad._descendants(pid) if p != pid]
        print(f"[census] PID {pid}: alive={pid in alive}  存活后代 {len(kids)} 个: {kids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
