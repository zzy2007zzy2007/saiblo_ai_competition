"""判断两个三网 ckpt 是否**逐张量相同**（用于"训练器早停回 epoch -1 ⇒ 产物与对照逐位相同"的自动识别）。

为什么要它：`train_value_net.py` 自带"按 val MSE 取最佳"的早停；在旧数据/新数据上都出现过
**最佳 = epoch -1（初始权重）** ⇒ 产物与对照 bit-identical ⇒ 那条臂的 32 局读数会是**逐字节复现**，
按构造没有信息（V1/V2 都发生过）。这个脚本让"要不要跑那一批对局"可以机械判断。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/same_ckpt.py --a A.pt --b B.pt
输出 `IDENTICAL` / `DIFFER <n 个 state 不同>`；退出码 0=相同、1=不同。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()

    if not Path(a.a).is_file() or not Path(a.b).is_file():
        print(f"文件缺失: {a.a} / {a.b}")
        return 2
    da = torch.load(a.a, map_location="cpu")
    db = torch.load(a.b, map_location="cpu")
    diff_states = []
    for st in ("class_state", "pos_state", "value_state"):
        x, y = da.get(st), db.get(st)
        if x is None or y is None:
            diff_states.append(f"{st}(缺)")
            continue
        if set(x.keys()) != set(y.keys()):
            diff_states.append(f"{st}(键不同)")
            continue
        n_bad = sum(0 if torch.equal(x[k], y[k]) else 1 for k in x)
        if n_bad:
            diff_states.append(f"{st}({n_bad}/{len(x)})")
    if diff_states:
        print("DIFFER " + " ".join(diff_states))
        return 1
    print("IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
