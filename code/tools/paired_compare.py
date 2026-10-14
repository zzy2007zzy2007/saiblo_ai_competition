"""配对检验：**同一批种子**下比较两条臂（逐局 McNemar 精确检验 + 逐对分数差的配对 bootstrap）。

为什么需要：`analyze_paired.py` 给的是**独立**的 cluster-bootstrap CI（±11pp ✗），
而两臂跑的是**同一批种子/同一对手** ⇒ **配对**能消掉地图与对手的方差 ⇒ CI 通常显著更窄 ✓。
解析**复用** `_tmp_ladder.read_game`（与分析器同源 ✓，绝不重写 ✗）。

用法：
  python -u code/tools/paired_compare.py --a=AB_MC1_dec_vs_baseline \
      --b=AB_MENU12_vs_baseline_vs_baseline --seeds=7-70
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "match_results" / "ladder_logs"


def load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_for_paired", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def won(ladder, tag: str, token: str):
    """我方是否赢；无效/缺失返回 None（按分析器口径只判 winner_side ✓）。"""
    p = LOG_ROOT / tag / f"{token}.log"
    if not p.is_file():
        return None
    g = ladder.read_game(p, token)
    m = re.search(r"winner_side=(\S+)", p.read_text(encoding="utf-8", errors="replace"))
    ws = m.group(1) if m else None
    if ws not in ("p0", "p1"):
        return None
    us_is_p0 = not token.endswith("r")
    return ws == ("p0" if us_is_p0 else "p1")


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线臂 tag（如 incumbent）")
    ap.add_argument("--b", required=True, help="候选臂 tag")
    ap.add_argument("--seeds", default="7-70")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20261005)
    args = ap.parse_args()

    ladder = load_ladder()
    seeds = parse_seeds(args.seeds)
    tokens = [f"{s}{suf}" for s in seeds for suf in ("", "r")]

    A, B = {}, {}
    for t in tokens:
        A[t] = won(ladder, args.a, t)
        B[t] = won(ladder, args.b, t)

    both = [t for t in tokens if A[t] is not None and B[t] is not None]
    na = sum(1 for v in A.values() if v is not None)
    nb = sum(1 for v in B.values() if v is not None)
    print(f"[paired] A={args.a}（有效 {na}/128，胜率 {sum(1 for v in A.values() if v) / max(na,1):.4f}）")
    print(f"[paired] B={args.b}（有效 {nb}/128，胜率 {sum(1 for v in B.values() if v) / max(nb,1):.4f}）")
    print(f"[paired] 两臂都有效的**同一批局** = {len(both)}/128 ✓（配对集 ✓）")
    if not both:
        print("[paired] 没有可配对的对局 ✗")
        return 1

    b_ = sum(1 for t in both if A[t] and not B[t])   # A 赢 B 输
    c_ = sum(1 for t in both if B[t] and not A[t])   # B 赢 A 输
    p = mcnemar_exact(b_, c_)
    print(f"\n  逐局 McNemar（{len(both)} 局配对）：A 赢B 输 = {b_}；B 赢A 输 = {c_}；"
          f"一致 = {len(both) - b_ - c_}")
    print(f"  ⇒ 精确二项检验 p = {p:.4f}  " + ("**显著（p<0.05）** ✓" if p < 0.05 else "不显著 ✗（±1 局的量级都算不上 ✓）"))
    print(f"  逐局胜率差（B − A）= {(sum(1 for t in both if B[t]) - sum(1 for t in both if A[t])) / len(both) * 100:+.2f}pp")

    # 逐对（每种子两局）分数：0 / 0.5 / 1 ⇒ 差的配对 bootstrap ✓
    diffs = []
    for s in seeds:
        pair = [f"{s}", f"{s}r"]
        if all(A[t] is not None and B[t] is not None for t in pair):
            sa = sum(1 for t in pair if A[t]) / 2.0
            sb = sum(1 for t in pair if B[t]) / 2.0
            diffs.append(sb - sa)
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(args.boot_seed)
    boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(args.iters)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  逐对分数差（B − A，{len(diffs)} 对完整配对）：均值 **{diffs.mean() * 100:+.2f}pp**，"
          f"配对 bootstrap 95% CI **[{lo * 100:+.2f}, {hi * 100:+.2f}]pp**")
    print("  " + ("CI 不含 0 ⇒ **配对下 B 优于 A** ✓" if lo > 0 or hi < 0 else "CI 含 0 ⇒ 配对下仍**不可分辨** ✗"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
