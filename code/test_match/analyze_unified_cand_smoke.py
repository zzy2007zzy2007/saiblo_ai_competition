"""统计 `AZAI_CAND_MODE=unified` 冒烟局的**每决策读数**（2026-10-29）。

输入 = 桥接的 stderr 日志（`AZAI_CAND_VERBOSE=1` 时每决策一行 `[cand] mode=unified ...`）。
输出 = 报告 §4 要的那几张表：
  * 每决策 `n_cand`（去重后）分布：min/p25/中位/p75/max + 直方图
  * 候选**含 HOLD** 的回合占比、**含升级(op 12)** 的回合占比
  * 升级被**选中**的次数（来自每决策的 `选中=` 字段 + 真实对局 `--trace-ops` 的 op 计数）
  * 决策耗时 p50/p95/max
  * 候选长度分布（0/1/2/3 op）与阶段 1 耗时

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/analyze_unified_cand_smoke.py \
    --stderr match_results/ai0_az_bridge_ai_seed8_UC.stderr.log \
    --stdout match_results/unified_cand/UNIFIED_CAND_FULL_seed8.log
```
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

RE_CAND = re.compile(
    r"\[cand\] mode=unified n_cand=(?P<n>-?\d+) 含HOLD=(?P<h>\d+) 含升级=(?P<u>\d+)"
    r"(?: 池=(?P<pool>\d+) 重复=(?P<dup>\d+) 长度分布=(?P<sizes>\{[^}]*\})"
    r" p1_ms=(?P<p1>[\d.]+) 决策_ms=(?P<ms>[\d.]+) 选中=(?P<pick>\S+))?")
RE_ROUNDOP = re.compile(r"^\s+\[round (\d+)\] p(\d)\(([^)]*)\) 原文='(.*)' -> (.*)$")


def pct(vals: list, q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return float(xs[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stderr", required=True)
    ap.add_argument("--stdout", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sp = Path(args.stderr)
    if not sp.is_absolute():
        sp = Path.cwd() / sp
    txt = sp.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in RE_CAND.finditer(txt):
        d = m.groupdict()
        rows.append(d)
    if not rows:
        print(f"!! {sp} 里没有 `[cand] mode=unified` 行（是不是没开 AZAI_CAND_VERBOSE=1？）",
              file=sys.stderr)
        return 2

    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    n_cand = [int(r["n"]) for r in rows]
    has_hold = [int(r["h"]) for r in rows]
    has_upg = [int(r["u"]) for r in rows]
    ms = [float(r["ms"]) for r in rows if r.get("ms")]
    p1s = [float(r["p1"]) for r in rows if r.get("p1")]
    picks = [r.get("pick", "?") for r in rows]
    lengths = {0: 0, 1: 0, 2: 0, 3: 0}
    n_len_total = 0
    for r in rows:
        if r.get("sizes"):
            d = ast.literal_eval(r["sizes"])
            for k, v in d.items():
                lengths[int(k)] = lengths.get(int(k), 0) + int(v)
                n_len_total += int(v)

    emit(f"[stat] 决策数 = {len(rows)}（每条 = 一次 `mcts.search`）")
    emit("")
    emit("## 每决策 n_cand（去重后）分布")
    emit(f"  min={min(n_cand)}  p25={pct(n_cand,0.25):.0f}  中位={pct(n_cand,0.5):.0f}  "
         f"p75={pct(n_cand,0.75):.0f}  p95={pct(n_cand,0.95):.0f}  max={max(n_cand)}  "
         f"均值={sum(n_cand)/len(n_cand):.1f}")
    from collections import Counter
    c = Counter(n_cand)
    emit("  直方图（n_cand: 次数 / 占比）:")
    for k in sorted(c):
        emit(f"    n_cand={k:<4} {c[k]:>4}  {100.0*c[k]/len(n_cand):5.1f}%")
    emit(f"  n_cand==1 的占比 = {100.0*c.get(1,0)/len(n_cand):.1f}%"
         f"（对照：现役 pin 口径 82.3%）")
    emit(f"  n_cand>=24 的占比 = {100.0*sum(v for k,v in c.items() if k>=24)/len(n_cand):.1f}%")
    emit("")
    emit("## 候选构成（每决策）")
    emit(f"  含 HOLD 的回合占比 = {100.0*sum(1 for x in has_hold if x)/len(rows):.1f}%"
         f"（{sum(1 for x in has_hold if x)}/{len(rows)}）")
    emit(f"  含升级(op12) 的回合占比 = {100.0*sum(1 for x in has_upg if x)/len(rows):.1f}%"
         f"（{sum(1 for x in has_upg if x)}/{len(rows)}）")
    emit(f"  每决策含升级条数：min={min(has_upg)} 中位={pct(has_upg,0.5):.0f} max={max(has_upg)}"
         f" 合计={sum(has_upg)}")
    if n_len_total:
        emit("  候选长度分布（全部候选合计）: " + "  ".join(
            f"{k}op {lengths.get(k,0)} ({100.0*lengths.get(k,0)/n_len_total:.1f}%)" for k in (0, 1, 2, 3)))
    emit("")
    emit("## 选中（argmax after phase-2 rollout）")
    pc = Counter(picks)
    for k, v in pc.most_common():
        emit(f"  选中={k:<6} {v:>4} 次  {100.0*v/len(rows):5.1f}%")
    emit("")
    emit("## 决策耗时（ms）")
    if ms:
        emit(f"  p50={pct(ms,0.5):.0f}  p95={pct(ms,0.95):.0f}  max={max(ms):.0f}  "
             f"mean={sum(ms)/len(ms):.0f}  n={len(ms)}")
    if p1s:
        emit(f"  阶段 1（候选生成 + 确定性 idle rollout）耗时：p50={pct(p1s,0.5):.1f}  "
             f"p95={pct(p1s,0.95):.1f}  max={max(p1s):.1f}  mean={sum(p1s)/len(p1s):.1f} ms")
        emit(f"  阶段 1 占整决策：{100.0*(sum(p1s)/len(p1s))/(sum(ms)/len(ms)):.1f}%")

    if args.stdout:
        fp = Path(args.stdout)
        if not fp.is_absolute():
            fp = Path.cwd() / fp
        g = fp.read_text(encoding="utf-8", errors="replace")
        emit("")
        emit("## 对局级读数（来自 match_via_sdk 的 stdout 日志）")
        for line in g.splitlines():
            if line.startswith("  RESULT"):
                emit(f"  {line.strip()}")
            if "illegal-summary" in line:
                emit(f"  {line.strip()[:400]}")
        # 我方真实发出的 op 计数（按 label 区分）
        cnt: dict = {}
        for line in g.splitlines():
            m = RE_ROUNDOP.match(line)
            if not m:
                continue
            label = m.group(3)
            if "az_bridge" not in label and "preunifiedcand" not in label:
                continue
            ops = re.findall(r"Operation\(op_type=<OperationType\.(\w+):", m.group(5))
            for op in ops:
                cnt[op] = cnt.get(op, 0) + 1
        emit(f"  我方实际发出的 op 计数：{dict(sorted(cnt.items(), key=lambda kv: -kv[1]))}")

    if args.out:
        op = Path(args.out)
        if not op.is_absolute():
            op = Path.cwd() / op
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[stat] 已写入 {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
