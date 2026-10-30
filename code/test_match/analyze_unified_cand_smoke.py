"""统计 `AZAI_CAND_MODE=unified` 冒烟局的**每决策读数**（2026-10-29，修正后格式）。

输入 = 桥接的 stderr 日志（`AZAI_CAND_VERBOSE=1` 时每决策一行 `[cand] mode=unified ...`）。
输出 = 报告要的那几张表：
  * 每决策 `n_cand`（去重后）分布：min/p25/中位/p75/p95/max + 直方图
  * 候选**含 HOLD** / **含升级(op 12)** 的回合占比；升级被**选中**的次数
  * **同 op 类型多 op**（`[build,build]` / `[downgrade,downgrade]`）的决策占比
  * 缓存命中 / 位置推进步数 / 形状数（"7 条 / 426 次重复"式塌缩是否还在）
  * 决策耗时 p50/p95/max（总 / 阶段 1）
  * 对局级：`RESULT`、`illegal-summary`、我方实际发出的 op 计数（含"同类型多 op"的回合）

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/analyze_unified_cand_smoke.py \
    --stderr match_results/ai0_az_bridge_ai_seed8_UC.stderr.log \
    --stdout match_results/unified_cand/UNIFIED_CAND_FIX_seed8.log
```
兼容修正前后的两种 `[cand]` 行（`重复=` 是旧字段名，`缓存命中=` 是新字段名）。
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

RE_HEAD = re.compile(r"\[cand\] mode=unified n_cand=(?P<n>-?\d+) 含HOLD=(?P<h>\d+) "
                     r"含升级=(?P<u>\d+)")
RE_SIZES = re.compile(r"长度分布=(?P<d>\{[^}]*\})")
RE_ROUNDOP = re.compile(r"^\s+\[round (\d+)\] p(\d)\(([^)]*)\) 原文='(.*)' -> (.*)$")
RE_OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def _field(line: str, *names: str):
    for name in names:
        m = re.search(re.escape(name) + r"=([^\s]+)", line)
        if m:
            return m.group(1)
    return None


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
    for line in txt.splitlines():
        m = RE_HEAD.search(line)
        if not m:
            continue
        d = {"n": int(m.group("n")), "h": int(m.group("h")), "u": int(m.group("u")),
             "pool": _field(line, "池"), "hit": _field(line, "缓存命中", "重复"),
             "steps": _field(line, "步数"), "shapes": _field(line, "形状"),
             "rep": _field(line, "重复类形状"), "multi": _field(line, "同类型多op"),
             "p1": _field(line, "p1_ms"), "ms": _field(line, "决策_ms"),
             "pick": _field(line, "选中")}
        ms_e = RE_SIZES.search(line)
        d["sizes"] = ms_e.group("d") if ms_e else None
        rows.append(d)
    if not rows:
        print(f"!! {sp} 里没有 `[cand] mode=unified` 行（是不是没开 AZAI_CAND_VERBOSE=1？）",
              file=sys.stderr)
        return 2

    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    n = [r["n"] for r in rows]
    has_hold = [r["h"] for r in rows]
    has_upg = [r["u"] for r in rows]
    ms = [float(r["ms"]) for r in rows if r["ms"]]
    p1s = [float(r["p1"]) for r in rows if r["p1"]]
    hits = [int(r["hit"]) for r in rows if r["hit"]]
    steps = [int(r["steps"]) for r in rows if r["steps"]]
    shapes = [int(r["shapes"].split("(")[0]) for r in rows if r["shapes"]]
    multi = [int(r["multi"]) for r in rows if r["multi"]]
    picks = [r["pick"] or "?" for r in rows]
    lengths: dict = {}
    n_len_total = 0
    for r in rows:
        if r["sizes"]:
            d = ast.literal_eval(r["sizes"])
            for k, v in d.items():
                lengths[int(k)] = lengths.get(int(k), 0) + int(v)
                n_len_total += int(v)

    emit(f"[stat] 决策数 = {len(rows)}（每条 = 一次 `mcts.search`）")
    emit("")
    emit("## 每决策 n_cand（去重后）分布")
    emit(f"  min={min(n)}  p25={pct(n,0.25):.0f}  中位={pct(n,0.5):.0f}  "
         f"p75={pct(n,0.75):.0f}  p95={pct(n,0.95):.0f}  max={max(n)}  "
         f"均值={sum(n)/len(n):.1f}")
    c = Counter(n)
    emit("  直方图（n_cand: 次数 / 占比）:")
    for k in sorted(c):
        emit(f"    n_cand={k:<4} {c[k]:>4}  {100.0*c[k]/len(n):5.1f}%")
    emit(f"  n_cand==1 的占比 = {100.0*c.get(1,0)/len(n):.1f}%"
         f"（对照：现役 pin 口径 82.3%）")
    emit(f"  n_cand>=24 的占比 = {100.0*sum(v for k,v in c.items() if k>=24)/len(n):.1f}%")
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
            f"{k}op {lengths.get(k,0)} ({100.0*lengths.get(k,0)/n_len_total:.1f}%)"
            for k in sorted(lengths)))
    emit("")
    emit("## 同 op 类型多 op（`[build,build]` / `[downgrade,downgrade]` 的证据）")
    if multi:
        n_dec = sum(1 for x in multi if x > 0)
        emit(f"  含\"同类型多 op\"候选的回合占比 = {100.0*n_dec/len(multi):.1f}%"
             f"（{n_dec}/{len(multi)}）")
        emit(f"  每决策条数：中位={pct(multi,0.5):.0f} p95={pct(multi,0.95):.0f} max={max(multi)}"
             f" 合计={sum(multi)}")
    emit("")
    emit("## 缓存 / 游标推进（修正 ④ 的读数）")
    if hits:
        emit(f"  缓存命中：中位={pct(hits,0.5):.0f} p95={pct(hits,0.95):.0f} max={max(hits)} "
             f"合计={sum(hits)}")
    if steps:
        emit(f"  位置推进步数：中位={pct(steps,0.5):.0f} p95={pct(steps,0.95):.0f} max={max(steps)}")
    if shapes:
        emit(f"  形状数：中位={pct(shapes,0.5):.0f} max={max(shapes)}")
    if hits:
        bad = [r for r in rows if r["pool"] and int(r["pool"]) <= 7
               and r["hit"] and int(r["hit"]) >= 400]
        emit(f"  \"池≤7 且缓存命中≥400\"（= 上一轮 7 条/426 次重复式塌缩）的决策数 = {len(bad)}")
    emit("")
    emit("## 选中（argmax after phase-2 rollout）")
    for k, v in Counter(picks).most_common():
        emit(f"  选中={k:<6} {v:>4} 次  {100.0*v/len(rows):5.1f}%")
    emit("")
    emit("## 决策耗时（ms）")
    if ms:
        emit(f"  总耗时：p50={pct(ms,0.5):.0f}  p95={pct(ms,0.95):.0f}  max={max(ms):.0f}  "
             f"mean={sum(ms)/len(ms):.0f}  n={len(ms)}")
    if p1s:
        emit(f"  阶段 1（候选生成 + 确定性 idle rollout）：p50={pct(p1s,0.5):.1f}  "
             f"p95={pct(p1s,0.95):.1f}  max={max(p1s):.1f}  mean={sum(p1s)/len(p1s):.1f} ms")
        if ms:
            emit(f"  阶段 1 占整决策（按均值）：{100.0*(sum(p1s)/len(p1s))/(sum(ms)/len(ms)):.1f}%")

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
                emit(f"  {line.strip()[:420]}")
        cnt: dict = {}
        same_rounds: list = []
        for line in g.splitlines():
            m = RE_ROUNDOP.match(line)
            if not m:
                continue
            label = m.group(3)
            if "az_bridge" not in label and "preunifiedcand" not in label:
                continue
            ops = [(int(o.group(2)), int(o.group(3)), int(o.group(4)))
                   for o in RE_OP.finditer(m.group(5))]
            for (t, _a0, _a1) in ops:
                cnt[t] = cnt.get(t, 0) + 1
            ts = [t for (t, _a, _b) in ops]
            if len(ts) >= 2 and len(set(ts)) < len(ts):
                same_rounds.append((int(m.group(1)), ops))
        names = {11: "BUILD", 12: "UPGRADE", 13: "DOWNGRADE", 21: "LIGHTNING",
                 22: "EMP", 23: "DEFLECTOR", 24: "EVASION", 31: "UP_SPEED", 32: "UP_ANT"}
        emit(f"  我方实际发出的 op 计数（按 op_type）：{dict(sorted(cnt.items()))}")
        emit(f"  我方实际发出的 op 计数（按名字）："
             f"{ {names.get(k, k): v for k, v in sorted(cnt.items())} }")
        emit(f"  **真的发出去了**\"同 op 类型多 op\"（如两手建塔/两手降级）的回合数 = "
             f"{len(same_rounds)}")
        for (rnd, ops) in same_rounds[:6]:
            emit(f"    例 round={rnd}: {ops}")

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
