"""从 `[candq]` 日志里回答三个问题（2026-10-31，用户要求）。

输入：`AZAI_CAND_VERBOSE=1` 时桥接 stderr 里的每决策三行：
```
[candq] 阶段1结束 β=3.0 c=1.0 N_total=… 合规类=… | Q/N（按 Q 降序）: c:Q…/N… | …
[candq] topk n=… 构成=dict({...}) 前8=[(bundle),+0.123] …
[candq] chosen (bundle) 类别=… 阶段2=+0.0000 回退=… | 非HOLD最高=+…(类别) (bundle) | HOLD阶段2=… | 非HOLD最高−HOLD=…
```

回答：
1. **预算投给谁**：`N[c]` 的分布（哪些类拿走了预算）、`Q[c]` 最高的类、以及"神手类是否被埋没"
   （= 该类有高阶段 1 分的候选、但 N[c] 很小）。
2. **进 top-k 的候选构成**：升级/建塔/降级/多op/HOLD 各占比；阶段 1 分的分布（中位/最大/最小）。
3. **选 HOLD 的回合里，是否有阶段 2 分更高的非 HOLD 动作**（这是判"评估器 vs 候选"的关键）：
   * 由构造，`chosen` 就是阶段 2 的 argmax ⇒ 选中 HOLD 时它的阶段 2 分**必然** ≥ 非 HOLD；
     关键是**差多少**：`非HOLD最高−HOLD` 的分布（<0 = 评估器确实更爱 HOLD；≈0 = 打平/退化回退；
     >0 = 选了 HOLD 却有更高分的非 HOLD（只可能出现在全等/全NaN 的回退分支里））。
   * 另外统计"该决策里到底有没有非 HOLD 候选"（若没有，选 HOLD 与评估器无关，是候选不足）。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/analyze_candq_log.py \
    --stderr match_results/ai0_az_bridge_ai_seed8_UCV.stderr.log \
    --out match_results/unified_cand/candq_report.txt
```
"""
from __future__ import annotations

import argparse
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RE_Q = re.compile(r"\[candq\] 阶段1结束 β=(?P<beta>[-\d.]+) c=(?P<c>[-\d.]+) "
                  r"N_total=(?P<nt>\d+) 合规类=(?P<nl>\d+) \| Q/N（按 Q 降序）: (?P<qs>.*)$")
RE_ITEM = re.compile(r"(?P<cls>\d+):Q(?P<q>[+-][\d.]+)/N(?P<n>\d+)")
RE_TOPK = re.compile(r"\[candq\] topk n=(?P<n>\d+) 构成=dict\((?P<comp>\{.*?\})\) 前8=(?P<top>.*)$")
RE_CHOSEN = re.compile(
    r"\[candq\] chosen (?P<b>.*?) 类别=(?P<cat>\S+) 阶段2=(?P<s2>[+-]?[\d.nan]+) 回退=(?P<deg>\S+)"
    r" \| 非HOLD最高=(?P<bn>[-+\d.na]+)\((?P<bnc>[^)]*)\) (?P<bnb>\(.*?\)|\(\))"
    r" \| HOLD阶段2=(?P<sh>[-+\d.na]+) \| 非HOLD最高−HOLD=(?P<gap>[-+\d.na]+)")
RE_P1 = re.compile(r"\[(?P<b>\(.*?\)),(?P<s>[+-][\d.]+)\]")


def pct(k: int, n: int) -> str:
    return f"{100.0*k/max(1,n):5.1f}%"


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:+.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stderr", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sp = Path(args.stderr)
    if not sp.is_absolute():
        sp = Path.cwd() / sp
    txt = sp.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()

    decs = []          # 每决策一条
    cur = None
    for line in lines:
        m = RE_Q.search(line)
        if m:
            if cur is not None:
                decs.append(cur)
            qn = [(int(i.group("cls")), float(i.group("q")), int(i.group("n")))
                  for i in RE_ITEM.finditer(m.group("qs"))]
            cur = {"beta": float(m.group("beta")), "c": float(m.group("c")),
                   "N_total": int(m.group("nt")), "n_legal": int(m.group("nl")),
                   "qn": qn, "topk": None, "comp": None, "chosen": None}
            continue
        m = RE_TOPK.search(line)
        if m and cur is not None:
            cur["topk"] = int(m.group("n"))
            cur["comp"] = m.group("comp")
            cur["p1"] = [float(x.group("s")) for x in RE_P1.finditer(m.group("top"))]
            continue
        m = RE_CHOSEN.search(line)
        if m and cur is not None:
            def _num(s: str):
                s = s.strip()
                try:
                    return float(s)
                except ValueError:
                    return float("nan")
            cur["chosen"] = dict(bundle=m.group("b"), cat=m.group("cat"),
                                 s2=_num(m.group("s2")), deg=m.group("deg"),
                                 best_nh=_num(m.group("bn")), best_nh_cat=m.group("bnc"),
                                 best_nh_b=m.group("bnb"), s_hold=_num(m.group("sh")),
                                 gap=_num(m.group("gap")))
            decs.append(cur)
            cur = None
    if cur is not None:
        decs.append(cur)
    if not decs:
        print(f"!! {sp} 里没有 `[candq]` 行（要 AZAI_CAND_VERBOSE=1）", file=sys.stderr)
        return 2

    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit(f"[candq] 决策数 = {len(decs)}  （日志 {sp.name}）")
    emit("")
    emit("## 1. bandit 把预算（N[c]）投给谁 / Q[c] 谁最高")
    Nsum: dict = defaultdict(int)
    Ntot = 0
    qbest: dict = defaultdict(list)
    for d in decs:
        for c, q, n in d["qn"]:
            Nsum[c] += n
            Ntot += n
            qbest[c].append(q)
    top_n = sorted(Nsum.items(), key=lambda kv: -kv[1])
    emit(f"  总 N（ΣN[c]）= {Ntot}；每决策 N_total 中位 = "
         f"{stats.median([d['N_total'] for d in decs]):.0f}")
    emit("  N[c] 占比（前 12 个类）：")
    for c, n in top_n[:12]:
        qs = qbest[c]
        emit(f"    class {c:<3} N={n:<7} ({100.0*n/max(1,Ntot):5.1f}%)  "
             f"Q 中位={stats.median(qs):+.4f}  Q 最大={max(qs):+.4f}")
    nz = [c for c, _ in top_n]
    nl = [d["n_legal"] for d in decs]
    emit(f"  被选中过的类数 = {len(nz)}；每决策**合规类数** n_legal：中位 {stats.median(nl):.0f} "
         f"min {min(nl)} max {max(nl)}（≤24）")
    # 预算是否"塌到少数类"：前 2 类占 N 的比例
    top2 = sum(n for _c, n in top_n[:2])
    emit(f"  **前 2 个类占掉 N 的 {100.0*top2/max(1,Ntot):.1f}%**"
         f"（class {top_n[0][0]} {100.0*top_n[0][1]/max(1,Ntot):.1f}% + "
         f"class {top_n[1][0]} {100.0*top_n[1][1]/max(1,Ntot):.1f}%）" if len(top_n) >= 2 else "")
    # Q 最高 vs N 最少（"神手类被埋没"的候选）
    qmax = {c: max(v) for c, v in qbest.items()}
    qorder = sorted(qmax.items(), key=lambda kv: -kv[1])[:6]
    emit("  Q[c] 最高（每类取历史最大）的前 6： " +
         "  ".join(f"{c}:{q:+.3f}(N={Nsum[c]})" for c, q in qorder))
    emit("")
    emit("## 2. 进阶段 2 的候选构成（top-k）")
    nk = [d["topk"] for d in decs if d["topk"]]
    emit(f"  n_cand：中位 {stats.median(nk):.0f}  min {min(nk)}  max {max(nk)}")
    tag: Counter = Counter()
    n_with = 0
    for d in decs:
        c = d.get("comp")
        if not c:
            continue
        n_with += 1
        for k, v in re.findall(r"'([^']+)': (\d+)", c):
            tag[k] += int(v)
    tot = sum(tag.values())
    emit(f"  含各类标签的候选条数合计 {tot}（一条候选可能多标签）：")
    for k, v in tag.most_common():
        emit(f"    {k:<6} {v:<7} ({100.0*v/max(1,tot):5.1f}%)")
    # 每个决策"有没有 HOLD / 有没有非 HOLD"
    have_hold = sum(1 for d in decs if d.get("comp") and "'HOLD'" in d["comp"])
    have_nonhold = sum(1 for d in decs
                       if d.get("comp") and re.search(r"'(?!HOLD')[^']+': [1-9]", d["comp"]))
    emit(f"  候选里**有 HOLD** 的决策：{have_hold}/{len(decs)} = {pct(have_hold, len(decs))}")
    emit(f"  候选里**有非 HOLD** 的决策：{have_nonhold}/{len(decs)} = {pct(have_nonhold, len(decs))}")
    p1all = [s for d in decs for s in (d.get("p1") or [])]
    if p1all:
        emit(f"  阶段 1 分（每决策前 8 条的汇总）：中位 {stats.median(p1all):+.4f}  "
             f"min {min(p1all):+.4f}  max {max(p1all):+.4f}")
    emit("")
    emit("## 3. 选 HOLD 的回合里，是否存在阶段 2 分更高的非 HOLD 动作？")
    chosen = [d["chosen"] for d in decs if d.get("chosen")]
    n_hold = sum(1 for c in chosen if c["cat"] == "HOLD")
    emit(f"  选中 HOLD 的决策：{n_hold}/{len(chosen)} = {pct(n_hold, len(chosen))}")
    # 分三种：没有非HOLD候选 / 有但评估器更爱HOLD / 有且更高(只可能回退)
    no_nh = gap_neg = gap_pos = nan_gap = 0
    gaps = []
    deg = 0
    for c in chosen:
        if c["cat"] != "HOLD":
            continue
        deg += int(c["deg"].startswith("是"))
        g = c["gap"]
        if c["best_nh_b"] in ("()", "") and c["best_nh"] != c["best_nh"]:
            no_nh += 1
            continue
        if g != g:
            nan_gap += 1
        elif g > 1e-9:
            gap_pos += 1
            gaps.append(g)
        else:
            gap_neg += 1
            gaps.append(g)
    emit(f"    其中：回退分支（全等/全NaN）= {deg}")
    emit(f"          候选里**没有非 HOLD**（只能选 HOLD）= {no_nh}")
    emit(f"          有非 HOLD 但评估器**更爱 HOLD**（gap ≤ 0）= {gap_neg}")
    emit(f"          有非 HOLD 且**分数更高却仍选了 HOLD**（gap > 0，只可能回退）= {gap_pos}")
    if gaps:
        emit(f"    gap（非HOLD最高 − HOLD）分布：中位 {_f(stats.median(gaps))}  "
             f"min {_f(min(gaps))}  max {_f(max(gaps))}")
        buckets = Counter()
        for g in gaps:
            if g <= -0.05:
                buckets["≤ -0.05（明显更爱 HOLD）"] += 1
            elif g < -0.01:
                buckets["(-0.05, -0.01)"] += 1
            elif g < 0:
                buckets["(-0.01, 0)（几乎打平）"] += 1
            elif g <= 0.01:
                buckets["(0, 0.01]（几乎打平）"] += 1
            else:
                buckets["> 0.01（评估器漏了好动作）"] += 1
        for k, v in buckets.most_common():
            emit(f"      {k:<34} {v:>4}  {pct(v, len(gaps))}")
    emit("")
    emit("  选中项的类别分布： " +
         "  ".join(f"{k}:{v}" for k, v in Counter(c["cat"] for c in chosen).most_common()))
    s2 = [c["s2"] for c in chosen if c["s2"] == c["s2"]]
    if s2:
        emit(f"  选中项阶段 2 分：中位 {stats.median(s2):+.4f}  min {min(s2):+.4f}  max {max(s2):+.4f}")

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[candq] 已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
