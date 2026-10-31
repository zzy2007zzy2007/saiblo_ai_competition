"""分析"现役 × 金币门控"侦察 A/B：胜负/血差 + **机制读数**（我们这侧每局的 op 构成）。

数据源：
  · 对局结果：`match_results/explore_20261102/coin_gate/<TAG>/<arm>/<token>.log` 里的 `RESULT` 行；
  · 对照臂：`match_results/value_ab_rv4/AB_RV4_FULL/incumbent/<token>.log`（同 seeds、同 BASE）；
  · 机制（每局 op 构成）：`training_history/progress/match_*/az_progress_seed<token><arm>_<token>.txt`
    里的逐回合 `P0=[OP(...)] P1=[...]`（token 不带 r ⇒ 我们执 P0；`r` 结尾 ⇒ 我们执 P1）。

用法：
  python code/test_match/analyze_coin_gate_ab.py
  python code/test_match/analyze_coin_gate_ab.py --tag EXPLORE_COIN_GATE
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COIN_GATE = "match_results/explore_20261102/coin_gate"
BASELINE = "match_results/value_ab_rv4/AB_RV4_FULL/incumbent"
PROGRESS = "training_history/progress"

RE_RESULT = re.compile(r"RESULT seed=(\d+) .*?winner=(\S+) .*?base_hp=(\d+),(\d+)")
RE_PROG = re.compile(r"round=(\d+) HP0=(\d+) HP1=(\d+) G0=(\d+) G1=(\d+) P0=\[(.*?)\]\s+P1=\[(.*?)\]")
RE_OP = re.compile(r"([A-Z_]+)\(")


def parse_result(log_path):
    """返回 (our_hp, opp_hp) 或 None。我们这侧由 token 决定（见 main）。"""
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    for line in txt.splitlines():
        m = RE_RESULT.search(line)
        if m:
            return dict(seed=int(m.group(1)), winner=m.group(2),
                        hp=(int(m.group(3)), int(m.group(4))),
                        line=line.strip())
    return None


def our_side_of(token: str) -> int:
    return 1 if token.endswith("r") else 0


def margin_of(res, token):
    p = our_side_of(token)
    hp = res["hp"]
    return hp[p] - hp[1 - p]


def op_profile(token, arm):
    """扫 progress 文件，统计我们这侧每局的 op 构成。"""
    p = our_side_of(token)
    pat = os.path.join(PROGRESS, "match_*", f"az_progress_seed{token}{arm}_{token}.txt")
    files = glob.glob(pat)
    if not files:
        # 兜底：按后缀模糊匹配
        files = [f for f in glob.glob(os.path.join(PROGRESS, "match_*", "*.txt"))
                 if f.endswith(f"_{token}.txt") and f"cg_{arm}_" in f]
    if not files:
        return None
    cnt = Counter()
    n_rounds = 0
    hp_first_bleed = None
    for line in open(files[0], encoding="utf-8", errors="replace"):
        m = RE_PROG.search(line)
        if not m:
            continue
        n_rounds = int(m.group(1))
        hp = [int(m.group(2)), int(m.group(3))]
        if hp_first_bleed is None and hp[p] < 50:
            hp_first_bleed = n_rounds
        ops = m.group(6) if p == 0 else m.group(7)
        for name in RE_OP.findall(ops):
            cnt[name] += 1
    return dict(rounds=n_rounds, n_ops=sum(cnt.values()), ops=dict(cnt),
                first_bleed=hp_first_bleed)


def report(arm_dir, arm_name, tokens, baseline=False):
    rows = []
    for tok in tokens:
        log = os.path.join(arm_dir, f"{tok}.log")
        if not os.path.exists(log):
            continue
        res = parse_result(log)
        if not res:
            continue
        rows.append(dict(token=tok, margin=margin_of(res, tok), res=res,
                         prof=op_profile(tok, arm_name) if not baseline else None))
    return rows


def report_multi(tags, arm, tokens):
    """把多个 TAG 目录里的同名 arm 合起来（分批跑的时候用）。"""
    out = {}
    for tag in tags:
        for r in report(os.path.join(COIN_GATE, tag, arm), arm, tokens):
            out[r["token"]] = r
    return list(out.values())


def summarize(tag, arms, tokens):
    tags = tag.split(",")
    base_rows = {r["token"]: r for r in report(BASELINE, "incumbent", tokens, baseline=True)}
    print(f"# 金币门控 A/B（TAG={tag}）  基线 = {BASELINE}")
    print(f"{'arm':6s} {'n':>2s} {'W-L':>5s} {'margin':>8s} {'闪电/局':>7s} {'建塔/局':>7s} "
          f"{'升级/局':>7s} {'降级/局':>7s} {'op/局':>6s} {'首掉血':>6s}")
    if base_rows:
        rws = list(base_rows.values())
        print(f"{'inc':6s} {len(rws):2d} "
              f"{sum(1 for r in rws if r['margin']>0)}-{sum(1 for r in rws if r['margin']<=0)} "
              f"{statistics.mean(r['margin'] for r in rws):+8.2f}")
    for arm in arms:
        rws = report_multi(tags, arm, tokens)
        if not rws:
            print(f"{arm:6s}  0  （还没有完赛的对局）")
            continue
        w = sum(1 for r in rws if r["margin"] > 0)
        profs = [r["prof"] for r in rws if r["prof"]]
        def avg(k):
            return statistics.mean(p["ops"].get(k, 0) for p in profs) if profs else float("nan")
        print(f"{arm:6s} {len(rws):2d} {w}-{len(rws)-w} "
              f"{statistics.mean(r['margin'] for r in rws):+8.2f} "
              f"{avg('LIGHTNING'):7.2f} {avg('BUILD'):7.1f} "
              f"{avg('UPGRADE'):7.1f} {avg('DOWNGRADE'):7.1f} "
              f"{(statistics.mean(p['n_ops'] for p in profs) if profs else float('nan')):6.1f} "
              f"{(statistics.mean(p['first_bleed'] for p in profs if p['first_bleed']) if any(p['first_bleed'] for p in profs) else float('nan')):6.1f}")
    # 配对对比
    print("\n## 配对（同 token）")
    for arm in arms:
        rws = report_multi(tags, arm, tokens)
        if not rws:
            continue
        deltas = [r["margin"] - base_rows[r["token"]]["margin"]
                  for r in rws if r["token"] in base_rows]
        if deltas:
            print(f"- {arm} − incumbent：Δhp = {statistics.mean(deltas):+.2f} "
                  f"(n={len(deltas)}；逐局 {[round(d,1) for d in deltas]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="EXPLORE_COIN_GATE")
    ap.add_argument("--arms", default="g60 g90")
    ap.add_argument("--tokens", default="100 100r 101 101r 102 102r 103 103r")
    a = ap.parse_args()
    summarize(a.tag, a.arms.split(), a.tokens.split())


if __name__ == "__main__":
    main()
