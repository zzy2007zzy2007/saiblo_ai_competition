# -*- coding: utf-8 -*-
"""升级目标回合的 s1/s2 排名：升级候选 vs 最佳非升级候选（HOLD/建塔/闪电）。"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
OP_NAMES = {11: "建塔", 12: "升级", 13: "降级", 21: "闪电", 22: "EMP",
            23: "偏转", 24: "闪避", 31: "升速", 32: "升蚁"}
UPG = 12
LIGHT = 21


def cat(bundle):
    ts = sorted({int(o[0]) for o in (bundle or ())})
    return "HOLD" if not ts else "+".join(OP_NAMES.get(t, f"op{t}") for t in ts)


def has_op(bundle, op):
    return any(int(o[0]) == op for o in (bundle or ()))


def rank_of(scores, idx):
    s = scores[idx]
    return 1 + sum(1 for x in scores if x > s)


data = json.loads((ROOT / "code" / "test_match" / "_target_rounds.json").read_text(encoding="utf-8"))
recs = data["upg_targets"]
print(f"升级目标回合共 {len(recs)} 个")

chosen_s1_ranks = Counter()
chosen_s2_ranks = Counter()
best_nonupg_cat = Counter()
s2_gaps = []
s1_gaps = []
gold = []
best_nonupg_is_hold = 0
best_nonupg_is_light = 0
n_degen = 0
for r in recs:
    gold.append(r["coins"])
    cand = r["cand"]; s1 = r["s1"]; s2 = r["s2"]
    ci = r["chosen_idx"]
    chosen_s1_ranks[rank_of(s1, ci)] += 1
    chosen_s2_ranks[rank_of(s2, ci)] += 1
    # 非升级候选：bundle 不含 op12
    nonupg = [i for i, b in enumerate(cand) if not has_op(b, UPG)]
    if not nonupg:
        continue
    bi = max(nonupg, key=lambda i: s2[i])
    best_nonupg_cat[cat(cand[bi])] += 1
    s2_gaps.append(s2[ci] - s2[bi])
    s1_gaps.append(s1[ci] - s1[bi])
    if len(cand[bi]) == 0:
        best_nonupg_is_hold += 1
    if has_op(cand[bi], LIGHT):
        best_nonupg_is_light += 1
    if max(s2) - min(s2) < 1e-9:
        n_degen += 1

q = statistics.quantiles(gold, n=4)
print(f"gold 分布（升级回合）: min={min(gold)} p25={q[0]:.0f} 中位={statistics.median(gold):.0f} "
      f"p75={q[2]:.0f} max={max(gold)} | 金币≥90 的升级回合={sum(1 for g in gold if g >= 90)}")
print(f"退化（s2 全等）回合数={n_degen}")
print("\nchosen(升级) 的 s1 排名分布:")
for k in sorted(chosen_s1_ranks):
    print(f"  第{k}名: {chosen_s1_ranks[k]}")
print("\nchosen(升级) 的 s2 排名分布:")
for k in sorted(chosen_s2_ranks):
    print(f"  第{k}名: {chosen_s2_ranks[k]}")
print(f"\n最佳非升级候选的类别分布（按 s2 最高）:")
for k, v in best_nonupg_cat.most_common():
    print(f"  {k}: {v}  ({100.0*v/len(recs):.1f}%)")
print(f"  其中最佳非升级=HOLD 的回合: {best_nonupg_is_hold}  ({100.0*best_nonupg_is_hold/len(recs):.1f}%)")
print(f"  其中最佳非升级=闪电 的回合: {best_nonupg_is_light}")
if s2_gaps:
    print(f"\ns2 差距 (升级_chosen - 最佳非升级): 均值={statistics.mean(s2_gaps):+.4f} "
          f"中位={statistics.median(s2_gaps):+.4f} min={min(s2_gaps):+.4f} max={max(s2_gaps):+.4f}")
if s1_gaps:
    print(f"s1 差距 (升级_chosen - 最佳非升级): 均值={statistics.mean(s1_gaps):+.4f} "
          f"中位={statistics.median(s1_gaps):+.4f}")
