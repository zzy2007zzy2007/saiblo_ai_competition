# -*- coding: utf-8 -*-
"""第 2 步细化：对目标回合扒 s1/s2 排名（读 _target_rounds.json，不重扫原始数据）。"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
OP_NAMES = {11: "建塔", 12: "升级", 13: "降级", 21: "闪电", 22: "EMP",
            23: "偏转", 24: "闪避", 31: "升速", 32: "升蚁"}
LIGHT = 21


def cat(bundle):
    ts = sorted({int(o[0]) for o in (bundle or ())})
    return "HOLD" if not ts else "+".join(OP_NAMES.get(t, f"op{t}") for t in ts)


def rank_of(scores, idx):
    """1-based rank of idx in scores (descending, stable)。"""
    s = scores[idx]
    return 1 + sum(1 for x in scores if x > s)


def analyze(recs, kind):
    print(f"\n{'='*70}\n[{kind}] 共 {len(recs)} 个目标回合\n{'='*70}")
    if not recs:
        return
    no_cand = [r for r in recs if not r["light_idxs"]]
    in_cand = [r for r in recs if r["light_idxs"]]
    print(f"闪电没进候选（阶段1 就没进 top-k）: {len(no_cand)}  ({100.0*len(no_cand)/len(recs):.1f}%)")
    print(f"闪电进了候选但没被选:             {len(in_cand)}  ({100.0*len(in_cand)/len(recs):.1f}%)")

    # 对 in_cand：闪电 s2 排名分布、s2 差距、s1 排名
    s2_ranks = Counter()
    s1_ranks = Counter()
    gaps = []
    s1_gaps = []
    chosen_cats = Counter()
    light_s1_ranks_of_s2rank1 = []
    for r in in_cand:
        s2 = r["s2"]; s1 = r["s1"]
        # 闪电候选里 s2 最高的那条
        best_li = max(r["light_idxs"], key=lambda i: s2[i])
        s2r = rank_of(s2, best_li)
        s1r = rank_of(s1, best_li)
        s2_ranks[s2r] += 1
        s1_ranks[s1r] += 1
        gaps.append(s2[r["chosen_idx"]] - s2[best_li])
        s1_gaps.append(s1[r["chosen_idx"]] - s1[best_li])
        chosen_cats[cat(r["chosen"])] += 1
        if s2r == 1:
            light_s1_ranks_of_s2rank1.append(s1r)
    print(f"\n闪电候选的 s2 排名分布（1=本应被选）:")
    for k in sorted(s2_ranks):
        print(f"  第{k}名: {s2_ranks[k]}  ({100.0*s2_ranks[k]/len(in_cand):.1f}%)")
    print(f"\n闪电候选的 s1 排名分布:")
    for k in sorted(s1_ranks):
        print(f"  第{k}名: {s1_ranks[k]}  ({100.0*s1_ranks[k]/len(in_cand):.1f}%)")
    if gaps:
        print(f"\ns2 差距 (chosen_s2 - 闪电_s2): 均值={statistics.mean(gaps):+.4f} "
              f"中位={statistics.median(gaps):+.4f} min={min(gaps):+.4f} max={max(gaps):+.4f}")
    if s1_gaps:
        print(f"s1 差距 (chosen_s1 - 闪电_s1): 均值={statistics.mean(s1_gaps):+.4f} "
              f"中位={statistics.median(s1_gaps):+.4f}")
    print(f"\n被选中的类别分布:")
    for k, v in chosen_cats.most_common():
        print(f"  {k}: {v}  ({100.0*v/len(in_cand):.1f}%)")

    # 没进候选的回合：被选中的是什么
    if no_cand:
        nc_cats = Counter(cat(r["chosen"]) for r in no_cand)
        print(f"\n[闪电没进候选] 被选中的类别分布:")
        for k, v in nc_cats.most_common():
            print(f"  {k}: {v}")

    # 几个典型样例（闪电 in cand，s2 排名靠前但没被选）
    print(f"\n样例（闪电在候选里、s2 排名前 3 但没被选，按 gap 升序取 6 条）:")
    examples = []
    for r in in_cand:
        s2 = r["s2"]
        best_li = max(r["light_idxs"], key=lambda i: s2[i])
        s2r = rank_of(s2, best_li)
        if s2r <= 3:
            examples.append((s2[r["chosen_idx"]] - s2[best_li], r, best_li, s2r))
    examples.sort(key=lambda x: x[0])
    for gap, r, best_li, s2r in examples[:6]:
        s2 = r["s2"]; s1 = r["s1"]
        s1r = rank_of(s1, best_li)
        print(f"  seed={r['seed']} round={r['round']} p{r['player']} coins={r['coins']} cd={r['cd']} "
              f"| 闪电s1第{s1r} s2第{s2r} | chosen={cat(r['chosen'])} s1={s1[r['chosen_idx']]:+.4f} "
              f"s2={s2[r['chosen_idx']]:+.4f} | 闪电s1={s1[best_li]:+.4f} s2={s2[best_li]:+.4f} gap={gap:+.4f}")


data = json.loads((ROOT / "code" / "test_match" / "_target_rounds.json").read_text(encoding="utf-8"))
analyze(data["light_targets"], "闪电目标（能放却没放）")
analyze(data["upg_targets"], "升级目标（chosen=升级）")
