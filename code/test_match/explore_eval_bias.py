"""离线：**评估器到底把哪类动作判成正分？**（246 局 unified 采集的 cand.jsonl，165k 决策）

背景（parent 的追问 + 我的竞技场结论）：
  · 246 局行为账本：现役把 69% 收入花在**升级**、35% 在武器，闪电只有 3.25 次/人/局（理论 9.6）；
  · 竞技场（N=192/臂）：拿掉闪电 −21.9 HP；`upg=always` −0.31 vs `upg=damaged` +5.8；
  · 网络裸策略（argmax、无搜索）基本"攒钱+闪电、不造塔"，只比调优规则差 1.5 HP
    ⇒ 怀疑**评估器（s1=idle+值网 / s2=flat rollout）系统性偏袒升级、压低闪电**。

本脚本直接量这件事：对每个决策，把候选按 op 类型打标，看
  ① 每类候选存在时，它在 s1/s2 里的**排名百分位**；
  ② 它和"被选中候选"的分数差（s2_best_light − s2_chosen）；
  ③ s1 与 s2 的 top-1 是否一致（两套评估器打架的比例）；
  ④ 按金币分桶（穷/富）看偏袒是否随"买得起"变化。
只读，不改任何东西。产出 match_results/explore_20261102/eval_bias/。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict


def tags(ops):
    codes = {int(o[0]) for o in ops} if ops else set()
    t = []
    if not ops:
        return ["hold"]
    if 11 in codes:
        t.append("build")
    if 12 in codes:
        t.append("upg")
    if 13 in codes:
        t.append("down")
    if 21 in codes:
        t.append("light")
    if 22 in codes:
        t.append("emp")
    if 23 in codes:
        t.append("defl")
    if 24 in codes:
        t.append("evas")
    if 31 in codes or 32 in codes:
        t.append("base")
    return t or ["other"]


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="training_history/collect_v2_s1/*.cand.jsonl")
    ap.add_argument("--out", default="match_results/explore_20261102/eval_bias")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    os.makedirs(a.out, exist_ok=True)
    print(f"[info] {len(files)} files")

    n_dec = 0
    n_hold = 0
    present = Counter()          # 该类候选出现在多少决策里
    rank_pct = defaultdict(list)  # 类 -> [排名百分位]（s2，越小越好）
    rank_pct_s1 = defaultdict(list)
    gap_s2 = defaultdict(list)    # 类 -> best 该类 s2 − chosen s2
    gap_s1 = defaultdict(list)
    top1_by_class = {"s1": Counter(), "s2": Counter()}   # 该类存在时，top1 是否该类
    chosen_type = Counter()
    s1s2_disagree = 0
    coin_buckets = defaultdict(lambda: defaultdict(list))
    top1_type = Counter()
    chosen_low_coins_share = {}

    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cand, s1, s2 = d.get("cand"), d.get("s1"), d.get("s2")
            if not cand or s1 is None or s2 is None or len(cand) != len(s2):
                continue
            n_dec += 1
            ci = int(d.get("chosen_idx", -1))
            ctags = [tags(c) for c in cand]
            coins = None
            # coins 不在 cand.jsonl 里；用 root_value 分不了金币 ⇒ 用"是否含 hold/便宜候选"代理不可靠，
            # 这里改用 Q/N 里的类集合粗判（不精确），金币分桶交给另一个脚本。
            if ci < 0 or ci >= len(cand):
                continue
            if ctags[ci] == ["hold"]:
                n_hold += 1
            chosen_type[ctags[ci][0]] += 1
            i1, i2 = argmax(s1), argmax(s2)
            top1_type[ctags[i2][0]] += 1
            if i1 != i2:
                s1s2_disagree += 1
            seen = set()
            for k, t in enumerate(ctags):
                for name in t:
                    if (name, k) in seen:
                        continue
                    seen.add((name, k))
                    present[name] += 1
                    # 该类里最好的那个的排名（按 s2 降序）
                    pct = 1.0 - (sorted(s2, reverse=True).index(s2[k]) / max(len(s2) - 1, 1))
                    rank_pct[name].append(pct)
                    pct1 = 1.0 - (sorted(s1, reverse=True).index(s1[k]) / max(len(s1) - 1, 1))
                    rank_pct_s1[name].append(pct1)
            # 每类"最好的 s2" 与选中候选的差
            for name in ("light", "upg", "down", "hold", "build"):
                idxs = [k for k, t in enumerate(ctags) if name in t]
                if idxs:
                    best = max(s2[k] for k in idxs)
                    gap_s2[name].append(best - s2[ci])
                    best1 = max(s1[k] for k in idxs)
                    gap_s1[name].append(best1 - s1[ci])
            # 该类存在时，两套评估器各自的 top-1 是不是该类
            for name in ("light", "upg", "down", "hold", "build"):
                if any(name in t for t in ctags):
                    top1_by_class["s1"][name] += int(name in ctags[i1])
                    top1_by_class["s2"][name] += int(name in ctags[i2])

    out = []
    def P(s=""):
        out.append(str(s))
        print(s)

    P(f"# 评估器偏袒体检（{n_dec} 决策，{len(files)} 局）")
    P(f"- HOLD 被选中：{n_hold} ({n_hold/n_dec:.1%})")
    P()
    P("## 1. 被选中候选的动作类型 / s2-top1 的类型")
    for k, v in chosen_type.most_common():
        P(f"- chosen {k:8s} {v:7d} {v/n_dec:6.2%}   |  s2-top1 {k:8s} {top1_type[k]:7d}")
    P()
    P("## 2. 每类候选存在时的**排名百分位**（1.0 = 第一名；同一决策去重）")
    P("类      出现决策   s2排名pct均值  s1排名pct均值   best_s2 − chosen_s2 (均值)")
    for name in ("light", "emp", "defl", "evas", "build", "upg", "down", "base", "hold"):
        if not present.get(name):
            continue
        gap = gap_s2.get(name) or []
        gap_s = f"{statistics.mean(gap):+.5f}  (n={len(gap)})" if gap else "(无)"
        P(f"{name:8s} {present[name]:7d}   {statistics.mean(rank_pct[name]):.3f}          "
          f"{statistics.mean(rank_pct_s1[name]):.3f}          {gap_s}")
    P()
    P("## 3. 两套评估器的**头对头**（该类存在时，top-1 落在该类的比例）")
    P("类      出现决策   s1-top1=该类   s2-top1=该类   best_s1−chosen_s1   best_s2−chosen_s2")
    for name in ("light", "upg", "down", "build", "hold"):
        if not present.get(name):
            continue
        g1 = gap_s1.get(name) or []
        g2 = gap_s2.get(name) or []
        P(f"{name:8s} {present[name]:7d}   "
          f"{top1_by_class['s1'][name]/present[name]:6.1%}        "
          f"{top1_by_class['s2'][name]/present[name]:6.1%}        "
          f"{(statistics.mean(g1) if g1 else float('nan')):+.5f}          "
          f"{(statistics.mean(g2) if g2 else float('nan')):+.5f}")
    P()
    P("## 4. s1（idle+值网）与 s2（random rollout）的 top-1 一致率")
    P(f"- 不一致 {s1s2_disagree}/{n_dec} = {s1s2_disagree/n_dec:.1%}（**最终选择永远跟 s2**）")
    P()
    P("## 5. 读法")
    P("- `s2排名pct` 高 ⇒ 评估器把它排得靠前（更常被选）；`best_s2 − chosen_s2` < 0 ⇒ 该类最好的一条也没赢过被选的。")
    P("- 头对头表里 s1 与 s2 的符号若相反，就是'两套评估器打架、s2 赢'的直接证据。")

    with open(os.path.join(a.out, "eval_bias.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[ok] {a.out}/eval_bias.txt")


if __name__ == "__main__":
    main()
