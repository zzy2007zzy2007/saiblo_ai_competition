"""把"金币"和"评估器的选择"接起来：**穷的时候评估器到底选什么？**

做法：
  1. 用引擎重放 `collect_v2_s1/*.replay.json`（seed + 双方 ops），在**每个决策时刻**
     记下 (round, player, coins, tower_count)；
  2. 与同局的 `*.cand.jsonl`（round, player, cand/s1/s2/chosen_idx/Q）按 (round, player) 对齐；
  3. 按金币分桶，看：候选供给（闪电在不在菜单里）、合法类集合（Q 的 key）、
     被选中的动作类型、各类的 s2 排名百分位。

回答三个问题：
  A. "贫困陷阱"（合法类坍缩成 {降级, HOLD}）在 252 局上占多少？
  B. 穷的时候是"评估器想降级"还是"只剩降级可选"？（看 Q 的 key 集 vs chosen）
  C. 闪电"进不了菜单"是因为穷（钱<90）还是因为候选生成？（按 coins 分桶看 present 率）

只读。产出 match_results/explore_20261102/poverty/。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

POOR = 15
BUCKETS = [(0, 14, "<15"), (15, 44, "15-44"), (45, 89, "45-89"),
           (90, 199, "90-199"), (200, 10 ** 9, ">=200")]


def bucket(c):
    for lo, hi, name in BUCKETS:
        if lo <= c <= hi:
            return name
    return "?"


def re_player(replay_path: str):
    """重放一局，返回 {(round, player): (coins, towers)}。"""
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    rp = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    st = GameStateFacade.initial(seed=int(rp["seed"]), cold_handle_rule_illegal=True)
    out = {}
    for step in rp["steps"]:
        if st.terminal:
            break
        rnd = int(step["round"])
        for p in (0, 1):
            if st.terminal:
                break
            out[(rnd, p)] = (int(st.coins[p]), len(st.towers_of(p)))
            ops = step.get(f"op{p}") or []
            if ops:
                st.apply_operation_list(
                    p, [Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops])
        if st.terminal:
            break
        st.advance_round()
    return out


def _job(fp):
    return fp, re_player(fp)


def tags(ops):
    if not ops:
        return "hold"
    codes = {int(o[0]) for o in ops}
    if 21 in codes:
        return "light"
    if 22 in codes:
        return "emp"
    if 23 in codes:
        return "defl"
    if 24 in codes:
        return "evas"
    if 12 in codes:
        return "upg"
    if 13 in codes:
        return "down"
    if 11 in codes:
        return "build"
    if 31 in codes or 32 in codes:
        return "base"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="training_history/collect_v2_s1")
    ap.add_argument("--out", default="match_results/explore_20261102/poverty")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    reps = sorted(glob.glob(os.path.join(a.dir, "*.replay.json")))
    print(f"[info] replay {len(reps)} 局")
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            coins_by_game = dict(ex.map(_job, reps, chunksize=2))
        # dict(ex.map(...)) 需要 (k,v) 对 ⇒ re_player 返回 dict，改成下面这样更稳
    else:
        coins_by_game = dict(_job(fp) for fp in reps)

    n_dec = 0
    by_bucket = defaultdict(lambda: dict(n=0, light_present=0, light_chosen=0,
                                         upg_present=0, upg_chosen=0,
                                         down_present=0, down_chosen=0,
                                         hold=0, trap=0,
                                         light_rank=[], upg_rank=[], down_rank=[],
                                         coins=[], towers=[]))
    trap_chosen = Counter()
    trap_coins = []
    missing = 0
    for cj in sorted(glob.glob(os.path.join(a.dir, "*.cand.jsonl"))):
        rep = cj.replace(".cand.jsonl", ".replay.json")
        table = coins_by_game.get(rep)
        if table is None:
            continue
        for line in open(cj, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (int(d.get("round", -1)), int(d.get("player", -1)))
            hit = table.get(key)
            if hit is None:
                missing += 1
                continue
            coins, ntowers = hit
            cand, s2 = d.get("cand"), d.get("s2")
            q = d.get("Q") or {}
            if not cand or s2 is None or len(cand) != len(s2):
                continue
            n_dec += 1
            b = bucket(coins)
            st = by_bucket[b]
            st["n"] += 1
            st["coins"].append(coins)
            st["towers"].append(ntowers)
            ct = [tags(c) for c in cand]
            ci = int(d.get("chosen_idx", 0))
            chosen = ct[ci] if 0 <= ci < len(ct) else "?"
            # 合法类集合（Q 的 key）
            keys = {int(k) for k in q}
            is_trap = keys <= {16, 23}
            if is_trap:
                st["trap"] += 1
                trap_chosen[chosen] += 1
                trap_coins.append(coins)
            for name in ("light", "upg", "down"):
                idxs = [k for k, t in enumerate(ct) if t == name]
                st[f"{name}_present"] += int(bool(idxs))
                st[f"{name}_chosen"] += int(chosen == name)
                if idxs:
                    best = max(s2[k] for k in idxs)
                    rank = 1.0 - sorted(s2, reverse=True).index(best) / max(len(s2) - 1, 1)
                    st[f"{name}_rank"].append(rank)
            st["hold"] += int(chosen == "hold")

    out = []
    def P(s=""):
        out.append(str(s))
        print(s)

    P(f"# 金币 × 评估器选择（{n_dec} 决策；对齐失败 {missing}）")
    P()
    P("## A. 按金币分桶")
    P("桶       决策数    占比    闪电在菜单  闪电被选   升级在菜单 升级被选  降级在菜单 降级被选  HOLD被选  {降,HLD}合法")
    for lo, hi, name in BUCKETS:
        st = by_bucket.get(name)
        if not st or not st["n"]:
            continue
        n = st["n"]
        P(f"{name:8s} {n:8d} {n/n_dec:6.1%}   "
          f"{st['light_present']/n:9.1%} {st['light_chosen']/n:8.2%}   "
          f"{st['upg_present']/n:8.1%} {st['upg_chosen']/n:7.1%}   "
          f"{st['down_present']/n:8.1%} {st['down_chosen']/n:7.1%}  "
          f"{st['hold']/n:6.1%}  {st['trap']/n:6.1%}")
    P()
    P("## B. 各类候选的 s2 排名百分位（1.0 = 第一）")
    P("桶       light     upg       down")
    for lo, hi, name in BUCKETS:
        st = by_bucket.get(name)
        if not st or not st["n"]:
            continue
        f = lambda xs: f"{statistics.mean(xs):.3f}" if xs else "  -  "
        P(f"{name:8s} {f(st['light_rank'])}   {f(st['upg_rank'])}   {f(st['down_rank'])}")
    P()
    P("## C. 贫困陷阱（合法类 ⊆ {降级16, HOLD23}）")
    tot_trap = sum(st["trap"] for st in by_bucket.values())
    P(f"- 占全部决策 **{tot_trap/n_dec:.1%}**（{tot_trap}/{n_dec}）")
    P(f"- 这些决策里被选中的动作：{dict(trap_chosen.most_common())}")
    if trap_coins:
        P(f"- 这些决策时的金币：mean={statistics.mean(trap_coins):.1f} "
          f"median={statistics.median(trap_coins)} max={max(trap_coins)}")
    P()
    P("## D. 闪电为什么进不了菜单")
    for lo, hi, name in BUCKETS:
        st = by_bucket.get(name)
        if not st or not st["n"]:
            continue
        P(f"- {name:8s}: 闪电在菜单里 {st['light_present']/st['n']:.1%}，"
          f"在菜单里时被选 {st['light_chosen']/max(st['light_present'],1):.1%}")
    P()
    P("读法：")
    P("- 若 `闪电在菜单` 的比例远低于 100%（钱够 90 时也不到 100%）⇒ 候选供给是瓶颈；")
    P("- 若 `降级在菜单` 几乎是 100% 而 `{降,HLD}合法` 很高 ⇒ 你说的贫困陷阱成立，")
    P("  但要看 `降级被选` 是不是因为评估器给降级高分（看 B 表的 down 排名）。")

    with open(os.path.join(a.out, "poverty.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[ok] {a.out}/poverty.txt")


if __name__ == "__main__":
    main()
