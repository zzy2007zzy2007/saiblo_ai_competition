"""探索性分析：把 collect_v2_s1 的 246 局自对弈 replay 逐回合统计成"行为画像"。

只读，不改任何东西。产出：
  - 每局一行 tsv
  - 汇总 txt（动作构成 / 武器使用 / 降级 churn / 塔数与胜负关系）
用法：
  python code/test_match/explore_replay_profile.py --glob 'training_history/collect_v2_s1/*.replay.json' \
      --out match_results/explore_20261102
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

# op codes
BUILD, UPGRADE, DOWNGRADE = 11, 12, 13
WEAPON = {21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}
BASE_UPG = {31: "speed", 32: "anthp"}


def classify(op):
    code = int(op[0])
    if code == BUILD:
        return "build"
    if code == UPGRADE:
        return "upgrade"
    if code == DOWNGRADE:
        return "downgrade"
    if code in WEAPON:
        return WEAPON[code]
    if code in BASE_UPG:
        return "base_" + BASE_UPG[code]
    return f"other:{code}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="training_history/collect_v2_s1/*.replay.json")
    ap.add_argument("--out", default="match_results/explore_20261102")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"[info] {len(files)} replay files")

    rows = []
    act_total = Counter()          # 所有 op 实例按类型
    act_by_player = {0: Counter(), 1: Counter()}
    weapon_rounds = defaultdict(list)   # type -> [round]
    weapon_games = Counter()       # type -> 有多少局至少用一次
    per_game_weapons = []          # (n_weapons, won, rounds)
    hold_rounds = {0: 0, 1: 0}
    build_rounds = {0: [], 1: []}  # 建造发生的回合
    tower_count_at = defaultdict(list)  # round bucket -> 塔数样本
    downgrade_gap = []             # 降级后到下一次建造的回合间隔

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        steps = d.get("steps") or []
        res = d.get("result") or {}
        if not steps:
            continue
        n_rounds = len(steps)
        tol = 0
        acts = {0: Counter(), 1: Counter()}
        tower_alive = {0: 0, 1: 0}
        # 逆向不好做，直接按 op 事件累计"净塔数"：build +1，downgrade 若目标是 basic 则 -1
        # 但 downgrade 的 id 是塔 id，需要跟踪类型。用 id->type 映射（build 分配新 id 未知）
        # 简化：只统计 op 数，塔数用 build - (downgrade 到 basic) 近似由 op 参数判断不了。
        for st in steps:
            r = int(st.get("round", 0))
            per_player_ops = {0: st.get("op0") or [], 1: st.get("op1") or []}
            for p in (0, 1):
                ops = per_player_ops[p]
                if not ops:
                    hold_rounds[p] += 1
                for op in ops:
                    kind = classify(op)
                    acts[p][kind] += 1
                    act_total[kind] += 1
                    act_by_player[p][kind] += 1
                    if kind in ("lightning", "emp", "deflector", "evasion"):
                        weapon_rounds[kind].append(r)
                    if kind == "build":
                        build_rounds[p].append(r)
            # 记录塔数近似：用事后回放里 type=-1 不可得，跳过
        winner = res.get("winner")
        hp = res.get("hp") or []
        n_w = sum(v for k, v in acts[0].items() if k in
                  ("lightning", "emp", "deflector", "evasion"))
        n_w1 = sum(v for k, v in acts[1].items() if k in
                   ("lightning", "emp", "deflector", "evasion"))
        per_game_weapons.append((n_w, n_w1, winner, n_rounds))
        for k in ("lightning", "emp", "deflector", "evasion"):
            if acts[0][k] + acts[1][k] > 0:
                weapon_games[k] += 1
        rows.append(dict(
            file=os.path.basename(fp), seed=d.get("seed"), rounds=n_rounds,
            winner=winner, hp=hp, terminal=res.get("terminal"),
            vt=res.get("value_target_p0"),
            act0=dict(acts[0]), act1=dict(acts[1]),
            hold0=sum(1 for st in steps if not (st.get("op0") or [])),
            hold1=sum(1 for st in steps if not (st.get("op1") or [])),
        ))

    # ---- 汇总 ----
    out = []
    def P(s=""):
        out.append(str(s))
        print(s)

    P(f"# replay 行为画像（{len(rows)} 局）")
    P()
    P("## 0. 对局")
    rr = [r["rounds"] for r in rows]
    P(f"- 回合数：mean={statistics.mean(rr):.1f} median={statistics.median(rr)} "
      f"min={min(rr)} max={max(rr)}")
    term = sum(1 for r in rows if r["terminal"])
    P(f"- terminal（打到基地爆）：{term}/{len(rows)} = {term/len(rows):.1%}；"
      f"其余为 512 回合按血量判")
    w0 = sum(1 for r in rows if r["winner"] == 0)
    w1 = sum(1 for r in rows if r["winner"] == 1)
    tie = len(rows) - w0 - w1
    P(f"- 胜方：P0={w0} P1={w1} 其它={tie}  (自对弈座位偏差 {w0/len(rows):.1%} P0)")

    P()
    P("## 1. 动作构成（全部 op 实例）")
    tot = sum(act_total.values())
    for k, v in act_total.most_common():
        P(f"- {k:12s} {v:7d}  {v/tot:6.2%}")
    P(f"- 合计 op 实例 {tot}")

    P()
    P("## 2. HOLD 率（回合里完全不出招的比例）")
    th = sum(r["hold0"] + r["hold1"] for r in rows)
    tr = sum(r["rounds"] for r in rows) * 2
    P(f"- HOLD 回合 / 全部玩家回合 = {th}/{tr} = {th/tr:.1%}")

    P()
    P("## 3. 超级武器")
    for k in ("lightning", "emp", "deflector", "evasion"):
        n = act_total.get(k, 0)
        g = weapon_games.get(k, 0)
        P(f"- {k:10s} 总次数={n:5d}  使用过它的局数={g:3d}/{len(rows)} "
          f"({g/len(rows):.1%})  每局平均={n/len(rows):.3f}"
          + (f"  首次使用回合中位={statistics.median(weapon_rounds[k])}" if weapon_rounds[k] else ""))
    P()
    P("## 4. 每局武器数 vs 胜负（自对弈，两侧合计）")
    nw = [a + b for a, b, *_ in per_game_weapons]
    P(f"- 每局武器总数：mean={statistics.mean(nw):.2f} "
      f"median={statistics.median(nw)} max={max(nw)} "
      f"零武器的局={sum(1 for x in nw if x == 0)}/{len(nw)}")
    has = [r["rounds"] for r, x in zip(rows, nw) if x > 0]
    non = [r["rounds"] for r, x in zip(rows, nw) if x == 0]
    if has and non:
        P(f"- 用过武器的局平均回合 {statistics.mean(has):.0f} vs 没用过 {statistics.mean(non):.0f}")

    P()
    P("## 5. 降级 churn")
    dn = act_total.get("downgrade", 0)
    bu = act_total.get("build", 0)
    up = act_total.get("upgrade", 0)
    P(f"- downgrade={dn} build={bu} upgrade={up}  (down/build={dn/max(bu,1):.2f})")
    per_game = [(r["act0"].get("downgrade", 0) + r["act1"].get("downgrade", 0),
                 r["act0"].get("build", 0) + r["act1"].get("build", 0),
                 r["act0"].get("upgrade", 0) + r["act1"].get("upgrade", 0)) for r in rows]
    P(f"- 每局均值：downgrade={statistics.mean([x[0] for x in per_game]):.1f} "
      f"build={statistics.mean([x[1] for x in per_game]):.1f} "
      f"upgrade={statistics.mean([x[2] for x in per_game]):.1f}")

    P()
    P("## 6. 建造发生的回合分布")
    br = sorted(build_rounds[0] + build_rounds[1])
    buckets = [(0, 24), (25, 49), (50, 99), (100, 199), (200, 400), (401, 600)]
    for lo, hi in buckets:
        c = sum(1 for x in br if lo <= x <= hi)
        P(f"- 回合 {lo:3d}-{hi:3d}: {c:6d} 次建造 ({c/max(len(br),1):.1%})")

    P()
    P("## 7. 基地升级")
    P(f"- speed 升级 {act_total.get('base_speed',0)} 次（{sum(1 for r in rows if r['act0'].get('base_speed',0)+r['act1'].get('base_speed',0)>0)} 局）")
    P(f"- anthp 升级 {act_total.get('base_anthp',0)} 次")

    # 落盘
    with open(os.path.join(args.out, "profile.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(args.out, "per_game.tsv"), "w", encoding="utf-8") as f:
        f.write("file\tseed\trounds\twinner\thp0\thp1\tterminal\tvt\tbuild\tdowngrade\tupgrade\tweapons\n")
        for r in rows:
            b = r["act0"].get("build", 0) + r["act1"].get("build", 0)
            dg = r["act0"].get("downgrade", 0) + r["act1"].get("downgrade", 0)
            ug = r["act0"].get("upgrade", 0) + r["act1"].get("upgrade", 0)
            w = sum(r["act0"].get(k, 0) + r["act1"].get(k, 0)
                    for k in ("lightning", "emp", "deflector", "evasion"))
            hp = r["hp"] or [None, None]
            f.write(f"{r['file']}\t{r['seed']}\t{r['rounds']}\t{r['winner']}\t{hp[0]}\t{hp[1]}\t"
                    f"{r['terminal']}\t{r['vt']}\t{b}\t{dg}\t{ug}\t{w}\n")
    print(f"[ok] wrote {args.out}/profile.txt per_game.tsv")


if __name__ == "__main__":
    main()
