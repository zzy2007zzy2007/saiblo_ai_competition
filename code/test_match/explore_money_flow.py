"""用官方确定性引擎重放 collect_v2_s1 的 246 局，做**精确的钱流账本**。

只读、不训练、不改任何已有文件。产出 `match_results/explore_20261102/money/`：
  - money_per_game.tsv   每局一行的钱流分类
  - money_summary.txt    汇总（收入构成 / 支出构成 / churn 烧掉多少 / 闲置金币 / 错过的机会）

为什么要它：`docs/unified_rootcause_deepdive_20261101.md` 说"降级 churn 烧钱"，
但历史上从没有把"钱从哪来、到哪去"逐币对上过。replay 里有 seed + 每回合双方完整 ops，
引擎是确定性的 ⇒ 可以把每个 op 的金币增量精确记下来。

用法：
  python code/test_match/explore_money_flow.py --limit 8          # 冒烟
  python code/test_match/explore_money_flow.py --jobs 8           # 全量
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

BUILD, UPGRADE, DOWNGRADE = 11, 12, 13
WEAPON_CODES = {21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}
BASE_UPG = {31: "speed", 32: "anthp"}
LIGHTNING_COST, LIGHTNING_CD = 90, 35
UPG_COST = {1: 60, 2: 200}   # level -> cost (rough; downgrade refunds use hp ratio too)


def analyze_one(fp: str) -> dict:
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    rp = json.loads(Path(fp).read_text(encoding="utf-8"))
    seed = int(rp["seed"])
    steps = rp["steps"]
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)

    acc = {
        "file": os.path.basename(fp), "seed": seed,
        "income": {0: 0, 1: 0},           # 回合结算带来的钱（基础 + 击杀 + 到达）
        "build_spend": {0: 0, 1: 0},      # 建塔花掉（正数表示花掉）
        "downgrade_refund": {0: 0, 1: 0},  # 降级/拆除退回（正数）
        "upgrade_spend": {0: 0, 1: 0},
        "weapon_spend": {0: 0, 1: 0},
        "base_spend": {0: 0, 1: 0},
        "n_ops": {0: 0, 1: 0},
        "weapon_use": {0: Counter(), 1: Counter()},
        "n_rounds_player": 0,
        "coins_snap": {0: [], 1: []},     # 每回合决策时刻的金币
        "tower_cnt": {0: [], 1: []},
        "missed_lightning": {0: 0, 1: 0},  # 钱够 & cd==0 & 这回合没放
        "missed_upgrade": {0: 0, 1: 0},    # 钱>=60 & 有塔 & 没升级
        "hold_turns": {0: 0, 1: 0},
        "tower_mix": {0: Counter(), 1: Counter()},
        "died_hp": None,
    }

    for step in steps:
        if st.terminal:
            break
        coins_after_ops = [0, 0]
        for p in (0, 1):
            if st.terminal:
                break
            ops = step.get(f"op{p}") or []
            c0 = int(st.coins[p])
            acc["coins_snap"][p].append(c0)
            acc["tower_cnt"][p].append(len(st.towers_of(p)))
            cds = st.weapon_cooldowns
            can_light = (cds[p][1] == 0) and (c0 >= LIGHTNING_COST)
            used_light = False
            acc["n_rounds_player"] += 1
            if not ops:
                acc["hold_turns"][p] += 1
            for (t, a0, a1) in ops:
                t = int(t)
                before = int(st.coins[p])
                st.apply_operation_list(p, [Operation(OperationType(t), int(a0), int(a1))])
                delta = int(st.coins[p]) - before
                acc["n_ops"][p] += 1
                if t == BUILD:
                    acc["build_spend"][p] += -delta
                elif t == UPGRADE:
                    acc["upgrade_spend"][p] += -delta
                elif t == DOWNGRADE:
                    acc["downgrade_refund"][p] += delta
                elif t in WEAPON_CODES:
                    acc["weapon_spend"][p] += -delta
                    acc["weapon_use"][p][WEAPON_CODES[t]] += 1
                    if t == 21:
                        used_light = True
                elif t in BASE_UPG:
                    acc["base_spend"][p] += -delta
            if can_light and not used_light:
                acc["missed_lightning"][p] += 1
            if c0 >= 60 and acc["tower_cnt"][p][-1] > 0 and not any(int(o[0]) == UPGRADE for o in ops):
                acc["missed_upgrade"][p] += 1
            coins_after_ops[p] = int(st.coins[p])
        if st.terminal:
            break
        st.advance_round()
        for p in (0, 1):
            acc["income"][p] += int(st.coins[p]) - coins_after_ops[p]

    for p in (0, 1):
        for tw in st.towers_of(p):
            acc["tower_mix"][p][str(tw.tower_type)] += 1
    acc["died_hp"] = [int(st.bases[0].hp), int(st.bases[1].hp)]
    acc["winner"] = None if st.winner is None else int(st.winner)
    acc["rounds"] = int(st.round_index)
    acc["terminal"] = bool(st.terminal)
    return acc


def _spent_this_round(acc, p, step):
    """turn-level 已单独记账；这里只是让 income 的口径显式化（spend 不计入 income）。"""
    return 0


def summarize(results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def P(s=""):
        lines.append(str(s))
        print(s)

    n = len(results)
    P(f"# 钱流账本（{n} 局 replay 精确重放）")
    P()

    def S(key):
        return sum(r[key][0] + r[key][1] for r in results) / n

    inc = S("income")
    bs, dr, us, ws, bsp = S("build_spend"), S("downgrade_refund"), S("upgrade_spend"), S("weapon_spend"), S("base_spend")
    P("## 1. 每局平均（两名玩家合计）")
    P(f"- 结算收入 income          : {inc:8.1f}")
    P(f"- 建塔支出 build            : {bs:8.1f}")
    P(f"- 降级退款 refund (−支出)   : {-dr:8.1f}")
    P(f"- 升级支出 upgrade          : {us:8.1f}")
    P(f"- 超级武器 weapon           : {ws:8.1f}")
    P(f"- 基地升级 base             : {bsp:8.1f}")
    net = inc - (bs - dr) - us - ws - bsp
    P(f"- 净留存（应≈终局金币和−100）: {net:8.1f}")
    P(f"- 终局金币和                : {sum(r['coins_snap'][0][-1] + r['coins_snap'][1][-1] for r in results)/n:8.1f}")
    P()
    P("## 2. churn（建塔+降级净值）")
    P(f"- 每局建塔 {bs:.1f}，退回 {dr:.1f} ⇒ 净花在建/降循环 {bs-dr:.1f}"
      f"（占收入 {(bs-dr)/max(inc,1):.1%}）")
    P(f"- 每局升级 {us:.1f}（占收入 {us/max(inc,1):.1%}）")
    P(f"- 每局武器 {ws:.1f}（占收入 {ws/max(inc,1):.1%}）")
    P(f"- 每局基地升级 {bsp:.1f}")
    P()
    P("## 3. 闲置 / 机会")
    mc = sum(r["missed_lightning"][0] + r["missed_lightning"][1] for r in results) / n
    mu = sum(r["missed_upgrade"][0] + r["missed_upgrade"][1] for r in results) / n
    turns = sum(r["n_rounds_player"] for r in results) / n
    P(f"- 每局玩家回合数：{turns:.1f}")
    P(f"- **错过闪电**（钱≥90 且 cd==0 却没用）：{mc:.1f} 次/局 = {mc/turns:.1%} 的回合")
    P(f"- **错过升级**（钱≥60 且有塔却没升级）：{mu:.1f} 次/局 = {mu/turns:.1%} 的回合")
    allc = [c for r in results for p in (0, 1) for c in r["coins_snap"][p]]
    P(f"- 决策时刻金币：mean={statistics.mean(allc):.1f} median={statistics.median(allc)} "
      f"p90={statistics.quantiles(allc, n=10)[8]:.0f} max={max(allc)}")
    P(f"- 金币>=90 的回合占 {sum(1 for c in allc if c>=90)/len(allc):.1%}；"
      f">=200 占 {sum(1 for c in allc if c>=200)/len(allc):.1%}；"
      f"<15 占 {sum(1 for c in allc if c<15)/len(allc):.1%}")
    P()
    P("## 4. 武器使用（两名玩家合计 / 局）")
    for k in ("lightning", "emp", "deflector", "evasion"):
        tot = sum(r["weapon_use"][0][k] + r["weapon_use"][1][k] for r in results) / n
        P(f"- {k:10s} {tot:.2f}/局")
    P()
    P("## 5. 塔数（决策时刻快照）")
    tc = [c for r in results for p in (0, 1) for c in r["tower_cnt"][p]]
    P(f"- 塔数 mean={statistics.mean(tc):.2f} median={statistics.median(tc)} max={max(tc)}")
    P(f"- 有塔回合占比 {sum(1 for x in tc if x>0)/len(tc):.1%}")
    P()
    P("## 6. 每局明细（前后各 5 局）")
    for r in results[:5] + results[-5:]:
        P(f"- {r['file']} rounds={r['rounds']} hp={r['died_hp']} "
          f"inc={r['income'][0]+r['income'][1]} build={r['build_spend'][0]+r['build_spend'][1]} "
          f"refund={r['downgrade_refund'][0]+r['downgrade_refund'][1]} "
          f"upg={r['upgrade_spend'][0]+r['upgrade_spend'][1]} "
          f"wpn={r['weapon_spend'][0]+r['weapon_spend'][1]}")

    with open(os.path.join(out_dir, "money_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "money_per_game.tsv"), "w", encoding="utf-8") as f:
        cols = ["file", "seed", "rounds", "winner", "hp0", "hp1", "terminal",
                "income", "build", "refund", "upgrade", "weapon", "base",
                "miss_light", "miss_upg", "coins_mean", "coins_final", "towers_mean"]
        f.write("\t".join(cols) + "\n")
        for r in results:
            cs = r["coins_snap"][0] + r["coins_snap"][1]
            f.write("\t".join(str(x) for x in [
                r["file"], r["seed"], r["rounds"], r["winner"],
                r["died_hp"][0], r["died_hp"][1], int(r["terminal"]),
                r["income"][0] + r["income"][1],
                r["build_spend"][0] + r["build_spend"][1],
                r["downgrade_refund"][0] + r["downgrade_refund"][1],
                r["upgrade_spend"][0] + r["upgrade_spend"][1],
                r["weapon_spend"][0] + r["weapon_spend"][1],
                r["base_spend"][0] + r["base_spend"][1],
                r["missed_lightning"][0] + r["missed_lightning"][1],
                r["missed_upgrade"][0] + r["missed_upgrade"][1],
                round(statistics.mean(cs), 1) if cs else 0,
                r["coins_snap"][0][-1] + r["coins_snap"][1][-1],
                round(statistics.mean(r["tower_cnt"][0] + r["tower_cnt"][1]), 2),
            ]) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="training_history/collect_v2_s1/*.replay.json")
    ap.add_argument("--out", default="match_results/explore_20261102/money")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if a.limit:
        files = files[: a.limit]
    print(f"[info] {len(files)} games, jobs={a.jobs}")
    if a.jobs <= 1:
        results = []
        for i, fp in enumerate(files):
            results.append(analyze_one(fp))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(analyze_one, files, chunksize=1))
    summarize(results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
