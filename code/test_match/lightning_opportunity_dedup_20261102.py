# -*- coding: utf-8 -*-
"""闪电机会去重统计（2026-11-02）。

背景：原始比例「放了闪电的回合 / 可放闪电的回合」= 1,660 / 3,811 ≈ 43.6%，
把「同一波机会里反复出现的可放闪电回合」合并后，看真实「放闪电机会」的命中率。

口径（与 code/test_match/find_lightning_upg_rounds.py 完全一致）：
  · 可放闪电回合 = coins[player] >= 90 且 weapon_cd[player][1] == 0（闪电冷却）。
  · 放了闪电       = chosen 束里含 op 21（OperationType.USE_LIGHTNING_STORM）。
  · player 0 与 1 都算；每个 (game, player) 独立聚类。
  · 跳过 search 的决策（cand.jsonl `skipped==1`）不计入（与 3811/1660 的参考口径一致）。

去重（单链接聚类）：
  · 对每个 (game, player)，把可放闪电回合按 round 升序排列；
  · 相邻两回合的 round 差 gap = r[i] - r[i-1]（"间隔" = 回合号之差）；
  · gap <= n 则合并到同一簇（传递合并）。n = ∞ ⇒ 不合并 = 原始口径。

只读数据：training_history/collect_v2_s1/*.cand.jsonl + *.ents.jsonl（1:1 对齐）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
DATA = ROOT / "training_history" / "collect_v2_s1"

OP_LIGHT = 21           # OperationType.USE_LIGHTNING_STORM
COST = 90
LIGHT_CD_IDX = 1        # weapon_cd[player][1] = 闪电冷却


def has_op(bundle, op: int) -> bool:
    return any(int(o[0]) == op for o in (bundle or ()))


def iter_decisions():
    files = sorted(DATA.glob("az_selfplay_seed*.cand.jsonl"))
    for cf in files:
        seed = int(cf.stem.replace("az_selfplay_seed", "").replace(".cand", ""))
        ef = DATA / f"az_selfplay_seed{seed}.ents.jsonl"
        if not ef.exists():
            print(f"⚠️ 缺 ents: {ef.name}", file=sys.stderr)
            continue
        clines = cf.read_text(encoding="utf-8").splitlines()
        elines = ef.read_text(encoding="utf-8").splitlines()
        assert len(clines) == len(elines), (seed, len(clines), len(elines))
        for cl, el in zip(clines, elines):
            c = json.loads(cl)
            e = json.loads(el)
            if c.get("round") != e.get("round") or c.get("player") != e.get("player"):
                print(f"⚠️ 对齐错位 seed={seed} cand={c.get('round')}/{c.get('player')} "
                      f"ent={e.get('round')}/{e.get('player')}", file=sys.stderr)
                continue
            yield seed, c, e


def single_linkage_clusters(rounds, n: int):
    """rounds: 已按 round 升序的 [(round, is_cast)]；n = 合并阈值（gap <= n 合并）。

    返回 list of {"rounds": [...], "cast": bool}（每个簇是否至少放了一次闪电）。
    n 传 None 或很大 = 不合并。
    """
    clusters = []
    for r, cast in rounds:
        if clusters and (n is not None) and (r - clusters[-1]["last"] <= n):
            clusters[-1]["last"] = r
            clusters[-1]["cast"] = clusters[-1]["cast"] or cast
            clusters[-1]["rounds"].append(r)
        else:
            clusters.append({"last": r, "cast": cast, "rounds": [r]})
    return clusters


def main() -> None:
    # 按 (seed, player) 收集可放闪电回合（round, is_cast），保持 round 升序
    ready_by_key = defaultdict(list)   # (seed, player) -> [(round, is_cast)]

    n_dec = 0
    n_skipped = 0
    n_ready = 0
    n_cast = 0
    ready_skipped = 0     # 被 skipped 但 coins/cd 仍满足"可放"的决策数（诊断）

    for seed, c, e in iter_decisions():
        n_dec += 1
        player = int(c["player"])
        rnd = int(c["round"])
        coins = int(e["coins"][player])
        cd = int(e["weapon_cd"][player][LIGHT_CD_IDX])
        if int(c.get("skipped", 0)):
            n_skipped += 1
            if coins >= COST and cd == 0:
                ready_skipped += 1
            continue
        if coins >= COST and cd == 0:
            n_ready += 1
            cast = has_op(c.get("chosen") or [], OP_LIGHT)
            if cast:
                n_cast += 1
            ready_by_key[(seed, player)].append((rnd, cast))

    print(f"决策总数 = {n_dec}  跳过 search = {n_skipped}")
    print(f"跳过决策里满足 coins>=90 & cd==0 的 = {ready_skipped}（诊断：这些被排除在口径外）")
    print(f"可放闪电回合（原始分母） = {n_ready}")
    print(f"  放了闪电（原始分子） = {n_cast}  ({100.0 * n_cast / max(1, n_ready):.3f}%)")
    print(f"  （参考口径：1,660 / 3,811 = 43.6%）")
    print()

    # 每个 key 内按 round 升序（数据按写出顺序已升序，显式 sort 保稳）
    for k in ready_by_key:
        ready_by_key[k].sort(key=lambda t: t[0])

    # 诊断：可放闪电回合的相邻 gap 分布（跨所有 key，仅对同 key 内相邻）
    gaps = []
    run_lengths = []      # 连续可放闪电段（gap==1）的长度
    for k, rounds in ready_by_key.items():
        cur = 1
        for i in range(1, len(rounds)):
            g = rounds[i][0] - rounds[i - 1][0]
            gaps.append(g)
            if g == 1:
                cur += 1
            else:
                run_lengths.append(cur)
                cur = 1
        if rounds:
            run_lengths.append(cur)
    print("=== 可放闪电回合的结构诊断 ===")
    print(f"相邻可放闪电回合的 gap 分布（n = 回合号之差）：")
    for g, cnt in sorted(Counter(gaps).items()):
        print(f"  gap={g:3d}: {cnt:6d}  ({100.0 * cnt / max(1, len(gaps)):.1f}%)")
    print(f"连续可放闪电段（gap==1）长度分布（前 20）：")
    for ln, cnt in sorted(Counter(run_lengths).items())[:20]:
        print(f"  段长={ln:3d}: {cnt:5d} 段")
    print()

    # ── 去重计算 ─────────────────────────────────────────────────────────
    print("=== 去重结果 ===")
    print(f"{'n':>6} | {'机会数(新分母)':>14} | {'放闪电机会(新分子)':>16} | {'比例':>9}")
    print("-" * 60)
    results = {}
    for n in (5, 10, 20):
        n_opp = 0
        n_opp_cast = 0
        for rounds in ready_by_key.values():
            for cl in single_linkage_clusters(rounds, n):
                n_opp += 1
                if cl["cast"]:
                    n_opp_cast += 1
        ratio = 100.0 * n_opp_cast / max(1, n_opp)
        results[n] = (n_opp, n_opp_cast, ratio)
        print(f"{n:>6} | {n_opp:>14} | {n_opp_cast:>16} | {ratio:>8.3f}%")

    # n = ∞（不合并）= 原始口径
    n_opp_inf = n_ready
    n_opp_cast_inf = n_cast
    ratio_inf = 100.0 * n_opp_cast_inf / max(1, n_opp_inf)
    results["inf"] = (n_opp_inf, n_opp_cast_inf, ratio_inf)
    print(f"{'∞':>6} | {n_opp_inf:>14} | {n_opp_cast_inf:>16} | {ratio_inf:>8.3f}%")
    print()

    # 结论辅助：n=5 时每个机会平均含多少个可放回合
    for n in (5, 10, 20):
        opp, opp_cast, _ = results[n]
        print(f"n={n}: 每个机会平均含 {n_ready / max(1, opp):.2f} 个可放闪电回合；"
              f"「至少放了一次」的机会占比 = {100.0 * opp_cast / max(1, opp):.2f}%")

    print("\n完成。")


if __name__ == "__main__":
    main()
