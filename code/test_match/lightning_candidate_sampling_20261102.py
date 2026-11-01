# -*- coding: utf-8 -*-
"""闪电候选「采样不足」猜想验证（2026-11-02）。

只读数据：training_history/collect_v2_s1/*.cand.jsonl + *.ents.jsonl（1:1 对齐，无需重放）。

背景：`docs/rollout_lightning_diagnostic_20261102.md` 已确立「2,151 个『能放却没放』回合里，
65.7% 闪电进了 top-k 却被 s2 压下去（闪电 s2 排名中位 = 第 8）」。
本脚本验证用户的新猜想：**是不是闪电候选被 bandit 采样得太少，导致只进了少量「差的」
闪电候选到第二阶段，所以 s2 分偏低**。

统计口径（全部用「能放闪电」= coins[player]>=90 且 weapon_cd[player][1]==0 的回合）：
  §1 N/Q：闪电类(17) vs 塔类(0-15) 的 bandit 选中次数/价值（放了 vs 没放 vs 合计）。
  §2 top-k 构成：闪电/塔/降级/HOLD/超武其他/基地升级 各几条、比例（放了 vs 没放）。
  §3 猜想验证：候选条数(1 vs 多条) + s1 排名分布；#候选 与 闪电 s2 排名 的关系。

op 类型（SDK.utils.constants.OperationType）：
  11 建塔 / 12 升级塔 / 13 降级 / 21 闪电 / 22 EMP / 23 偏转 / 24 闪避 / 31 升速 / 32 升蚁。
闪电 = SuperWeaponType.LIGHTNING_STORM = 1 ⇒ weapon_cd[player][1]，cost=90，cooldown=35。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
DATA = ROOT / "training_history" / "collect_v2_s1"

OP_BUILD = 11
OP_UPGRADE = 12
OP_DOWNGRADE = 13
OP_LIGHT = 21
OP_EMP = 22
OP_DEFLECT = 23
OP_EVASION = 24
OP_BASE_SPEED = 31
OP_BASE_ANT = 32

LIGHT_CH = 17
TOWER_CHS = tuple(range(16))          # 0-15 塔型（建塔/升级）
DOWNGRADE_CH = 16
SUPER_OTHER_CHS = (18, 19, 20)         # EMP/偏转/闪避
BASE_CHS = (21, 22)                    # 升速/升蚁
HOLD_CH = 23

COST = 90
LIGHT_CD_IDX = 1                       # weapon_cd[player][1] = 闪电冷却


def has_op(bundle, op: int) -> bool:
    return any(int(o[0]) == op for o in (bundle or ()))


def classify(bundle) -> tuple:
    """候选束的类别标签（多标签：一个多 op 束可同时算多个桶）。"""
    ts = sorted({int(o[0]) for o in (bundle or ())})
    if not ts:
        return ("HOLD",)
    tags = []
    for t in ts:
        if t in (OP_BUILD, OP_UPGRADE):
            tags.append("塔")
        elif t == OP_DOWNGRADE:
            tags.append("降级")
        elif t == OP_LIGHT:
            tags.append("闪电")
        elif t in (OP_EMP, OP_DEFLECT, OP_EVASION):
            tags.append("超武其他")
        elif t in (OP_BASE_SPEED, OP_BASE_ANT):
            tags.append("基地升级")
        else:
            tags.append(f"op{t}")
    return tuple(tags)


def rank_of(scores, idx: int) -> int:
    """1-based 降序排名（同分不占坑，稳定）。"""
    s = scores[idx]
    return 1 + sum(1 for x in scores if x > s)


def iter_decisions():
    files = sorted(DATA.glob("az_selfplay_seed*.cand.jsonl"))
    for cf in files:
        seed = int(cf.stem.replace("az_selfplay_seed", "").replace(".cand", ""))
        ef = DATA / f"az_selfplay_seed{seed}.ents.jsonl"
        if not ef.exists():
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


def q_hist(vals, edges):
    """按 edges（右闭开）分桶的计数 + 占比。"""
    n = len(vals)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = sum(1 for v in vals if lo <= v < hi)
        out.append((f"[{lo},{hi})", k, 100.0 * k / max(1, n)))
    return out


def summarize(vals, label, pad=0):
    """打印一组数值的 mean/median/p25/p75/min/max + 计数。"""
    if not vals:
        print(f"{'':{pad}}{label}: n=0")
        return
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"{'':{pad}}{label}: n=0(全NaN)")
        return
    print(f"{'':{pad}}{label}: n={a.size} 均值={a.mean():.3f} 中位={np.median(a):.3f} "
          f"p25={np.percentile(a,25):.3f} p75={np.percentile(a,75):.3f} "
          f"min={a.min():.3f} max={a.max():.3f}")


def main() -> None:
    n_dec = 0
    n_skipped = 0
    n_ready = 0                 # 能放闪电
    n_cast = 0                  # 放了闪电（chosen 含 op21）
    n_light_in_cand = 0         # 闪电进了 top-k

    # 每回合指标（按组存）
    groups = {"cast": [], "no_cast": [], "all": []}

    # top-k 构成：每回合各标签条数
    comp = {"cast": defaultdict(list), "no_cast": defaultdict(list), "all": defaultdict(list)}
    TAG_KEYS = ("闪电", "塔", "降级", "HOLD", "超武其他", "基地升级")

    # §3 明细
    hyp = {"in_cand_no_cast": [], "cast": []}   # 每项：dict

    for seed, c, e in iter_decisions():
        n_dec += 1
        if int(c.get("skipped", 0)):
            n_skipped += 1
            continue
        player = int(c["player"])
        coins = int(e["coins"][player])
        cd = int(e["weapon_cd"][player][LIGHT_CD_IDX])
        if not (coins >= COST and cd == 0):
            continue
        n_ready += 1
        chosen = c.get("chosen") or []
        cand = c.get("cand") or []
        s1 = [float(x) for x in (c.get("s1") or [])]
        s2 = [float(x) for x in (c.get("s2") or [])]
        chosen_idx = int(c.get("chosen_idx", -1))
        N = c.get("N") or {}
        Q = c.get("Q") or {}

        chosen_light = has_op(chosen, OP_LIGHT)
        if chosen_light:
            n_cast += 1
        li_idx = [i for i, b in enumerate(cand) if has_op(b, OP_LIGHT)]
        if li_idx:
            n_light_in_cand += 1

        # §1 N/Q
        n_light = int(N.get(str(LIGHT_CH), 0))
        q_light = Q.get(str(LIGHT_CH))
        n_tower_sum = sum(int(N.get(str(ch), 0)) for ch in TOWER_CHS)
        q_tower_present = [float(Q[str(ch)]) for ch in TOWER_CHS if str(ch) in Q]
        q_tower_mean = float(np.mean(q_tower_present)) if q_tower_present else float("nan")
        n_downgrade = int(N.get(str(DOWNGRADE_CH), 0))
        n_hold = int(N.get(str(HOLD_CH), 0))
        n_super_other = sum(int(N.get(str(ch), 0)) for ch in SUPER_OTHER_CHS)
        n_base = sum(int(N.get(str(ch), 0)) for ch in BASE_CHS)
        n_total = int(c.get("N_total", 0))

        rec = {
            "n_light": n_light, "q_light": q_light,
            "n_tower_sum": n_tower_sum, "n_tower_mean": n_tower_sum / 16.0,
            "q_tower_mean": q_tower_mean,
            "n_downgrade": n_downgrade, "n_hold": n_hold,
            "n_super_other": n_super_other, "n_base": n_base, "n_total": n_total,
        }
        grp = "cast" if chosen_light else "no_cast"
        groups[grp].append(rec)
        groups["all"].append(rec)

        # §2 top-k 构成
        cnt = defaultdict(int)
        for b in cand:
            for tag in classify(b):
                cnt[tag] += 1
        for k in TAG_KEYS:
            comp[grp][k].append(cnt.get(k, 0))
            comp["all"][k].append(cnt.get(k, 0))
        comp[grp]["n_cand"].append(len(cand))
        comp["all"]["n_cand"].append(len(cand))

        # §3 明细
        if li_idx:
            best_s2 = max(li_idx, key=lambda i: s2[i])
            best_s1 = max(li_idx, key=lambda i: s1[i])
            detail = {
                "n_light_cand": len(li_idx),
                "s1_rank_best": rank_of(s1, best_s1),
                "s1_val_best": s1[best_s1],
                "s2_rank_best": rank_of(s2, best_s2),
                "s2_val_best": s2[best_s2],
                "s1_rank_of_s2best": rank_of(s1, best_s2),
                "chosen_s2": s2[chosen_idx] if 0 <= chosen_idx < len(s2) else float("nan"),
                "chosen_s1": s1[chosen_idx] if 0 <= chosen_idx < len(s1) else float("nan"),
            }
            if chosen_light:
                hyp["cast"].append(detail)
            else:
                hyp["in_cand_no_cast"].append(detail)

    # ── 宏观 sanity check ────────────────────────────────────────────────
    print(f"决策总数={n_dec}  跳过={n_skipped}")
    print(f"能放闪电(金币≥90 & cd==0)={n_ready}")
    print(f"  放了闪电={n_cast} ({100.0*n_cast/max(1,n_ready):.1f}%)")
    print(f"  闪电进了 top-k={n_light_in_cand} ({100.0*n_light_in_cand/max(1,n_ready):.1f}%)")
    print(f"  能放却没放={n_ready-n_cast}  其中闪电进候选但没选={len(hyp['in_cand_no_cast'])}")

    # ── §1 N/Q ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("§1 N/Q（bandit 选中次数/价值）—— 闪电类(17) vs 塔类(0-15)")
    print("=" * 78)
    for grp, label in (("all", "合计"), ("cast", "放了闪电"), ("no_cast", "没放闪电")):
        rs = groups[grp]
        if not rs:
            continue
        print(f"\n--- {label}（n={len(rs)} 回合）---")
        summarize([r["n_light"] for r in rs], "N[17] 闪电选中次数", 2)
        summarize([r["q_light"] for r in rs if r["q_light"] is not None], "Q[17] 闪电价值", 2)
        summarize([r["n_tower_sum"] for r in rs], "N[塔类合计] (0-15 之和)", 2)
        summarize([r["n_tower_mean"] for r in rs], "N[塔类每类均值] (合计/16)", 2)
        summarize([r["q_tower_mean"] for r in rs], "Q[塔类均值]", 2)
        summarize([r["n_downgrade"] for r in rs], "N[16] 降级", 2)
        summarize([r["n_hold"] for r in rs], "N[23] HOLD", 2)
        summarize([r["n_super_other"] for r in rs], "N[18-20] 其他超武合计", 2)
        summarize([r["n_total"] for r in rs], "N_total", 2)

    # 选中总量对比（直接回答"闪电 vs 塔被选中多少次"）
    for grp, label in (("all", "合计"), ("cast", "放了"), ("no_cast", "没放")):
        rs = groups[grp]
        if not rs:
            continue
        tot_light = sum(r["n_light"] for r in rs)
        tot_tower = sum(r["n_tower_sum"] for r in rs)
        tot_hold = sum(r["n_hold"] for r in rs)
        tot_dg = sum(r["n_downgrade"] for r in rs)
        print(f"\n[{label}] bandit 选中总量：闪电={tot_light} 塔类合计={tot_tower} "
              f"（每类均={tot_tower/16:.1f}） 降级={tot_dg} HOLD={tot_hold}")
        print(f"          塔类/闪电 比值={tot_tower/max(1,tot_light):.1f}；"
              f"塔类每类均/闪电={tot_tower/16/max(1,tot_light):.2f}")

    # N[17] 分布
    print("\nN[17] 分布（合计回合）：")
    vals = [r["n_light"] for r in groups["all"]]
    dist = Counter(vals)
    for k in sorted(dist):
        print(f"  N[17]={k}: {dist[k]}  ({100.0*dist[k]/len(vals):.1f}%)")

    # ── §2 top-k 构成 ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("§2 top-k 候选构成（每回合各类候选条数）")
    print("=" * 78)
    for grp, label in (("all", "合计"), ("cast", "放了闪电"), ("no_cast", "没放闪电")):
        d = comp[grp]
        n = len(d["n_cand"])
        if not n:
            continue
        print(f"\n--- {label}（n={n} 回合）---")
        tot = sum(d["n_cand"])
        print(f"  {'类别':<10} {'条数合计':>8} {'占比(条数)':>10} {'每回合均值':>10}")
        for k in TAG_KEYS:
            s = sum(d[k])
            print(f"  {k:<10} {s:>8} {100.0*s/max(1,tot):>9.1f}% {s/n:>10.2f}")
        print(f"  {'候选合计':<10} {tot:>8} {'100.0%':>10} {tot/n:>10.2f}")

    # ── §3 猜想验证 ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("§3 猜想验证：闪电候选条数 + s1 分布")
    print("=" * 78)

    def count_dist(items, key):
        c = Counter(int(d[key]) for d in items)
        n = len(items)
        return {k: (v, 100.0 * v / max(1, n)) for k, v in sorted(c.items())}

    for grp, label in (("in_cand_no_cast", "闪电进了 top-k 但没被选"),
                       ("cast", "放了闪电（闪电被选）")):
        items = hyp[grp]
        if not items:
            continue
        print(f"\n--- {label}（n={len(items)} 回合）---")
        print(f"  候选里闪电条数分布（1 vs 多条）：")
        for k, (v, p) in count_dist(items, "n_light_cand").items():
            print(f"    {k} 条: {v}  ({p:.1f}%)")
        print(f"  闪电候选的 s1 排名分布（best by s1）：")
        for k, (v, p) in count_dist(items, "s1_rank_best").items():
            print(f"    第{k}名: {v}  ({p:.1f}%)")
        print(f"  闪电候选的 s2 排名分布（best by s2）：")
        for k, (v, p) in count_dist(items, "s2_rank_best").items():
            print(f"    第{k}名: {v}  ({p:.1f}%)")
        summarize([d["s1_val_best"] for d in items], "闪电候选 s1 值(best by s1)", 2)
        summarize([d["s2_val_best"] for d in items], "闪电候选 s2 值(best by s2)", 2)

    # 关键交叉：候选条数 与 闪电 s2 排名（没被选回合）——多采样是否提升 s2 排名
    print("\n[关键交叉] 没被选回合：候选里闪电条数 vs 闪电 s2 排名（best by s2）")
    items = hyp["in_cand_no_cast"]
    by_n = defaultdict(list)
    for d in items:
        by_n[min(d["n_light_cand"], 3)].append(d["s2_rank_best"])  # 3=3+
    for k in sorted(by_n):
        rs = by_n[k]
        label = f"{k} 条" if k < 3 else "3+ 条"
        print(f"  {label}: n={len(rs)} s2排名 均值={np.mean(rs):.2f} 中位={np.median(rs):.1f} "
              f"p25={np.percentile(rs,25):.1f} p75={np.percentile(rs,75):.1f}")

    # 反证点：s1 排名第 1 的闪电仍然输给 s2（=「好闪电」也进二阶段了，仍被压）
    print("\n[反证] 没被选回合里，闪电 s1 排名第 1 的回合（「好闪电」也输了）：")
    good_s1 = [d for d in items if d["s1_rank_best"] == 1]
    print(f"  s1 第1名但没被选：{len(good_s1)} / {len(items)} = "
          f"{100.0*len(good_s1)/max(1,len(items)):.1f}%")
    if good_s1:
        s2r = [d["s2_rank_best"] for d in good_s1]
        print(f"  这些回合里闪电的 s2 排名：均值={np.mean(s2r):.2f} 中位={np.median(s2r):.1f} "
              f"p25={np.percentile(s2r,25):.1f} p75={np.percentile(s2r,75):.1f}")
        n1 = sum(1 for r in s2r if r == 1)
        print(f"  其中 s2 仍第 1（= 平局退化被 chosen 抢走）的：{n1}")

    # 相关：N[17] 与 候选条数（全体能放闪电且闪电进候选的回合）
    print("\n[N[17] 与 候选里闪电条数 的相关]")
    # 需要把 §1 的 N[17] 与 §3 的候选条数对齐；这里单独再扫一遍（只存必要字段）
    # 简化：直接从 groups['all'] 与 hyp 数量对齐不可靠，重扫一次
    corr = []
    for seed, c, e in iter_decisions():
        if int(c.get("skipped", 0)):
            continue
        player = int(c["player"])
        if not (int(e["coins"][player]) >= COST and int(e["weapon_cd"][player][LIGHT_CD_IDX]) == 0):
            continue
        cand = c.get("cand") or []
        N = c.get("N") or {}
        n_light = int(N.get(str(LIGHT_CH), 0))
        n_li_cand = sum(1 for b in cand if has_op(b, OP_LIGHT))
        corr.append((n_light, n_li_cand))
    if corr:
        x = np.array([a for a, _ in corr], dtype=np.float64)
        y = np.array([b for _, b in corr], dtype=np.float64)
        r = np.corrcoef(x, y)[0, 1]
        print(f"  能放闪电回合 N[17] vs 候选闪电条数：Pearson r={r:.3f}（n={len(corr)}）")
        # 分层桶：6-10 / 11-13 / 14-16 / 17-19 / 20+
        def nbuck(a):
            if a <= 10:
                return "6-10"
            if a <= 13:
                return "11-13"
            if a <= 16:
                return "14-16"
            if a <= 19:
                return "17-19"
            return "20+"
        by_nl = defaultdict(list)
        for a, b in corr:
            by_nl[nbuck(a)].append(b)
        print("  N[17] 分层 → 候选闪电条数均值：")
        for k in ("6-10", "11-13", "14-16", "17-19", "20+"):
            if k in by_nl:
                print(f"    N[17]∈{k}: n={len(by_nl[k])} 候选闪电条数均值={np.mean(by_nl[k]):.2f}")

    # 每类候选产量：塔类每类 vs 闪电类（直接检验"按类算是否采样不足"）
    print("\n[每类候选产量] 塔类 vs 闪电类（合计回合）")
    d = comp["all"]
    n = len(d["n_cand"])
    if n:
        tower_per_class = sum(d["塔"]) / 16.0 / n
        light_per_class = sum(d["闪电"]) / n
        print(f"  塔类每类候选条数/回合 = {tower_per_class:.2f}")
        print(f"  闪电类候选条数/回合 = {light_per_class:.2f}")
        print(f"  比值（塔类每类 / 闪电类）= {tower_per_class/max(1e-9, light_per_class):.2f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
