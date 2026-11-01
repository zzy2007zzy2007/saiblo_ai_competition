# -*- coding: utf-8 -*-
"""机制级诊断 第 1/2 步：从 collect_v2_s1 里筛「能放闪电却没放」与「选升级」的回合，
并对每个目标回合扒两阶段分数（s1 idle / s2 flat rollout）。

只读数据：training_history/collect_v2_s1/*.cand.jsonl + *.ents.jsonl
（ents.jsonl 在每决策点记录了 coins / weapon_cd，与 cand.jsonl 1:1 同序，
  由 az_selfplay.collect_game 的 samples 循环逐条写出 ⇒ 无需重放即可对齐）。

op 类型（SDK.utils.constants.OperationType）：
  11 建塔 / 12 升级塔 / 13 降级 / 21 闪电 / 22 EMP / 23 偏转 / 24 闪避 / 31 升速 / 32 升蚁
闪电 = SuperWeaponType.LIGHTNING_STORM = 1 ⇒ weapon_cd[player][1]，cost=90，cooldown=35。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
DATA = ROOT / "training_history" / "collect_v2_s1"

OP_NAMES = {11: "建塔", 12: "升级", 13: "降级", 21: "闪电", 22: "EMP",
            23: "偏转", 24: "闪避", 31: "升速", 32: "升蚁"}
LIGHT = 21          # OperationType.USE_LIGHTNING_STORM
UPG = 12            # OperationType.UPGRADE_TOWER
COST = 90
LIGHT_CD_IDX = 1    # weapon_cd[player][1] = 闪电冷却


def cat(bundle) -> str:
    ts = sorted({int(o[0]) for o in (bundle or ())})
    if not ts:
        return "HOLD"
    return "+".join(OP_NAMES.get(t, f"op{t}") for t in ts)


def has_op(bundle, op) -> bool:
    return any(int(o[0]) == op for o in (bundle or ()))


def iter_decisions():
    """遍历所有 seed 的所有决策，yield (seed, cand_rec, ents_rec)。"""
    files = sorted(DATA.glob("az_selfplay_seed*.cand.jsonl"))
    for cf in files:
        seed = int(cf.stem.replace("az_selfplay_seed", "").replace(".cand", ""))
        ef = DATA / f"az_selfplay_seed{seed}.ents.jsonl"
        if not ef.exists():
            continue
        with open(cf, encoding="utf-8") as fh:
            cand_lines = fh.read().splitlines()
        with open(ef, encoding="utf-8") as fh:
            ent_lines = fh.read().splitlines()
        assert len(cand_lines) == len(ent_lines), (seed, len(cand_lines), len(ent_lines))
        for cl, el in zip(cand_lines, ent_lines):
            c = json.loads(cl)
            e = json.loads(el)
            if c.get("round") != e.get("round") or c.get("player") != e.get("player"):
                print(f"⚠️ 对齐错位 seed={seed} cand={c.get('round')}/{c.get('player')} "
                      f"ent={e.get('round')}/{e.get('player')}", file=sys.stderr)
                continue
            yield seed, c, e


def main():
    light_targets = []   # 闪电目标回合
    upg_targets = []     # 升级目标回合
    n_dec = 0
    n_skipped = 0
    n_light_ready = 0          # 能放闪电（金币≥90 且 cd==0）的决策数
    n_light_ready_cast = 0     # 其中真的放了闪电
    n_light_in_cand = 0        # 能放闪电且闪电进了候选
    n_light_in_cand_not_chosen = 0

    for seed, c, e in iter_decisions():
        n_dec += 1
        player = int(c["player"])
        if int(c.get("skipped", 0)):
            n_skipped += 1
            continue
        coins = int(e["coins"][player])
        cd = int(e["weapon_cd"][player][LIGHT_CD_IDX])
        cand = c.get("cand") or []
        s1 = c.get("s1") or []
        s2 = c.get("s2") or []
        chosen = c.get("chosen") or []
        chosen_idx = int(c.get("chosen_idx", -1))

        # 闪电目标：能放但没放
        if coins >= COST and cd == 0:
            n_light_ready += 1
            chosen_light = has_op(chosen, LIGHT)
            if chosen_light:
                n_light_ready_cast += 1
            # 闪电候选是否在 cand 里
            li_idx = [i for i, b in enumerate(cand) if has_op(b, LIGHT)]
            if li_idx:
                n_light_in_cand += 1
                if not chosen_light:
                    n_light_in_cand_not_chosen += 1
            if not chosen_light:
                light_targets.append((seed, c, e, li_idx))

        # 升级目标：chosen 是升级 op
        if has_op(chosen, UPG):
            upg_targets.append((seed, c, e))

    print(f"决策总数={n_dec}  跳过={n_skipped}")
    print(f"\n=== 闪电 ===")
    print(f"能放闪电(金币≥90 & cd==0)决策数 = {n_light_ready}")
    print(f"  其中真放了闪电 = {n_light_ready_cast}  ({100.0*n_light_ready_cast/max(1,n_light_ready):.1f}%)")
    print(f"  其中闪电进了候选 = {n_light_in_cand}")
    print(f"  其中闪电进了候选但没选 = {n_light_in_cand_not_chosen}")
    print(f"「能放却没放」目标回合数 = {len(light_targets)}")
    print(f"\n=== 升级 ===")
    print(f"chosen 是升级 op 的回合数 = {len(upg_targets)}")

    # 保存目标回合清单（供 dump_rollout_tree.py 用）
    out = ROOT / "code" / "test_match" / "_target_rounds.json"
    def slim(seed, c, e, li):
        cand = c.get("cand") or []
        s1 = c.get("s1") or []
        s2 = c.get("s2") or []
        return {
            "seed": seed, "round": int(c["round"]), "player": int(c["player"]),
            "coins": int(e["coins"][int(c["player"])]),
            "cd": int(e["weapon_cd"][int(c["player"])][LIGHT_CD_IDX]),
            "chosen_idx": int(c.get("chosen_idx", -1)),
            "chosen": c.get("chosen"),
            "light_idxs": li,
            "n_cand": len(cand),
            "cand": cand,
            "s1": [float(x) for x in s1] if s1 else [],
            "s2": [float(x) for x in s2] if s2 else [],
        }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "light_targets": [slim(*x) for x in light_targets],
            "upg_targets": [slim(*x[:2], x[2], []) for x in upg_targets],
        }, fh, ensure_ascii=False)
    print(f"\n已保存目标回合清单 -> {out}")
    return light_targets, upg_targets


if __name__ == "__main__":
    main()
