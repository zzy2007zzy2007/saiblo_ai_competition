"""诊断（修正版，2026-10-11）：**钉类之后一个决策到底有几个格能解码成功？为什么少？**

⚠️ 修正了上一版的**两个真 bug**（都是我自己写的）：
  1. **局面从不推进**：上一版只对 `post = st.clone()` 落子/推进，`st` 本身永不前进 ⇒ 一局里所有决策看同一个冻结局面，
     于是"每决策只有 2 格、行数=3头×2玩家×256"全是假象。历史那个**有效**的采集器（`collect_value_prior.py:195-198`）
     是 `ops = decode_network_output(o, st, pl, ...)` → `st.apply_operation_list(pl, ops)` → 每轮 `st.advance_round()`。
  2. **坐标轴序错**：位置掩码是 `mask[ch, x, y]`（`decoder.py:192` 用 `HIGHLAND_CELLS` 的 `(x, y)` 直接索引）
     ⇒ `np.argwhere(pm[c])` 给的**就是 (x, y)** ⇒ 强行置 `am[c, x, y]`，并把**同一个 (x,y)** 交给 `_op_matches_cell`。
     上一版写成 `(y, x)` 双交换 ⇒ 只有 x==y 的对角格能过"位置相符"检查（实测 32/33 判"位置不符"）。

本轮要回答的问题：
  * 每个决策：钉到哪个类、该类掩码多少格、其中**空格/我方塔/敌方塔**各多少、能解码成功几个（建/升）；
  * 失败的三种死法（decode 返回 None / 不可执行 / 位置不符）各多少；
  * 按回合段看：highland 格数、空格数、双方塔数、金币 ⇒ 判断"格子少"是**局面填满**、**类不可达**、还是**掩码把敌塔当空格**。

用法：python -u code/my_ai/az_intent/diag_pos_cells.py --games 2 --seed 24000
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HOLD = 23


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=24000)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--probe-classes", type=int, default=8)
    ap.add_argument("--out", default="training_history/posroll/diag_pos_cells2.json")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(2)
    from SDK.utils.features import FeatureExtractor
    from SDK.backend.model import OperationType as OT

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.collect_value_prior import _op_matches_cell
    from my_ai.decoder import (HIGHLAND_CELLS, decode_head, decode_network_output,
                               make_class_mask, make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    pol, _ = make_net_fn_from_ckpt(args.checkpoint, feat)

    per_class = collections.defaultdict(collections.Counter)
    per_bucket = collections.defaultdict(collections.Counter)
    pin_hist = collections.Counter()
    succ_hist = collections.Counter()
    cat_ok = collections.Counter()      # 按格子类别的成率
    cat_all = collections.Counter()
    rounds_seen = 0
    axis_proof = []

    for g in range(args.games):
        st = make_initial_state(args.seed + g, True)
        for rnd in range(args.max_rounds):
            if st.terminal:
                break
            rounds_seen = rnd
            for pl in (0, 1):
                if st.terminal:
                    break
                obs = feat.encode_observation(st, pl, np.zeros(96))
                with torch.no_grad():
                    o = pol(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                am = o["action_map"][0].numpy()
                hl = [o[f"head{h}_logits"].squeeze(0).numpy() for h in (1, 2, 3)]
                pm = make_position_masks(st, pl, intent_decoding=True)
                cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)

                high = list(HIGHLAND_CELLS[pl])
                my_tw = {(t.x, t.y) for t in st.towers_of(pl)}
                op_tw = {(t.x, t.y) for t in st.towers_of(1 - pl)}
                n_empty = sum(1 for c in high if c not in my_tw and c not in op_tw)
                b = per_bucket[(rnd // 64) * 64]
                b["decisions"] += 1; b["highland"] += len(high); b["empty"] += n_empty
                b["my_towers"] += len(my_tw); b["enemy_towers"] += len(op_tw)
                b["coins"] += int(st.coins[pl])

                # 轴序证明（只做一次）：类 16 = 降级，掩码 True 的应正好是我方塔位置
                if not axis_proof and my_tw:
                    cells16 = {tuple(int(v) for v in xy) for xy in np.argwhere(pm[16])}
                    axis_proof.append({
                        "my_towers_xy": sorted(my_tw)[:4],
                        "pm16_argwhere": sorted(cells16)[:4],
                        "matches_as_xy": cells16 == my_tw,
                    })

                # 钉类：复刻 playable 扫描
                c_pin = None
                for c in np.argsort(-hl[0])[:max(1, int(args.probe_classes))]:
                    c = int(c)
                    if c == HOLD or not (0 <= c < len(cm)) or not cm[c] or not pm[c].any():
                        continue
                    probe = decode_head(hl[0], am, cm.copy(), pm.copy(), st, pl, class_id=c,
                                        temperature=0.0, pos_temperature=0.0, intent_decoding=True)
                    if probe is not None and st.can_apply_operation(pl, probe, ()):
                        c_pin = c
                        break
                if c_pin is None:
                    b["no_pin"] += 1
                else:
                    pin_hist[c_pin] += 1
                    pc = per_class[c_pin]
                    cells = np.argwhere(pm[c_pin])          # (x, y) —— 与 [ch, x, y] 口径一致
                    pc["decisions"] += 1
                    pc["mask_cells"] += len(cells)
                    ok = 0
                    for (x, y) in cells:
                        x, y = int(x), int(y)
                        if (x, y) in my_tw:
                            cat = "my_tower"; pc["cell_my_tower"] += 1
                        elif (x, y) in op_tw:
                            cat = "enemy_tower"; pc["cell_enemy_tower"] += 1
                        else:
                            cat = "empty"; pc["cell_empty"] += 1
                        cat_all[cat] += 1
                        am2 = am.copy(); am2[c_pin] = -1e9; am2[c_pin, x, y] = 1e9
                        op = decode_head(hl[0], am2, cm.copy(), pm.copy(), st, pl,
                                         class_id=int(c_pin), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None:
                            pc["f_none"] += 1; pc[f"none_{cat}"] += 1; continue
                        if not st.can_apply_operation(pl, op, ()):
                            pc["f_apply"] += 1; continue
                        if not _op_matches_cell(op, st, pl, int(c_pin), x, y):
                            pc["f_match"] += 1; continue
                        cat_ok[cat] += 1
                        ok += 1
                        if int(op.op_type) == int(OT.UPGRADE_TOWER):
                            pc["ok_upgrade"] += 1
                        else:
                            pc["ok_build"] += 1
                    succ_hist[ok] += 1

                # ⚠️ 关键修正：把网络的束**真的打到局面上**，并按历史口径推进回合
                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                st.apply_operation_list(pl, ops)
            if not st.terminal:
                st.advance_round()

    def fin(c: collections.Counter) -> dict:
        d = dict(c)
        n = max(d.get("decisions", 0), 1)
        d["mask_per_decision"] = round(d.get("mask_cells", 0) / n, 2)
        d["ok_per_decision"] = round((d.get("ok_build", 0) + d.get("ok_upgrade", 0)) / n, 2)
        return d

    cat_rate = {k: {"cells": cat_all[k], "ok": cat_ok[k],
                    "rate": round(cat_ok[k] / max(cat_all[k], 1), 4)} for k in cat_all}
    summary = {
        "games": args.games, "seed": args.seed, "rounds_seen": rounds_seen,
        "axis_proof": axis_proof,
        "pin_hist": dict(pin_hist.most_common(8)),
        "succ_per_decision_hist": dict(sorted(succ_hist.items())),
        "cell_category_success": cat_rate,
        "per_class": {str(k): fin(v) for k, v in sorted(per_class.items())},
        "per_bucket": {str(k): fin(v) for k, v in sorted(per_bucket.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== 钉类掩码/解码诊断（修正版，局面真实推进）=====")
    print(f"  轴序证明：{axis_proof}")
    print(f"  钉到的类分布：{summary['pin_hist']}")
    print(f"  **每决策解码成功数直方图**：{summary['succ_per_decision_hist']}")
    print(f"  按格子类别的成率：{cat_rate}")
    print("\n  按类：")
    for c, d in summary["per_class"].items():
        print(f"    类{c:>3} 决策={d['decisions']:>5} 掩码/决策={d['mask_per_decision']:>6.2f} 成功/决策={d['ok_per_decision']:>5.2f} "
              f"[空{d.get('cell_empty',0)} 我塔{d.get('cell_my_tower',0)} 敌塔{d.get('cell_enemy_tower',0)}] "
              f"失败[None {d.get('f_none',0)}(其中敌塔 {d.get('none_enemy_tower',0)}) 不可执行 {d.get('f_apply',0)} 位置不符 {d.get('f_match',0)}] "
              f"成功[建{d.get('ok_build',0)} 升{d.get('ok_upgrade',0)}]")
    print("\n  按回合段：")
    for b, d in summary["per_bucket"].items():
        n = max(d.get("decisions", 0), 1)
        print(f"    回合 {b:>3}+ 决策={d['decisions']:>5} highland/决策={d.get('highland',0)/n:>5.1f} 空格/决策={d.get('empty',0)/n:>5.1f} "
              f"我塔/决策={d.get('my_towers',0)/n:>5.1f} 敌塔/决策={d.get('enemy_towers',0)/n:>5.1f} coins={d.get('coins',0)/n:>6.0f} "
              f"→ 成功/决策={d['ok_per_decision']:>5.2f} 掩码/决策={d['mask_per_decision']:>5.2f}")
    print(f"\n  摘要：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
