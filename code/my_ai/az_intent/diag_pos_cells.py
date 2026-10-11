"""诊断（2026-10-11 修正版 v2）：钉类之后的**可采格数**、失败原因、以及"只采头 0 会损失多少"。

修掉了原版两个真 bug（用户追问逼出，见 `docs/experiment_log.md` 86b）：
  1. **局面从不推进** ⇒ 原版只对 `post = st.clone()` 落子。现按历史**有效**口径
     （`collect_value_prior.py:195-198`）：`ops = decode_network_output(o, st, pl, ...)` → `st.apply_operation_list(pl, ops)` → 每轮 `st.advance_round()`。
  2. **坐标轴序写反**：位置掩码是 `mask[ch, x, y]`（`decoder.py:192` 用 `HIGHLAND_CELLS` 的 `(x, y)` 索引）
     ⇒ `np.argwhere(pm[c])` 给的就是 **(x, y)**。全程统一 `[x, y]`（设格 `am[c,x,y]`、校验 `_op_matches_cell(..., x, y)`）。

本轮还回答用户提的两个问题：
  * **"多动作束很少吗？只采头 0 影响大吗？"** ⇒ 统计每回合**束大小直方图**，以及**每个头**各自贡献多少"可采决策点"
    （钉到塔类 0–16 且该类合法格 ≥2）；
  * **"为什么格少？"** ⇒ 按类、按回合段、按格子类别（空格/我方塔/敌方塔）统计掩码格数与解码成功率、失败三分类。

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
PIN_MAX_CLASS = 16          # 与采集器一致：只钉 0..16（部署 MC 菜单的经济类范围）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=24000)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--probe-classes", type=int, default=8)
    ap.add_argument("--out", default="training_history/posroll2/diag_pos_cells.json")
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

    bundle_hist = collections.Counter()          # 每回合束大小
    head_dec = collections.Counter()             # 每个头贡献的"可采决策点"
    head_pin_fail = collections.Counter()        # 每个头：没钉到 / 类合法格 <2
    head_few = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    per_bucket = collections.defaultdict(collections.Counter)
    pin_hist = collections.Counter()
    succ_hist = collections.Counter()
    cat_ok = collections.Counter()
    cat_all = collections.Counter()
    axis_proof = []
    turns = 0
    final_round = 0

    for g in range(args.games):
        st = make_initial_state(args.seed + g, True)
        for rnd in range(args.max_rounds):
            if st.terminal:
                break
            final_round = rnd
            for pl in (0, 1):
                if st.terminal:
                    break
                turns += 1
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
                b["my_towers"] += len(my_tw); b["enemy_towers"] += len(op_tw); b["coins"] += int(st.coins[pl])

                if not axis_proof and my_tw:
                    c16 = {tuple(int(v) for v in xy) for xy in np.argwhere(pm[16])}
                    axis_proof.append({"my_towers_xy": sorted(my_tw)[:4],
                                       "pm16_argwhere": sorted(c16)[:4], "matches_as_xy": c16 == my_tw})

                for h in range(3):
                    c_pin = None
                    for c in np.argsort(-hl[h])[:max(1, int(args.probe_classes))]:
                        c = int(c)
                        if c == HOLD or not (0 <= c <= PIN_MAX_CLASS) or not cm[c] or not pm[c].any():
                            continue
                        probe = decode_head(hl[h], am, cm.copy(), pm.copy(), st, pl, class_id=c,
                                            temperature=0.0, pos_temperature=0.0, intent_decoding=True)
                        if probe is not None and st.can_apply_operation(pl, probe, ()):
                            c_pin = c
                            break
                    if c_pin is None:
                        head_pin_fail[h] += 1
                        continue
                    cells = np.argwhere(pm[c_pin])            # (x, y)
                    if len(cells) < 2:
                        head_few[h] += 1
                        continue
                    head_dec[h] += 1
                    pin_hist[c_pin] += 1
                    pc = per_class[c_pin]
                    pc["decisions"] += 1; pc["mask_cells"] += len(cells)
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
                        op = decode_head(hl[h], am2, cm.copy(), pm.copy(), st, pl,
                                         class_id=int(c_pin), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None:
                            pc["f_none"] += 1; b["f_none"] += 1; continue
                        if not st.can_apply_operation(pl, op, ()):
                            pc["f_apply"] += 1; b["f_apply"] += 1; continue
                        if not _op_matches_cell(op, st, pl, int(c_pin), x, y):
                            pc["f_match"] += 1; b["f_match"] += 1; continue
                        cat_ok[cat] += 1; ok += 1
                        if int(op.op_type) == int(OT.UPGRADE_TOWER):
                            pc["ok_upgrade"] += 1; b["ok_upgrade"] += 1
                        else:
                            pc["ok_build"] += 1; b["ok_build"] += 1
                    succ_hist[ok] += 1

                # 真实对局推进（历史口径）
                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                bundle_hist[len(ops)] += 1
                st.apply_operation_list(pl, ops)
            if not st.terminal:
                st.advance_round()

    def fin(c: collections.Counter) -> dict:
        d = dict(c)
        n = max(d.get("decisions", 0), 1)
        d["mask_per_decision"] = round(d.get("mask_cells", 0) / n, 2)
        d["ok_per_decision"] = round((d.get("ok_build", 0) + d.get("ok_upgrade", 0)) / n, 2)
        return d

    dec_total = sum(head_dec.values())
    summary = {
        "games": args.games, "seed": args.seed, "turns": turns, "final_round": final_round,
        "axis_proof": axis_proof,
        "bundle_size_hist": dict(sorted(bundle_hist.items())),
        "head_decisions": {str(h): head_dec[h] for h in (0, 1, 2)},
        "head_decisions_share": {str(h): round(head_dec[h] / max(dec_total, 1), 4) for h in (0, 1, 2)},
        "head_pin_fail": {str(h): head_pin_fail[h] for h in (0, 1, 2)},
        "head_few_cells": {str(h): head_few[h] for h in (0, 1, 2)},
        "pin_hist": dict(sorted(pin_hist.items(), key=lambda kv: -kv[1])[:8]),
        "succ_per_decision_hist": dict(sorted(succ_hist.items())),
        "cell_category_success": {k: {"cells": cat_all[k], "ok": cat_ok[k],
                                      "rate": round(cat_ok[k] / max(cat_all[k], 1), 4)} for k in cat_all},
        "per_class": {str(k): fin(v) for k, v in sorted(per_class.items())},
        "per_bucket": {str(k): fin(v) for k, v in sorted(per_bucket.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== 诊断 v2（真实对局 + 全程 [x,y]）=====")
    print(f"  回合数={turns}（末回合 {final_round}）；轴序证明：{axis_proof}")
    print(f"  **每回合束大小直方图**：{summary['bundle_size_hist']}")
    print(f"  **每个头贡献的可采决策点**：{summary['head_decisions']}  占比 {summary['head_decisions_share']}")
    print(f"     （没钉到：{summary['head_pin_fail']}；钉到但合法格<2：{summary['head_few_cells']}）")
    print(f"  钉类分布：{summary['pin_hist']}")
    print(f"  每决策成功格数直方图：{summary['succ_per_decision_hist']}")
    print(f"  按格子类别成率：{summary['cell_category_success']}")
    print("\n  按类：")
    for c, d in summary["per_class"].items():
        print(f"    类{c:>3} 决策={d['decisions']:>5} 掩码/决策={d['mask_per_decision']:>5.1f} 成功/决策={d['ok_per_decision']:>5.1f} "
              f"[空{d.get('cell_empty',0)} 我塔{d.get('cell_my_tower',0)} 敌塔{d.get('cell_enemy_tower',0)}] "
              f"失败[None {d.get('f_none',0)} 不可执行 {d.get('f_apply',0)} 位置不符 {d.get('f_match',0)}] 成功[建{d.get('ok_build',0)} 升{d.get('ok_upgrade',0)}]")
    print("\n  按回合段：")
    for k, d in summary["per_bucket"].items():
        n = max(d.get("decisions", 0), 1)
        print(f"    回合 {k:>3}+ 决策={d['decisions']:>5} highland/决策={d.get('highland',0)/n:>5.1f} 空格/决策={d.get('empty',0)/n:>5.1f} "
              f"我塔/决策={d.get('my_towers',0)/n:>5.1f} 敌塔/决策={d.get('enemy_towers',0)/n:>5.1f} coins={d.get('coins',0)/n:>6.0f} "
              f"→ 成功[建{d.get('ok_build',0)} 升{d.get('ok_upgrade',0)}] 失败[None {d.get('f_none',0)} 位置不符 {d.get('f_match',0)}]")
    print(f"\n  摘要：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
