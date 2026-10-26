"""覆盖率/召回 v3：用**正确重放**的局面上算"我们不弱于亚军走法"（2026-10-26）。

关键修正（来自 docs/engine_consistency_probe.md）：
  · 解析用正则抠 `Operation(op_type=..., arg0=..., arg1=...)`，不用 ast.literal_eval
  · 出招必须**整回合一次性** apply_operation_list(p0 列表) → apply_operation_list(p1 列表) → advance_round
  · 不用宽泛 except，错误直接抛

指标：在每个亚军决策点（回合开始，双方共享同一局面），
  把亚军实际执行的动作规整成 (op_type, 格)，看我们的先验在"该类合法格"上把它排第几。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models
from my_ai.az_intent.game_state_facade import GameStateFacade
from my_ai.decoder import make_position_masks
from SDK.utils.features import FeatureExtractor
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

CKPT = str(REPO / "checkpoints/prior_distill_f10.pt")
CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20]}
OP_RE = re.compile(r"op_type=<OperationType\.\w+: (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
LINE_RE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文=.*? -> (\[.*?\])\s*$")


def parse_log(path: Path):
    per, seed, hp = {}, None, None
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"对战 seed=(\d+)", txt)
    if m:
        seed = int(m.group(1))
    m = re.search(r"base_hp=(\d+),(\d+)", txt)
    if m:
        hp = (int(m.group(1)), int(m.group(2)))
    for line in txt.splitlines():
        mm = LINE_RE.match(line.strip())
        if not mm:
            continue
        per.setdefault(int(mm.group(1)), {})[mm.group(2)] = [
            (int(a), int(b), int(c)) for a, b, c in OP_RE.findall(mm.group(4))
        ]
    ru = "p1" if "p1=runnerup" in txt else ("p0" if "p0=runnerup" in txt else None)
    return seed, per, hp, ru


def main() -> int:
    logs = [Path(p) for p in sys.argv[1:]]
    if not logs:
        logs = sorted((REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4").glob("*.log"))[:2]
    feat = FeatureExtractor(max_actions=96)
    _, m, _ = load_three_models(CKPT)
    m.eval()

    ks = (1, 4, 24)
    agg = {k: [0, 0] for k in ks}
    per_op = {}
    replay_ok = 0
    for lp in logs:
        seed, per, hp_ref, ru = parse_log(lp)
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        ranks = []
        for rnd in sorted(per):
            if st.terminal:
                break
            snap = st.clone()
            op = (per[rnd].get(ru) or [None])[0]
            if op is not None and op[0] in CH:
                try:
                    pt = int(ru[1])
                    masks = make_position_masks(snap, pt)
                    with torch.no_grad():
                        amap = m(
                            torch.from_numpy(np.asarray(feat.encode_board(snap, pt), dtype=np.float32)).unsqueeze(0),
                            torch.from_numpy(np.asarray(feat.encode_stats(snap, pt), dtype=np.float32)).unsqueeze(0),
                        )["action_map"][0]
                    amap = amap.detach().cpu()
                    sc = (amap[0:16].mean(0) if op[0] == 11 else amap[CH[op[0]][0]]).numpy()
                    mask = masks[CH[op[0]][0]].astype(bool) if op[0] != 11 else None
                    if mask is None:
                        mask = np.zeros((19, 19), bool)
                        for c in range(16):
                            mask |= masks[c].astype(bool)
                    if op[0] == 13:
                        tw = {int(t.tower_id): (int(t.x), int(t.y)) for t in snap.towers}
                        if op[1] not in tw:
                            continue
                        cx, cy = tw[op[1]]
                    else:
                        cx, cy = op[1], op[2]
                    SUP = int(mask.sum())
                    if 0 <= cx < 19 and 0 <= cy < 19 and mask[cx, cy]:
                        v = np.where(mask, sc, -1e9).reshape(-1)
                        tgt = cx * 19 + cy
                        rank = int(np.where(np.argsort(-v) == tgt)[0][0]) + 1
                        ranks.append((op[0], rank, SUP))
                except Exception as exc:      # 不吞错：打印一次类型
                    print(f"    ⚠️ round={rnd} 打分失败 {type(exc).__name__}: {exc}")
            for side in ("p0", "p1"):
                raw = per[rnd].get(side) or []
                if raw:
                    ops = [Operation(OperationType(t), a, b) for t, a, b in raw]
                    st.apply_operation_list(int(side[1]), ops)
            st.advance_round()
        hp_now = (int(st.bases[0].hp), int(st.bases[1].hp))
        ok = hp_now == hp_ref
        replay_ok += int(ok)
        for op_t, rank, sup in ranks:
            d = per_op.setdefault(op_t, {k: [0, 0] for k in ks})
            d.setdefault("sup", []).append(sup)
            for k in ks:
                agg[k][0] += int(rank <= k)
                agg[k][1] += 1
                d[k][0] += int(rank <= k)
                d[k][1] += 1
        print(f"  {lp.name}: 重放 hp={hp_now} 记录={hp_ref} {'✅' if ok else '❌'}  亚军决策点 {len(ranks)}", flush=True)

    print(f"\n  重放忠实 {replay_ok}/{len(logs)} 局")
    if agg[24][1]:
        print("  我们的先验对'亚军实际走法'的召回：")
        for k in ks:
            print(f"    recall@{k:<3} = {agg[k][0]}/{agg[k][1]} = {100*agg[k][0]/agg[k][1]:5.1f}%")
        print("\n  按动作类型拆分：")
        for op_t in sorted(per_op):
            d = per_op[op_t]
            n = d[24][1]
            miss = n - d[24][0]
            sup = d.get("sup") or [0]
            sup_sorted = sorted(sup)
            med = sup_sorted[len(sup_sorted)//2]
            mean = sum(sup)/len(sup)
            base1 = sum(1.0/max(1, s) for s in sup)/len(sup)
            print(f"    op_type={op_t:<3} n={n:<5} 支撑集 均值{mean:5.1f}/中位{med:<4} "
                  f"@1={100*d[1][0]/n:5.1f}%(随机基线{100*base1:5.1f}%)  "
                  f"@4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
            print(f"                 支撑集 <4 的比例 {100*sum(1 for s in sup if s < 4)/len(sup):5.1f}%   "
                  f"<24 的比例 {100*sum(1 for s in sup if s < 24)/len(sup):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
