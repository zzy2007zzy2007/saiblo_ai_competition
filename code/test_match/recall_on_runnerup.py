"""离线检验：我们的先验能否覆盖"亚军实际走的棋"（2026-10-26）。

流程（单局）：
  1. 从 ladder 日志解析双方每回合的 ops（格式 [[op_type,arg0,arg1],...]）
  2. 用 GameStateFacade 从 seed 重放；**用 RESULT 行的 base_hp / rounds 校验忠实性**
  3. 在亚军的每个决策点（回合开始，双方共享同一局面），取它实际执行的 (op_type, x, y)
  4. 用我们的先验给该局面打分，在"该类动作的合法格"上算它排第几 → recall@1/@4/@24

用法：python recall_on_runnerup.py <log> [更多 log ...]
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

CKPT = str(REPO / "checkpoints/prior_distill_f10.pt")
OP2CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20], 31: [21], 32: [22]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}

LINE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文='[^']*' -> (\[.*\])")


def parse_log(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    seed = int(re.search(r"对战 seed=(\d+)", txt).group(1))
    per = {}
    for rnd, side, who, ops in LINE.findall(txt):
        try:
            lst = ast.literal_eval(ops)
        except Exception:
            continue
        per.setdefault(int(rnd), {})[side] = lst
    res = re.search(r"base_hp=(\d+),(\d+)", txt)
    hp = (int(res.group(1)), int(res.group(2))) if res else None
    ru_side = "p1" if "p1=runnerup" in txt else "p0"
    return seed, per, hp, ru_side


def main() -> int:
    logs = [Path(p) for p in sys.argv[1:]] or sorted(
        (REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4").glob("*.log"))[:1]
    feat = FeatureExtractor(max_actions=96)
    _, m, _ = load_three_models(CKPT)
    m.eval()

    ks = (1, 4, 24)
    _ERR = []
    agg = {k: [0, 0] for k in ks}
    per_group = {}
    ok_games = 0
    for lp in logs:
        seed, per, hp_ref, ru = parse_log(lp)
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        ranks = []
        for rnd in sorted(per):
            if st.terminal:
                break
            snapshot = st.clone()
            for side in ("p0", "p1"):
                ops = per[rnd].get(side) or []
                applied = []
                for op in ops:
                    try:
                        applied.append((int(op[0]), int(op[1]), int(op[2])))
                    except Exception:
                        pass
                if side == ru and applied:
                    op_type, a0, a1 = applied[0]
                    if op_type in OP2CH:
                        try:
                            masks = make_position_masks(snapshot, int(ru[1]))
                            ch = masks[OP2CH[op_type][0]] if op_type != 11 else None
                            with torch.no_grad():
                                amap = m(torch.from_numpy(
                                    np.asarray(feat.encode_board(snapshot, int(ru[1])), dtype=np.float32)
                                ).unsqueeze(0),
                                    torch.from_numpy(
                                        np.asarray(feat.encode_stats(snapshot, int(ru[1])), dtype=np.float32)
                                    ).unsqueeze(0))["action_map"][0]
                            sc = (amap[0:16].mean(0) if op_type == 11 else amap[OP2CH[op_type][0]]).numpy()
                            mask = masks[OP2CH[op_type][0]].astype(bool) if op_type != 11 else None
                            if mask is None:
                                mask = np.zeros((19, 19), bool)
                                for c in range(16):
                                    mask |= masks[c].astype(bool)
                            if op_type == 11:
                                cx, cy = a0, a1
                            else:
                                tw = {int(t.tower_id): (int(t.x), int(t.y)) for t in snapshot.towers}
                                if op_type == 13:
                                    if a0 not in tw:
                                        continue
                                    cx, cy = tw[a0]
                                else:
                                    cx, cy = a0, a1
                            if not (0 <= cx < 19 and 0 <= cy < 19) or not mask[cx, cy]:
                                continue
                            v = np.where(mask, sc, -1e9).reshape(-1)
                            target = cx * 19 + cy
                            order = np.argsort(-v)
                            rank = int(np.where(order == target)[0][0]) + 1
                            ranks.append(rank)
                            for k in ks:
                                agg[k][0] += int(rank <= k); agg[k][1] += 1
                            d = per_group.setdefault(NAME.get(op_type, str(op_type)), {k: [0, 0] for k in ks})
                            for k in ks:
                                d[k][0] += int(rank <= k); d[k][1] += 1
                        except Exception as _e:
                            _ERR.append(f"{type(_e).__name__}: {_e}")
                            continue
            # 真正应用双方操作（用 facade 的 apply）
            for side in ("p0", "p1"):
                for op in per[rnd].get(side) or []:
                    try:
                        op_t, x, y = int(op[0]), int(op[1]), int(op[2])
                        if op_t == 13:
                            tw = {int(t.tower_id): int(t.tower_id) for t in st.towers}
                            if x not in tw:
                                continue
                        st.apply_operation_list(int(side[1]), [(op_t, x, y)])
                    except Exception:
                        pass
            st.advance_round()
        hp_now = (int(st.bases[0].hp), int(st.bases[1].hp))
        match = (hp_ref is not None and hp_now == hp_ref)
        ok_games += int(match)
        print(f"  {lp.name}: seed={seed} 重放 base_hp={hp_now} 记录={hp_ref} "
              f"{'✅一致' if match else '❌不一致'}  亚军决策点 {len(ranks)}", flush=True)

    print(f"\n  忠实重放 {ok_games}/{len(logs)} 局")
    if _ERR:
        print(f"  内部异常 {len(_ERR)} 次，前 3 条：")
        for e in _ERR[:3]:
            print(f"    {e}")
    if agg[24][1]:
        print("  我先验对'亚军实际走法'的召回：")
        for k in ks:
            print(f"    recall@{k:<3} = {agg[k][0]}/{agg[k][1]} = {100*agg[k][0]/agg[k][1]:5.1f}%")
        for g, d in sorted(per_group.items()):
            n = d[24][1]
            if n:
                print(f"    {g:<10} n={n:<5} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
