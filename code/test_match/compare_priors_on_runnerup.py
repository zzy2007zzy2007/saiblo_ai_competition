"""多网对比：在"亚军实际走法"上比较多个先验的召回（2026-10-26）。

同一批日志、同一批决策点，只换被评估的 checkpoint ⇒ 可比较。
指标按动作类型拆分，并给出随机基线（1/支撑集、24/支撑集）。

用法：python compare_priors_on_runnerup.py [log ...]
     不给参数则用 VS_RUNNERUP_cells4 前 16 局。
"""
from __future__ import annotations

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
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
from SDK.utils.features import FeatureExtractor

NETS = [
    ("posnet_A(未训练)", "checkpoints/posnet_A_k5_m32.pt"),
    ("f10", "checkpoints/prior_distill_f10.pt"),
    ("prior_z_80g", "checkpoints/prior_z_80g.best.pt"),
    ("hybrid", "checkpoints/prior_z_hybrid_light.pt"),
]
CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}
OP_RE = re.compile(r"op_type=<OperationType\.\w+: (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
LINE_RE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文=.*? -> (\[.*?\])\s*$")


def parse_log(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    per, seed = {}, None
    m = re.search(r"对战 seed=(\d+)", txt)
    if m:
        seed = int(m.group(1))
    for line in txt.splitlines():
        mm = LINE_RE.match(line.strip())
        if not mm:
            continue
        per.setdefault(int(mm.group(1)), {})[mm.group(2)] = [
            (int(a), int(b), int(c)) for a, b, c in OP_RE.findall(mm.group(4))
        ]
    ru = "p1" if "p1=runnerup" in txt else ("p0" if "p0=runnerup" in txt else None)
    return seed, per, ru


def collect_decisions(logs):
    """→ [(状态快照, 我方player, 亚军侧player, 亚军首个操作, 该类mask, 支撑集)]"""
    feat = FeatureExtractor(max_actions=96)
    out = []
    for lp in logs:
        seed, per, ru = parse_log(lp)
        if ru is None or seed is None:
            continue
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        for rnd in sorted(per):
            if st.terminal:
                break
            op = (per[rnd].get(ru) or [None])[0]
            if op is not None and op[0] in CH:
                pt = int(ru[1])
                own = 1 - pt
                snap = st.clone()
                masks = make_position_masks(snap, pt)
                if op[0] == 13:
                    tw = {int(t.tower_id): (int(t.x), int(t.y)) for t in snap.towers}
                    cell = tw.get(op[1])
                else:
                    cell = (op[1], op[2])
                if op[0] == 11:
                    mask = np.zeros((19, 19), bool)
                    for c in range(16):
                        mask |= masks[c].astype(bool)
                else:
                    mask = masks[CH[op[0]][0]].astype(bool)
                if cell and 0 <= cell[0] < 19 and 0 <= cell[1] < 19 and mask[cell[0], cell[1]]:
                    board = np.asarray(feat.encode_board(snap, pt), dtype=np.float32)
                    stats = np.asarray(feat.encode_stats(snap, pt), dtype=np.float32)
                    out.append((board, stats, op[0], cell, mask))
            for side in ("p0", "p1"):
                raw = per[rnd].get(side) or []
                if raw:
                    st.apply_operation_list(int(side[1]), [Operation(OperationType(t), a, b) for t, a, b in raw])
            st.advance_round()
    return out


def evaluate(net_path: str, decisions, ks=(1, 4, 24)):
    _, m, _ = load_three_models(str(REPO / net_path))
    m.eval()
    per_op = {}
    for board, stats, op_t, cell, mask in decisions:
        with torch.no_grad():
            amap = m(torch.from_numpy(board).unsqueeze(0), torch.from_numpy(stats).unsqueeze(0))["action_map"][0]
        amap = amap.detach().cpu()
        sc = (amap[0:16].mean(0) if op_t == 11 else amap[CH[op_t][0]]).numpy()
        v = np.where(mask, sc, -1e9).reshape(-1)
        tgt = cell[0] * 19 + cell[1]
        rank = int(np.where(np.argsort(-v) == tgt)[0][0]) + 1
        sup = int(mask.sum())
        d = per_op.setdefault(NAME[op_t], {"hit": {k: 0 for k in ks}, "n": 0, "sup": []})
        d["n"] += 1
        d["sup"].append(sup)
        for k in ks:
            d["hit"][k] += int(rank <= k)
    return per_op


def main() -> int:
    logs = [Path(p) for p in sys.argv[1:]] or sorted(
        (REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4").glob("*.log"))[:16]
    print(f"  日志 {len(logs)} 局", flush=True)
    decisions = collect_decisions(logs)
    print(f"  可用决策点 {len(decisions)}", flush=True)

    by_net = {}
    for tag, path in NETS:
        by_net[tag] = evaluate(path, decisions)
        print(f"  ✓ 评估完 {tag}", flush=True)

    ops = ["build", "lightning", "downgrade", "evasion"]
    for op in ops:
        print(f"\n=== {op} ===")
        print(f"  {'网络':<18} {'n':>5} {'支撑集中位':>10} {'@1':>7} {'@4':>7} {'@24':>7}   {'随机@1':>7} {'随机@24':>8}")
        for tag, _ in NETS:
            d = by_net[tag].get(op)
            if not d or not d["n"]:
                continue
            sup = sorted(d["sup"])
            med = sup[len(sup) // 2]
            b1 = sum(1.0 / max(1, s) for s in sup) / len(sup)
            b24 = sum(min(24, s) / max(1, s) for s in sup) / len(sup)
            n = d["n"]
            print(f"  {tag:<18} {n:>5} {med:>10} "
                  f"{100*d['hit'][1]/n:>6.1f}% {100*d['hit'][4]/n:>6.1f}% {100*d['hit'][24]/n:>6.1f}%   "
                  f"{100*b1:>6.1f}% {100*b24:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
