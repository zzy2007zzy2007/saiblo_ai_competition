"""直接量位置网通道质量：位置图排序 vs "1-ply 值网最优"（oracle）。

对每个真实中局局面、每个通道（0=建基本塔、17=闪电）：
  · 列出该通道全部**合法**格；
  · 每个格：打下去 → 推进 H 回合 → 值网打分（= 该格的 1-ply 值）；
  · 看位置图 argmax 的格在这些格里的**排名**，以及位置图 top-1 与 oracle top-1 是否一致。

若位置网通道靠谱，top-1 一致率应显著高于随机（1/n）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.decoder import make_class_mask, make_position_masks  # noqa: E402

H = 0                # 0 = 文档里的 adv 目标（打完后不推进，直接看值网差分）
SEED = 732
CKPT = str(_REPO / "checkpoints/posnet_A_k5_m32.pt")

LOGS = _REPO / "match_results"


def load_ops(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if '"rank":1,' not in line or "base_heuristic" not in line:
            continue
        rec = json.loads(line)
        ops = []
        if rec["key"] != "hold":
            for seg in rec["key"].split("|"):
                if seg.startswith("F:"):
                    continue
                parts = seg.split(":")
                if len(parts) < 2:
                    continue
                v = [int(x) for x in parts[:3]] + [-1] * (3 - min(3, len(parts)))
                ops.append((v[0], v[1], v[2]))
        out[(rec["round"], rec["player"])] = ops
    return out


feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(CKPT, feat, value_tanh=True, value_rel_to_abs=0.0)
_, _, value_model = load_three_models(CKPT)
value_model.eval()
ops0 = load_ops(LOGS / f"ai0_runnerup_seed{SEED}.stderr.log")
ops1 = load_ops(LOGS / f"ai1_ai3_linear_seed{SEED}.stderr.log")

CLS_TO_OP = {0: OperationType.BUILD_TOWER, 17: OperationType.USE_LIGHTNING_STORM}
OP_ARG = {0: (0, 0), 17: (0, 0)}       # Build(x,y) / Lightning(x,y) 都是 (x,y)


def leaf_value(state, player: int) -> float:
    sim = state.clone()
    for _ in range(H):
        if sim.terminal:
            break
        sim.advance_round()
    return float(value_oracle([sim], player)[0])


def value_oracle(states, player: int):
    zeros = np.zeros(96)
    boards, stats = [], []
    for st in states:
        obs = feat.encode_observation(st, player, zeros)
        boards.append(obs["board"])
        stats.append(obs["stats"])
    import torch
    b = torch.from_numpy(np.stack(boards)).float()
    s = torch.from_numpy(np.stack(stats)).float()
    with torch.no_grad():
        v = torch.tanh(value_model(b, s)["value"].squeeze(-1))
    return v.numpy()


facade = GameStateFacade.initial(SEED)
results = {0: {"top1": 0, "n": 0, "ranks": []}, 17: {"top1": 0, "n": 0, "ranks": []}}
states_seen = 0
for r in range(320):
    if facade.terminal:
        break
    for player, table in ((0, ops0), (1, ops1)):
        ops = table.get((r, player), [])
        if ops:
            facade.apply_operation_list(player, [
                Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops])
    facade.advance_round()
    if r % 40 != 0 or r < 60:
        continue
    for player in (0, 1):
        out = net_fn(facade, player)
        am = np.asarray(out["action_map"], dtype=np.float32)
        masks = make_position_masks(facade, player, intent_decoding=True)
        cmask = make_class_mask(facade, player, position_mask=masks, intent_decoding=True)
        for cid, op_type in CLS_TO_OP.items():
            if not cmask[cid]:
                continue
            cells = [(int(x), int(y)) for y, x in zip(*np.where(masks[cid]))]
            if len(cells) < 3:
                continue
            # 只取位置图前 6 个格 + oracle 需要全量（合法格通常 ≤40）
            trials, vals = [], []
            for (x, y) in cells:
                sim = facade.clone()
                try:
                    sim.apply_operation_list(player, [Operation(op_type, x, y)])
                except Exception:
                    continue
                trials.append((x, y))
                vals.append(None)
            if len(trials) < 3:
                continue
            leaves = []
            for (x, y) in trials:
                sim = facade.clone()
                sim.apply_operation_list(player, [Operation(op_type, x, y)])
                leaves.append(sim)
            values = value_oracle(leaves, player)
            prior_vals = [float(am[cid][y, x]) for (x, y) in trials]
            if values.size >= 3:
                ra = np.argsort(np.argsort(prior_vals)).astype(float)
                rb = np.argsort(np.argsort(values)).astype(float)
                ra -= ra.mean(); rb -= rb.mean()
                den = (ra*ra).sum() ** 0.5 * (rb*rb).sum() ** 0.5
                if den > 1e-9:
                    results[cid].setdefault("rhos", []).append(float((ra*rb).sum()/den))
            order = np.argsort(-values)
            rank_of_prior_top = int(np.where(order == int(np.argmax(prior_vals)))[0][0]) + 1
            results[cid]["n"] += 1
            results[cid]["ranks"].append(rank_of_prior_top)
            if rank_of_prior_top == 1:
                results[cid]["top1"] += 1
            states_seen += 1

print(f"[样本] {states_seen} 个 (局面, 通道) 组合，rollout H={H}")
for cid, name in ((0, "建塔通道(0)"), (17, "闪电通道(17)")):
    d = results[cid]
    if d["n"] == 0:
        print(f"  {name}: 无样本")
        continue
    ranks = np.array(d["ranks"])
    print(f"\n  {name}: n={d['n']}")
    print(f"    位置图 top-1 与 1-ply 值网 top-1 **一致率 = {100*d['top1']/d['n']:.1f}%**")
    print(f"    位置图 top-1 的排名分布：中位 {np.median(ranks):.0f}，均值 {ranks.mean():.1f}，"
          f"前3名内 {100*(ranks <= 3).mean():.0f}%")
    rhos = d.get("rhos", [])
    if rhos:
        print(f"    位置图 vs adv 的 **Spearman 中位 = {np.median(rhos):+.3f}**"
              f"（均值 {np.mean(rhos):+.3f}，n={len(rhos)}）")
