"""端到端实验：**我们的确定性候选集 + 亚军自己的评分器** ⇒ 决策一致率。

流程
----
1. 从 12 个**留出局**里抽 N 个状态（一半有动作、一半 hold）；
2. 3 头生成器给确定性候选集：每头「hold + top-3 类 × 每类位置 top-16」；
   物化 = 全 hold + 单 op(约49) + 双 op(49×2) ≈ 148 条 ⇒ 与亚军 ~152 个候选同量级；
   升级类的目标塔型模型不预测 ⇒ 对该塔**枚举全部合法升级目标**（类穷举哲学的延伸）；
3. 交给 `dump_plan_prior.exe --mode score_ops`，用**亚军自己的评分器**打分
   （8 回合 rollout 均值 + hold+80 / 降级狙击 −400 / C1 静态加成）；
4. 比 `argmax(score)` 与**亚军真实下发**的动作组，拆成覆盖率 × 条件一致率。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path.insert(0, str(_REPO / "code"))
from my_ai.entity_transformer.train_gen_bundles import ThreeHeadGenerator, to_batch

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
HOLD, BUILD, UPGRADE, DOWNGRADE, LIGHTNING = 0, 1, 2, 3, 4
UPGRADE_TREE = {0: (1, 2, 3, 4), 1: (11, 12, 13), 2: (21, 22, 23),
                3: (31, 32, 33), 4: (41, 42, 43)}


def make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def load_states(seed: int) -> list[dict]:
    base = _REPO / "training_history/distill_top2/abl_linear2_raw"
    path = base / f"ents_seed{seed:05d}.jsonl"
    if not path.exists():
        cands = list(base.glob(f"ents_*{seed}.jsonl"))
        if not cands:
            return []
        path = cands[0]
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    rollouts = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    head2_width = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    k_cls, n_pos = 3, 16

    device = torch.device("cuda")
    raw = np.load(_REPO / "training_history/distill_top2/q8/gen_ds_64games.npz", allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels, games = data["labels"], data["game"]
    rounds, players = data["round"], data["player"]
    CKPT = sys.argv[4] if len(sys.argv) > 4 else "checkpoints/gen_bundles_3head.pt"
    ckpt = torch.load(_REPO / CKPT, map_location="cpu", weights_only=False)
    print(f"[ckpt] {CKPT}")
    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(0)
    heldout = rng.permutation(np.unique(games))[:12]
    test_index = np.flatnonzero(np.isin(games, heldout))
    act_idx = test_index[labels[test_index, 0] > 0]
    hold_idx = test_index[labels[test_index, 0] == 0]
    take_act = rng.choice(act_idx, size=min(len(act_idx), n_states // 2), replace=False)
    take_hold = rng.choice(hold_idx, size=min(len(hold_idx), n_states - len(take_act)), replace=False)
    chosen = np.sort(np.concatenate([take_act, take_hold]))
    print(f"[样本] {len(chosen)} 个决策（有动作 {len(take_act)} / hold {len(take_hold)}）", flush=True)

    with torch.no_grad():
        heads, cell, tower = model(to_batch(data, chosen, device))
    cls_logp = torch.stack([F.log_softmax(c, -1) for c in heads], 1).cpu()
    cell_logp = F.log_softmax(cell, -1).cpu()
    tower_logp = F.log_softmax(tower, -1).cpu()
    own_mask = torch.from_numpy(data["tower_mask"][chosen]
                               & (data["tower_cont"][chosen][:, :, 0] > 0.5))
    tower_logp = tower_logp.masked_fill(~own_mask.unsqueeze(1), -1e9)

    cache: dict[int, list[dict]] = {}
    requests, bundles_all, truth_all = [], [], []
    for row, sample in enumerate(chosen):
        seed, round_index, player = int(games[sample]), int(rounds[sample]), int(players[sample])
        if seed not in cache:
            cache[seed] = load_states(seed)
        state = dict(cache[seed][round_index])
        towers = state.get("towers") or []

        def options(slot: int, limit: int):
            logp = cls_logp[row, slot].clone()
            logp[HOLD] = -1e9
            keep = torch.topk(logp, k=k_cls).indices.tolist()
            out = [(float(cls_logp[row, slot, HOLD]), ())]
            for cls_id in keep:
                if cls_id in (BUILD, LIGHTNING):
                    channel = 1 if cls_id == LIGHTNING else 0
                    table = torch.topk(cell_logp[row, channel], k=n_pos).indices.tolist()
                    op_type = 21 if cls_id == LIGHTNING else 11
                    out += [(float(cls_logp[row, slot, cls_id] + cell_logp[row, channel, c]),
                             ((op_type, c // 19, c % 19),)) for c in table]
                elif cls_id == DOWNGRADE:
                    cand = [(float(tower_logp[row, 1, i]), i) for i in range(tower_logp.shape[2])]
                    cand = [item for item in cand if item[0] > -1e8]
                    cand.sort(reverse=True)
                    out += [(score, ((13, int(towers[i][0]), -1),)) for score, i in cand[:n_pos]]
                elif cls_id == UPGRADE:
                    cand = [(float(tower_logp[row, 0, i]), i) for i in range(tower_logp.shape[2])]
                    cand = [item for item in cand if item[0] > -1e8]
                    cand.sort(reverse=True)
                    for score, i in cand[:3]:
                        for target in UPGRADE_TREE.get(int(towers[i][4]), ()):
                            out.append((score, ((12, int(towers[i][0]), target),)))
            out.sort(key=lambda item: -item[0])
            return out[:limit]

        head1 = options(0, 1 + k_cls * n_pos + 16)  # 放宽截断：升级枚举会超出 49
        head2 = options(1, head2_width)
        bundles = [list(item[1]) for item in head1]
        for _, ops1 in head1:
            if not ops1:
                continue
            for _, ops2 in head2:
                if ops2:
                    bundles.append(list(ops1) + list(ops2))

        truth = []
        for slot in range(3):
            cls_id = int(labels[sample][1 + slot * 4])
            if cls_id == HOLD:
                break
            tgt = int(labels[sample][2 + slot * 4])
            arg1 = int(labels[sample][4 + slot * 4])
            if cls_id in (BUILD, LIGHTNING):
                truth.append((21 if cls_id == LIGHTNING else 11, tgt // 19, tgt % 19))
            elif cls_id == DOWNGRADE:
                truth.append((13, int(towers[tgt][0]), -1))
            else:
                truth.append((12, int(towers[tgt][0]), arg1))

        state.update(seed=seed, player=player, mode="score_ops", rollouts=rollouts,
                     candidates=[[list(op) for op in cand] for cand in bundles])
        requests.append(state)
        bundles_all.append([tuple(cand) for cand in bundles])
        truth_all.append(tuple(truth))

    print(f"[候选] 平均 {np.mean([len(b) for b in bundles_all]):.0f} 条/状态（head2 宽 {head2_width}）", flush=True)
    payload = "".join(json.dumps(item) + "\n" for item in requests)
    proc = subprocess.run([str(EXE)], input=payload.encode("utf-8"), capture_output=True, env=make_env())
    if proc.returncode != 0:
        print(proc.stderr.decode("utf-8", "replace")[:800])
        return 1
    results = [json.loads(line) for line in proc.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    print(f"[评分] 返回 {len(results)} 条", flush=True)

    covered = agree = act_total = act_covered = 0
    for result, bundles, truth in zip(results, bundles_all, truth_all):
        has_act = len(truth) > 0
        act_total += int(has_act)
        if result.get("ok") != 1:
            continue
        present = truth in bundles
        covered += int(present)
        act_covered += int(present and has_act)
        scored = [(row["score"], idx) for idx, row in enumerate(result["candidates"]) if row["valid"] and row["score"] is not None]
        if not scored or not present:
            continue
        best = max(scored)[1]
        if bundles[best] == truth:
            agree += 1

    total = len(results)
    print(f"\n=== 端到端（{total} 个状态；评分 = 亚军自己的 8 回合 rollout × {rollouts}）===")
    print(f"  覆盖率（真值在候选集里）      : {covered}/{total} = {100 * covered / max(total, 1):.1f}%"
          f"   [有动作] {act_covered}/{act_total} = {100 * act_covered / max(act_total, 1):.1f}%")
    print(f"  条件一致率（在的情况下挑对）  : {agree}/{max(covered, 1)} = {100 * agree / max(covered, 1):.1f}%")
    print(f"  **端到端决策一致率**          : {agree}/{total} = {100 * agree / max(total, 1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
