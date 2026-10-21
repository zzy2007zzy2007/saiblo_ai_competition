"""诊断 head1/head2/head3 的能力差异：是"类"不行还是"位置"不行，以及是否训练不足。

三个问题：
1. 每个头的**类覆盖@k**（真类在不在前 k 个类里）——对照 head1 看是否明显更差；
2. 每个头在类猜对时的**位置覆盖@n**（按类别分别看格子/塔）；
3. 用**同一批状态**（都有 op1 的那些）比较 head1 与 head2，排除"状态分布不同"的解释。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path.insert(0, str(_REPO / "code"))
from my_ai.entity_transformer.train_gen_bundles import ThreeHeadGenerator, to_batch

HOLD, BUILD, UPGRADE, DOWNGRADE, LIGHTNING = 0, 1, 2, 3, 4
NAMES = {HOLD: "hold", BUILD: "建塔", UPGRADE: "升级", DOWNGRADE: "降级", LIGHTNING: "闪电"}
ACT = (BUILD, UPGRADE, DOWNGRADE, LIGHTNING)


def main() -> int:
    device = torch.device("cuda")
    raw = np.load(_REPO / "training_history/distill_top2/q8/gen_ds_64games.npz", allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels, games = data["labels"], data["game"]
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gen_bundles_3head.pt"
    ckpt = torch.load(_REPO / ckpt_name, map_location="cpu", weights_only=False)
    print(f"[ckpt] {ckpt_name}  args={ {k: v for k, v in ckpt.get('args', {}).items() if k in ('epochs', 'batch', 'lr')} }")
    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(0)
    heldout = rng.permutation(np.unique(games))[:12]
    test_index = np.flatnonzero(np.isin(games, heldout))
    rows = labels[test_index]
    n = len(test_index)

    cls_parts, cell_parts, tower_parts = [], [], []
    with torch.no_grad():
        for start in range(0, n, 512):
            chunk = test_index[start:start + 512]
            heads, cell, tower = model(to_batch(data, chunk, device))
            cls_parts.append(torch.stack([F.log_softmax(c, -1) for c in heads], 1).cpu())
            cell_parts.append(F.log_softmax(cell, -1).cpu())
            tower_parts.append(F.log_softmax(tower, -1).cpu())
    cls_logp = torch.cat(cls_parts)
    cell_logp = torch.cat(cell_parts)
    tower_logp = torch.cat(tower_parts)
    own_mask = torch.from_numpy(data["tower_mask"][test_index]
                               & (data["tower_cont"][test_index][:, :, 0] > 0.5))
    tower_logp = tower_logp.masked_fill(~own_mask.unsqueeze(1), -1e9)

    def head_cols(slot: int):
        return (torch.from_numpy(rows[:, 1 + slot * 4]).long(),
                torch.from_numpy(rows[:, 2 + slot * 4]).long())

    # ---- 1) 每个头的类覆盖@k（只在该头真值有动作的样本上）----
    print("\n=== 每个头：真值有动作时，类覆盖@k ===")
    print(f"{'头':>4} {'样本':>6} " + " ".join(f"{'@'+str(k):>7}" for k in (1, 2, 3)))
    for slot in range(3):
        truth_cls, _ = head_cols(slot)
        sel = truth_cls != HOLD
        if not sel.any():
            continue
        logp = cls_logp[:, slot].clone()
        logp[:, HOLD] = -1e9
        order = torch.argsort(logp, dim=1, descending=True)
        line = f"{'head'+str(slot+1):>4} {int(sel.sum()):>6} "
        for k in (1, 2, 3):
            covered = (order[:, :k] == truth_cls.unsqueeze(1)).any(1)
            line += f"{100 * covered[sel].float().mean():>6.1f}% "
        print(line)

    # ---- 2) 位置覆盖（类猜对时）----
    print("\n=== 每个头：类猜对时，位置覆盖@n ===")
    for slot in range(3):
        truth_cls, truth_tgt = head_cols(slot)
        sel = truth_cls != HOLD
        logp = cls_logp[:, slot].clone()
        logp[:, HOLD] = -1e9
        argmax_cls = logp.argmax(1)
        right = sel & (argmax_cls == truth_cls)
        if not right.any():
            continue
        print(f"  head{slot + 1}: 类猜对 {int(right.sum())} 条")
        for cls_id in ACT:
            sub = right & (truth_cls == cls_id)
            if sub.sum() < 3:
                continue
            if cls_id in (BUILD, LIGHTNING):
                pos = cell_logp[:, 1 if cls_id == LIGHTNING else 0]
            else:
                pos = tower_logp[:, 1 if cls_id == DOWNGRADE else 0]
            hits = []
            for n_pos in (1, 4, 16, 32):
                table = torch.topk(pos, k=min(n_pos, pos.shape[1]), dim=1).indices
                hits.append(f"@{n_pos}={100 * (table == truth_tgt.unsqueeze(1)).any(1)[sub].float().mean():.0f}%")
            print(f"    {NAMES[cls_id]:<3} n={int(sub.sum()):>4}  " + "  ".join(hits))

    # ---- 3) 同一批状态上比较 head1 vs head2 ----
    print("\n=== 同一批状态（真值有 op1）上，head1 vs head2 的类覆盖 ===")
    truth0, _ = head_cols(0)
    same = truth0 != HOLD
    for slot in (0, 1):
        truth_cls, _ = head_cols(slot)
        sel = same & (truth_cls != HOLD)
        logp = cls_logp[:, slot].clone()
        logp[:, HOLD] = -1e9
        order = torch.argsort(logp, dim=1, descending=True)
        line = f"  head{slot + 1}（其中真值有该 op 的 {int(sel.sum())} 条）: "
        for k in (1, 2, 3):
            covered = (order[:, :k] == truth_cls.unsqueeze(1)).any(1)
            line += f"@ {k}={100 * covered[sel].float().mean():.1f}%  "
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
