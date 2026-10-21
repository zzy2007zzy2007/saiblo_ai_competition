"""确定性候选生成器的**动作组覆盖率**扫描：每个头「hold + top-k 类 × 每类位置 top-n」。

这就是"能不能替代亚军规则候选生成器"的直接实验：
  真值动作组（1–3 个 op）能不能被这套确定性候选集合**表示出来**。

对照三件事：
  1. 每个头保留 k 个类、每类 n 个位置 ⇒ 每头 1 + k·n 个 (类,目标) 选项；
  2. 顺序生成成本 = 3 个头相加（不是相乘）；
  3. 若强行物化 = 1 + a + a·b + a·b·c（a=1+k·n），会爆炸。

用法：
    python eval_top3_staged.py
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
NAMES = {BUILD: "建塔", UPGRADE: "升级", DOWNGRADE: "降级", LIGHTNING: "闪电"}
ACT = (BUILD, UPGRADE, DOWNGRADE, LIGHTNING)


def main() -> int:
    device = torch.device("cuda")
    raw = np.load(_REPO / "training_history/distill_top2/q8/gen_ds_64games.npz", allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels, games = data["labels"], data["game"]
    ckpt = torch.load(_REPO / "checkpoints/gen_bundles_3head.pt", map_location="cpu", weights_only=False)
    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(0)
    test_games = rng.permutation(np.unique(games))[:12]
    test_index = np.flatnonzero(np.isin(games, test_games))
    n = len(test_index)
    rows = labels[test_index]

    cls_all, cell_all, tower_all = [], [], []
    with torch.no_grad():
        for start in range(0, n, 512):
            chunk = test_index[start:start + 512]
            cls, cell, tower = model(to_batch(data, chunk, device))
            cls_all.append(torch.stack([F.log_softmax(c, -1) for c in cls], 1).cpu())
            cell_all.append(F.log_softmax(cell, -1).cpu())
            tower_all.append(F.log_softmax(tower, -1).cpu())
    cls_logp = torch.cat(cls_all)        # (N,3,5)
    cell_logp = torch.cat(cell_all)      # (N,2,361)
    tower_logp = torch.cat(tower_all)    # (N,2,48)
    own_mask = torch.from_numpy(data["tower_mask"][test_index]
                               & (data["tower_cont"][test_index][:, :, 0] > 0.5))
    tower_logp = tower_logp.masked_fill(~own_mask.unsqueeze(1), -1e9)

    n_ops = torch.from_numpy(rows[:, 0]).long()
    act = n_ops > 0
    print(f"[测试] {n} 个决策；hold {100.0 * (~act).float().mean():.1f}%；"
          f"1op {int((n_ops == 1).sum())} / 2op {int((n_ops == 2).sum())} / 3op {int((n_ops == 3).sum())}")

    def head_truth(slot: int):
        col = 1 + slot * 4
        return (torch.from_numpy(rows[:, col]).long(),
                torch.from_numpy(rows[:, col + 1]).long())

    def head_cover(slot: int, k_cls: int, n_pos: int) -> torch.Tensor:
        """第 slot 个头的真值 (类,目标) 是否被确定性候选集合覆盖。"""
        truth_cls, truth_tgt = head_truth(slot)
        logp = cls_logp[:, slot]
        # 去掉 hold 后的类排序
        masked = logp.clone()
        masked[:, HOLD] = -1e9
        keep_cls = torch.topk(masked, k=k_cls, dim=1).indices      # (N,k_cls)
        covered = truth_cls == HOLD                                # hold 恒在
        cls_kept = (keep_cls == truth_cls.unsqueeze(1)).any(1)
        for cls_id in ACT:
            sel = (truth_cls == cls_id) & cls_kept
            if not sel.any():
                continue
            if cls_id in (BUILD, LIGHTNING):
                table = torch.topk(cell_logp[:, 1 if cls_id == LIGHTNING else 0],
                                   k=n_pos, dim=1).indices
            else:
                table = torch.topk(tower_logp[:, 1 if cls_id == DOWNGRADE else 0],
                                   k=min(n_pos, 16), dim=1).indices
            covered |= sel & (table == truth_tgt.unsqueeze(1)).any(1)
        return covered

    print("\n=== 确定性命中率：每头 top-k 类 × 每类位置 top-n ===")
    print(f"{'k':>2} {'n':>3} | {'每头候选':>7} {'顺序成本':>8} {'物化条数':>10} | "
          f"{'动作组全对':>10} {'[有动作]':>9} {'首op':>7} {'2op':>7} {'3op':>7}")
    for k_cls in (2, 3, 4):
        for n_pos in (8, 16, 32):
            per_head = 1 + k_cls * n_pos
            seq_cost = 3 * per_head
            material = 1 + (per_head - 1) + (per_head - 1) ** 2 + (per_head - 1) ** 3
            c0 = head_cover(0, k_cls, n_pos)
            c1 = head_cover(1, k_cls, n_pos)
            c2 = head_cover(2, k_cls, n_pos)
            full = c0 & c1 & c2
            print(f"{k_cls:>2} {n_pos:>3} | {per_head:>7} {seq_cost:>8} {material:>10} | "
                  f"{100 * full.float().mean():>9.1f}% {100 * full[act].float().mean():>8.1f}% "
                  f"{100 * c0[act].float().mean():>6.1f}% "
                  f"{100 * full[n_ops == 2].float().mean() if (n_ops == 2).any() else float('nan'):>6.1f}% "
                  f"{100 * full[n_ops == 3].float().mean() if (n_ops == 3).any() else float('nan'):>6.1f}%")

    print("\n=== 推荐档位 k=3 / n=16 的细分（每头 49 个候选）===")
    for slot in range(3):
        c = head_cover(slot, 3, 16)
        truth_cls, _ = head_truth(slot)
        print(f"  slot{slot}: 覆盖 {100 * c.float().mean():.1f}%   "
              f"[真值有该 op 的 {int((truth_cls != HOLD).sum())} 条里覆盖 "
              f"{100 * c[truth_cls != HOLD].float().mean():.1f}%]")

    print("\n=== 只生成第一个 op（不算多 op 组合）的覆盖率 ===")
    for k_cls, n_pos in ((3, 16), (3, 32), (4, 16)):
        c0 = head_cover(0, k_cls, n_pos)
        print(f"  k={k_cls} n={n_pos} 每头 {1 + k_cls * n_pos} 个：首 op 覆盖 "
              f"{100 * c0[act].float().mean():.1f}%（有动作）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
