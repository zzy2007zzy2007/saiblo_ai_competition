"""头间不对称分配：把"物化候选条数"压到亚军量级（~150-250），看覆盖率掉多少。

需要区分两个 K：
  · K_head  = 每个头各自的选项数（1 个 hold + 4 类 × n 个位置）
  · K_bundle= 你要交给搜索的**显式动作组条数** = 1 + 4n1 + 4n1·4n2 + 4n1·4n2·4n3（按"头1 hold ⇒ 后面必须 hold"剪枝）

三个头都用同一个宽集合会相乘爆炸（129³≈215 万）；所以给头 1 宽、头 2/3 窄。
本脚本扫若干 (n1,n2,n3) 组合，同时报 **K_bundle** 和 **覆盖率**。
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

    # 每个头各自的位置图（3 头结构里每个头没有独立位置图 ⇒ 这里用同一个位置头，
    # 但用"第几个头"的类分布来区分，评估时按头分别取 top-n）
    cell_all, tower_all, cls_all = [], [], []
    with torch.no_grad():
        for start in range(0, n, 512):
            chunk = test_index[start:start + 512]
            cls, cell, tower = model(to_batch(data, chunk, device))
            cls_all.append(torch.stack([F.log_softmax(c, -1) for c in cls], 1).cpu())
            cell_all.append(F.log_softmax(cell, -1).cpu())
            tower_all.append(F.log_softmax(tower, -1).cpu())
    cls_all = torch.cat(cls_all)        # (N,3,5)
    cell_all = torch.cat(cell_all)      # (N,2,361)
    tower_all = torch.cat(tower_all)    # (N,2,48)

    rows = labels[test_index]
    own_mask = torch.from_numpy(data["tower_mask"][test_index]
                               & (data["tower_cont"][test_index][:, :, 0] > 0.5))
    # 每个头、每个类的目标排序（格子/塔）
    top_cell = torch.topk(cell_all, k=64, dim=2).indices        # (N,2,64)
    top_tower = torch.topk(tower_all.masked_fill(~own_mask.unsqueeze(1), -1e9), k=16, dim=2).indices

    def op_in_head_set(slot: int, n_pos: int) -> torch.Tensor:
        """第 slot 个头的 (类,目标) 是否在 top-n_pos 集合里（hold 恒在）。"""
        col = 1 + slot * 4
        cls = torch.from_numpy(rows[:, col]).long()
        tgt = torch.from_numpy(rows[:, col + 1]).long()
        ok = cls == HOLD
        for c in (BUILD, LIGHTNING):
            table = top_cell[:, 1 if c == LIGHTNING else 0][:, :n_pos]
            ok |= (cls == c) & (table == tgt.unsqueeze(1)).any(1)
        for c in (UPGRADE, DOWNGRADE):
            table = top_tower[:, 1 if c == DOWNGRADE else 0][:, :min(n_pos, 16)]
            ok |= (cls == c) & (table == tgt.unsqueeze(1)).any(1)
        return ok

    print(f"[测试] {n} 个决策，真值 hold {100.0 * (rows[:, 1] == HOLD).mean():.1f}%")
    print("\n n1  n2  n3 | 物化条数 | 顺序条数 | 覆盖率  [有动作]  2op全对  3op全对")
    print("             | (相乘)   | (相加)   |")
    settings = [(1, 0, 0), (8, 1, 0), (8, 8, 0), (16, 16, 0), (16, 16, 8),
                (32, 32, 0), (32, 32, 8), (32, 32, 32), (48, 48, 48)]
    for n1, n2, n3 in settings:
        # 剪枝：头1 hold ⇒ 后面必须 hold
        b1 = 4 * n1
        b2 = b1 * 4 * n2
        b3 = b2 * 4 * n3
        k_material = 1 + b1 + b2 + b3
        # 顺序生成：每阶段各自出集合，按阶段决策 ⇒ 成本相加
        k_sequential = (1 + 4 * n1) + (1 + 4 * n2 if n2 else 1) + (1 + 4 * n3 if n3 else 1)
        ok1 = op_in_head_set(0, n1)
        ok2 = op_in_head_set(1, n2) if n2 else (torch.from_numpy(rows[:, 5]).long() == HOLD)
        ok3 = op_in_head_set(2, n3) if n3 else (torch.from_numpy(rows[:, 9]).long() == HOLD)
        ok = ok1 & ok2 & ok3
        act = torch.from_numpy(rows[:, 0]).long() > 0
        two = torch.from_numpy(rows[:, 0]).long() == 2
        three = torch.from_numpy(rows[:, 0]).long() == 3
        print(f"{n1:>3} {n2:>3} {n3:>3} | {k_material:>8} | {k_sequential:>8} | "
              f"{100 * ok.float().mean():>6.1f}% {100 * ok[act].float().mean():>7.1f}%  "
              f"{100 * ok[two].float().mean():>6.1f}%  {100 * ok[three].float().mean():>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
