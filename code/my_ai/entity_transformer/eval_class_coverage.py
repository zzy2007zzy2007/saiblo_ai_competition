"""类头的**覆盖率**（不是 argmax 准确率）：真值动作类在不在 top-k 个类里。

为什么必须换成这个指标：类 argmax 准确率只有 24.1%（有动作时），
但"候选生成"要的是**覆盖**——真类只要在保留的 k 个类里就够了。
如果 top-2/top-3 的覆盖率很高，那类头**不需要更准**，只需要"不剪掉"。

同时给 joint（类 top-k × 位置 top-n）的首个 op 召回，和"每阶段成本"。
"""
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
    cls_logp = torch.cat(cls_all)          # (N,3,5)
    cell_logp = torch.cat(cell_all)        # (N,2,361)
    tower_logp = torch.cat(tower_all)      # (N,2,48)
    own_mask = torch.from_numpy(data["tower_mask"][test_index]
                               & (data["tower_cont"][test_index][:, :, 0] > 0.5))

    truth_cls = torch.from_numpy(rows[:, 1]).long()
    truth_tgt = torch.from_numpy(rows[:, 2]).long()
    act = truth_cls != HOLD
    print(f"[测试] {n} 个决策；有动作 {int(act.sum())} 条")

    # ---- 1) 类覆盖@k（只在有动作的真值上算，且不含 hold 类）----
    print("\n=== 类覆盖：真类是否落在「去掉 hold 后」的前 k 个类里 ===")
    order = torch.argsort(cls_logp[:, 0], dim=1, descending=True)
    # 去掉 hold 后的排序
    order_no_hold = torch.stack([row[row != HOLD] for row in order])
    for k in (1, 2, 3, 4):
        hit = (order_no_hold[:, :k] == truth_cls.unsqueeze(1)).any(1)
        print(f"  top-{k} 类覆盖：{100 * hit[act].float().mean():.1f}%  （有动作 {int(act.sum())} 条）")
        # 分项
        parts = []
        for c in ACT:
            sel = truth_cls == c
            if sel.any():
                parts.append(f"{NAMES[c]}:{100 * hit[sel].float().mean():.0f}%")
        print(f"        分项 " + "  ".join(parts))

    # ---- 2) joint：类 top-k × 该类位置 top-n ----
    print("\n=== 首个 op 召回（类 top-k × 位置 top-n）与「每阶段候选数」 ===")
    print("  k代表保留几个类，n代表每个类取几个位置；候选数 = 1(hold) + k×n")
    print(f"  {'k':>2} {'n':>3} {'每阶段候选数':>10} {'召回(有动作)':>12}")
    for k in (1, 2, 3):
        for n_pos in (4, 8, 16, 32):
            keep = order_no_hold[:, :k]                       # (N,k) 保留的类
            hit_cls = (keep == truth_cls.unsqueeze(1)).any(1)
            # 位置命中：真类在保留集合里，且真目标在该类的 top-n 内
            hit_pos = torch.zeros(n, dtype=torch.bool)
            for c in ACT:
                sel = truth_cls == c
                if not sel.any():
                    continue
                if c in (BUILD, LIGHTNING):
                    table = torch.topk(cell_logp[:, 1 if c == LIGHTNING else 0], k=n_pos, dim=1).indices
                else:
                    masked = tower_logp[:, 1 if c == DOWNGRADE else 0].masked_fill(~own_mask, -1e9)
                    table = torch.topk(masked, k=min(n_pos, 16), dim=1).indices
                hit_pos |= sel & (table == truth_tgt.unsqueeze(1)).any(1)
            joint = hit_cls & hit_pos
            print(f"  {k:>2} {n_pos:>3} {1 + k * n_pos:>10} {100 * joint[act].float().mean():>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
