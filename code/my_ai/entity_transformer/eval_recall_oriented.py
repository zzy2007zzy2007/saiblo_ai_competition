"""召回导向的候选生成：**类穷举 + 位置网排序裁剪**，测 recall@K。

## 为什么这样设计（由诊断数据决定）
3 头联合概率取 top-K 效果很差（有动作时 K=192 只有 17.8%），诊断显示瓶颈在**类头**：
  · 类准确率：整体 77.2%，但"有动作"时只有 24.1%
  · 位置准确率（类猜对时）：塔 top-2 = 99.2%，格子 top-16 = 81.6%
⇒ 位置网很强、类头很弱。而类别只有 5 种 ⇒ **干脆穷举类别，只用网络给位置排序**。

## 候选集构造（K = 1 + 4n）
    hold                                         1 个
    downgrade @ 我方每座塔（按位置网排序取前 n）   n 个
    upgrade   @ 同上                             n 个
    build     @ 位置网 top-n 格子                n 个
    lightning @ 位置网 top-n 格子                n 个
n = 1/2/4/8/16/32/48 ⇒ K = 5/9/17/33/65/129/193（覆盖用户要的"个位数到一两百"）

指标：**首个 op** 的 recall@K（类别 + 目标都要对）。多 op 的完整动作组另算。
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
CELL_CLASSES = (BUILD, LIGHTNING)
TOWER_CLASSES = (UPGRADE, DOWNGRADE)


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
    print(f"[测试] {n} 个决策")

    # 取位置头输出（只要有正类通道即可，不需要类头）
    cell_all, tower_all = [], []
    with torch.no_grad():
        for start in range(0, n, 512):
            chunk = test_index[start:start + 512]
            _, cell, tower = model(to_batch(data, chunk, device))
            cell_all.append(cell.cpu())
            tower_all.append(tower.cpu())
    cell_all = torch.cat(cell_all)       # (N,2,361) 0=build 1=lightning
    tower_all = torch.cat(tower_all)     # (N,2,48)  0=upgrade 1=downgrade

    rows = labels[test_index]
    truth_cls = torch.from_numpy(rows[:, 1]).long()
    truth_tgt = torch.from_numpy(rows[:, 2]).long()
    truth_ops = torch.from_numpy(rows[:, 0]).long()

    # 我方塔掩码（塔通道上的合法位置）
    own_mask = torch.from_numpy(data["tower_mask"][test_index]
                               & (data["tower_cont"][test_index][:, :, 0] > 0.5))
    tower_logp_down = F.log_softmax(tower_all[:, 1].masked_fill(~own_mask, -1e9), -1)
    tower_logp_up = F.log_softmax(tower_all[:, 0].masked_fill(~own_mask, -1e9), -1)
    cell_logp_build = F.log_softmax(cell_all[:, 0], -1)
    cell_logp_light = F.log_softmax(cell_all[:, 1], -1)
    n_own = own_mask.sum(1)

    print(f"[规模] 真值 hold {100.0 * (truth_cls == HOLD).float().mean():.1f}%, "
          f"有动作 {int((truth_cls != HOLD).sum())} 条；每状态我方塔数中位 {int(n_own.median())}")
    for cls, name in ((DOWNGRADE, "降级"), (BUILD, "建塔"), (UPGRADE, "升级"), (LIGHTNING, "闪电")):
        count = int((truth_cls == cls).sum())
        print(f"  {name}: {count} 条 ({100.0 * count / n:.1f}%)")

    print("\n=== 首个 op 的 recall@K（穷举类别 + 位置网取前 n）===")
    print(f"{'n':>3} {'K':>5} {'全量':>7} {'[有动作]':>9}   分项（有动作命中率）")
    for n_pos in (1, 2, 4, 8, 16, 32, 48):
        k = 1 + 4 * n_pos
        # 逐类取 top-n 位置的（类,目标）集合
        top_down = torch.topk(tower_logp_down, k=n_pos, dim=1).indices
        top_up = torch.topk(tower_logp_up, k=n_pos, dim=1).indices
        top_build = torch.topk(cell_logp_build, k=n_pos, dim=1).indices
        top_light = torch.topk(cell_logp_light, k=n_pos, dim=1).indices

        hit = torch.zeros(n, dtype=torch.bool)
        per_class = {}
        for cls, table in ((DOWNGRADE, top_down), (UPGRADE, top_up),
                           (BUILD, top_build), (LIGHTNING, top_light)):
            is_cls = truth_cls == cls
            in_set = (table == truth_tgt.unsqueeze(1)).any(1)
            hit |= is_cls & in_set
            if is_cls.any():
                per_class[cls] = float(in_set[is_cls].float().mean())
        hit |= truth_cls == HOLD
        act = truth_cls != HOLD
        parts = " ".join(f"{name}:{100 * per_class.get(cls, float('nan')):.0f}%"
                         for cls, name in ((DOWNGRADE, "降级"), (UPGRADE, "升级"),
                                           (BUILD, "建塔"), (LIGHTNING, "闪电")))
        print(f"{n_pos:>3} {k:>5} {100 * hit.float().mean():>6.1f}% "
              f"{100 * hit[act].float().mean():>8.1f}%   {parts}")

    print("\n=== 完整动作组（1–3 个 op 全对）的 recall@K ===")
    print("  做法：第 1 个 op 用上面的集合；再从第 2/3 个头的位置网取同规模候选做笛卡尔积后排序")
    # 简化：对 2-op / 3-op 真值，要求每个 op 都在各自头的 top-n 集合里
    for n_pos in (4, 8, 16, 32):
        k = 1 + 4 * n_pos
        top_down = torch.topk(tower_logp_down, k=n_pos, dim=1).indices
        top_up = torch.topk(tower_logp_up, k=n_pos, dim=1).indices
        top_build = torch.topk(cell_logp_build, k=n_pos, dim=1).indices
        top_light = torch.topk(cell_logp_light, k=n_pos, dim=1).indices
        tables = {DOWNGRADE: top_down, UPGRADE: top_up, BUILD: top_build, LIGHTNING: top_light}
        ok = torch.ones(n, dtype=torch.bool)
        for slot in range(3):
            col = 1 + slot * 4
            cls = torch.from_numpy(rows[:, col]).long()
            tgt = torch.from_numpy(rows[:, col + 1]).long()
            slot_ok = torch.zeros(n, dtype=torch.bool)
            for c, table in tables.items():
                slot_ok |= (cls == c) & (table == tgt.unsqueeze(1)).any(1)
            slot_ok |= cls == HOLD
            ok &= slot_ok
        print(f"  n={n_pos:<3} K={k:<4} 动作组全对 {100 * ok.float().mean():.1f}%  "
              f"[有动作] {100 * ok[truth_ops > 0].float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
