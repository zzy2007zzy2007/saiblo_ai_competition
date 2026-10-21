"""纯粹测「类网」的好坏，完全排除位置网：
   ① 每个头在其自己的子集上：类覆盖@k vs「永远猜最常见的类」基线
   ② **同一批状态**上横向比较各头（≥2 op 子集上 head1 vs head2）
   ③ 计划级：整个动作组的**类序列**是否都被各自头的 top-k 覆盖（不含位置）
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

HOLD = 0
NAMES = {0: "hold", 1: "建塔", 2: "升级", 3: "降级", 4: "闪电"}


def main() -> int:
    device = torch.device("cuda")
    raw = np.load(_REPO / "training_history/distill_top2/q8/gen_ds_64games.npz", allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels, games = data["labels"], data["game"]
    CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gen_bundles_3head.pt"
    ckpt = torch.load(_REPO / CKPT, map_location="cpu", weights_only=False)
    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[ckpt] {CKPT}")

    rng = np.random.default_rng(0)
    heldout = rng.permutation(np.unique(games))[:12]
    test_index = np.flatnonzero(np.isin(games, heldout))
    rows = labels[test_index]
    n = len(test_index)
    parts = []
    with torch.no_grad():
        for start in range(0, n, 512):
            chunk = test_index[start:start + 512]
            heads, _, _ = model(to_batch(data, chunk, device))
            parts.append(torch.stack([F.log_softmax(c, -1) for c in heads], 1).cpu())
    cls_logp = torch.cat(parts)

    def cls_rank(slot: int) -> torch.Tensor:
        """返回每个样本里真类的排名（0 = 第一名；hold 已被排除）。"""
        logp = cls_logp[:, slot].clone()
        logp[:, HOLD] = -1e9
        order = torch.argsort(logp, dim=1, descending=True)
        truth = torch.from_numpy(rows[:, 1 + slot * 4]).long()
        # 排名 = 真类在 order 里的位置
        pos = (order == truth.unsqueeze(1)).float().argmax(1)
        return pos

    # ---- ① 每个头自己子集上：@k vs 多数类基线 ----
    print("\n=== ① 每个头（纯类，无位置）：@k 与「永远猜最常见类」基线 ===")
    print(f"{'头':>6} {'子集样本':>8} {'最常见类':>10} {'基线':>7} {'@1':>7} {'@2':>7} {'@3':>7} {'@1−基线':>9}")
    for slot in range(3):
        truth = torch.from_numpy(rows[:, 1 + slot * 4]).long()
        sel = truth != HOLD
        if sel.sum() == 0:
            continue
        sub = truth[sel]
        values, counts = torch.unique(sub, return_counts=True)
        majority = int(values[counts.argmax()])
        base = float(counts.max()) / float(sel.sum())
        rank = cls_rank(slot)[sel]
        at = {k: float((rank < k).float().mean()) for k in (1, 2, 3)}
        print(f"{'head'+str(slot+1):>6} {int(sel.sum()):>8} {NAMES[majority]:>10} "
              f"{100*base:>6.1f}% {100*at[1]:>6.1f}% {100*at[2]:>6.1f}% {100*at[3]:>6.1f}% "
              f"{100*(at[1]-base):>+8.1f}pp")

    # ---- ② 同一批状态上横向比较 ----
    print("\n=== ② 同一批状态上比较（消除子集差异）===")
    n_ops = torch.from_numpy(rows[:, 0]).long()
    for name, mask in (("≥2 op 子集", n_ops >= 2), ("3 op 子集", n_ops == 3)):
        if not mask.any():
            continue
        print(f"  {name}（{int(mask.sum())} 条）：")
        for slot in range(3):
            truth = torch.from_numpy(rows[:, 1 + slot * 4]).long()
            sel = mask & (truth != HOLD)
            if sel.sum() < 5:
                continue
            rank = cls_rank(slot)[sel]
            print(f"    head{slot + 1}: n={int(sel.sum()):>4}  "
                  f"@1={100*float((rank<1).float().mean()):.1f}%  "
                  f"@2={100*float((rank<2).float().mean()):.1f}%  "
                  f"@3={100*float((rank<3).float().mean()):.1f}%")

    # ---- ③ 计划级：整个动作组的类序列是否都被覆盖 ----
    print("\n=== ③ 计划级：动作组的全部 op 的「类」都被各自头 top-k 覆盖（不含位置）===")
    act = n_ops > 0
    ranks = [cls_rank(slot) for slot in range(3)]
    truth_cls = [torch.from_numpy(rows[:, 1 + slot * 4]).long() for slot in range(3)]
    for k in (1, 2, 3):
        ok = torch.ones(n, dtype=torch.bool)
        for slot in range(3):
            slot_ok = (truth_cls[slot] == HOLD) | (ranks[slot] < k)
            ok &= slot_ok
        print(f"  top-{k} 类：有动作命中 {100 * ok[act].float().mean():.1f}%"
              f"（{int(ok[act].sum())}/{int(act.sum())}）；全量 {100 * ok.float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
