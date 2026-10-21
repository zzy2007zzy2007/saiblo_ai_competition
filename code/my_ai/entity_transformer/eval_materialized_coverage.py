"""在全量测试集上精确测「物化候选清单里是否含亚军真实动作组」。

与端到端脚本用**完全相同**的候选构造：
  每头 = hold + top-k 类 × 每类位置 top-n（位置取自该类通道）
  清单 = 全 hold + head1 的单 op + (head1 × head2 的前 w 个) 的双 op
只在最后一步判断"真值动作组是否逐字出现在清单里"。

⚠️ 这里不做任何 rollout，所以是全量、无采样噪声的数。

用法：
    python eval_materialized_coverage.py
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
UPGRADE_TREE = {0: (1, 2, 3, 4), 1: (11, 12, 13), 2: (21, 22, 23),
                3: (31, 32, 33), 4: (41, 42, 43)}


def main() -> int:
    device = torch.device("cuda")
    raw = np.load(_REPO / "training_history/distill_top2/q8/gen_ds_64games.npz", allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels, games = data["labels"], data["game"]
    rounds, players = data["round"], data["player"]
    CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gen_bundles_3head.pt"
    ckpt = torch.load(_REPO / CKPT, map_location="cpu", weights_only=False)
    print(f"[ckpt] {CKPT}")
    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(0)
    heldout = rng.permutation(np.unique(games))[:12]
    test_index = np.flatnonzero(np.isin(games, heldout))
    n_ops = labels[test_index, 0]
    act = n_ops > 0
    print(f"[测试] {len(test_index)} 个决策：hold {int((~act).sum())} / "
          f"1op {int((n_ops == 1).sum())} / 2op {int((n_ops == 2).sum())} / 3op {int((n_ops == 3).sum())}")

    # 逐块前向
    cls_parts, cell_parts, tower_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(test_index), 512):
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

    # 需要每状态的塔信息（tower_id / tower_type）——从实体张量反推不方便，改从 npz 里的 plan_feats 拿不到；
    # 这里用「塔下标」表示，真值也用同一套下标，只需要可比较即可。
    def build_options(row: int, slot: int, k_cls: int, n_pos: int, limit: int):
        logp = cls_logp[row, slot].clone()
        logp[HOLD] = -1e9
        keep = torch.topk(logp, k=k_cls).indices.tolist()
        out = [(float(cls_logp[row, slot, HOLD]), ())]
        for cls_id in keep:
            if cls_id in (BUILD, LIGHTNING):
                channel = 1 if cls_id == LIGHTNING else 0
                table = torch.topk(cell_logp[row, channel], k=n_pos).indices.tolist()
                out += [(float(cls_logp[row, slot, cls_id] + cell_logp[row, channel, c]),
                         ((cls_id, c),)) for c in table]
            elif cls_id == DOWNGRADE:
                cand = sorted(((float(tower_logp[row, 1, i]), i) for i in range(tower_logp.shape[2])
                               if float(tower_logp[row, 1, i]) > -1e8), reverse=True)
                out += [(score, ((cls_id, i),)) for score, i in cand[:n_pos]]
            elif cls_id == UPGRADE:
                cand = sorted(((float(tower_logp[row, 0, i]), i) for i in range(tower_logp.shape[2])
                               if float(tower_logp[row, 0, i]) > -1e8), reverse=True)
                for score, i in cand[:3]:
                    out.append((score, ((cls_id, i),)))   # 塔型不计（模型不预测），只匹配塔
        out.sort(key=lambda item: -item[0])
        return out[:limit]

    print(f"\n{'k':>2} {'n':>3} {'head2宽':>7} {'清单条数(均)':>11} | {'覆盖率(全量)':>12} {'[有动作]':>9} "
          f"{'1op':>7} {'2op':>7}")
    for k_cls, n_pos in ((3, 16), (3, 32)):
        for width in (2, 4, 8, 16):
            hit_all = hit_1 = hit_2 = 0
            sizes = []
            for row in range(len(test_index)):
                head1 = build_options(row, 0, k_cls, n_pos, 1 + k_cls * n_pos + 16)
                head2 = build_options(row, 1, k_cls, n_pos, width)
                bundles = {item[1] for item in head1}
                for _, ops1 in head1:
                    if not ops1:
                        continue
                    for _, ops2 in head2:
                        if ops2:
                            bundles.add(ops1 + ops2)
                sizes.append(len(bundles))
                truth = []
                for slot in range(3):
                    cls_id = int(labels[test_index[row]][1 + slot * 4])
                    if cls_id == HOLD:
                        break
                    truth.append((cls_id, int(labels[test_index[row]][2 + slot * 4])))
                present = tuple(truth) in bundles
                hit_all += int(present)
                if n_ops[row] == 1:
                    hit_1 += int(present)
                if n_ops[row] == 2:
                    hit_2 += int(present)
            act_hit = hit_1 + hit_2
            print(f"{k_cls:>2} {n_pos:>3} {width:>7} {np.mean(sizes):>11.0f} | "
                  f"{100 * hit_all / len(test_index):>11.1f}% "
                  f"{100 * act_hit / int(act.sum()):>8.1f}% "
                  f"{100 * hit_1 / int((n_ops == 1).sum()):>6.1f}% "
                  f"{100 * hit_2 / int((n_ops == 2).sum()):>6.1f}%")
    print("\n（3-op 真值不在这里的清单里构造 ⇒ 上限就是 1op+2op 的比例；表里 [有动作] 的分母含 3op）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
