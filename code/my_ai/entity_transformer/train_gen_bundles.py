"""用「动作类网 + 位置网」的方式生成 K 个候选**动作组**，测 recall@K。

## 和上一个脚本的区别（重要，之前搞错了）
上一个脚本只生成**单个 op**，所以 R@64 卡在 91.4%（= hold 82% + 18%×单 op 52%）。
但亚军的一个"动作"其实是一个**动作组**（1–3 个 op），而我们的网络本来就是
**3 个类头 × 位置图**，所以正确的形态是：

    head1 → op1,  head2 → op2,  head3 → op3   （不足 3 个 op 的头输出 HOLD）

⇒ 不需要自回归：每个头独立给出 (类, 位置)，**一个动作组 = 3 个头的一次组合**。

## 候选枚举（"网络输出 + 规则"）
每个头取 top-m 个 (类, 目标)，然后组合成动作组，并用结构规则剪枝：
  - 头 1 是 HOLD ⇒ 头 2/3 必须也是 HOLD（不能"先不动再动"）
  - 头 2 是 HOLD ⇒ 头 3 必须 HOLD
得分 = Σ_头 [log p(类) + log p(目标|类)]，按分排序取前 K 个。

## 指标
recall@K：亚军真实动作组 ∈ 前 K 个候选的比例（K 从 1 扫到 192，覆盖亚军自己的候选量级 ~152）。
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))
from my_ai.entity_transformer.model import (
    ANT_CONT_DIM, N_ANT_BEHAVIORS, N_ANT_KINDS, N_TOWER_TYPES, TOWER_CONT_DIM,
)

TENSOR_KEYS = ("tower_cont", "tower_type", "tower_pos", "tower_mask",
               "ant_cont", "ant_kind", "ant_behavior", "ant_pos", "ant_mask", "state_feats")
INT_KEYS = {"tower_type", "tower_pos", "ant_kind", "ant_behavior", "ant_pos"}
BOOL_KEYS = {"tower_mask", "ant_mask"}
N_TEST = 12
C_CLASS = 5          # 0=hold 1=build 2=upgrade 3=downgrade 4=lightning
N_CELL = 361
MAX_TOWER = 48
CELL_CLASSES = (1, 4)      # 目标是格子
TOWER_CLASSES = (2, 3)     # 目标是"我方第几座塔"
HOLD = 0


class ThreeHeadGenerator(torch.nn.Module):
    """实体编码器 + 3 个动作类头 + 按类分开的位置图（格子 / 塔下标）。"""

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 state_dim: int = 63):
        super().__init__()
        self.tower_cont = torch.nn.Linear(TOWER_CONT_DIM, d_model)
        self.tower_type = torch.nn.Embedding(N_TOWER_TYPES, d_model)
        self.ant_cont = torch.nn.Linear(ANT_CONT_DIM, d_model)
        self.ant_kind = torch.nn.Embedding(N_ANT_KINDS, d_model)
        self.ant_behavior = torch.nn.Embedding(N_ANT_BEHAVIORS, d_model)
        self.pos = torch.nn.Embedding(N_CELL + 1, d_model)
        self.type_emb = torch.nn.Embedding(4, d_model)
        self.state_proj = torch.nn.Linear(state_dim, d_model)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        # 3 个类头（对应 3 个 op 槽位）
        self.head_class = torch.nn.ModuleList(
            [torch.nn.Linear(d_model, C_CLASS) for _ in range(3)])
        # 位置图：按类分开（build/lightning 用格子；upgrade/downgrade 用塔下标）
        self.cell_map = torch.nn.Linear(d_model, len(CELL_CLASSES) * N_CELL)
        self.tower_map = torch.nn.Linear(d_model, len(TOWER_CLASSES) * MAX_TOWER)

    def forward(self, batch):
        size = batch["state_feats"].shape[0]
        device = batch["state_feats"].device
        toks, masks = [], []
        cls = self.cls.expand(size, 1, -1) + self.type_emb.weight[0]
        toks.append(cls)
        masks.append(torch.ones(size, 1, dtype=torch.bool, device=device))
        state = self.state_proj(batch["state_feats"]).unsqueeze(1) + self.type_emb.weight[1]
        toks.append(state)
        masks.append(torch.ones(size, 1, dtype=torch.bool, device=device))
        tower = (self.tower_cont(batch["tower_cont"]) + self.tower_type(batch["tower_type"])
                 + self.pos(batch["tower_pos"]) + self.type_emb.weight[2])
        toks.append(tower)
        masks.append(batch["tower_mask"])
        ant = (self.ant_cont(batch["ant_cont"]) + self.ant_kind(batch["ant_kind"])
               + self.ant_behavior(batch["ant_behavior"]) + self.pos(batch["ant_pos"])
               + self.type_emb.weight[3])
        toks.append(ant)
        masks.append(batch["ant_mask"])
        out = self.norm(self.encoder(torch.cat(toks, 1), src_key_padding_mask=~torch.cat(masks, 1)))
        hidden = out[:, 0]
        class_logits = [head(hidden) for head in self.head_class]
        cell = self.cell_map(hidden).view(size, len(CELL_CLASSES), N_CELL)
        tower = self.tower_map(hidden).view(size, len(TOWER_CLASSES), MAX_TOWER)
        return class_logits, cell, tower


def to_batch(data, index, device):
    batch = {}
    for key in TENSOR_KEYS:
        tensor = torch.from_numpy(np.ascontiguousarray(data[key][index]))
        if key in INT_KEYS:
            tensor = tensor.long()
        elif key in BOOL_KEYS:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        batch[key] = tensor.to(device)
    return batch


def head_label(labels, slot: int):
    """取第 slot 个头 (类, 目标, 目标种类)；不足则 HOLD。"""
    col = 1 + slot * 4
    cls = labels[:, col]
    tgt = labels[:, col + 1]
    return cls, tgt


def enumerate_bundles(class_logps, cell_logp, tower_logp, top_pos: int, top_head: int):
    """每个头取 top 个 (类, 目标) 候选，组合成动作组并按总分排序。

    class_logps: (3, N, C) ; cell_logp: (N, 2, N_CELL) ; tower_logp: (N, 2, MAX_TOWER)
    通道约定：cell[0]=build, cell[1]=lightning；tower[0]=upgrade, tower[1]=downgrade
    """
    n = class_logps[0].shape[0]
    build_cell = cell_logp[:, 0]
    light_cell = cell_logp[:, 1]
    up_tower = tower_logp[:, 0]
    down_tower = tower_logp[:, 1]

    head_entries = []          # [slot][sample] -> 排序后的 [(score, cls, tgt)]
    for slot in range(3):
        cl = class_logps[slot]
        per_sample = []
        for i in range(n):
            entries = [(float(cl[i, HOLD]), HOLD, -1)]
            for c in CELL_CLASSES + TOWER_CLASSES:
                pos_logp = (light_cell if c == 4 else build_cell) if c in CELL_CLASSES else \
                           (up_tower if c == 2 else down_tower)
                top = torch.topk(pos_logp[i], k=min(top_pos, pos_logp.shape[1]))
                for value, tgt in zip(top.values.tolist(), top.indices.tolist()):
                    entries.append((float(cl[i, c]) + value, c, int(tgt)))
            entries.sort(key=lambda item: -item[0])
            per_sample.append(entries[:top_head])
        head_entries.append(per_sample)

    bundles = []
    for i in range(n):
        h1, h2, h3 = head_entries[0][i], head_entries[1][i], head_entries[2][i]
        hold_score = [item[0] for item in h1 if item[1] == HOLD]
        hold2 = [item[0] for item in h2 if item[1] == HOLD]
        hold3 = [item[0] for item in h3 if item[1] == HOLD]
        s_hold = (hold_score[0] if hold_score else 0.0)
        s2 = (hold2[0] if hold2 else 0.0)
        s3 = (hold3[0] if hold3 else 0.0)
        local = [(s_hold + s2 + s3, ())]
        ops1 = [item for item in h1 if item[1] != HOLD]
        ops2 = [item for item in h2 if item[1] != HOLD]
        ops3 = [item for item in h3 if item[1] != HOLD]
        for a in ops1:
            local.append((a[0] + s2 + s3, ((a[1], a[2]),)))
            for b in ops2:
                local.append((a[0] + b[0] + s3, ((a[1], a[2]), (b[1], b[2]))))
                for c in ops3:
                    local.append((a[0] + b[0] + c[0],
                                  ((a[1], a[2]), (b[1], b[2]), (c[1], c[2]))))
        local.sort(key=lambda item: -item[0])
        bundles.append([item[1] for item in local])
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_history/distill_top2/q8/gen_ds_64games.npz")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-games", type=int, default=12)
    parser.add_argument("--top-m", type=int, default=4,
                        help="每个类取多少个目标位置")
    parser.add_argument("--top-head", type=int, default=10,
                        help="每个头保留多少个 (类,目标) 候选（组合前）")
    parser.add_argument("--eval-samples", type=int, default=3000,
                        help="训练中每次评估用的测试样本数（最终评估用全量）")
    parser.add_argument("--out", default="checkpoints/gen_three_head.pt")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels = data["labels"]
    games = data["game"]
    print(f"[数据] {len(labels)} 个决策, {len(np.unique(games))} 局")
    op_counts = labels[:, 0]
    print(f"[标签] op 数分布: " + ", ".join(
        f"{k}个:{100.0 * (op_counts == k).mean():.1f}%" for k in range(4)))

    unique_games = np.unique(games)
    rng = np.random.default_rng(0)
    test_games = rng.permutation(unique_games)[:args.test_games]
    test_mask = np.isin(games, test_games)
    train_index = np.flatnonzero(~test_mask)
    test_index = np.flatnonzero(test_mask)
    print(f"[切分] train {len(train_index)} ({len(unique_games) - args.test_games} 局) / "
          f"test {len(test_index)} ({args.test_games} 局)")
    eval_index = test_index[:: max(1, len(test_index) // args.eval_samples)]

    model = ThreeHeadGenerator(state_dim=data["state_feats"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log_handle = open(args.log, "w", encoding="utf-8") if args.log else None

    def log(message):
        print(message, flush=True)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    # 类别权重：降级占多数 ⇒ 用 inverse-sqrt 重加权，避免只会输出 hold/降级
    all_cls = torch.from_numpy(labels[:, 1]).long()
    freq = torch.bincount(all_cls, minlength=C_CLASS).float()
    weight = (freq.sum() / (freq + 1e-6)) ** 0.5
    weight = weight / weight.mean()
    log(f"[类别权重] " + ", ".join(f"{c}:{w:.2f}" for c, w in enumerate(weight.tolist())))

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(epoch).permutation(train_index)
        total = 0.0
        batches = 0
        t0 = time.time()
        for start in range(0, len(order), args.batch):
            chunk = order[start:start + args.batch]
            batch = to_batch(data, chunk, device)
            class_logits, cell, tower = model(batch)
            loss = 0.0
            for slot in range(3):
                cls, tgt = head_label(labels[chunk], slot)
                cls_t = torch.from_numpy(cls).long().to(device)
                tgt_t = torch.from_numpy(tgt).long().to(device)
                loss = loss + F.cross_entropy(class_logits[slot], cls_t, weight=weight.to(device))
                cell_sel = torch.isin(cls_t, torch.tensor(CELL_CLASSES, device=device))
                if cell_sel.any():
                    channel = torch.where(cls_t[cell_sel] == 4,
                                          torch.ones_like(cls_t[cell_sel]),
                                          torch.zeros_like(cls_t[cell_sel]))
                    logits = cell[cell_sel][torch.arange(int(cell_sel.sum()), device=device), channel]
                    loss = loss + F.cross_entropy(logits, tgt_t[cell_sel])
                tower_sel = torch.isin(cls_t, torch.tensor(TOWER_CLASSES, device=device))
                if tower_sel.any():
                    channel = torch.where(cls_t[tower_sel] == 3,
                                          torch.ones_like(cls_t[tower_sel]),
                                          torch.zeros_like(cls_t[tower_sel]))
                    logits = tower[tower_sel][torch.arange(int(tower_sel.sum()), device=device), channel]
                    # ⚠️ 必须用已搬到 device 的 batch 张量做掩码（拿 numpy 数组去索引 CUDA 张量会报错）
                    mask = batch["tower_mask"][tower_sel] & (batch["tower_cont"][tower_sel][:, :, 0] > 0.5)
                    logits = logits.masked_fill(~mask, -1e9)
                    loss = loss + F.cross_entropy(logits, tgt_t[tower_sel])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        scheduler.step()
        if epoch % 2 == 0 or epoch == args.epochs:
            metrics = evaluate(model, data, eval_index, device, args.top_m, args.top_head)
            log(f"[ep {epoch:3d}] loss={total / max(batches, 1):.4f} "
                + " ".join(f"R@{k}={100 * metrics[k]:.1f}%" for k in (1, 8, 32, 128, 192))
                + f"  [有动作] R@8={100 * metrics['act@8']:.1f}% R@192={100 * metrics['act@192']:.1f}%"
                + f"  ({time.time() - t0:.0f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_path)
    log(f"[保存] {out_path}")

    final = evaluate(model, data, test_index, device, args.top_m, args.top_head)
    log(f"\n=== 全量测试集 recall@K（{len(test_index)} 个决策）===")
    log(f"  hold 占比 {100 * final['hold_rate']:.1f}%")
    log("  K      全量      [有动作时]")
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 192):
        log(f"  {k:<4} {100 * final[k]:6.1f}%   {100 * final.get(f'act@{k}', float('nan')):6.1f}%")
    if log_handle:
        log_handle.close()
    return 0


def evaluate(model, data, index, device, top_pos: int, top_head: int) -> dict:
    model.eval()
    labels = data["labels"]
    class_list, cell_list, tower_list = [], [], []
    with torch.no_grad():
        for start in range(0, len(index), 512):
            chunk = index[start:start + 512]
            class_logits, cell, tower = model(to_batch(data, chunk, device))
            class_list.append(torch.stack([F.log_softmax(c, -1) for c in class_logits], 1).cpu())
            cell_list.append(F.log_softmax(cell, -1).cpu())
            tower_list.append(F.log_softmax(tower, -1).cpu())
    class_logps = torch.cat(class_list).permute(1, 0, 2)     # (3, N, C)
    cell_logp = torch.cat(cell_list)
    tower_logp = torch.cat(tower_list)
    bundles = enumerate_bundles(class_logps, cell_logp, tower_logp, top_pos, top_head)

    # 真值动作组
    truth = []
    for row in labels[index]:
        ops = []
        for slot in range(3):
            col = 1 + slot * 4
            cls = int(row[col])
            if cls == HOLD:
                break
            ops.append((cls, int(row[col + 1])))
        truth.append(tuple(ops))

    results = {}
    act_mask = [len(item) > 0 for item in truth]
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 192):
        hit = np.array([int(item in bundles[i][:k]) for i, item in enumerate(truth)])
        results[k] = float(hit.mean())
        if any(act_mask):
            results[f"act@{k}"] = float(hit[np.array(act_mask)].mean())
    results["hold_rate"] = 1.0 - float(np.mean(act_mask))
    return results


if __name__ == "__main__":
    raise SystemExit(main())
