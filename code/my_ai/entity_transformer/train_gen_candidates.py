"""用实体 Transformer 生成「候选动作」，用 **recall@K** 评估 —— 目标是替代亚军的规则候选生成器。

和前面所有实验的区别（重要）：
- 之前做的是"给候选打分/估值"，候选集来自亚军自己的 `generate_root_plans`；
- 这里做的是**生成候选**：网络自己吐出 K 个候选动作，只要亚军真实选择的那个在里面就算命中。

输出空间（因子化，来自实测分布）：
  cls ∈ {hold, build, upgrade, downgrade, lightning}
  target = 格子 id（build/lightning）或"我方第几座塔"（upgrade/downgrade）
  候选打分 = log p(cls) + log p(target | cls)（hold 只有一项）

指标：
  recall@K = P(真实动作 ∈ 前 K 个候选)
  基线：① 只给 {hold}；② 用训练集边际分布排序（不用状态信息）；③ 网络

用法：
    python train_gen_candidates.py --data training_history/distill_top2/q8/gen_ds_64games.npz \
        --epochs 15 --test-games 12
"""
from __future__ import annotations

import argparse
import json
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
N_CELL = 361
MAX_TOWER = 48
N_CLASS = 5


class CandidateGenerator(torch.nn.Module):
    """实体编码器 + 因子化动作头（类别 / 格子 / 我方塔下标）。"""

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 state_dim: int = 63):
        super().__init__()
        self.tower_cont = torch.nn.Linear(TOWER_CONT_DIM, d_model)
        self.tower_type = torch.nn.Embedding(N_TOWER_TYPES, d_model)
        self.ant_cont = torch.nn.Linear(ANT_CONT_DIM, d_model)
        self.ant_kind = torch.nn.Embedding(N_ANT_KINDS, d_model)
        self.ant_behavior = torch.nn.Embedding(N_ANT_BEHAVIORS, d_model)
        self.pos = torch.nn.Embedding(N_CELL + 1, d_model)
        self.type_emb = torch.nn.Embedding(4, d_model)   # 0=CLS 1=STATE 2=TOWER 3=ANT
        self.state_proj = torch.nn.Linear(state_dim, d_model)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head_class = torch.nn.Linear(d_model, N_CLASS)
        self.head_cell = torch.nn.Linear(d_model, N_CELL)
        self.head_tower = torch.nn.Linear(d_model, MAX_TOWER)

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
        x = torch.cat(toks, dim=1)
        keep = torch.cat(masks, dim=1)
        out = self.norm(self.encoder(x, src_key_padding_mask=~keep))
        hidden = out[:, 0]
        return self.head_class(hidden), self.head_cell(hidden), self.head_tower(hidden)


def to_batch(data, index, device):
    batch = {}
    for key in TENSOR_KEYS:
        array = data[key][index]
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if key in INT_KEYS:
            tensor = tensor.long()
        elif key in BOOL_KEYS:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        batch[key] = tensor.to(device)
    return batch


def own_tower_mask(data, index, device):
    """哪些"我方塔下标"在当前状态里是合法的（用 tower_cont 的 is_own 推）。"""
    tower_cont = data["tower_cont"][index]       # (B,48,3)
    mask = data["tower_mask"][index] & (tower_cont[:, :, 0] > 0.5)
    return torch.from_numpy(mask).to(device)


def evaluate(model, data, index, device, ks=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    model.eval()
    labels = data["labels"]
    n = len(index)
    class_logits, cell_logits, tower_logits = [], [], []
    with torch.no_grad():
        for start in range(0, n, 1024):
            chunk = index[start:start + 1024]
            c, ce, t = model(to_batch(data, chunk, device))
            class_logits.append(c.cpu())
            cell_logits.append(ce.cpu())
            tower_logits.append(t.cpu())
    class_logits = torch.cat(class_logits)
    cell_logits = torch.cat(cell_logits)
    tower_logits = torch.cat(tower_logits)

    log_p_class = F.log_softmax(class_logits, dim=-1)                    # (N,5)
    log_p_cell = F.log_softmax(cell_logits, dim=-1)                      # (N,361)
    tower_mask = data["tower_mask"][index] & (data["tower_cont"][index][:, :, 0] > 0.5)
    tower_mask_t = torch.from_numpy(tower_mask)
    logits_masked = tower_logits.masked_fill(~tower_mask_t, -1e9)
    log_p_tower = F.log_softmax(logits_masked, dim=-1)

    truth_class = torch.from_numpy(labels[index][:, 1]).long()
    truth_tgt = torch.from_numpy(labels[index][:, 2]).long()
    truth_kind = torch.from_numpy(labels[index][:, 3]).long()

    # 候选打分：hold 单独一条，其余 = log p(class) + log p(target|class)
    top_cell_per_class = torch.topk(log_p_cell, k=32, dim=-1).indices  # (N,32)
    top_tower_per_class = torch.topk(log_p_tower, k=16, dim=-1).indices
    candidate_scores = []
    candidate_ids = []
    hold_scores = log_p_class[:, 0:1]
    candidate_scores.append(hold_scores)
    candidate_ids.append(torch.zeros(len(index), 1, dtype=torch.long) - 1)  # -1 表示 hold
    for cls_id in (1, 2, 3, 4):
        base = log_p_class[:, cls_id:cls_id + 1]
        if cls_id in (1, 4):   # 格子
            scores = base + torch.gather(log_p_cell, 1, top_cell_per_class)
            ids = top_cell_per_class.clone()
        else:                  # 塔下标
            scores = base + torch.gather(log_p_tower, 1, top_tower_per_class)
            ids = top_tower_per_class.clone()
        candidate_scores.append(scores)
        candidate_ids.append(ids * 0 + cls_id * 10000 + ids)   # 编码 (cls, tgt)
    all_scores = torch.cat(candidate_scores, dim=1)            # (N, 1 + 4*64)
    all_ids = torch.cat(candidate_ids, dim=1)
    order = torch.argsort(all_scores, dim=-1, descending=True)
    sorted_ids = torch.gather(all_ids, 1, order)

    truth_id = torch.where(
        truth_class == 0, torch.zeros_like(truth_class) - 1,
        truth_class * 10000 + truth_tgt)
    results = {"n": len(index)}
    act_truth = truth_class > 0
    for k in ks:
        top_k = sorted_ids[:, :k]
        hit = (top_k == truth_id.unsqueeze(1)).any(dim=1)
        results[f"recall@{k}"] = hit.float().mean().item()
        # 只在"真实有动作"的决策上算召回——这才是替代规则生成器时要看的数
        if act_truth.any():
            results[f"recall_act@{k}"] = hit[act_truth].float().mean().item()
    # 分类/目标单点准确率（诊断用）
    results["class_acc"] = (class_logits.argmax(-1) == truth_class).float().mean().item()
    # 诊断：目标位置本身的准确率（给定类别正确的前提下）
    cell_classes = torch.tensor([1, 4])
    tower_classes = torch.tensor([2, 3])
    cell_mask = cell_classes.unsqueeze(0).eq(truth_class.unsqueeze(1)).any(1)
    tower_mask_sel = tower_classes.unsqueeze(0).eq(truth_class.unsqueeze(1)).any(1)
    if cell_mask.any():
        results["tgt_acc_cell"] = (log_p_cell.argmax(-1)[cell_mask] == truth_tgt[cell_mask]).float().mean().item()
    else:
        results["tgt_acc_cell"] = float("nan")
    if tower_mask_sel.any():
        results["tgt_acc_tower"] = (log_p_tower.argmax(-1)[tower_mask_sel]
                                    == truth_tgt[tower_mask_sel]).float().mean().item()
    else:
        results["tgt_acc_tower"] = float("nan")
    results["hold_rate_truth"] = (truth_class == 0).float().mean().item()
    results["n_ops_mean"] = float(labels[index][:, 0].mean())
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_history/distill_top2/q8/gen_ds_64games.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--test-games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/gen_candidates.pt")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    labels = data["labels"]
    games = data["game"]
    print(f"[数据] {len(labels)} 个决策, {len(np.unique(games))} 局, device={device}")
    print(f"[标签] hold 占比 {100.0 * (labels[:, 1] == 0).mean():.1f}%, "
          f"平均 op 数 {labels[:, 0].mean():.2f}")

    unique_games = np.unique(games)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_games)
    test_games = shuffled[:args.test_games]
    test_mask = np.isin(games, test_games)
    train_index = np.flatnonzero(~test_mask)
    test_index = np.flatnonzero(test_mask)
    print(f"[切分] 按局: train {len(train_index)} ({len(unique_games) - args.test_games} 局) / "
          f"test {len(test_index)} ({args.test_games} 局)")

    model = CandidateGenerator(d_model=args.d_model, n_layers=args.layers,
                               state_dim=data["state_feats"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log_handle = open(args.log, "w", encoding="utf-8") if args.log else None

    def log(message: str) -> None:
        print(message, flush=True)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed * 100 + epoch).permutation(train_index)
        total_loss = 0.0
        batches = 0
        t0 = time.time()
        for start in range(0, len(order), args.batch):
            chunk = order[start:start + args.batch]
            batch = to_batch(data, chunk, device)
            class_logits, cell_logits, tower_logits = model(batch)
            truth_class = torch.from_numpy(labels[chunk][:, 1]).long().to(device)
            truth_tgt = torch.from_numpy(labels[chunk][:, 2]).long().to(device)
            truth_kind = torch.from_numpy(labels[chunk][:, 3]).long().to(device)

            loss = F.cross_entropy(class_logits, truth_class)
            cell_mask = torch.isin(truth_class, torch.tensor([1, 4], device=device))
            if cell_mask.any():
                loss = loss + F.cross_entropy(cell_logits[cell_mask], truth_tgt[cell_mask])
            tower_mask = torch.isin(truth_class, torch.tensor([2, 3], device=device))
            if tower_mask.any():
                mask = own_tower_mask(data, chunk, device)
                # ⚠️ 没有自家塔的行整行都被 mask（-1e9），算 CE 会炸出天文数字的 loss
                has_tower = mask.any(dim=1)
                tower_sel = tower_mask & has_tower
                if tower_sel.any():
                    masked_logits = tower_logits.masked_fill(~mask, -1e9)
                    loss = loss + F.cross_entropy(masked_logits[tower_sel], truth_tgt[tower_sel])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        if epoch % 3 == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate(model, data, test_index, device)
            log(f"[ep {epoch:3d}] loss={total_loss / max(batches, 1):.4f} "
                f"class_acc={100 * metrics['class_acc']:.1f}% "
                f"(hold基线 {100 * metrics['hold_rate_truth']:.1f}%) "
                + " ".join(f"R@{k}={100 * metrics[f'recall@{k}']:.1f}%" for k in (1, 4, 8, 16, 32, 64))
                + "  [有动作时] "
                + " ".join(f"R@{k}={100 * metrics.get(f'recall_act@{k}', float('nan')):.1f}%"
                           for k in (4, 8, 16, 32))
                + f"  tgt_cell={100 * metrics['tgt_acc_cell']:.1f}% "
                  f"tgt_tower={100 * metrics['tgt_acc_tower']:.1f}%"
                  f"  ({time.time() - t0:.0f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "state_dim": int(data["state_feats"].shape[1])}, out_path)
    log(f"[保存] {out_path}")

    # ---- 基线：状态无关的边际排序（不看棋盘，只按训练集里 (类别,目标) 的出现频率）----
    train_labels = labels[train_index]
    test_labels = labels[test_index]
    hold_rate = float((test_labels[:, 1] == 0).mean())
    counts: dict[tuple[int, int], int] = {}
    for row in train_labels:
        key = (int(row[1]), int(row[2]))
        counts[key] = counts.get(key, 0) + 1
    ranked = [pair for pair, _ in sorted(counts.items(), key=lambda item: -item[1])]
    print(f"\n[基线] 只给 {{hold}}: recall = {100 * hold_rate:.1f}%")
    print("[基线] 状态无关的边际排序（hold 打头，其余按训练集频率）:")
    test_pairs = [(int(row[1]), int(row[2])) for row in test_labels]
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        top = set(ranked[:max(0, k - 1)])
        hits = sum(1 for pair in test_pairs if pair[0] == 0 or pair in top)
        act_hits = sum(1 for pair in test_pairs if pair[0] != 0 and pair in top)
        act_total = sum(1 for pair in test_pairs if pair[0] != 0)
        print(f"   K={k:<4} recall {100 * hits / len(test_pairs):5.1f}%   "
              f"[有动作时] {100 * act_hits / max(act_total, 1):5.1f}%")
    if log_handle:
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
