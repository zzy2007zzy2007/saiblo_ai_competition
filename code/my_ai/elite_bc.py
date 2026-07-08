"""Elite Behavior Cloning — supervised learning from top-K individuals' game data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from my_ai.network import AntWarNetwork


# ════════════════════════════════════════════════
# EliteSelector 接口 & 实现
# ════════════════════════════════════════════════


class EliteSelector(ABC):
    """从种群中选择哪些个体的对战数据用于监督学习。"""

    @abstractmethod
    def select(self, fitness: list[float]) -> list[int]:
        """返回被选中的个体 index 列表。"""
        ...


class TopKSelector(EliteSelector):
    """按 fitness 排序取 top-K 个体。"""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def select(self, fitness: list[float]) -> list[int]:
        return sorted(range(len(fitness)), key=lambda i: -fitness[i])[:self.k]


# ════════════════════════════════════════════════
# BC 数据集（含 HOLD 过滤）
# ════════════════════════════════════════════════


class BCDataset(Dataset):
    """从 .npz 加载对战数据，以 p_hold 概率保留 HOLD 回合。

    HOLD 过滤逻辑：全部头都选 HOLD 的回合，以 ``p_hold`` 概率保留。
    默认 p_hold=0.1，保留 10% 的 HOLD 回合，既防止 HOLD 主导 loss，
    又保留少量 HOLD 样本让模型学到何时该停。

    每个 .npz 含一局对战中所有回合的数据：
      - board:  (T, 28, 19, 19) float16
      - stats:  (T, 42)         float16
      - class_: (T, N_heads)    int64（每个头 argmax class, 23=HOLD）
      - map_:   (T, N_heads)    int64（每个头 argmax position, 索引 0…8302）
    """

    HOLD_CLASS = 23

    def __init__(self, npz_paths: list[Path], p_hold: float = 0.1) -> None:
        boards, statss, classes, maps = [], [], [], []
        for p in npz_paths:
            data = np.load(p)
            boards.append(torch.from_numpy(data["board"]).float())
            statss.append(torch.from_numpy(data["stats"]).float())
            classes.append(torch.from_numpy(data["class_"]).long())
            maps.append(torch.from_numpy(data["map_"]).long())
        self.board = torch.cat(boards, dim=0)
        self.stats = torch.cat(statss, dim=0)
        self.class_label = torch.cat(classes, dim=0)
        self.map_label = torch.cat(maps, dim=0)

        # ── HOLD 降采样：全部 HOLD 的回合以 p_hold 保留 ──
        all_hold = (self.class_label == self.HOLD_CLASS).all(dim=1)
        keep_hold = torch.rand(len(self.class_label)) < p_hold
        valid = (~all_hold) | (all_hold & keep_hold)
        n_before = len(self.board)
        self.board = self.board[valid]
        self.stats = self.stats[valid]
        self.class_label = self.class_label[valid]
        self.map_label = self.map_label[valid]
        n_after = len(self.board)
        if n_before > 0 and n_after < n_before:
            print(f"  [BC] HOLD downsampled: {n_before}→{n_after} "
                  f"({100 * (n_before - n_after) // n_before}% removed)")

    def __len__(self) -> int:
        return len(self.board)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "board": self.board[idx],
            "stats": self.stats[idx],
            "class_label": self.class_label[idx],
            "map_label": self.map_label[idx],
        }


# ════════════════════════════════════════════════
# 监督训练（BC = 唯一优化步骤）
# ════════════════════════════════════════════════


def supervised_update(
    model: AntWarNetwork,
    dataset: BCDataset,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    lambda_map: float = 1.0,
    lambda_class: float = 1.0,
    weight_decay: float = 0.0,
) -> dict[str, float | int]:
    """用 elite 数据监督训练 mean 模型。BC 完全替代 ES 梯度。

    Loss = λ_class × CE(class_logits, class_label)  (平均各头)
         + λ_map   × CE(map_logits, map_label)

    返回 {class_loss, map_loss, total_loss, samples}。
    """
    if len(dataset) == 0:
        return {"class_loss": 0.0, "map_loss": 0.0, "total_loss": 0.0, "samples": 0}

    model.train()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_cls = 0.0
    total_map = 0.0
    total = 0.0
    n = 0

    for _ in range(epochs):
        for batch in loader:
            board = batch["board"].to(device)
            stats = batch["stats"].to(device)
            cls_label = batch["class_label"].to(device)
            map_label = batch["map_label"].to(device)

            optimizer.zero_grad()
            output = model(board, stats)

            # Class loss: average over all policy heads
            cls_loss = 0.0
            for i in range(model.num_heads):
                cls_loss += F.cross_entropy(output[f"head{i+1}_logits"], cls_label[:, i])
            cls_loss /= model.num_heads

            # Map loss: shared action_map (B, 23, 19, 19) → (B, 8303)
            map_flat = output["action_map"].view(output["action_map"].size(0), -1)
            map_loss = F.cross_entropy(map_flat, map_label[:, 0])

            loss = lambda_class * cls_loss + lambda_map * map_loss
            loss.backward()
            optimizer.step()

            total_cls += cls_loss.item()
            total_map += map_loss.item()
            total += loss.item()
            n += 1

    model.cpu()
    model.eval()
    return {
        "class_loss": total_cls / n,
        "map_loss": total_map / n,
        "total_loss": total / n,
        "samples": len(dataset),
    }


# ════════════════════════════════════════════════
# 数据写入 & 读取
# ════════════════════════════════════════════════


def write_bc_npz(
    path: Path,
    boards: list[np.ndarray],
    statss: list[np.ndarray],
    class_labels: list[np.ndarray],
    map_labels: list[np.ndarray],
) -> None:
    """保存 board(stats)/class/map 为压缩 .npz 文件。

    board 和 stats 转为 float16 以节省磁盘空间。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        board=np.concatenate(boards, axis=0).astype(np.float16),
        stats=np.concatenate(statss, axis=0).astype(np.float16),
        class_=np.concatenate(class_labels, axis=0),
        map_=np.concatenate(map_labels, axis=0),
    )


def collect_bc_data(bc_dir: Path, selected_indices: list[int], gen: int) -> list[Path]:
    """Glob 匹配 gen_NNNN_ind{idx}_*.npz 返回选中个体的全部 .npz 路径。"""
    paths: list[Path] = []
    for idx in selected_indices:
        paths.extend(sorted(bc_dir.glob(f"gen_{gen:04d}_ind{idx:03d}_*.npz")))
    return paths


def cleanup_gen_npz(bc_dir: Path, gen: int) -> None:
    """删除指定代数所有 gen_NNNN_*.npz 文件（读完 top-K 后清理）。"""
    for f in bc_dir.glob(f"gen_{gen:04d}_*.npz"):
        f.unlink()


# ════════════════════════════════════════════════
# BC 配置
# ════════════════════════════════════════════════


class BCConfig:
    """Elite Behavior Cloning 配置容器。"""

    def __init__(
        self,
        enabled: bool = False,
        k: int = 5,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
        lambda_map: float = 1.0,
        lambda_class: float = 1.0,
        weight_decay: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.enabled = enabled
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_map = lambda_map
        self.lambda_class = lambda_class
        self.weight_decay = weight_decay
        # 设备自动回退：如果参数为 "cuda" 但 CUDA 不可用，则使用 CPU
        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    @classmethod
    def from_args(cls, args) -> "BCConfig":
        """从 argparse.Namespace 创建配置。"""
        return cls(
            enabled=args.bc,
            k=args.bc_k,
            epochs=args.bc_epochs,
            lr=args.bc_lr,
            batch_size=args.bc_batch_size,
            lambda_map=args.bc_lambda_map,
            lambda_class=args.bc_lambda_class,
            weight_decay=args.bc_weight_decay,
            device=args.bc_device,
        )
