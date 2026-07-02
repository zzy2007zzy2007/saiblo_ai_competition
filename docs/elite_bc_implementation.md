# Elite BC 实现方案 — 详细代码改动

## 概要

每代 ES 评估后，从 top-K 个体的对战中提取 (state, action) 数据，用监督学习**替代** ES 梯度更新来优化 mean 模型。

```
传统 ES:
  mean → 采样 100 个体 → 评估 → ES 梯度更新 → 新 mean

Elite BC (CEM-style):
  mean → 采样 N 个体 → 评估（全量写 .npz）→ 选 top-K → BC 监督训练 mean → 新 mean
  (ES 梯度更新完全取消，由 BC 替代)
```

**改动文件**：
- `my_ai/elite_bc.py` — **新增**
- `my_ai/es_train.py` — 核心改动
- `my_ai/agent.py` — 小改动

---

## 关键设计决策

针对方案评审发现的 4 个问题，修正如下：

### 问题 1：HOLD 类污染训练数据 ⚠️

top-K 个体大部分输出 HOLD，BC 直接学 HOLD 会强化 HOLD 陷阱。

**解法**：BCDataset 加载时以概率 `p_hold` 保留 HOLD 回合，其余丢弃。默认 `p_hold=0.1`（保留 10%）。既防止 HOLD 主导 loss，又保留少量 HOLD 样本让模型知道什么时候该停。

```python
# 过滤逻辑：全部头都是 HOLD 的回合，以 p_hold 概率保留
HOLD_CLASS = 23
all_hold = (self.class_label == HOLD_CLASS).all(dim=1)
keep_hold = torch.rand(len(self.class_label)) < p_hold
valid = (~all_hold) | (all_hold & keep_hold)
```

典型过滤效果：97% HOLD 的个体 → 保留 ~9.7% HOLD + ~3% 有效 = ~12.7% 回合。loss 中 HOLD 占比从 97% 降到 ~76%，有效操作占比从 3% 提升到 ~24%。

### 问题 2：BC 和 ES 梯度冲突

**解法**：BC **完全替代** ES 的梯度更新。ES 只负责采样和评估，不更新 mean。Elite BC 训练就是唯一的优化步骤。

```
每代:
  1. mean → 采样 N 个体
  2. 评估（全量写 .npz）
  3. 选 top-K
  4. BC 监督训练 mean  ← 唯一更新
  5. 下一代
```

优点：
- 无冲突。ES 的 velocity/momentum 全部废弃（BC 中 torch.optim.Adam 自带动量）
- 框架从 ES 退化为 CEM（Cross-Entropy Method），BC 替代了 CEM 的 MLE 步骤
- `step()` 中 ES 梯度计算代码用 `if bc_enabled` 跳过

### 问题 3：`_eval_worker` 缺 `gen` 参数

函数签名新增 `gen`，写入文件名时用。

### 问题 4：候选浪费

全量写 .npz（所有个体），主进程按 rank 读入 top-K。数据收集在 worker 内是前向传播后"顺便"的操作，零额外时间成本。

```
原始数据量: 每回合 board(28,19,19)×float16≈20KB + stats(42)×4≈168B
            每局 ≈ 250 回合 × 20KB ≈ 5MB
            每个个体 18 局 = 90MB
            100 个体 = 9GB/代（原始）
            savez_compressed 压缩后: ~1-2GB/代
读取后立即清理: 每代占用降至 ~50-100MB（仅当前代 top-K 的压缩数据）
磁盘峰值 ≈ 存 2 代（当前写 + 上一代残留）≈ 2-4GB，安全
```

---

## 1. 新增 `my_ai/elite_bc.py`

```python
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
    @abstractmethod
    def select(self, fitness: list[float]) -> list[int]:
        ...

class TopKSelector(EliteSelector):
    def __init__(self, k: int = 5): self.k = k
    def select(self, fitness):
        return sorted(range(len(fitness)), key=lambda i: -fitness[i])[:self.k]


# ════════════════════════════════════════════════
# BC 数据集（含 HOLD 过滤）
# ════════════════════════════════════════════════

class BCDataset(Dataset):
    """从 .npz 加载对战数据，以 p_hold 概率保留 HOLD 回合。

    HOLD 过滤逻辑：全部头都选 HOLD 的回合，以 `p_hold` 概率保留。
    默认 p_hold=0.1，保留 10% 的 HOLD 回合，既防止 HOLD 主导 loss，
    又保留少量 HOLD 样本让模型学到何时该停。

    每个 .npz 含一局对战中所有回合的数据：
      - board:  (T, 28, 19, 19) float16
      - stats:  (T, 42)         float16
      - class:  (T, N_heads)    int64（每个头 argmax class, 23=HOLD）
      - map:    (T, N_heads)    int64（每个头 argmax position, 索引 0…8302）
    """
    HOLD_CLASS = 23

    def __init__(self, npz_paths: list[Path], p_hold: float = 0.1):
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

    def __len__(self): return len(self.board)
    def __getitem__(self, idx):
        return {"board": self.board[idx], "stats": self.stats[idx],
                "class_label": self.class_label[idx], "map_label": self.map_label[idx]}


# ════════════════════════════════════════════════
# 监督训练（BC = 唯一优化步骤）
# ════════════════════════════════════════════════

def supervised_update(model: AntWarNetwork, dataset: BCDataset, device: torch.device,
                      epochs: int = 3, lr: float = 1e-3, batch_size: int = 64,
                      lambda_map: float = 1.0, lambda_class: float = 1.0) -> dict:
    """用 elite 数据监督训练 mean 模型。BC 完全替代 ES 梯度。

    Loss = λ_class × CE(class_logits, class_label)
         + λ_map   × CE(map_logits, map_label)
    """
    if len(dataset) == 0:
        return {"class_loss": 0.0, "map_loss": 0.0, "total_loss": 0.0, "samples": 0}

    model.train()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_cls = total_map = total = 0.0; n = 0

    for epoch in range(epochs):
        for batch in loader:
            board = batch["board"].to(device)
            stats = batch["stats"].to(device)
            cls_label = batch["class_label"].to(device)
            map_label = batch["map_label"].to(device)

            optimizer.zero_grad()
            output = model(board, stats)

            # Class loss: average over heads
            cls_loss = 0.0
            for i in range(model.num_heads):
                cls_loss += F.cross_entropy(output[f"head{i+1}_logits"], cls_label[:, i])
            cls_loss /= model.num_heads

            # Map loss: shared action_map
            map_flat = output["action_map"].view(output["action_map"].size(0), -1)
            map_loss = F.cross_entropy(map_flat, map_label[:, 0])

            loss = lambda_class * cls_loss + lambda_map * map_loss
            loss.backward(); optimizer.step()

            total_cls += cls_loss.item(); total_map += map_loss.item()
            total += loss.item(); n += 1

    model.cpu(); model.eval()
    return {"class_loss": total_cls / n, "map_loss": total_map / n,
            "total_loss": total / n, "samples": len(dataset)}


# ════════════════════════════════════════════════
# 数据写入 & 读取
# ════════════════════════════════════════════════

def write_bc_npz(path: Path, boards: list[np.ndarray], statss: list[np.ndarray],
                 class_labels: list[np.ndarray], map_labels: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        board=np.concatenate(boards, axis=0).astype(np.float16),
        stats=np.concatenate(statss, axis=0).astype(np.float16),
        class_=np.concatenate(class_labels, axis=0),
        map_=np.concatenate(map_labels, axis=0),
    )

def collect_bc_data(bc_dir: Path, selected_indices: list[int], gen: int) -> list[Path]:
    paths = []
    for idx in selected_indices:
        paths.extend(sorted(bc_dir.glob(f"gen_{gen:04d}_ind{idx:03d}_*.npz")))
    return paths

def cleanup_gen_npz(bc_dir: Path, gen: int):
    """删除指定代数的所有 .npz 文件（读完 top-K 后清理）。"""
    for f in bc_dir.glob(f"gen_{gen:04d}_*.npz"):
        f.unlink()


# ════════════════════════════════════════════════
# BC 配置
# ════════════════════════════════════════════════

class BCConfig:
    def __init__(self, enabled=False, k=5, epochs=3, lr=1e-3, batch_size=64,
                 lambda_map=1.0, lambda_class=1.0, device="cuda"):
        self.enabled = enabled; self.k = k; self.epochs = epochs
        self.lr = lr; self.batch_size = batch_size
        self.lambda_map = lambda_map; self.lambda_class = lambda_class
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    @classmethod
    def from_args(cls, args) -> "BCConfig":
        return cls(enabled=args.bc, k=args.bc_k, epochs=args.bc_epochs, lr=args.bc_lr,
                   batch_size=args.bc_batch_size, lambda_map=args.bc_lambda_map,
                   lambda_class=args.bc_lambda_class, device=args.bc_device)
```


## 2. `es_train.py` 改动

### 2a. 文件头部新增 import

```python
from my_ai.elite_bc import TopKSelector, BCDataset, supervised_update, write_bc_npz, collect_bc_data, cleanup_gen_npz, BCConfig
```

### 2b. `main()` argparse 新增参数

```python
parser.add_argument("--bc", action="store_true", help="enable Elite Behavior Cloning")
parser.add_argument("--bc-k", type=int, default=5)
parser.add_argument("--bc-epochs", type=int, default=3)
parser.add_argument("--bc-lr", type=float, default=1e-3)
parser.add_argument("--bc-batch-size", type=int, default=64)
parser.add_argument("--bc-lambda-map", type=float, default=1.0)
parser.add_argument("--bc-lambda-class", type=float, default=1.0)
parser.add_argument("--bc-device", type=str, default="cuda")
```

### 2c. `ESTrainer.__init__` 新增字段

```python
def __init__(self, ..., bc_config: BCConfig | None = None):
    # ... 原有代码 ...
    self.bc_config = bc_config or BCConfig()
    self.bc_dir: Path | None = None
```

### 2d. `main()` 创建 trainer 后初始化 BC

```python
trainer = ESTrainer(...)
if args.bc:
    bc_dir = out_dir / "bc_data"
    bc_dir.mkdir(parents=True, exist_ok=True)
    trainer.bc_dir = bc_dir
    trainer.bc_config = BCConfig.from_args(args)
```

### 2e. `_eval_worker` 新增 gen 参数 + 全量数据收集

**函数签名**（新增 `ind`, `gen`；去掉 `collect_data` 条件判断——现在所有个体都收集）：

```python
def _eval_worker(params_flat, opp_params_flat, seed, num_heads=3,
                 synthetic_target=None, bc_dir=None, gen=0, ind=0):
```

**函数体内**，初始化 BC buffer：

```python
bc_boards, bc_stats, bc_class_labels, bc_map_labels = [], [], [], []
```

每回合 `ops_us = agent._choose_operations(...)` 之后插入：

```python
# ── BC data collection (all individuals write) ──
feat = agent.feature_extractor.encode_observation(
    state, our_player, np.zeros(agent.max_actions))
bc_boards.append(feat["board"].copy())
bc_stats.append(feat["stats"].copy())
# action_map 是所有头共享的，算一次
map_arg = agent.last_output["action_map"].reshape(-1).argmax().item()
cls_labels = [agent.last_output[f"head{hi+1}_logits"].argmax().item()
              for hi in range(num_heads)]
bc_class_labels.append(cls_labels)
bc_map_labels.append([map_arg] * num_heads)
```

循环结束后写入 .npz（**全量写入，所有个体都写**）：

```python
if bc_dir and bc_boards:
    from my_ai.elite_bc import write_bc_npz
    boards_arr = np.stack(bc_boards, axis=0)
    stats_arr = np.stack(bc_stats, axis=0)
    write_bc_npz(Path(bc_dir) / f"gen_{gen:04d}_ind{ind:03d}_seed{seed}.npz",
                 [boards_arr], [stats_arr],
                 [np.array(bc_class_labels)], [np.array(bc_map_labels)])
```

### 2f. `step()` 中 all_args 传入 ind/gen

BC 模式下所有个体都写 .npz，不需要候选筛选，每回合直接传参数：

```python
all_args = []
for idx in range(self.population_size):
    for k in range(k_per_ind):
        opp_params = opp_params_list[k]
        base_seed = self.seed + generation * self.population_size * self.games_per_individual + (idx * self.games_per_individual + k * 2)
        for s in range(2):  # first/second player
            all_args.append((
                params_list[idx], opp_params, base_seed + s,
                self.num_heads, self.synthetic_target,
                str(self.bc_dir) if self.bc_config.enabled else None,  # bc_dir
                gen,                                                    # gen
                idx,                                                    # ind
            ))
```

### 2g. `step()` 修改：BC 模式下跳过 ES 梯度更新

`step()` 中，在计算出 `fitness` 后，BC 模式下跳过梯度计算：

```python
# 在 step() 末尾，ES 梯度更新代码前：
if not self.bc_config.enabled:
    # 原有 ES 梯度更新代码（不变）
    ranks = np.argsort(np.argsort(fitness))
    shaped = (ranks + 1) / (self.population_size + 1) - 0.5
    shaped_pairs = shaped.reshape(n_noise, 2)
    pair_diffs = shaped_pairs[:, 0] - shaped_pairs[:, 1]
    gradient = (noise.T @ pair_diffs) / (self.population_size * self.sigma)
    if self.velocity is None:
        self.velocity = np.zeros_like(self.mean)
    self.velocity = self.momentum * self.velocity + self.lr * gradient.astype(self.mean.dtype)
    self.mean += self.velocity
    self.model.set_parameters_from_vector(self.mean)
else:
    # BC 模式：ES 不更新 mean，BC 在主循环中做
    pass
```

### 2h. 主循环修改：BC 监督训练

**注意**：BC 训练**在 step() 之后**、**checkpoint 保存之前**进行。step() 内的 ES 梯度已被跳过。

```python
result = trainer.step(gen, pool)

# ── Elite Behavior Cloning（替代 ES 梯度）──
if trainer.bc_config.enabled and trainer.bc_dir is not None:
    fitness_arr = np.array([d["score"] for d in result.get("ind_details", [])])
    if len(fitness_arr) > 0:
        selector = TopKSelector(k=trainer.bc_config.k)
        selected = selector.select(fitness_arr.tolist())
        npz_paths = collect_bc_data(trainer.bc_dir, selected, gen)
        if npz_paths:
            ds = BCDataset(npz_paths)
            if len(ds) > 0:
                bc_ret = supervised_update(trainer.model, ds,
                    device=trainer.bc_config.device,
                    epochs=trainer.bc_config.epochs,
                    lr=trainer.bc_config.lr,
                    batch_size=trainer.bc_config.batch_size,
                    lambda_map=trainer.bc_config.lambda_map,
                    lambda_class=trainer.bc_config.lambda_class)
                trainer.mean = trainer.model.get_parameters_as_vector()
                log.print(key="bc",
                    value=f"top{selected} samples={bc_ret['samples']} "
                          f"loss={bc_ret['total_loss']:.4f} "
                          f"cls={bc_ret['class_loss']:.4f} map={bc_ret['map_loss']:.4f}")
            else:
                log.print(key="bc", value="all-HOLD data filtered, no valid samples")
        else:
            log.print(key="bc", value="no data files for top-K")

    # 清理本代所有 .npz（BC 数据已读入内存，磁盘不再需要）
    cleanup_gen_npz(trainer.bc_dir, gen)

trainer.step_count = gen + 1
```

**注意**：BC 后不重置 velocity（已在 step() 中跳过 ES 梯度更新，velocity 未被修改或无意义）。

### 2i. `save_checkpoint` / `load_checkpoint` 保存 BC 超参

```python
# save 时：
if self.bc_config.enabled:
    data["bc_config"] = {"k": self.bc_config.k, "epochs": self.bc_config.epochs,
                         "lr": self.bc_config.lr, "batch_size": self.bc_config.batch_size,
                         "lambda_map": self.bc_config.lambda_map,
                         "lambda_class": self.bc_config.lambda_class}

# load 时：
if "bc_config" in ckpt and self.bc_config.enabled:
    for k, v in ckpt["bc_config"].items():
        setattr(self.bc_config, k, v)
```


## 3. `agent.py` 改动

**`NeuralAgent.__init__`** 新增 `self.last_output = None`

**`NeuralAgent._choose_operations`** 末尾新增 `self.last_output = output`


## 4. 变更汇总

| 文件 | 改动 |
|------|------|
| `my_ai/elite_bc.py` | 新增：EliteSelector, BCDataset（含 HOLD 过滤）, supervised_update, write_bc_npz, collect_bc_data, BCConfig |
| `my_ai/es_train.py` | import / args / _eval_worker 全量收集 / step() 跳过 ES 梯度 / __init__ / main loop BC 训练 / checkpoint |
| `my_ai/agent.py` | `__init__` + `self.last_output = None`，`_choose_operations` 末尾保存 |


## 5. 使用方式

```bash
# 启用 BC（默认 top-5, 3 epoch, lr=1e-3, 自动 GPU）:
python code/my_ai/es_train.py --pop-size 64 --games 18 --workers 24 --bc

# 自定义:
python code/my_ai/es_train.py --pop-size 64 --games 18 --workers 24 --bc \
    --bc-k 10 --bc-epochs 5 --bc-lr 5e-4 --bc-batch-size 128

# CPU only:
python code/my_ai/es_train.py --pop-size 64 --games 18 --workers 24 --bc --bc-device cpu

# 不启用 BC（完全恢复传统 ES）:
python code/my_ai/es_train.py --pop-size 64 --games 18 --workers 24
```


## 6. 每代流程

```
1. mean → 采样 N 个体（mirrored sampling，和原来一样）
2. 所有 N 个体打比赛 + 全量写 .npz（board/stats/class/map）
3. 所有 fitness 返回
4. 选 top-K（按 fitness 排序）
5. 读取 bc_data/ 中 top-K 的 .npz
6. BCDataset 过滤掉全 HOLD 回合
7. GPU 上 Adam 监督训练 mean 模型 3 epoch
8. 更新 mean，进入下一代
   （ES 梯度更新完全跳过）
```

**额外时间开销**：
- 写 .npz（全量压缩 ~1-2GB/代，读完即删，峰值 2-4GB）
- GPU 训练（45K 样本过滤后 ~5-10K，3 epoch）：~1-2s
- 总代时间增加 ~5s（~2%），几乎可忽略
