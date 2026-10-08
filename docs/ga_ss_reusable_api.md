# GA + ss_train 混合训练 — 可复用接口文档

## 概述

手写 `ga_ss_train.py`，自己实现 GA 核心逻辑（选择、交叉），复用 ss_train.py 中的评估、数据收集、BC 训练等模块。

### 分工

| 部分 | 谁写 | 说明 |
|------|------|------|
| GA 核心（选择、交叉、主循环骨架） | **你** | 手写 |
| 个体评估（_eval_worker） | **复用 ss_train** | 不动 |
| 数据加载（SSDataset） | **复用 ss_train** | 不动 |
| BC 底层更新（ss_supervised_update） | **复用 ss_train** | 不动，作为 bc_train 的内部实现 |
| **BC 统一接口（bc_train）** | **我写** | 封装 ss_supervised_update，输入→输出参数向量 |
| 数据 IO（write_npz/collect_npz） | **复用 ss_train** | 不动 |
| 对手池（Leaderboard） | **复用 leaderboard.py** | 不动 |
| 存档/日志/参数/Ctrl+C 等样板 | **我写** | 提给 ga_ss_train.py |
| 接口汇总文件（ga_ss_interface.py） | **我写** | import 中转 |

### 使用方式

ga_ss_train.py 只 import 已有模块，不复制代码：

```python
from my_ai.ss_train import _eval_worker, SSDataset
from my_ai.ss_train import write_npz, collect_npz, save_checkpoint
from my_ai.ss_train import build_eval_args, run_eval, reload_config
from my_ai.leaderboard import Leaderboard
from my_ai.ga_ss_boilerplate import bc_train  # 我封装的 BC 统一接口
# + 其他样板
```

---

## 一、直接可复用的函数

### 1.1 `_eval_worker` — 评估个体（单局比赛）

```python
def _eval_worker(
    params_flat: np.ndarray,        # 模型参数向量
    opp_params_flat: np.ndarray,    # 对手模型参数向量
    seed: int,                      # 随机种子（决定先/后手）
    num_heads: int = 3,             # 策略头数
    bc_dir: str | Path | None = None,  # 保存.npz 的目录，None=不保存
    gen: int = 0,                   # 当前代数（文件名用）
    ind: int = 0,                   # 个体编号（文件名用）
    action_dropout: float = 0.0,    # 动作丢弃率
    small: bool = False,            # 是否用小模型
) -> dict:
    """运行一局比赛，返回 {'score': 0.0|0.5|1.0, 'our_player': 0|1}"""
```

**注意**：函数内部会设置 `CUDA_VISIBLE_DEVICES=""` / `OMP_NUM_THREADS=1`，worker 纯 CPU。

### 1.2 `build_eval_args` — 构建评估任务列表

```python
def build_eval_args(
    params_list: list[np.ndarray],  # 所有个体的参数向量
    opp_params_list: list[np.ndarray],  # 对手参数向量
    pop_size: int,
    games: int,                     # 每人打多少局（偶数）
    num_heads: int,
    bc_dir: str | None,             # None = 不保存数据
    gen: int,
    seed: int,
    only_idx: list[int] | None = None,  # None=评估所有个体
    seed_offset: int = 0,
    action_dropout: float = 0.0,
    small: bool = False,
) -> list[tuple]:
    """返回可传给 pool.starmap(_eval_worker, ...) 的元组列表"""
```

返回元组数 = `pop_size × len(opp_params_list) × 2`（每个对手先手/后手各一局）。

### 1.3 `run_eval` — 提交并收集评估结果

```python
def run_eval(
    pool: mp.Pool,
    all_args: list[tuple],
) -> list[dict]:
    """提交任务到进程池，返回每局结果列表 [{'score': ..., 'our_player': ...}, ...]"""
```

**注意**：内部处理 `mp.TimeoutError` 和 `BrokenProcessPool`。`KeyboardInterrupt` 通过全局变量 `interrupted` 传递。

### 1.4 `SSDataset` — 加载游戏帧数据

```python
class SSDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[Path],
        p_hold: float = 1.0,            # HOLD 帧保留比例
        seed: int = 42,
        oversample_alpha: float = 0.0,  # 0=关闭过采样
        oversample_max_dup: int = 20,
    ):
        """加载 .npz 文件，HOLD 降采样 + 稀有类过采样"""
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> dict:
        # 返回:
        #   board:       (28, 19, 19) float
        #   stats:       (42,) float
        #   class_label: (N_heads,) int64  (argmax class per head)
        #   action_map:  (24, 19, 19) float
        #   head_logits: (N_heads, 24) float
        #   value:       (2,) float32  (hp_us, hp_opp)  可选
```

### 1.5 `ss_supervised_update` — BC 训练

```python
def ss_supervised_update(
    model: AntWarNetwork,
    dataset: SSDataset,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    lambda_class: float = 1.0,
    lambda_map: float = 1.0,
    lambda_div: float = 0.0,
    lambda_soft: float = 0.0,
    soft_temperature: float = 1.0,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
    bias_decay: float = 0.0,
    class_weights: torch.Tensor | None = None,
    log = None,
) -> dict:
    """训练模型（原地修改），返回 {class_loss, map_loss, total_loss, samples}"""
```

**注意**：函数内部会把 model 放到 device 上训练，训练完移回 CPU 并设为 eval 模式。

### 1.6 `bc_train` — BC 统一接口（**我写**）

在 `ss_supervised_update` 之上封装一层，把"输入参数向量 → 输出参数向量"做成纯函数接口。
变异、交叉、常规迭代共用一个 BC 入口。

```python
def bc_train(
    init_params: np.ndarray,            # 起始参数向量
    model_template: AntWarNetwork,      # 模型结构模板（clone 用）
    dataset: SSDataset,                 # 训练数据
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    lambda_class: float = 1.0,
    lambda_map: float = 1.0,
    lambda_div: float = 0.0,
    lambda_soft: float = 0.0,
    bias_decay: float = 0.0,
    log = None,
) -> np.ndarray:
    """用 BC 在 dataset 上训练，从 init_params 开始，返回训练后的参数向量。

    内部实现：
      1. model = create_model(...)  # 新建一个模板模型
      2. model.load_state_dict(...)  # 用 init_params 初始化
      3. ss_supervised_update(model, dataset, ...)  # 原有的训练循环
      4. return model.get_parameters_as_vector()
    
    不修改传入的 model_template，也不修改 init_params。
    """
```

#### 使用场景

| 场景 | init_params | dataset |
|------|------------|---------|
| 常规 GA 迭代（更新 mean） | mean 参数 | top-K 精英的比赛数据 |
| **Function-Space Crossover**（你的想法） | 任意起点 | **两个亲本的数据 pooled 在一起** |
| 变异后微调 | 变异后的参数向量 | 原始训练数据 |
| 从随机初始化蒸馏 | 随机参数 | 收集的外部数据 |

#### 为什么封装这一层

1. **变异和交叉共享同一个 BC 入口** — 你的交叉操作本质也是"把两个亲本的数据放一起 BC"，
   调 `bc_train(init_params, data_parentA + data_parentB, ...)` 就行了，不需要额外逻辑。
2. **不修改传入对象** — 输入参数向量、输出新参数向量，纯函数式，组合方便。
3. **底层实现可换** — 未来想改 BC 算法（比如加数据增强），只改 `bc_train` 内部，所有调用方不变。

### 1.7 `mutate_class_labels` — 标签变异（**我写**）

从 `bc_mutate_population` 中提取出的标签变异逻辑。对一批数据的类别标签做随机变异，
用对应 head 的 logits + 温度采样决定变异结果。纯数据变换函数，不涉及模型前向/反向。

```python
def mutate_class_labels(
    cls_labels: torch.Tensor,       # (B, N_heads) 原始类别标签
    head_logits: torch.Tensor,      # (B, N_heads, 24) 各 head 的 logits
    p_mutate: float = 0.1,         # 每个标签被变异的概率
    temperature: float = 3.0,      # 采样温度（越高越均匀）
    seed: int = 0,                 # 随机种子
    category_weights: list[float] | None = None,  # 24 类的重平衡权重
) -> torch.Tensor:                 # (B, N_heads) 变异后的标签
    """对一批数据的类别标签做随机变异。

    内部逻辑（来自 ss_train.py bc_mutate_population 552-576 行）：
      1. 克隆原始标签
      2. 逐 head 随机选 p_mutate 比例的样本（mask）
      3. 对选中样本：
         a. 取出对应 head_logits
         b. clamp(logits, -50, 50)           ← 防止极端 logit 让温度采样失效
         c. logits += log(category_weights) * temperature  ← 重平衡偏置
         d. softmax(logits / temperature) → 多项式采样
      4. 替换对应位置的标签

    category_weights 默认 None 时使用 ss_train 的原始配置：
      - 塔类 (0-15):     权重 1.0  (16 类)
      - 拆除 (16):       权重 2.5  (1 类)
      - 超级武器 (17-20): 权重 2.5  (4 类)
      - 基地升级 (21-22): 权重 2.5  (2 类)
      - HOLD (23):       权重 2.5  (1 类)
    """
```

#### 使用场景

| 场景 | 说明 |
|------|------|
| `bc_mutate_population` 内部 | 替换原有内联代码 |
| GA 交叉后变异 | 对 pooled 数据做标签变异，增加数据多样性 |
| 数据增强 | 单独对 BC 数据集做变异，提升泛化 |

#### 保留的优化点

| 优化 | 在 `mutate_class_labels` 中 | 在 `bc_train`（→`ss_supervised_update`）中 |
|------|---------------------------|------------------------------------------|
| 类别重平衡权重（稀有类 2.5×） | `category_weights` 参数，默认与原代码相同 | 不涉及（BC 训练侧） |
| logits clamp [-50, 50] | 内部逻辑步骤 3b ✓ | 不涉及 |
| 逐 head 独立变异 | 内部逻辑步骤 2 ✓ | 不涉及 |
| per-head 梯度归一化 | **暂不处理**（梯度线性组合部分） | 不涉及 |
| 逐 head loss 计算 | 不涉及 | `ss_supervised_update` 中逐 head 算 CE ✓ |
| label smoothing | 不涉及 | `ss_supervised_update` 支持 ✓ |
| weight decay (AdamW) | 不涉及 | `ss_supervised_update` 支持 ✓ |
| map loss (KL) | 不涉及 | `ss_supervised_update` 支持 ✓ |

#### 和 `bc_train` 的关系

```
原始数据 → mutate_class_labels → 变异后的数据 → bc_train(init_params, data) → 新参数向量
                                                  ↑
                                             变异/交叉的结果作为 init_params
```

两阶段相互独立：先变异数据，再训练模型。可以只变异不训练，也可以只训练不变异。

### 1.7 `write_npz` — 保存游戏数据

```python
def write_npz(
    path: Path,
    board: np.ndarray,          # (T, 28, 19, 19)
    stats: np.ndarray,          # (T, 42)
    class_: np.ndarray,         # (T, N_heads)
    action_map: np.ndarray,     # (T, 24, 19, 19)
    head_logits: np.ndarray,    # (T, N_heads, 24)
    value: np.ndarray | None = None,  # (T, 2) hp_us, hp_opp
):
    """压缩保存。board/stats/action_map/head_logits 存为 float16"""
```

### 1.8 `collect_npz` — 收集某代的 .npz 文件

```python
def collect_npz(
    bc_dir: Path,
    selected_indices: list[int],  # 哪些个体的数据
    gen: int,
) -> list[Path]:
    """返回匹配 gen_NNNN_ind{idx}_*.npz 的文件路径列表"""
```

### 1.9 `save_checkpoint` — 保存训练存档

```python
def save_checkpoint(
    path: Path,
    mean: np.ndarray,           # 模型参数向量
    model: AntWarNetwork,       # 模型（存 state_dict）
    generation: int,
    config: dict | None = None,
    leaderboard: Leaderboard | None = None,
):
```

### 1.10 `reload_config` — 热加载配置

```python
def reload_config(
    config_path: str,
    args,
) -> bool:
    """读取 config.txt，更新 args 中对应属性。返回是否改变"""
```

### 1.11 `Leaderboard` — 对手池

```python
# from my_ai.leaderboard import Leaderboard
lb = Leaderboard(max_size=20, param_count=param_count)
lb.get_opponents(k=3)          # → [entry, ...]   rank-weighted 采样
lb.add_candidate(gen, params, match_fn)  # → bool
lb.get_ranked_entries()        # → 按分数排序的列表
lb.state_dict() / lb.load_state_dict()  # 存档/读档
```

---

## 二、我帮你写的样板代码

以下内容我会写成 `code/my_ai/ga_ss_boilerplate.py`，ga_ss_train.py 直接 import 使用：

### 2.1 `ga_ss_boilerplate.py` 包含的内容

```python
# 路径设置（sys.path 添加 Ant-Game/ 和 code/）
# 模型创建（create_model + 参数统计）
# 输出目录创建（auto timestamp / 指定 out_dir）
# 日志初始化 + CSV 初始化
# 中断处理（Ctrl+C 信号）
# argparse 参数定义（所有 GA 需要的 CLI 参数）

def setup_output_dir(args) -> tuple[Path, Path, Logger]:
    """创建输出目录、返回 out_dir / bc_dir / log"""
    
def init_csv(out_dir, csv_header, first_row=None):
    """创建/初始化 history.csv"""
    
def get_device():
    """返回 cuda / cpu"""
```

### 2.2 参数列表

```
--pop-size       种群大小
--games          每人每代比赛局数
--workers        进程数
--generations    总代数
--save-every     每 N 代保存存档
--num-heads      策略头数
--k              BC 训练选取 top-K
--epochs         BC 训练轮数
--lr             学习率
--batch-size     批次大小
--action-dropout 动作丢弃率
--p-hold         HOLD 帧保留比例
--seed           随机种子
--out-dir        输出目录（默认自动时间戳）
--checkpoint     从存档恢复
--small          使用小模型
--oversample-alpha  稀有类过采样
--no-lb          禁用 Leaderboard
--lb-inject      注入 Leaderboard 条目数
```

（你手写 GA 自己的参数可以再加，比如交叉率、变异率等）

---

## 三、GA 主循环中你想自己写的部分

```python
def main():
    # ── 初始化（我写） ──
    args = parse_args()
    out_dir, bc_dir, log = setup_output_dir(args)
    model, mean = create_model(...)
    leaderboard = Leaderboard(...) if not args.no_lb else None
    pool = mp.Pool(args.workers)
    
    # ── GA 主循环（你写） ──
    for gen in range(start_gen, args.generations):
        # (1) 生成种群 ← 你的 GA 核心
        #     - 选择  (selection)
        #     - 交叉  (crossover)
        #     - 变异  (mutation)
        #     生成 params_list
        #
        # (2) 评估 ← 复用接口
        #     all_args = build_eval_args(...)
        #     results = run_eval(pool, all_args)
        #     fitness = ...
        #
        # (3) 收集 top-K 数据 ← 复用接口
        #     npz_paths = collect_npz(bc_dir, top_k_idx, gen)
        #     ds = SSDataset(npz_paths, ...)
        #
        # (4) BC 训练 ← bc_train 统一接口
        #     mean = bc_train(mean, model, ds, device, ...)
        #     # bc_train 内部: create_model → load → ss_supervised_update → get_parameters
        #
        # (5) 更新 Leaderboard ← 复用 leaderboard.py
        #     leaderboard.add_candidate(gen, mean, match_fn=...)
        #
        # (6) 日志 + CSV + 存档 ← 我写样板
```

---

## 四、文件组织

```
code/my_ai/
├── ss_train.py              # 源文件：不动，纯 import
├── leaderboard.py           # 源文件：不动，纯 import
├── network.py / agent.py / decoder.py  # 底层：不动
├── ga_ss_boilerplate.py     # 我写：样板代码
└── ga_ss_train.py           # 你写：GA 核心逻辑
```

---

## 五、待确认的问题

1. **params_list 的组成** — ss_train 的每代流程是：
   - `bc_mutate_population` → 生成 N 个个体的参数向量
   - `head_swap` → 交换部分个体的 head
   - `mean_injection` → 在列表末尾放 mean
   - `elite_retention` → 在列表末尾放历史精英
   - `lb_inject` → 在列表末尾放 Leaderboard 强者
   - 这些操作都在**参数向量层面**进行，不涉及模型前向
   - GA 你打算如何生成种群？纯参数空间交叉（传统GA）还是用你的 Function-Space Crossover 想法？

2. **BC 训练时机** — ss_train 每代都收集数据 → BC 训练。你的 GA 是否也每代 BC？还是隔几代 BC 一次？

3. **新增参数** — 你手写的 GA 部分需要哪些 CLI 参数？
