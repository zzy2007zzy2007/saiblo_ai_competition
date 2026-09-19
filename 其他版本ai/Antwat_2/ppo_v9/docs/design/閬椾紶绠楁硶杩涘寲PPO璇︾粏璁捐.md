# 基于遗传算法进化策略网络的训练框架 —— 详细设计文档

> 本文档基于《遗传算法进化PPO概要设计》，提供可直接指导编码实施的详细设计。
> 凡标注"直接复用 ppo_v2"的模块，需按指定步骤操作；凡标注"新建"的模块，给出完整类/函数签名与算法逻辑。

---

## 1. 项目结构与模块总览

### 1.1 目录结构

```
ppo_v9/
├── train_ga.py                              # 训练入口脚本
├── requirements.txt                         # 依赖声明
├── configs/
│   └── ga_evo.yaml                          # 默认配置文件
├── src/
│   └── ppo_ga/                              # 主包
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── config_parser.py             # 配置解析器（改造自 ppo_v2）
│       │   └── path_config.py              # 路径管理器（改造自 ppo_v2）
│       ├── network/
│       │   ├── __init__.py
│       │   ├── ant_war_policy_value_network.py  # 直接复用 ppo_v2
│       │   ├── hex_cnn_encoder.py               # 直接复用 ppo_v2
│       │   ├── hex_conv.py                      # 直接复用 ppo_v2
│       │   ├── mlp_encoder.py                   # 直接复用 ppo_v2
│       │   └── heads.py                         # 直接复用 ppo_v2
│       ├── genome/
│       │   ├── __init__.py
│       │   ├── genome.py                    # 新建：基因组数据结构与操作
│       │   └── weight_init.py              # 新建：随机初始化策略
│       ├── evolution/
│       │   ├── __init__.py
│       │   ├── population_manager.py        # 新建：种群管理（初始化/杂交/变异/精英保留）
│       │   ├── crossover.py                # 新建：层级杂交
│       │   └── mutation.py                 # 新建：全局高斯变异
│       ├── selection/
│       │   ├── __init__.py
│       │   ├── battle_selection.py          # 新建：对战选拔（ELO 评分 + 种子选拔）
│       │   └── elo_rating.py              # 新建：ELO 评分系统
│       ├── league/
│       │   ├── __init__.py
│       │   ├── opponent_pool.py            # 直接复用 ppo_v2
│       │   ├── battle_shared_payoff.py     # 直接复用 ppo_v2
│       │   ├── trueskill_rating.py         # 直接复用 ppo_v2
│       │   └── opponent_selector.py        # 直接复用 ppo_v2
│       ├── battle/
│       │   ├── __init__.py
│       │   ├── ant_war_agent.py            # 直接复用 ppo_v2
│       │   ├── rule_based_agents.py        # 直接复用 ppo_v2
│       │   ├── agent_loader.py             # 直接复用 ppo_v2
│       │   ├── battle_simulator.py         # 直接复用 ppo_v2
│       │   ├── battle_config.py            # 改造自 ppo_v2（扩展字段）
│       │   ├── battle_coordinator.py       # 改造自 ppo_v2（适配 GA 场景）
│       │   ├── result_aggregator.py        # 直接复用 ppo_v2
│       │   ├── report_generator.py         # 直接复用 ppo_v2
│       │   ├── ppo_war_agent.py            # 直接复用 ppo_v2
│       │   ├── opponent_agent.py           # 直接复用 ppo_v2
│       │   └── ga_war_agent.py            # 新建：GA 个体驱动的对战 Agent
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── baseline_evaluator.py       # 新建：基线验证
│       ├── trainer/
│       │   ├── __init__.py
│       │   ├── genetic_evolution_trainer.py # 新建：主训练循环
│       │   ├── checkpoint_manager.py       # 改造自 ppo_v2（适配 GA checkpoint）
│       │   └── logging_subsystem.py        # 新建：GA 日志子系统
│       ├── env/
│       │   ├── __init__.py
│       │   ├── observation.py              # 直接复用 ppo_v2
│       │   └── action_mask.py             # 直接复用 ppo_v2
│       ├── compat/
│       │   ├── __init__.py
│       │   └── adapters.py                # 直接复用 ppo_v2
│       ├── monitor/
│       │   ├── __init__.py
│       │   ├── ga_logger.py               # 新建：GA 训练主日志
│       │   ├── battle_log_writer.py        # 新建：对战日志写入器
│       │   └── generation_stats_writer.py  # 新建：种群统计写入器
│       └── utils/
│           ├── __init__.py
│           └── action_constants.py         # 直接复用 ppo_v2
├── docs/
│   ├── design/
│   │   ├── 遗传算法进化PPO概要设计.md
│   │   └── 遗传算法进化PPO详细设计.md      # 本文档
│   ├── game/
│   │   ├── antwar_rules.md
│   │   ├── antwar_rules_economy.md
│   │   └── antwar_rules_review.md
│   └── requirement/
│       └── 遗传算法进化PPO需求文档.md
└── tests/
    └── temp_test/                          # 临时测试代码
```

### 1.2 模块复用策略总览

| 分类 | 模块 | 复用方式 | 说明 |
|------|------|---------|------|
| **直接复用** | network/*, league/*, battle/ant_war_agent.py, battle/rule_based_agents.py, battle/agent_loader.py, battle/battle_simulator.py, battle/result_aggregator.py, battle/report_generator.py, battle/ppo_war_agent.py, battle/opponent_agent.py, env/*, compat/*, utils/* | 拷贝源文件，仅修改 import 路径 | 逻辑不变，包名从 `ppo_antwar` 改为 `ppo_ga` |
| **改造复用** | config/config_parser.py, config/path_config.py, battle/battle_config.py, battle/battle_coordinator.py, trainer/checkpoint_manager.py | 拷贝后按本文档指定修改 | 增加字段或适配 GA 场景 |
| **新建** | genome/*, evolution/*, selection/*, evaluation/*, trainer/genetic_evolution_trainer.py, trainer/logging_subsystem.py, battle/ga_war_agent.py, monitor/* | 按本文档从零实现 | GA 特有逻辑 |

### 1.3 直接复用操作步骤

对于所有标记为"直接复用"的文件，执行以下统一操作：

1. **拷贝文件**：从 `ppo_v2/src/ppo_antwar/` 对应目录拷贝到 `ppo_v9/src/ppo_ga/` 对应目录
2. **修改 import 路径**：将所有 `from ..xxx` 和 `from ppo_antwar.xxx` 中的 `ppo_antwar` 替换为 `ppo_ga`
3. **不修改任何业务逻辑**

具体需要批量替换的 import 模式：
```
ppo_antwar.config  → ppo_ga.config
ppo_antwar.network → ppo_ga.network
ppo_antwar.battle  → ppo_ga.battle
ppo_antwar.env     → ppo_ga.env
ppo_antwar.league  → ppo_ga.league
ppo_antwar.trainer → ppo_ga.trainer
ppo_antwar.utils   → ppo_ga.utils
ppo_antwar.compat  → ppo_ga.compat
ppo_antwar.monitor → ppo_ga.monitor
ppo_antwar._agent  → ppo_ga._agent
```

> **注意**：`ppo_v2/src/ppo_antwar/_agent/protocol.py` 中的 `Agent` 和 `RecordingAgent` 基类在 GA 框架中不直接使用（GA 不需要 `act_record`），但 `NeuralAgent` 继承了 `RecordingAgent`，因此仍需保留该文件。

---

## 2. 网络层与基因组模块

### 2.1 网络层（直接复用）

**操作步骤**：

1. 拷贝以下文件到 `ppo_v9/src/ppo_ga/network/`：
   - `ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py`
   - `ppo_v2/src/ppo_antwar/network/hex_cnn_encoder.py`
   - `ppo_v2/src/ppo_antwar/network/hex_conv.py`
   - `ppo_v2/src/ppo_antwar/network/mlp_encoder.py`
   - `ppo_v2/src/ppo_antwar/network/heads.py`
   - `ppo_v2/src/ppo_antwar/network/__init__.py`

2. 修改 import 路径（按 1.3 节统一规则）

3. **不修改任何业务逻辑**

**关键接口备忘**（GA 框架中需要使用的）：

```python
# 创建网络实例
network = AntWarPolicyValueNetwork(hidden_dim=256, enable_auxiliary=True)

# 获取所有参数（用于提取基因组）
for name, param in network.named_parameters():
    # name: str, 如 "policy_cnn.stage1.0.weight"
    # param.data: Tensor, 参数值
    pass

# 加载基因组到网络（用于对战时从基因组恢复网络）
# state_dict = genome_to_state_dict(genome)
# network.load_state_dict(state_dict)

# 前向推理（用于对战时选择动作）
# 需通过 GAWarAgent 封装，见 6.2 节
```

### 2.2 基因组数据结构（新建）

**文件**：`ppo_v9/src/ppo_ga/genome/genome.py`

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass
class LayerBoundary:
    """层边界信息，用于层级杂交时定位切点"""
    start_idx: int          # 在扁平向量中的起始位置
    end_idx: int            # 在扁平向量中的结束位置（不含）
    layer_type: str         # conv | linear | norm | bias | head
    layer_name: str         # 层名，如 policy_cnn.stage1.0


@dataclass
class Genome:
    """基因组 —— 策略网络所有可训练参数的扁平化表示"""
    weights: np.ndarray                              # 一维 float32 权重向量
    shapes: List[Tuple[int, ...]]                    # 各参数的原始形状
    layer_boundaries: List[LayerBoundary]             # 层边界索引
    param_count: int                                  # 总参数量
    param_names: List[str]                            # 各参数名（与 shapes 一一对应）


def extract_genome(network: nn.Module) -> Genome:
    """从网络实例提取基因组。

    遍历 network.named_parameters()，将每个参数展平拼接为一维向量，
    同时记录形状和层边界信息。

    Args:
        network: AntWarPolicyValueNetwork 实例

    Returns:
        Genome 实例
    """
    weights_list = []
    shapes = []
    param_names = []
    layer_boundaries = []

    offset = 0
    for name, param in network.named_parameters():
        flat = param.data.cpu().numpy().flatten()
        shapes.append(tuple(param.shape))
        param_names.append(name)

        # 判定层类型
        layer_type = _classify_layer(name, param)

        boundary = LayerBoundary(
            start_idx=offset,
            end_idx=offset + len(flat),
            layer_type=layer_type,
            layer_name=name,
        )
        layer_boundaries.append(boundary)

        weights_list.append(flat)
        offset += len(flat)

    all_weights = np.concatenate(weights_list).astype(np.float32)

    return Genome(
        weights=all_weights,
        shapes=shapes,
        layer_boundaries=layer_boundaries,
        param_count=len(all_weights),
        param_names=param_names,
    )


def genome_to_state_dict(genome: Genome) -> Dict[str, torch.Tensor]:
    """将基因组转换回 state_dict 格式，用于加载到网络。

    按记录的 shapes 将一维权重向量切分并 reshape 为原始形状。

    Args:
        genome: Genome 实例

    Returns:
        {param_name: Tensor} 字典
    """
    state_dict = {}
    offset = 0
    for name, shape in zip(genome.param_names, genome.shapes):
        size = 1
        for s in shape:
            size *= s
        tensor = torch.from_numpy(
            genome.weights[offset:offset + size].copy()
        ).reshape(shape)
        state_dict[name] = tensor
        offset += size
    return state_dict


def load_genome_to_network(network: nn.Module, genome: Genome) -> None:
    """将基因组加载到网络实例（in-place 修改网络参数）。

    Args:
        network: AntWarPolicyValueNetwork 实例
        genome: Genome 实例
    """
    state_dict = genome_to_state_dict(genome)
    network.load_state_dict(state_dict)


def _classify_layer(name: str, param: nn.Parameter) -> str:
    """根据参数名和形状判定层类型。"""
    if 'weight' in name:
        if len(param.shape) >= 3:
            return 'conv'       # Conv2d 权重 (out, in, kH, kW) 或 HexConv 权重 (out, in, 6)
        elif len(param.shape) == 2:
            return 'linear'     # Linear 权重 (out, in)
        elif len(param.shape) == 1:
            return 'norm'       # BatchNorm/LayerNorm weight
    elif 'bias' in name:
        return 'bias'
    else:
        return 'other'
```

**关键设计说明**：

1. `Genome.weights` 使用 `np.ndarray`（float32），而非 `torch.Tensor`，因为 GA 操作（杂交、变异）不需要 GPU 加速，numpy 更轻量
2. `layer_boundaries` 在提取时一次性计算并保存，后续杂交操作直接使用，无需重复计算
3. `genome_to_state_dict` 是对战时从基因组恢复网络的关键函数，每个对战 worker 调用此函数将基因组加载到网络
4. `_classify_layer` 用于区分层类型，虽然当前杂交算法不区分类型（所有层平等对待），但保留分类信息供未来扩展

### 2.3 随机初始化策略（新建）

**文件**：`ppo_v9/src/ppo_ga/genome/weight_init.py`

```python
import math
import torch
import torch.nn as nn
import numpy as np


def random_init_network(network: nn.Module) -> None:
    """对网络的所有可训练参数执行完全随机初始化（均匀分布）。

    初始化策略：U(-k, k)，其中 k = 1 / sqrt(fan_in)
    - Conv2d 权重：fan_in = in_channels * kernel_height * kernel_width
    - Linear 权重：fan_in = in_features
    - Conv2d/Linear 偏置：0
    - LayerNorm/BatchNorm weight：1
    - LayerNorm/BatchNorm bias：0

    注意：不使用正交初始化，因为正交初始化引入先验偏向，与"完全随机"初衷不符。

    Args:
        network: AntWarPolicyValueNetwork 实例
    """
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue

        if 'weight' in name:
            if isinstance(param, (nn.Conv2d, nn.Linear)) or _is_conv_or_linear_weight(name, param):
                fan_in = _compute_fan_in(param)
                k = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.uniform_(param, -k, k)
            elif _is_norm_weight(name):
                nn.init.ones_(param)
            else:
                # HexConv.weight 等自定义参数
                fan_in = _compute_fan_in(param)
                k = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.uniform_(param, -k, k)
        elif 'bias' in name:
            if _is_norm_bias(name):
                nn.init.zeros_(param)
            else:
                nn.init.zeros_(param)
        else:
            # 其他参数（如 BatchNorm running_mean 等非可训练参数不会进入此分支）
            pass


def _compute_fan_in(param: torch.Tensor) -> int:
    """计算 fan_in（输入维度大小）。

    对于 Conv2d 权重 (out, in, kH, kW)：fan_in = in * kH * kW
    对于 Linear 权重 (out, in)：fan_in = in
    对于 HexConv 权重 (out, in, 6)：fan_in = in * 6
    """
    shape = param.shape
    if len(shape) >= 3:
        # Conv2d 或 HexConv: fan_in = dim[1] * prod(dim[2:])
        fan_in = shape[1]
        for d in shape[2:]:
            fan_in *= d
        return fan_in
    elif len(shape) == 2:
        # Linear: fan_in = shape[1]
        return shape[1]
    else:
        return shape[0]


def _is_norm_weight(name: str) -> bool:
    """判断是否为归一化层的 weight 参数"""
    return any(k in name for k in ['LayerNorm', 'BatchNorm', 'layer_norm', 'batch_norm'])


def _is_norm_bias(name: str) -> bool:
    """判断是否为归一化层的 bias 参数"""
    return _is_norm_weight(name)


def _is_conv_or_linear_weight(name: str, param: torch.Tensor) -> bool:
    """判断是否为 Conv2d 或 Linear 的 weight 参数（通过形状推断）"""
    return len(param.shape) in (2, 4)
```

**关键设计说明**：

1. 初始化函数直接操作 `nn.Module` 的参数（in-place），而非操作 Genome 的 weights 向量。这是因为初始化在网络创建后立即执行，此时参数仍在网络中，提取 Genome 之前完成初始化
2. `_compute_fan_in` 通过形状推断而非 `nn.init._calculate_fan_in_and_fan_out`，因为 HexConv 的参数形状 `(out, in, 6)` 不符合标准 Conv2d 格式
3. 初始化流程：`创建网络 → random_init_network → extract_genome`，三步顺序执行

---

## 3. 种群管理模块

### 3.1 种群数据结构（新建）

**文件**：`ppo_v9/src/ppo_ga/evolution/population_manager.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ..genome.genome import Genome, extract_genome, load_genome_to_network
from ..genome.weight_init import random_init_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork


@dataclass
class Individual:
    """个体 —— 种群中的一个成员"""
    id: str                                          # 唯一标识，如 "gen_001_ind_042"
    generation: int                                  # 所属代次
    genome: Genome                                   # 权重向量 + 形状/边界信息
    elo_rating: float = 1200.0                       # 代内 ELO 评分
    seed_rank: int = -1                              # 种子排名（-1 表示非种子）
    parent_ids: List[str] = field(default_factory=list)  # 亲本 ID
    mutation_desc: str = ""                           # 变异描述


@dataclass
class Population:
    """种群 —— 一代的所有个体"""
    generation: int                                  # 当代代次
    individuals: List[Individual]                    # 所有个体
    best_elo: float = 0.0                            # 本代最高 ELO
    mean_elo: float = 0.0                            # 本代平均 ELO
    diversity: float = 0.0                           # 种群多样性指标


class PopulationManager:
    """种群管理器 —— 负责种群初始化、杂交、变异、精英保留"""

    def __init__(
        self,
        population_size: int = 80,
        elitism_count: int = 2,
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
    ):
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary

    def init_population(self) -> Population:
        """初始化第一代种群（完全随机初始化）。

        流程：
        1. 创建 population_size 个 AntWarPolicyValueNetwork 实例
        2. 对每个实例执行 random_init_network（完全随机初始化）
        3. 提取基因组
        4. 生成 Individual

        Returns:
            Population 实例
        """
        individuals = []
        for i in range(self.population_size):
            network = AntWarPolicyValueNetwork(
                hidden_dim=self.hidden_dim,
                enable_auxiliary=self.enable_auxiliary,
            )
            random_init_network(network)
            genome = extract_genome(network)
            ind = Individual(
                id=f"gen_000_ind_{i:03d}",
                generation=0,
                genome=genome,
            )
            individuals.append(ind)
        return Population(generation=0, individuals=individuals)

    def evolve(
        self,
        seeds: List[Individual],
        generation: int,
        crossover_fn,
        mutation_fn,
        current_mutation_scale: float,
    ) -> Population:
        """从种子生成下一代种群。

        流程：
        1. 精英保留：top elitism_count 个种子直接复制到下一代
        2. 从种子池中随机配对
        3. 每对执行层级杂交，生成 2 个子代
        4. 对子代施加全局高斯变异
        5. 重复直到填满 population_size

        Args:
            seeds: 上一代选拔出的种子列表（按 ELO 降序排列）
            generation: 新一代的代次编号
            crossover_fn: 杂交函数，签名 (Genome, Genome) -> (Genome, Genome)
            mutation_fn: 变异函数，签名 (Genome, float, float) -> Genome
            current_mutation_scale: 当前变异幅度（已含衰减）

        Returns:
            新一代 Population 实例
        """
        new_individuals: List[Individual] = []

        # 1. 精英保留
        for i in range(min(self.elitism_count, len(seeds))):
            elite = seeds[i]
            elite_child = Individual(
                id=f"gen_{generation:03d}_ind_{i:03d}",
                generation=generation,
                genome=Genome(
                    weights=elite.genome.weights.copy(),
                    shapes=elite.genome.shapes,
                    layer_boundaries=elite.genome.layer_boundaries,
                    param_count=elite.genome.param_count,
                    param_names=elite.genome.param_names,
                ),
                elo_rating=1200.0,  # 新代重置 ELO
                parent_ids=[elite.id],
                mutation_desc="elite_copy",
            )
            new_individuals.append(elite_child)

        # 2. 从种子中随机配对，杂交 + 变异，填满种群
        import random
        child_idx = len(new_individuals)

        while len(new_individuals) < self.population_size:
            # 随机选择两个不同的亲本
            parent_a, parent_b = random.sample(seeds, min(2, len(seeds)))
            if len(seeds) < 2:
                parent_b = parent_a  # 退化为仅变异

            # 杂交
            child_a_genome, child_b_genome = crossover_fn(
                parent_a.genome, parent_b.genome
            )

            # 变异
            child_a_genome = mutation_fn(child_a_genome, rate=0.1, scale=current_mutation_scale)
            child_b_genome = mutation_fn(child_b_genome, rate=0.1, scale=current_mutation_scale)

            # 生成子代个体
            for child_genome in [child_a_genome, child_b_genome]:
                if len(new_individuals) >= self.population_size:
                    break
                child = Individual(
                    id=f"gen_{generation:03d}_ind_{child_idx:03d}",
                    generation=generation,
                    genome=child_genome,
                    elo_rating=1200.0,
                    parent_ids=[parent_a.id, parent_b.id],
                    mutation_desc=f"crossover+mutation(scale={current_mutation_scale:.4f})",
                )
                new_individuals.append(child)
                child_idx += 1

        return Population(generation=generation, individuals=new_individuals)

    @staticmethod
    def compute_diversity(population: Population) -> float:
        """计算种群多样性指标（权重向量间平均欧氏距离）。

        为避免 O(n²) 全量计算，随机采样 min(100, n*(n-1)/2) 对计算平均距离。

        Args:
            population: Population 实例

        Returns:
            多样性指标（平均欧氏距离）
        """
        import random

        individuals = population.individuals
        n = len(individuals)
        if n < 2:
            return 0.0

        max_pairs = n * (n - 1) // 2
        sample_size = min(100, max_pairs)

        distances = []
        for _ in range(sample_size):
            i, j = random.sample(range(n), 2)
            diff = individuals[i].genome.weights - individuals[j].genome.weights
            dist = np.linalg.norm(diff)
            distances.append(dist)

        return float(np.mean(distances))
```

### 3.2 层级杂交（新建）

**文件**：`ppo_v9/src/ppo_ga/evolution/crossover.py`

```python
import random
from typing import Tuple
from ..genome.genome import Genome


def layer_wise_crossover(
    parent_a: Genome,
    parent_b: Genome,
    crossover_rate: float = 0.8,
) -> Tuple[Genome, Genome]:
    """层级单点杂交。

    算法：
    1. 获取网络的层边界列表 layer_boundaries
    2. 以 crossover_rate 概率决定是否执行杂交，否则直接复制亲本
    3. 在层序列中随机选择一个切点索引 k
    4. 子代 C：前 k 层权重 = 亲本 A，后 N-k 层权重 = 亲本 B
    5. 子代 D：前 k 层权重 = 亲本 B，后 N-k 层权重 = 亲本 A

    Args:
        parent_a: 亲本 A 的基因组
        parent_b: 亲本 B 的基因组
        crossover_rate: 杂交概率（默认 0.8）

    Returns:
        (子代 C 基因组, 子代 D 基因组)
    """
    # 不杂交则直接复制亲本
    if random.random() > crossover_rate:
        return (
            _copy_genome(parent_a),
            _copy_genome(parent_b),
        )

    boundaries = parent_a.layer_boundaries
    n_layers = len(boundaries)

    # 随机选择切点（1 到 n_layers-1 之间，确保双方都有贡献）
    k = random.randint(1, n_layers - 1)

    # 切点在扁平向量中的位置
    split_idx = boundaries[k].start_idx

    # 生成子代
    child_a_weights = np_concat_split(
        parent_a.weights, parent_b.weights, split_idx
    )
    child_b_weights = np_concat_split(
        parent_b.weights, parent_a.weights, split_idx
    )

    child_a = Genome(
        weights=child_a_weights,
        shapes=parent_a.shapes,
        layer_boundaries=parent_a.layer_boundaries,
        param_count=parent_a.param_count,
        param_names=parent_a.param_names,
    )
    child_b = Genome(
        weights=child_b_weights,
        shapes=parent_b.shapes,
        layer_boundaries=parent_b.layer_boundaries,
        param_count=parent_b.param_count,
        param_names=parent_b.param_names,
    )

    return child_a, child_b


def np_concat_split(a_weights, b_weights, split_idx):
    """拼接 a 的前半部分和 b 的后半部分。

    Args:
        a_weights: np.ndarray, 亲本 A 的权重
        b_weights: np.ndarray, 亲本 B 的权重
        split_idx: 切点位置

    Returns:
        拼接后的 np.ndarray
    """
    import numpy as np
    return np.concatenate([
        a_weights[:split_idx],
        b_weights[split_idx:],
    ]).astype(np.float32)


def _copy_genome(genome: Genome) -> Genome:
    """深拷贝基因组（权重数组需要 copy）"""
    return Genome(
        weights=genome.weights.copy(),
        shapes=genome.shapes,
        layer_boundaries=genome.layer_boundaries,
        param_count=genome.param_count,
        param_names=genome.param_names,
    )
```

### 3.3 全局高斯变异（新建）

**文件**：`ppo_v9/src/ppo_ga/evolution/mutation.py`

```python
import numpy as np
from ..genome.genome import Genome


def gaussian_mutation(
    genome: Genome,
    rate: float = 0.1,
    scale: float = 0.03,
) -> Genome:
    """全局高斯变异。

    对扁平权重向量的每个元素，以概率 rate 决定是否变异，
    被选中的元素加上高斯噪声 N(0, scale²)。

    Args:
        genome: 待变异的基因组（会被 in-place 修改）
        rate: 变异率（每个参数被变异的概率，默认 0.1）
        scale: 变异幅度（高斯噪声标准差，默认 0.03）

    Returns:
        变异后的基因组（与输入为同一对象）
    """
    # 1. 生成变异掩码
    mask = np.random.random(genome.weights.shape) < rate

    # 2. 生成高斯噪声
    noise = np.random.randn(*genome.weights.shape).astype(np.float32) * scale

    # 3. 施加变异（仅 mask=True 的位置）
    genome.weights += noise * mask

    return genome
```

**变异衰减说明**：

变异衰减不在 `gaussian_mutation` 函数内部实现，而是在 `GeneticEvolutionTrainer` 的主循环中管理：

```python
# 在主循环中
current_mutation_scale = config.mutation.scale * (config.mutation.decay ** generation)
```

这样设计的原因：衰减是训练策略的一部分，不属于变异操作本身。

---

## 4. 对战选拔与 ELO 模块

### 4.1 ELO 评分系统（新建）

**文件**：`ppo_v9/src/ppo_ga/selection/elo_rating.py`

```python
from typing import Dict, Optional


class ELORating:
    """ELO 评分系统 —— 用于代内个体排名"""

    def __init__(
        self,
        initial_rating: float = 1200.0,
        k_factor: float = 64.0,
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}

    def ensure_player(self, player_id: str) -> None:
        """注册玩家（若未注册则使用初始评分）"""
        if player_id not in self._ratings:
            self._ratings[player_id] = self.initial_rating

    def update(self, player_a: str, player_b: str, result: float) -> None:
        """更新双方 ELO 评分。

        Args:
            player_a: 个体 ID
            player_b: 参照物 ID
            result: 1.0 = A 胜, 0.5 = 平, 0.0 = A 负
        """
        self.ensure_player(player_a)
        self.ensure_player(player_b)

        ra = self._ratings[player_a]
        rb = self._ratings[player_b]

        # 期望得分
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

        # 更新评分
        self._ratings[player_a] = ra + self.k_factor * (result - ea)
        # 参照物评分不更新（固定参照物保持不变）

    def get_rating(self, player_id: str) -> float:
        """获取玩家评分"""
        return self._ratings.get(player_id, self.initial_rating)

    def get_rankings(self) -> list:
        """获取按评分降序排列的 (player_id, rating) 列表"""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def reset_individuals(self) -> None:
        """重置所有非参照物的评分（每代开始时调用）"""
        self._ratings.clear()
```

**关键设计说明**：

1. ELO 评分每代重新计算（`reset_individuals` 在每代开始时调用），因为每代的个体不同
2. 参照物的 ELO 评分固定（`update` 方法中只更新 `player_a` 即个体的评分，不更新参照物评分）
3. K 因子设为 64（比标准 32 大），因为个体与参照物的对战次数少（每参照物仅 2 局），需要大 K 快速收敛

### 4.2 对战选拔（新建）

**文件**：`ppo_v9/src/ppo_ga/selection/battle_selection.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from loguru import logger

from ..evolution.population_manager import Individual, Population
from ..battle.battle_simulator import BattleSimulator
from ..battle.battle_config import GABattleConfig
from ..battle.ga_war_agent import GAWarAgent
from ..battle.rule_based_agents import BasicRandomAI, BasicTowerAI, MediumRuleAI
from ..battle.agent_loader import AgentLoader
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from .elo_rating import ELORating


# ── 参照物定义 ──────────────────────────────────────────────────────────────

@dataclass
class ReferenceAgent:
    """参照物 Agent 定义"""
    name: str           # 名称
    elo: float          # 固定 ELO 评分
    agent_factory: Any  # 创建 Agent 的工厂（player_id -> Agent 实例）


# 锚定参照物（永不更换）
ANCHOR_REFERENCES = [
    ReferenceAgent("BasicRandomAI", 400.0, lambda pid: BasicRandomAI(pid)),
    ReferenceAgent("BasicTowerAI", 1000.0, lambda pid: BasicTowerAI(pid)),
    ReferenceAgent("MediumRuleAI", 1600.0, lambda pid: MediumRuleAI(pid)),
]


class BattleSelection:
    """对战选拔 —— 编排个体与参照物的对战、ELO 评分、种子选拔"""

    def __init__(
        self,
        config: GABattleConfig,
        device: str = "cuda",
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
        num_seeds: int = 8,
        elo_k_factor: float = 64.0,
        elo_initial: float = 1200.0,
    ):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary
        self.num_seeds = num_seeds
        self.elo = ELORating(initial_rating=elo_initial, k_factor=elo_k_factor)
        self._simulator = BattleSimulator(config)

    def evaluate_population(
        self,
        population: Population,
        dynamic_references: Optional[List[ReferenceAgent]] = None,
    ) -> List[Individual]:
        """评估种群中所有个体的 ELO 评分。

        流程：
        1. 构建参照物列表 = 锚定参照物 + 动态参照物
        2. 对每个个体 vs 每个参照物执行 2 局对战（先手/后手）
        3. 根据对战结果更新 ELO 评分
        4. 按 ELO 降序排列

        Args:
            population: 待评估的种群
            dynamic_references: 动态参照物列表（从对手池选取）

        Returns:
            按 ELO 降序排列的个体列表
        """
        # 重置 ELO
        self.elo.reset_individuals()

        # 构建参照物列表
        references = list(ANCHOR_REFERENCES)
        if dynamic_references:
            references.extend(dynamic_references)

        # 注册参照物到 ELO 系统（使用固定评分）
        for ref in references:
            self.elo.ensure_player(ref.name)
            self.elo._ratings[ref.name] = ref.elo

        # 遍历每个个体 vs 每个参照物
        for individual in population.individuals:
            self.elo.ensure_player(individual.id)

            # 创建个体对应的 Agent
            ga_agent = self._create_ga_agent(individual)

            for ref in references:
                # 创建参照物 Agent
                ref_agent = ref.agent_factory(player_id=1)

                # 执行 2 局对战（先手/后手）
                battle_results = self._simulator.run_battles(
                    ga_agent, ref_agent, n_battles=1  # n_battles=1 会产生 2 局（先手+后手）
                )

                # 更新 ELO
                for result in battle_results:
                    if "error" in result:
                        # 对战异常，判为平局
                        self.elo.update(individual.id, ref.name, 0.5)
                        continue

                    game_result = result.get("result")
                    if game_result == "agent1_win":
                        self.elo.update(individual.id, ref.name, 1.0)
                    elif game_result == "agent2_win":
                        self.elo.update(individual.id, ref.name, 0.0)
                    else:
                        self.elo.update(individual.id, ref.name, 0.5)

            # 更新个体的 ELO 评分
            individual.elo_rating = self.elo.get_rating(individual.id)

        # 按 ELO 降序排列
        ranked = sorted(population.individuals, key=lambda ind: ind.elo_rating, reverse=True)
        return ranked

    def select_seeds(self, ranked_individuals: List[Individual]) -> List[Individual]:
        """从排名列表中选拔种子。

        Args:
            ranked_individuals: 按 ELO 降序排列的个体列表

        Returns:
            种子列表（前 num_seeds 个）
        """
        seeds = ranked_individuals[:self.num_seeds]
        for i, seed in enumerate(seeds):
            seed.seed_rank = i + 1
        return seeds

    def _create_ga_agent(self, individual: Individual) -> GAWarAgent:
        """从个体基因组创建 GAWarAgent（用于对战）。

        Args:
            individual: 个体

        Returns:
            GAWarAgent 实例
        """
        import torch
        device = torch.device(self.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.hidden_dim,
            enable_auxiliary=self.enable_auxiliary,
        ).to(device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(
            player_id=0,
            network=network,
            device=device,
        )
```

**对战局数计算**：

```
80 个体 × (3 锚定 + 2 动态 + 1 上代冠军 = 6 参照物) × 2 局（先手+后手）= 960 局/代
```

**关键设计说明**：

1. 每个个体对战时创建一个新的 `AntWarPolicyValueNetwork` 实例并加载基因组，对战结束后丢弃。虽然开销较大，但避免了多进程序列化网络的复杂性
2. `BattleSimulator.run_battles(agent1, agent2, n_battles=1)` 会产生 2 局对战（先手+后手各一局），这是 ppo_v2 的 BattleSimulator 的固有行为
3. 参照物的 ELO 评分在 `evaluate_population` 开始时被设置为固定值，且 `elo.update` 只更新个体的评分，不更新参照物

---

## 5. 对手池与 TrueSkill 模块

### 5.1 直接复用文件

从 `ppo_v2/src/ppo_antwar/league/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/league/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `trueskill_rating.py` | `trueskill_rating.py` | 直接拷贝 + 修改 import |
| `battle_shared_payoff.py` | `battle_shared_payoff.py` | 直接拷贝 + 修改 import |
| `opponent_pool.py` | `opponent_pool.py` | 直接拷贝 + 修改 import |
| `opponent_selector.py` | `opponent_selector.py` | 直接拷贝 + 修改 import |
| `__init__.py` | `__init__.py` | 直接拷贝 |

### 5.2 GA 框架中使用对手池的方式

对手池在 GA 框架中的用途与 ppo_v2 不同：

| 维度 | ppo_v2 | ppo_v9 (GA) |
|------|--------|-------------|
| 添加时机 | 每个 episode 后添加当前策略 | 每代选拔后添加种子 |
| 对手选择 | 训练时选择对手进行自对弈 | 从对手池选取动态参照物 |
| TrueSkill 更新 | 每局对战更新 | 种子间对战更新（可选） |

**在 `GeneticEvolutionTrainer` 中的使用方式**：

```python
# 初始化
self.payoff = BattleSharedPayoff(decay=0.99, min_win_rate_games=8)
self.opponent_pool = OpponentPool(
    max_size=20,
    min_games_threshold=8,
    payoff=self.payoff,
)
self.opponent_selector = OpponentSelector(
    payoff=self.payoff,
    opponent_pool=self.opponent_pool,
    exploit_prob=0.7,
)

# 每代选拔后
for seed in seeds:
    # 保存种子模型
    checkpoint_path = self._save_seed_model(seed, generation)
    # 添加到对手池
    self.opponent_pool.add(seed.id, checkpoint_path, generation)

# 获取动态参照物
def _build_dynamic_references(self, generation):
    dynamic_refs = []
    # 从对手池选 2 个动态参照物
    for _ in range(2):
        opp_id = self.opponent_selector.select()
        if opp_id:
            checkpoint_path = self.opponent_pool.get_checkpoint_path(opp_id)
            if checkpoint_path:
                # 从 checkpoint 加载网络创建参照物
                ref = self._create_reference_from_checkpoint(opp_id, checkpoint_path)
                dynamic_refs.append(ref)
    return dynamic_refs
```

### 5.3 对手池状态持久化

直接复用 ppo_v2 的 `save_state` / `load_state` 方法，持久化到 `outputs/{run_id}/league/` 目录：

- `pool.json`：对手池状态（opponents, checkpoint_paths, games_played, added_episodes）
- `payoff.json`：TrueSkill 评分矩阵（players, players_ids, data, trueskill_ratings）

---

## 6. 对战系统模块

### 6.1 直接复用文件

从 `ppo_v2/src/ppo_antwar/battle/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/battle/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `ant_war_agent.py` | `ant_war_agent.py` | 直接拷贝 + 修改 import |
| `rule_based_agents.py` | `rule_based_agents.py` | 直接拷贝 + 修改 import |
| `agent_loader.py` | `agent_loader.py` | 直接拷贝 + 修改 import |
| `battle_simulator.py` | `battle_simulator.py` | 直接拷贝 + 修改 import |
| `result_aggregator.py` | `result_aggregator.py` | 直接拷贝 + 修改 import |
| `report_generator.py` | `report_generator.py` | 直接拷贝 + 修改 import |
| `ppo_war_agent.py` | `ppo_war_agent.py` | 直接拷贝 + 修改 import |
| `opponent_agent.py` | `opponent_agent.py` | 直接拷贝 + 修改 import |

### 6.2 GA 对战 Agent（新建）

**文件**：`ppo_v9/src/ppo_ga/battle/ga_war_agent.py`

```python
from typing import Any, List
import numpy as np
import torch
from loguru import logger

from .ant_war_agent import AntWarAgent
from ..env.observation import ObservationEncoder
from ..env.action_mask import ActionMaskHandler
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..utils.obs_utils import observation_to_tensors


class GAWarAgent(AntWarAgent):
    """GA 个体驱动的对战 Agent。

    与 ppo_v2 的 PPOWarAgent 类似，但直接持有网络实例（而非通过 NeuralAgent 间接引用），
    因为 GA 个体不需要 log_prob/value 等训练元数据。
    """

    def __init__(
        self,
        player_id: int,
        network: AntWarPolicyValueNetwork,
        device: torch.device,
    ):
        super().__init__(player_id, name="GA")
        self._network = network
        self._device = device
        self._observation_encoder = ObservationEncoder()
        self._action_mask_handler = ActionMaskHandler()

    def choose_operations(self, state) -> List[Any]:
        """根据当前状态选择操作（确定性策略）。

        流程：
        1. 编码观测
        2. 网络前向推理，获取动作
        3. 转换为 Operation

        Args:
            state: 游戏状态

        Returns:
            Operation 列表
        """
        obs = self._observation_encoder.encode(state, self.player_id)
        with np.errstate(invalid="ignore"):
            action_id = self._act(obs)

        op = self._action_mask_handler._action_id_to_op(
            action_id, state, self.player_id
        )
        if op is not None:
            return [op]

        # 非法动作回退到随机合法动作
        logger.warning(
            f"[GAWarAgent] Illegal action_id={action_id} for player={self.player_id}, "
            f"falling back to random valid action"
        )
        mask = self._action_mask_handler.get_action_mask(state, self.player_id)
        valid = np.where(mask > 0.5)[0]
        if len(valid) > 0:
            action_id = int(np.random.choice(valid))
            op = self._action_mask_handler._action_id_to_op(
                action_id, state, self.player_id
            )
            return [op] if op is not None else []
        return []

    def _act(self, observation) -> int:
        """网络推理获取动作 ID（确定性）"""
        tensors = observation_to_tensors(observation, self._device)
        with torch.no_grad():
            action_id, _, _, _ = self._network.get_action(
                tensors["board"],
                tensors["global"],
                tensors["action_mask"],
                deterministic=True,
                exploration_epsilon=0.0,
            )
        return action_id
```

**关键设计说明**：

1. GA 对战使用**确定性策略**（`deterministic=True, exploration_epsilon=0.0`），因为对战评估应反映个体的真实水平，不需要探索
2. 与 ppo_v2 的 `PPOWarAgent` 的区别：PPOWarAgent 通过 `NeuralAgent.act()` 间接调用网络，GAWarAgent 直接持有网络实例并调用 `get_action`
3. 非法动作回退逻辑与 PPOWarAgent 一致

### 6.3 对战配置（改造自 ppo_v2）

**文件**：`ppo_v9/src/ppo_ga/battle/battle_config.py`

**操作步骤**：

1. 拷贝 `ppo_v2/src/ppo_antwar/battle/battle_config.py` 到 `ppo_v9/src/ppo_ga/battle/battle_config.py`
2. 修改 import 路径
3. 在 `BaselineBattleConfig` 基础上新增 `GABattleConfig`

```python
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaselineBattleConfig:
    """对战配置 - 管理对战评估的所有可调参数（与 ppo_v2 保持一致）"""
    max_rounds: int
    n_battles: int
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    log_to_console: bool = True
    enable_timing: bool = True
    timing_warning_threshold: float = 2.0
    ppo_checkpoint_path: Optional[str] = None
    baseline_agents: List[str] = field(default_factory=list)
    device: str = "auto"


@dataclass
class GABattleConfig(BaselineBattleConfig):
    """GA 对战配置 - 扩展 BaselineBattleConfig，增加 GA 特有参数"""
    hidden_dim: int = 256
    enable_auxiliary: bool = True
```

### 6.4 对战协调器（改造自 ppo_v2）

**文件**：`ppo_v9/src/ppo_ga/battle/battle_coordinator.py`

**操作步骤**：

1. 拷贝 `ppo_v2/src/ppo_antwar/battle/battle_coordinator.py` 到 `ppo_v9/src/ppo_ga/battle/battle_coordinator.py`
2. 修改 import 路径
3. 保留 `BattleCoordinator` 类的所有方法不变

GA 框架中 `BattleCoordinator` 的使用方式：

```python
# 在 BattleSelection.evaluate_population 中
# 不使用 BattleCoordinator，而是直接使用 BattleSimulator
# 因为 GA 的对战场景更简单（个体 vs 参照物），不需要 BattleCoordinator 的复杂封装

# 在 BaselineEvaluator 中
# 使用 BattleCoordinator.run_ppo_evaluation_with_agent 进行基线评估
# 但需要将 GAWarAgent 适配为 BattleCoordinator 期望的接口
```

---

## 7. 基线验证模块

### 7.1 基线评估器（新建）

**文件**：`ppo_v9/src/ppo_ga/evaluation/baseline_evaluator.py`

```python
from typing import List, Dict, Any, Optional
from loguru import logger

from ..evolution.population_manager import Individual
from ..battle.battle_simulator import BattleSimulator
from ..battle.battle_config import GABattleConfig
from ..battle.ga_war_agent import GAWarAgent
from ..battle.rule_based_agents import BasicRandomAI, BasicTowerAI, MediumRuleAI
from ..battle.result_aggregator import aggregate_results
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork


# 基线 Agent 列表
BASELINE_AGENTS = [
    ("BasicRandomAI", lambda pid: BasicRandomAI(pid)),
    ("BasicTowerAI", lambda pid: BasicTowerAI(pid)),
    ("MediumRuleAI", lambda pid: MediumRuleAI(pid)),
]


class BaselineEvaluator:
    """基线验证器 —— 种子与固定 baseline AI 对战，验证绝对质量"""

    def __init__(
        self,
        config: GABattleConfig,
        device: str = "cuda",
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
        enabled: bool = True,
        interval: int = 1,
        n_battles: int = 12,
    ):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary
        self.enabled = enabled
        self.interval = interval
        self.n_battles = n_battles
        self._simulator = BattleSimulator(config)

    def should_evaluate(self, generation: int) -> bool:
        """判断当前代是否需要执行基线验证"""
        if not self.enabled:
            return False
        return generation % self.interval == 0

    def evaluate_seeds(
        self,
        seeds: List[Individual],
        generation: int,
    ) -> Dict[str, Dict[str, Any]]:
        """对种子选手执行基线验证。

        Args:
            seeds: 种子列表
            generation: 当前代次

        Returns:
            {seed_id: {baseline_name: {wins, losses, draws, win_rate, ...}, ...}, ...}
        """
        if not self.should_evaluate(generation):
            return {}

        results = {}
        for seed in seeds:
            seed_results = self._evaluate_single_seed(seed)
            results[seed.id] = seed_results

        return results

    def _evaluate_single_seed(self, seed: Individual) -> Dict[str, Dict[str, Any]]:
        """评估单个种子 vs 所有 baseline Agent"""
        import torch

        device = torch.device(self.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.hidden_dim,
            enable_auxiliary=self.enable_auxiliary,
        ).to(device)
        load_genome_to_network(network, seed.genome)
        network.eval()

        ga_agent = GAWarAgent(player_id=0, network=network, device=device)

        results = {}
        for baseline_name, baseline_factory in BASELINE_AGENTS:
            baseline_agent = baseline_factory(player_id=1)

            battle_results = self._simulator.run_battles(
                ga_agent, baseline_agent, self.n_battles
            )

            aggregated = aggregate_results(battle_results)
            results[baseline_name] = aggregated

            win_rate = aggregated.get("agent1_wins", 0) / max(aggregated.get("total_battles", 1), 1)
            logger.info(
                f"Seed {seed.id} vs {baseline_name}: "
                f"win_rate={win_rate:.2%} "
                f"({aggregated.get('agent1_wins', 0)}W/"
                f"{aggregated.get('agent2_wins', 0)}L/"
                f"{aggregated.get('draws', 0)}D)"
            )

        return results
```

---

## 8. 主训练循环与日志模块

### 8.1 主训练循环（新建）

**文件**：`ppo_v9/src/ppo_ga/trainer/genetic_evolution_trainer.py`

```python
from typing import Dict, Any, Optional, List
import time
import os

import torch
from loguru import logger

from ..evolution.population_manager import PopulationManager, Population, Individual
from ..evolution.crossover import layer_wise_crossover
from ..evolution.mutation import gaussian_mutation
from ..selection.battle_selection import BattleSelection, ReferenceAgent
from ..evaluation.baseline_evaluator import BaselineEvaluator
from ..league.opponent_pool import OpponentPool
from ..league.battle_shared_payoff import BattleSharedPayoff
from ..league.opponent_selector import OpponentSelector
from ..genome.genome import load_genome_to_network, genome_to_state_dict
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..battle.opponent_agent import OpponentAgent
from .checkpoint_manager import GACheckpointManager
from .logging_subsystem import GALoggingSubsystem


class GeneticEvolutionTrainer:
    """遗传进化训练器 —— 编排整个进化流程"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config["system"]["device"])

        # ── 核心组件 ──
        self.population_mgr = PopulationManager(
            population_size=config["evolution"]["population_size"],
            elitism_count=config["evolution"]["elitism_count"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
        )

        self.battle_selection = BattleSelection(
            config=self._create_battle_config(),
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
            num_seeds=config["evolution"]["num_seeds"],
            elo_k_factor=config["selection"]["elo_k_factor"],
            elo_initial=config["selection"]["elo_initial"],
        )

        self.baseline_evaluator = BaselineEvaluator(
            config=self._create_battle_config(),
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
            enabled=config["baseline_battle"]["enabled"],
            interval=config["baseline_battle"]["interval"],
            n_battles=config["baseline_battle"]["n_battles"],
        )

        # ── 对手池 ──
        self.payoff = BattleSharedPayoff(
            decay=config["opponent_pool"].get("decay", 0.99),
            min_win_rate_games=config["opponent_pool"].get("min_win_rate_games", 8),
        )
        self.opponent_pool = OpponentPool(
            max_size=config["opponent_pool"]["max_size"],
            min_games_threshold=config["opponent_pool"]["min_games_threshold"],
            payoff=self.payoff,
        )
        self.opponent_selector = OpponentSelector(
            payoff=self.payoff,
            opponent_pool=self.opponent_pool,
            exploit_prob=config["opponent_pool"]["exploit_prob"],
        )

        # ── 日志与检查点 ──
        self.logging = GALoggingSubsystem(config)
        self.checkpoint_mgr = GACheckpointManager(self.device)

        # ── 训练状态 ──
        self._generation = 0
        self._best_elo_history: List[float] = []
        self._start_time = time.time()

    def train(self) -> None:
        """主训练循环"""
        max_generations = self.config["evolution"]["max_generations"]
        convergence_window = self.config["evolution"]["convergence_window"]
        convergence_threshold = self.config["evolution"]["convergence_threshold"]

        logger.info(f"GA Training started: max_generations={max_generations}")

        for gen in range(1, max_generations + 1):
            self._generation = gen
            gen_start_time = time.time()

            # ── 阶段一：种群生成 ──
            if gen == 1:
                population = self.population_mgr.init_population()
            else:
                population = self.population_mgr.evolve(
                    seeds=self._last_seeds,
                    generation=gen,
                    crossover_fn=layer_wise_crossover,
                    mutation_fn=gaussian_mutation,
                    current_mutation_scale=self._current_mutation_scale(gen),
                )

            # ── 阶段二：对战评估 ──
            dynamic_refs = self._build_dynamic_references(gen)
            ranked = self.battle_selection.evaluate_population(population, dynamic_refs)

            # ── 阶段三：种子选拔 ──
            seeds = self.battle_selection.select_seeds(ranked)
            self._last_seeds = seeds

            # 更新种群统计
            population.best_elo = ranked[0].elo_rating if ranked else 0.0
            population.mean_elo = sum(ind.elo_rating for ind in ranked) / max(len(ranked), 1)
            population.diversity = PopulationManager.compute_diversity(population)

            # ── 阶段四：基线验证 ──
            baseline_results = self.baseline_evaluator.evaluate_seeds(seeds, gen)

            # ── 阶段五：对手池更新 ──
            for seed in seeds:
                checkpoint_path = self._save_seed_model(seed, gen)
                self.opponent_pool.add(seed.id, checkpoint_path, gen)

            # ── 阶段六：日志与持久化 ──
            gen_duration = time.time() - gen_start_time
            self._best_elo_history.append(population.best_elo)

            self.logging.log_generation(
                generation=gen,
                population=population,
                seeds=seeds,
                baseline_results=baseline_results,
                duration=gen_duration,
            )

            # 定期保存 checkpoint
            save_interval = self.config["logging"]["save_interval"]
            if gen % save_interval == 0:
                self._save_checkpoint(gen, population)

            # 持久化对手池
            self._save_league_state()

            # ── 收敛判断 ──
            if gen >= convergence_window:
                recent = self._best_elo_history[-convergence_window:]
                if max(recent) - min(recent) < convergence_threshold:
                    logger.info(f"Converged at generation {gen}")
                    break

        # 训练完成
        total_time = time.time() - self._start_time
        self.logging.log_training_complete(self._generation, total_time)
        self.logging.close()

    def _current_mutation_scale(self, generation: int) -> float:
        """计算当前变异幅度（含衰减）"""
        base_scale = self.config["mutation"]["scale"]
        decay = self.config["mutation"]["decay"]
        return base_scale * (decay ** (generation - 1))

    def _build_dynamic_references(self, generation: int) -> List[ReferenceAgent]:
        """构建动态参照物列表"""
        dynamic_refs = []

        # 从对手池选 2 个动态参照物
        for _ in range(2):
            opp_id = self.opponent_selector.select()
            if opp_id:
                checkpoint_path = self.opponent_pool.get_checkpoint_path(opp_id)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    ref = self._create_reference_from_checkpoint(
                        opp_id, checkpoint_path
                    )
                    if ref:
                        dynamic_refs.append(ref)

        # 上一代冠军（如果有）
        if hasattr(self, '_last_seeds') and self._last_seeds:
            champion = self._last_seeds[0]
            ref = ReferenceAgent(
                name=f"champion_gen_{generation-1}",
                elo=champion.elo_rating,
                agent_factory=lambda pid, g=champion: self._create_agent_from_individual(g, pid),
            )
            dynamic_refs.append(ref)

        return dynamic_refs

    def _create_reference_from_checkpoint(
        self, opponent_id: str, checkpoint_path: str
    ) -> Optional[ReferenceAgent]:
        """从对手池 checkpoint 创建参照物"""
        try:
            def factory(pid, cp=checkpoint_path):
                agent = OpponentAgent.from_checkpoint(
                    checkpoint_path=cp,
                    device=self.device,
                    hidden_dim=self.config["network"]["hidden_dim"],
                )
                # OpponentAgent 有 act() 方法，需要包装为 AntWarAgent 接口
                from ..battle.ppo_war_agent import PPOWarAgent
                from ..env.observation import ObservationEncoder
                from ..env.action_mask import ActionMaskHandler
                from ..trainer.neural_agent import NeuralAgent

                # OpponentAgent 内部已有 NeuralAgent，直接使用
                # 但 BattleSimulator 需要 AntWarAgent 接口
                # 使用 PPOWarAgent 包装
                return PPOWarAgent(
                    player_id=pid,
                    ppo_agent=agent,
                    observation_encoder=ObservationEncoder(),
                    action_mask_handler=ActionMaskHandler(),
                )

            rating = self.payoff._trueskill_ratings.get(opponent_id)
            elo_approx = rating.mu * (1600 / 25) if rating else 1200.0  # 粗略换算

            return ReferenceAgent(
                name=opponent_id,
                elo=elo_approx,
                agent_factory=factory,
            )
        except Exception as e:
            logger.warning(f"Failed to create reference from {checkpoint_path}: {e}")
            return None

    def _create_agent_from_individual(self, individual: Individual, player_id: int):
        """从 Individual 创建对战 Agent"""
        from ..battle.ga_war_agent import GAWarAgent

        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config["network"]["hidden_dim"],
            enable_auxiliary=self.config.get("network", {}).get("enable_auxiliary", True),
        ).to(self.device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(player_id=player_id, network=network, device=self.device)

    def _save_seed_model(self, seed: Individual, generation: int) -> str:
        """保存种子模型到文件"""
        output_dir = self.config["_output_dir"]
        seed_dir = os.path.join(output_dir, "generations", f"gen_{generation:04d}", "seed_models")
        os.makedirs(seed_dir, exist_ok=True)

        filepath = os.path.join(seed_dir, f"seed_{seed.seed_rank}.pt")

        state_dict = genome_to_state_dict(seed.genome)
        torch.save({
            "generation": generation,
            "individual_id": seed.id,
            "seed_rank": seed.seed_rank,
            "elo_rating": seed.elo_rating,
            "policy_state_dict": state_dict,
            "parent_ids": seed.parent_ids,
        }, filepath)

        return filepath

    def _save_checkpoint(self, generation: int, population: Population) -> None:
        """保存完整训练状态"""
        output_dir = self.config["_output_dir"]
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        filepath = os.path.join(ckpt_dir, f"checkpoint_gen_{generation:04d}.pt")
        self.checkpoint_mgr.save(
            filepath=filepath,
            generation=generation,
            population=population,
            opponent_pool=self.opponent_pool,
            payoff=self.payoff,
            config=self.config,
        )

    def _save_league_state(self) -> None:
        """持久化对手池状态"""
        output_dir = self.config["_output_dir"]
        league_dir = os.path.join(output_dir, "league")
        os.makedirs(league_dir, exist_ok=True)

        self.opponent_pool.save_state(os.path.join(league_dir, "pool.json"))
        self.payoff.save_state(os.path.join(league_dir, "payoff.json"))

    def _create_battle_config(self):
        """创建对战配置"""
        from ..battle.battle_config import GABattleConfig
        return GABattleConfig(
            max_rounds=self.config["env"]["max_steps"],
            n_battles=self.config["baseline_battle"]["n_battles"],
            device=self.config["system"]["device"],
        )
```

### 8.2 GA 检查点管理器（改造自 ppo_v2）

**文件**：`ppo_v9/src/ppo_ga/trainer/checkpoint_manager.py`

**操作步骤**：

1. 拷贝 `ppo_v2/src/ppo_antwar/trainer/checkpoint_manager.py` 到 `ppo_v9/src/ppo_ga/trainer/checkpoint_manager.py`
2. 修改 import 路径
3. 新增 `GACheckpointManager` 类

```python
from typing import Any, Dict, Optional, List
import torch
from loguru import logger

from ..evolution.population_manager import Population, Individual
from ..genome.genome import Genome


class GACheckpointManager:
    """GA 训练检查点管理器"""

    def __init__(self, device: torch.device):
        self._device = device

    def save(
        self,
        filepath: str,
        generation: int,
        population: Population,
        opponent_pool: Any,
        payoff: Any,
        config: Dict[str, Any],
    ) -> None:
        """保存完整训练状态到 checkpoint 文件。

        保存内容：
        - generation: 当前代次
        - population: 种群数据（所有个体的基因组 + ELO + 元信息）
        - opponent_pool_state: 对手池状态
        - payoff_state: TrueSkill 评分矩阵状态
        - config: 训练配置
        """
        # 序列化种群
        population_data = []
        for ind in population.individuals:
            population_data.append({
                "id": ind.id,
                "generation": ind.generation,
                "genome_weights": ind.genome.weights,
                "genome_shapes": ind.genome.shapes,
                "genome_param_names": ind.genome.param_names,
                "elo_rating": ind.elo_rating,
                "seed_rank": ind.seed_rank,
                "parent_ids": ind.parent_ids,
                "mutation_desc": ind.mutation_desc,
            })

        checkpoint = {
            "generation": generation,
            "population_data": population_data,
            "population_stats": {
                "best_elo": population.best_elo,
                "mean_elo": population.mean_elo,
                "diversity": population.diversity,
            },
            "config": config,
        }

        # 对手池和 payoff 状态保存到单独的 JSON 文件（在 trainer 中处理）
        # checkpoint 中仅保存文件路径引用
        output_dir = config.get("_output_dir", "")
        league_dir = os.path.join(output_dir, "league") if output_dir else ""
        checkpoint["league_dir"] = league_dir

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} at generation {generation}")

    def load(
        self,
        filepath: str,
    ) -> Dict[str, Any]:
        """从 checkpoint 文件加载训练状态。

        Returns:
            包含 generation, population, ... 的字典
        """
        import os
        checkpoint = torch.load(filepath, map_location=self._device)

        # 重建种群
        individuals = []
        for ind_data in checkpoint["population_data"]:
            # 重建 layer_boundaries（需要网络实例）
            # 简化方案：从 weights 和 shapes 重建
            from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
            from ..genome.genome import extract_genome

            network = AntWarPolicyValueNetwork()
            temp_genome = extract_genome(network)

            genome = Genome(
                weights=ind_data["genome_weights"],
                shapes=[tuple(s) for s in ind_data["genome_shapes"]],
                layer_boundaries=temp_genome.layer_boundaries,
                param_count=len(ind_data["genome_weights"]),
                param_names=ind_data["genome_param_names"],
            )

            individual = Individual(
                id=ind_data["id"],
                generation=ind_data["generation"],
                genome=genome,
                elo_rating=ind_data["elo_rating"],
                seed_rank=ind_data["seed_rank"],
                parent_ids=ind_data["parent_ids"],
                mutation_desc=ind_data["mutation_desc"],
            )
            individuals.append(ind_data)

        population = Population(
            generation=checkpoint["generation"],
            individuals=individuals,
            **checkpoint.get("population_stats", {}),
        )

        return {
            "generation": checkpoint["generation"],
            "population": population,
            "config": checkpoint.get("config", {}),
        }
```

### 8.3 GA 日志子系统（新建）

**文件**：`ppo_v9/src/ppo_ga/trainer/logging_subsystem.py`

```python
import os
import json
import time
from typing import Dict, Any, List, Optional
from loguru import logger

from ..evolution.population_manager import Population, Individual


class GALoggingSubsystem:
    """GA 日志子系统 —— 管理所有日志输出"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._output_dir = config.get("_output_dir", "outputs/ga_run")
        self._setup_loguru()

    def _setup_loguru(self):
        """配置 loguru 日志输出"""
        training_dir = os.path.join(self._output_dir, "training")
        os.makedirs(training_dir, exist_ok=True)

        # 移除默认 handler
        from loguru import logger as loguru_logger
        loguru_logger.remove()

        # 控制台输出
        loguru_logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.get("logging", {}).get("log_level", "INFO"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        # 主日志文件
        loguru_logger.add(
            os.path.join(training_dir, "ga_training_{time}.log"),
            level="INFO",
            rotation="100 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

        # 错误日志文件
        loguru_logger.add(
            os.path.join(training_dir, "ga_training_error_{time}.log"),
            level="ERROR",
            rotation="50 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    def log_generation(
        self,
        generation: int,
        population: Population,
        seeds: List[Individual],
        baseline_results: Dict[str, Dict[str, Any]],
        duration: float,
    ) -> None:
        """记录每代的日志"""
        # 1. 控制台输出
        logger.info(
            f"Generation {generation}: "
            f"best_elo={population.best_elo:.1f}, "
            f"mean_elo={population.mean_elo:.1f}, "
            f"diversity={population.diversity:.4f}, "
            f"duration={duration:.1f}s"
        )

        # 2. 保存种群统计 JSON
        self._save_population_stats(generation, population, seeds)

        # 3. 保存对战结果 JSON
        self._save_battle_results(generation, population, seeds)

        # 4. 保存基线验证结果 JSON
        if baseline_results:
            self._save_evaluation_results(generation, baseline_results)

        # 5. TensorBoard（如果启用）
        if self.config.get("logging", {}).get("tensorboard", False):
            self._write_tensorboard(generation, population, baseline_results)

    def log_training_complete(self, final_generation: int, total_time: float) -> None:
        """记录训练完成"""
        logger.info(
            f"GA Training complete: {final_generation} generations, "
            f"total_time={total_time:.1f}s"
        )

        # 保存训练完成日志
        training_dir = os.path.join(self._output_dir, "training")
        filepath = os.path.join(training_dir, "training_complete.log")
        with open(filepath, "w") as f:
            f.write(f"Final generation: {final_generation}\n")
            f.write(f"Total time: {total_time:.1f}s\n")

    def close(self) -> None:
        """关闭日志子系统"""
        pass  # loguru 自动管理

    def _save_population_stats(self, generation: int, population: Population, seeds: List[Individual]) -> None:
        """保存种群统计到 JSON"""
        gen_dir = os.path.join(self._output_dir, "generations", f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        stats = {
            "generation": generation,
            "population_size": len(population.individuals),
            "best_elo": population.best_elo,
            "mean_elo": population.mean_elo,
            "diversity": population.diversity,
            "num_seeds": len(seeds),
            "seed_elo_range": (
                seeds[-1].elo_rating if seeds else 0.0,
                seeds[0].elo_rating if seeds else 0.0,
            ),
        }

        filepath = os.path.join(gen_dir, "population_stats.json")
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)

    def _save_battle_results(self, generation: int, population: Population, seeds: List[Individual]) -> None:
        """保存对战结果到 JSON"""
        gen_dir = os.path.join(self._output_dir, "generations", f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        # ELO 排名表
        rankings = [
            {"id": ind.id, "elo": ind.elo_rating, "seed_rank": ind.seed_rank}
            for ind in sorted(population.individuals, key=lambda x: x.elo_rating, reverse=True)
        ]

        # 种子名单
        seed_list = [
            {"id": s.id, "elo": s.elo_rating, "rank": s.seed_rank, "parents": s.parent_ids}
            for s in seeds
        ]

        results = {
            "generation": generation,
            "rankings": rankings,
            "seeds": seed_list,
        }

        filepath = os.path.join(gen_dir, "battle_results.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def _save_evaluation_results(self, generation: int, baseline_results: Dict) -> None:
        """保存基线验证结果到 JSON"""
        gen_dir = os.path.join(self._output_dir, "generations", f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        filepath = os.path.join(gen_dir, "evaluation.json")
        with open(filepath, "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)

    def _write_tensorboard(self, generation: int, population: Population, baseline_results: Dict) -> None:
        """写入 TensorBoard 数据"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self._output_dir, "tensorboard")
            writer = SummaryWriter(tb_dir)

            writer.add_scalar("ga/best_elo", population.best_elo, generation)
            writer.add_scalar("ga/mean_elo", population.mean_elo, generation)
            writer.add_scalar("ga/diversity", population.diversity, generation)

            # 基线胜率
            if baseline_results:
                for seed_id, seed_results in baseline_results.items():
                    for baseline_name, stats in seed_results.items():
                        win_rate = stats.get("agent1_wins", 0) / max(stats.get("total_battles", 1), 1)
                        writer.add_scalar(f"ga/baseline_win_rate/{baseline_name}", win_rate, generation)

            writer.close()
        except ImportError:
            logger.warning("TensorBoard not available, skipping tensorboard logging")
```

---

## 9. 配置系统与入口脚本

### 9.1 配置解析器（改造自 ppo_v2）

**文件**：`ppo_v9/src/ppo_ga/config/config_parser.py`

**操作步骤**：

1. 拷贝 `ppo_v2/src/ppo_antwar/config/config_parser.py` 到 `ppo_v9/src/ppo_ga/config/config_parser.py`
2. 修改 import 路径
3. 替换 `_STRUCTURAL_DEFAULTS` 和 `_TYPE_MAPPING` 为 GA 版本
4. 将 `PPORLConfigParser` 重命名为 `GAConfigParser`
5. 将 `create_ppo_config` 重命名为 `create_ga_config`

**需要修改的内容**：

```python
# 替换 _STRUCTURAL_DEFAULTS
_STRUCTURAL_DEFAULTS: Dict[str, Any] = {
    "system": {"device": _get_default_device()},
    "network": {"hidden_dim": 256, "enable_auxiliary": True},
    "evolution": {
        "max_generations": 200,
        "population_size": 80,
        "num_seeds": 8,
        "elitism_count": 2,
        "convergence_window": 20,
        "convergence_threshold": 5.0,
    },
    "crossover": {"method": "layer_wise", "rate": 0.8},
    "mutation": {"method": "gaussian", "rate": 0.1, "scale": 0.03, "decay": 0.99},
    "selection": {
        "method": "fixed_reference_elo",
        "anchor_references": ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"],
        "dynamic_references": 2,
        "include_previous_champion": True,
        "elo_k_factor": 64,
        "elo_initial": 1200,
    },
    "opponent_pool": {
        "max_size": 20,
        "min_games_threshold": 8,
        "exploit_prob": 0.7,
        "decay": 0.99,
        "min_win_rate_games": 8,
    },
    "baseline_battle": {"enabled": True, "interval": 1, "n_battles": 12},
    "env": {"player_id": 0, "backend_type": "python", "max_steps": 512},
    "logging": {"log_level": "INFO", "save_interval": 10, "tensorboard": True},
}

# 替换 _TYPE_MAPPING
_TYPE_MAPPING: Dict[str, type] = {
    "evolution.max_generations": int,
    "evolution.population_size": int,
    "evolution.num_seeds": int,
    "evolution.elitism_count": int,
    "evolution.convergence_window": int,
    "evolution.convergence_threshold": float,
    "crossover.rate": float,
    "mutation.rate": float,
    "mutation.scale": float,
    "mutation.decay": float,
    "selection.elo_k_factor": float,
    "selection.elo_initial": float,
    "selection.dynamic_references": int,
    "opponent_pool.max_size": int,
    "opponent_pool.min_games_threshold": int,
    "opponent_pool.exploit_prob": float,
    "opponent_pool.decay": float,
    "baseline_battle.enabled": bool,
    "baseline_battle.interval": int,
    "baseline_battle.n_battles": int,
    "network.hidden_dim": int,
    "network.enable_auxiliary": bool,
    "env.max_steps": int,
    "logging.save_interval": int,
    "logging.tensorboard": bool,
}
```

### 9.2 路径管理器（改造自 ppo_v2）

**文件**：`ppo_v9/src/ppo_ga/config/path_config.py`

**操作步骤**：

1. 拷贝 `ppo_v2/src/ppo_antwar/config/path_config.py` 到 `ppo_v9/src/ppo_ga/config/path_config.py`
2. 修改 import 路径
3. 新增 GA 特有的子目录常量和方法

**需要新增的内容**：

```python
_SUBDIR_GENERATIONS = "generations"
_SUBDIR_LEAGUE = "league"

# 在 PathConfig 类中新增方法：

def get_generations_dir(self, run_id: str) -> Path:
    path = self._base_output_dir / run_id / _SUBDIR_GENERATIONS
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_league_dir(self, run_id: str) -> Path:
    path = self._base_output_dir / run_id / _SUBDIR_LEAGUE
    path.mkdir(parents=True, exist_ok=True)
    return path
```

### 9.3 默认配置文件

**文件**：`ppo_v9/configs/ga_evo.yaml`

```yaml
# ============================================================
# GA 进化训练配置文件 - AntWar v9
# ============================================================

# 系统配置
system:
  device: "cuda"
  seed: 42

# 网络结构配置
network:
  hidden_dim: 256
  enable_auxiliary: true

# 进化参数
evolution:
  max_generations: 200
  population_size: 80
  num_seeds: 8
  elitism_count: 2
  convergence_window: 20
  convergence_threshold: 5.0

# 杂交参数
crossover:
  method: "layer_wise"
  rate: 0.8

# 变异参数
mutation:
  method: "gaussian"
  rate: 0.1
  scale: 0.03
  decay: 0.99

# 选拔参数
selection:
  method: "fixed_reference_elo"
  anchor_references:
    - "BasicRandomAI"
    - "BasicTowerAI"
    - "MediumRuleAI"
  dynamic_references: 2
  include_previous_champion: true
  elo_k_factor: 64
  elo_initial: 1200

# 对手池参数
opponent_pool:
  max_size: 20
  min_games_threshold: 8
  exploit_prob: 0.7
  decay: 0.99
  min_win_rate_games: 8

# 基线验证参数
baseline_battle:
  enabled: true
  interval: 1
  n_battles: 12

# 环境参数
env:
  player_id: 0
  backend_type: "python"
  max_steps: 512

# 日志参数
logging:
  log_level: "INFO"
  save_interval: 10
  tensorboard: true
```

### 9.4 训练入口脚本

**文件**：`ppo_v9/train_ga.py`

```python
"""GA 进化训练入口脚本"""

import argparse
import sys
import os
import time

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="GA Evolution Training for AntWar")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--max_generations", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--mutation_rate", type=float, default=None)
    parser.add_argument("--mutation_scale", type=float, default=None)
    parser.add_argument("--crossover_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # 确定配置文件路径
    if args.config:
        yaml_path = args.config
    else:
        yaml_path = os.path.join(
            os.path.dirname(__file__), "configs", "ga_evo.yaml"
        )

    # 构建 CLI 覆盖
    cli_overrides = {}
    if args.population_size is not None:
        cli_overrides.setdefault("evolution", {})["population_size"] = args.population_size
    if args.max_generations is not None:
        cli_overrides.setdefault("evolution", {})["max_generations"] = args.max_generations
    if args.num_seeds is not None:
        cli_overrides.setdefault("evolution", {})["num_seeds"] = args.num_seeds
    if args.mutation_rate is not None:
        cli_overrides.setdefault("mutation", {})["rate"] = args.mutation_rate
    if args.mutation_scale is not None:
        cli_overrides.setdefault("mutation", {})["scale"] = args.mutation_scale
    if args.crossover_rate is not None:
        cli_overrides.setdefault("crossover", {})["rate"] = args.crossover_rate
    if args.device is not None:
        cli_overrides.setdefault("system", {})["device"] = args.device

    # 加载配置
    from ppo_ga.config.config_parser import create_ga_config
    config = create_ga_config(yaml_path, cli_overrides if cli_overrides else None)

    # 设置输出目录
    run_id = f"ga_run_{time.strftime('%Y%m%d_%H%M%S')}"
    from ppo_ga.config.path_config import PathConfig
    path_config = PathConfig()
    output_dir = str(path_config.get_run_dir(run_id))
    config["_output_dir"] = output_dir

    # 记录启动配置
    logger.info(f"GA Training config: {config}")

    # 创建训练器
    from ppo_ga.trainer.genetic_evolution_trainer import GeneticEvolutionTrainer
    trainer = GeneticEvolutionTrainer(config)

    # 断点续训
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: 实现断点续训加载逻辑

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
```

---

## 10. 断点续训与错误处理

### 10.1 断点续训

**Checkpoint 保存内容**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `generation` | int | 当前代次 |
| `population_data` | List[Dict] | 所有个体的基因组 + 元信息 |
| `population_stats` | Dict | 种群统计（best_elo, mean_elo, diversity） |
| `config` | Dict | 训练配置 |
| `league_dir` | str | 对手池状态文件目录 |

**恢复流程**：

1. 加载 checkpoint 文件
2. 重建种群（从 `population_data` 恢复所有 Individual）
3. 从 `league_dir` 加载对手池状态（`pool.json` + `payoff.json`）
4. 从 `generation + 1` 继续训练

**恢复时需要重建 `layer_boundaries`**：

由于 `layer_boundaries` 依赖于网络结构（参数名和形状），而 checkpoint 中保存了 `genome_weights`、`genome_shapes`、`genome_param_names`，恢复时需要创建一个临时网络实例来获取 `layer_boundaries`：

```python
# 恢复 layer_boundaries
temp_network = AntWarPolicyValueNetwork(hidden_dim=config["network"]["hidden_dim"])
temp_genome = extract_genome(temp_network)
# temp_genome.layer_boundaries 即为所需的层边界信息
```

### 10.2 异常处理策略

遵循项目原则"不做容错，让错误尽早暴露"，但合理捕获并记录异常：

| 场景 | 处理方式 |
|------|---------|
| 单局对战崩溃 | 该局判负，记录到 error 日志，不影响其他对战 |
| 对战超时（300s） | BattleSimulator 已有超时机制，超时返回 error 结果 |
| 网络前向推理 NaN | GAWarAgent 中回退到随机合法动作（与 PPOWarAgent 一致） |
| checkpoint 加载失败 | 抛出异常，终止训练 |
| 对手池状态加载失败 | 抛出异常，终止训练 |
| 配置文件缺失/格式错误 | 抛出异常，终止训练 |

---

## 11. 环境与工具模块

### 11.1 直接复用的环境模块

从 `ppo_v2/src/ppo_antwar/env/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/env/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `observation.py` | `observation.py` | 直接拷贝 + 修改 import |
| `action_mask.py` | `action_mask.py` | 直接拷贝 + 修改 import |

> 注意：`antwar_env.py`、`player_state.py`、`state_snapshot.py` 在 GA 框架中不需要，因为 GA 不使用 Gym 风格的 step/reward 循环。对战通过 `BattleSimulator` 直接调用 SDK。

### 11.2 直接复用的工具模块

从 `ppo_v2/src/ppo_antwar/utils/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/utils/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `action_constants.py` | `action_constants.py` | 直接拷贝 + 修改 import |
| `obs_utils.py` | `obs_utils.py` | 直接拷贝 + 修改 import |

> 注意：`gae.py`、`aux_labels.py` 在 GA 框架中不需要。

### 11.3 直接复用的兼容模块

从 `ppo_v2/src/ppo_antwar/compat/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/compat/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `adapters.py` | `adapters.py` | 直接拷贝 + 修改 import |

### 11.4 Agent 协议模块

从 `ppo_v2/src/ppo_antwar/_agent/` 拷贝以下文件到 `ppo_v9/src/ppo_ga/_agent/`：

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `protocol.py` | `protocol.py` | 直接拷贝 + 修改 import |

### 11.5 NeuralAgent 复用

从 `ppo_v2/src/ppo_antwar/trainer/neural_agent.py` 拷贝到 `ppo_v9/src/ppo_ga/trainer/neural_agent.py`。

NeuralAgent 在 GA 框架中的用途：`OpponentAgent.from_checkpoint` 内部创建 NeuralAgent 实例。

---

## 12. 依赖声明

**文件**：`ppo_v9/requirements.txt`

```
torch>=2.0
numpy>=1.24
trueskill>=0.4
loguru>=0.7
pyyaml>=6.0
cloudpickle>=2.0
tensorboard>=2.12
```

---

## 13. 实施顺序建议

按以下顺序实施，每步完成后可独立验证：

| 步骤 | 内容 | 验证方式 |
|------|------|---------|
| 1 | 创建目录结构 + 所有 `__init__.py` | 目录存在 |
| 2 | 拷贝直接复用文件 + 修改 import | `python -c "from ppo_ga.network import AntWarPolicyValueNetwork"` |
| 3 | 实现 genome/genome.py | 提取/加载基因组往返测试 |
| 4 | 实现 genome/weight_init.py | 初始化后检查参数分布 |
| 5 | 实现 evolution/crossover.py + mutation.py | 单元测试杂交/变异 |
| 6 | 实现 evolution/population_manager.py | 初始化种群 + 进化一代 |
| 7 | 实现 selection/elo_rating.py | ELO 更新测试 |
| 8 | 实现 battle/ga_war_agent.py + battle_config.py | 单局对战测试 |
| 9 | 实现 selection/battle_selection.py | 小规模种群评估 |
| 10 | 实现 evaluation/baseline_evaluator.py | 基线验证测试 |
| 11 | 实现 trainer/logging_subsystem.py + checkpoint_manager.py | 日志输出测试 |
| 12 | 实现 trainer/genetic_evolution_trainer.py | 完整训练 2 代 |
| 13 | 实现 config/ + train_ga.py | CLI 启动测试 |
| 14 | 端到端测试：运行 5 代完整进化 | 检查日志/模型输出 |

---

## 14. 关键数据流图

```
train_ga.py
  │
  ▼
GeneticEvolutionTrainer.train()
  │
  ├── 阶段一：PopulationManager.init_population() / evolve()
  │     │
  │     ├── AntWarPolicyValueNetwork() → random_init_network() → extract_genome()
  │     │
  │     └── layer_wise_crossover() + gaussian_mutation()
  │
  ├── 阶段二：BattleSelection.evaluate_population()
  │     │
  │     ├── _build_dynamic_references() → OpponentSelector.select()
  │     │
  │     ├── _create_ga_agent() → GAWarAgent(network=...)
  │     │
  │     └── BattleSimulator.run_battles() → ELO 评分更新
  │
  ├── 阶段三：BattleSelection.select_seeds()
  │
  ├── 阶段四：BaselineEvaluator.evaluate_seeds()
  │     │
  │     └── BattleSimulator.run_battles() → aggregate_results()
  │
  ├── 阶段五：OpponentPool.add() → _save_seed_model()
  │
  └── 阶段六：GALoggingSubsystem.log_generation() → GACheckpointManager.save()
```

---

## 15. 与 ppo_v2 的关键差异总结

| 维度 | ppo_v2 | ppo_v9 (GA) |
|------|--------|-------------|
| 优化方式 | PPO 梯度下降 | 遗传算法（无梯度） |
| 训练循环 | episode → collect → PPO update | generation → evaluate → select → evolve |
| 参数更新 | 逐 batch 梯度更新 | 整体基因组杂交/变异 |
| Agent 推理 | NeuralAgent（含 log_prob/value） | GAWarAgent（仅 action） |
| 对战用途 | 自对弈收集训练数据 | 评估个体适应度 |
| 评分系统 | TrueSkill（训练中持续更新） | ELO（代内）+ TrueSkill（跨代对手池） |
| 环境使用 | AntWarEnv（step/reward 循环） | BattleSimulator（直接调用 SDK） |
| 辅助损失 | 塔伤害/金币/基地伤害 MSE | 不使用（无梯度） |
| 检查点 | policy + optimizer 状态 | 种群基因组 + 对手池状态 |
