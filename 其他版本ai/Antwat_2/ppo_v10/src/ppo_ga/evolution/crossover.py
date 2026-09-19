import random
from typing import Tuple
import numpy as np

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
    child_a_weights = np.concatenate([
        parent_a.weights[:split_idx],
        parent_b.weights[split_idx:],
    ]).astype(np.float32)

    child_b_weights = np.concatenate([
        parent_b.weights[:split_idx],
        parent_a.weights[split_idx:],
    ]).astype(np.float32)

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


def _copy_genome(genome: Genome) -> Genome:
    """深拷贝基因组（权重数组需要 copy）"""
    return Genome(
        weights=genome.weights.copy(),
        shapes=genome.shapes,
        layer_boundaries=genome.layer_boundaries,
        param_count=genome.param_count,
        param_names=genome.param_names,
    )
