from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random

import numpy as np
from loguru import logger

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
        logger.info(f"Population initialized: {len(individuals)} individuals, genome_dim={individuals[0].genome.param_count}")
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

        n_elite = min(self.elitism_count, len(seeds))
        n_crossover = len(new_individuals) - n_elite
        logger.info(
            f"Evolved gen {generation}: {n_elite} elite + {n_crossover} crossover/mutation = {len(new_individuals)} total"
        )

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
