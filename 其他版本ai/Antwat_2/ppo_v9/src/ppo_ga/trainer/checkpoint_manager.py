from typing import Any, Dict, Optional, List
import os

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
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
    ) -> Dict[str, Any]:
        """从 checkpoint 文件加载训练状态。

        Args:
            filepath: checkpoint 文件路径
            hidden_dim: 网络隐藏层维度（用于重建 layer_boundaries）
            enable_auxiliary: 是否启用辅助头

        Returns:
            包含 generation, population, ... 的字典
        """
        checkpoint = torch.load(filepath, map_location=self._device, weights_only=False)

        # 重建种群
        individuals = []
        for ind_data in checkpoint["population_data"]:
            # 重建 layer_boundaries（需要网络实例）
            from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
            from ..genome.genome import extract_genome

            network = AntWarPolicyValueNetwork(
                hidden_dim=hidden_dim,
                enable_auxiliary=enable_auxiliary,
            )
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
            individuals.append(individual)

        population = Population(
            generation=checkpoint["generation"],
            individuals=individuals,
            **checkpoint.get("population_stats", {}),
        )

        logger.info(
            f"Checkpoint loaded: gen={checkpoint['generation']}, population_size={len(individuals)}"
        )

        return {
            "generation": checkpoint["generation"],
            "population": population,
            "config": checkpoint.get("config", {}),
        }
