"""GA 进化训练入口脚本 (v10)

本脚本供手动命令行直接调用：
    python3 train_ga.py --config configs/ga_evo_v10.yaml [--max_generations 50 ...]

如需通过 manager.sh 远程启动训练，请使用同目录下的 train.py：
    bash tools/manager.sh --server server9 --model ppo_v10 train_start --config configs/ga_evo_v10.yaml
train.py 是 train_ga.py 的兼容层，会静默忽略 manager.sh 传入的 PPO 参数（--episodes / --batch-size 等），
最终加载同一个 GeneticEvolutionTrainer。
"""

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
            os.path.dirname(__file__), "configs", "ga_evo_v10.yaml"
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

    # 初始化 PathConfig 和 run_id（与 ppo_v2 对齐：outputs/{run_id}/）
    run_id = f"ga_run_{time.strftime('%Y%m%d_%H%M%S')}"
    from ppo_ga.config.path_config import PathConfig
    path_config = PathConfig()
    config["_output_dir"] = str(path_config.get_run_dir(run_id))

    # 设置 AgentLoader 的 baselines 路径（支持加载 sample 等外部 agent）
    from ppo_ga.battle.agent_loader import AgentLoader
    AgentLoader.set_baselines_path(str(path_config.baselines_path))

    # 创建训练器（传入 PathConfig 和 run_id，内部由 PathConfig 管理所有子目录）
    from ppo_ga.trainer.genetic_evolution_trainer import GeneticEvolutionTrainer
    trainer = GeneticEvolutionTrainer(config, path_config, run_id)

    # 断点续训
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: 实现断点续训加载逻辑

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
