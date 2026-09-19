"""ppov9 训练入口 —— 兼容 ppo_v2 的 train_start 部署方式

对 PPO 模式参数（--episodes / --batch-size / --n-envs / --save-interval 等）
做 parse_known_args 静默忽略，只读取 --config / --device / --resume。
"""

import argparse
import sys
import os
import time

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="ppov9 Training Entry")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    # PPO 模式参数由 start_training.sh 传入，parse_known_args 自动忽略
    args, _ = parser.parse_known_args()
    return args


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
    if args.device is not None:
        cli_overrides.setdefault("system", {})["device"] = args.device

    # 加载配置
    from ppo_ga.config.config_parser import create_ga_config
    config = create_ga_config(yaml_path, cli_overrides if cli_overrides else None)

    # 初始化 PathConfig 和 run_id
    run_id = time.strftime('%Y%m%d_%H%M%S')
    from ppo_ga.config.path_config import PathConfig
    path_config = PathConfig()
    config["_output_dir"] = str(path_config.get_run_dir(run_id))

    # 设置 AgentLoader 的 baselines 路径
    from ppo_ga.battle.agent_loader import AgentLoader
    AgentLoader.set_baselines_path(str(path_config.baselines_path))

    # 创建训练器
    from ppo_ga.trainer.genetic_evolution_trainer import GeneticEvolutionTrainer
    trainer = GeneticEvolutionTrainer(config, path_config, run_id)

    # 断点续训
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
