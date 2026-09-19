#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import yaml
import traceback
from pathlib import Path
from easydict import EasyDict

# 在导入任何其他可能使用 multiprocessing 的模块之前设置 start_method
if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    except ImportError:
        pass

REPO_ROOT = Path(__file__).resolve().parent / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger
from ppo_antwar import PPOTrainer, SelfPlayTrainer, make_antwar_env
from ppo_antwar.config.config_parser import create_ppo_config
from ppo_antwar.callbacks import CheckpointCallback, create_ppo_callbacks
from ppo_antwar.config.path_config import PathConfig, PathConfigError


def parse_args():
    parser = argparse.ArgumentParser(description="Train AntWar PPO Agent")
    parser.add_argument("--episodes", type=int, default=None, help="Total training episodes")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO batch size")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--lambda", dest="gae_lambda", type=float, default=None, help="GAE lambda")
    parser.add_argument("--model-save-interval", type=int, default=None, help="Model save interval")
    parser.add_argument("--config", type=str, default=None, help="Config file path (YAML format)")
    
    # SelfPlay specific parameters
    parser.add_argument("--opponent-update-interval", type=int, default=None, help="Opponent update interval")
    parser.add_argument("--battle-interval", type=int, default=None, help="Battle evaluation interval")
    parser.add_argument("--n-battles", type=int, default=None, help="Number of battles per evaluation")
    parser.add_argument("--opponent-pool-size", type=int, default=None, help="Size of opponent pool")
    parser.add_argument("--min-opponent-games", type=int, default=None, help="Minimum games per opponent")
    parser.add_argument("--exploit-prob", type=float, default=None, help="Exploit probability for opponent selection")
    parser.add_argument("--baseline-agents", type=str, nargs='*', default=None, help="List of baseline agents to use (e.g., BasicTowerAI BasicRandomAgent sample)")
    return parser.parse_args()


def main():
    log_dir = None
    try:
        args = parse_args()

        # REPO_ROOT 定义于 train.py:L22: REPO_ROOT = Path(__file__).resolve().parent / "src"
        yaml_path = args.config or str(REPO_ROOT / "ppo_antwar" / "configs" / "ppo_antwar.yaml")
        config = create_ppo_config(
            yaml_path=yaml_path,
            cli_overrides={
                "ppo": {
                    "lr": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                },
                "training": {
                    "total_episodes": args.episodes,
                    "n_envs": args.n_envs,
                    "save_interval": args.model_save_interval,
                    "opponent_update_interval": args.opponent_update_interval,
                    "battle_interval": args.battle_interval,
                    "n_battles": args.n_battles,
                },
                "selfplay": {
                    "opponent_pool_size": args.opponent_pool_size,
                    "min_opponent_games": args.min_opponent_games,
                    "exploit_prob": args.exploit_prob,
                    "baseline_agents": args.baseline_agents,
                },
            },
        )

        # 使用 PathConfig 管理路径（SDK 和 Baseline 路径会在此处验证）
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        path_config = PathConfig(f"./outputs/{timestamp}")

        log_dir = str(path_config.base_dir)
        logger.info(f"Log directory: {log_dir}")

        config_path = path_config.training_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dict(config), f)

        logger.info("Creating environment factory...")
        env_factory = make_antwar_env(player_id=0)

        logger.info("Creating PPOTrainer...")
        # Create PPOTrainer first
        ppo_trainer = PPOTrainer(
            env_factory=env_factory,
            config=config,
            device=config.system.device,
            base_dir=log_dir,
        )
        logger.info("PPOTrainer created successfully")

        logger.info("Creating SelfPlayTrainer...")
        selfplay_trainer = SelfPlayTrainer(
            trainer=ppo_trainer,
            opponent_pool_size=config.selfplay.opponent_pool_size,
            min_opponent_games=config.selfplay.min_opponent_games,
            exploit_prob=config.selfplay.exploit_prob,
            n_battles=config.training.n_battles,
            battle_interval=config.training.battle_interval,
            baseline_agents=config.selfplay.baseline_agents,
        )
        logger.info("SelfPlayTrainer created successfully")

        logger.info("Creating callbacks...")
        callbacks = create_ppo_callbacks(
            save_interval=config.training.save_interval,
            log_interval=1,
            checkpoint_dir=str(path_config.checkpoint_dir),
            path_config=path_config,
            enable_checkpoint=True,
            enable_metrics_logging=True,
        )
        logger.info(f"Callbacks created successfully")

        logger.info("Starting training...")
        # Start training
        selfplay_trainer.train(
            num_episodes=config.training.total_episodes,
            opponent_update_interval=config.training.opponent_update_interval,
            checkpoint_callback=callbacks,
        )
        logger.info("Training completed successfully")

    except PathConfigError as e:
        logger.error(f"路径配置错误: {e}")
        logger.error("请确保以下目录存在:")
        logger.error("  - Ant-Game/SDK (SDK目录)")
        logger.error("  - baselines (Baseline目录)")
        sys.exit(1)

    except Exception as e:
        error_msg = f"Fatal error during training: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        # 尝试记录到 log_dir
        if log_dir and os.path.exists(log_dir):
            try:
                error_log_path = Path(log_dir) / "system" / "init_error.log"
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log_path, 'w') as f:
                    f.write(f"{error_msg}\n\n")
                    f.write(traceback.format_exc())
            except Exception as log_err:
                logger.error(f"Failed to write error log: {log_err}")

        sys.exit(1)


if __name__ == "__main__":
    main()
