import argparse
from datetime import datetime

import torch

from loguru import logger

from src.ppo_antwar import (
    AntWarEnv,
    PPOTrainer,
    SelfPlayTrainer,
    BaselineBattleConfig,
    create_ppo_config,
    PathConfig,
    create_ppo_callbacks,
    Logger as AppLogger,
    TimeTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AntWar PPO Training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument("--episodes", type=int, default=None,
                        help="总训练 episode 数")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="学习率")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="并行环境数")
    parser.add_argument("--opponent-pool-size", type=int, default=None,
                        help="对手池大小")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="模型保存间隔")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")

    parser.add_argument("--battle-interval", type=int, default=None,
                        help="对战评估间隔 (每 N 个 episode)")
    parser.add_argument("--opponent-update-interval", type=int, default=None,
                        help="对手更新间隔 (episode)")
    parser.add_argument("--n-battles", type=int, default=None,
                        help="每次评估的对战次数")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="单局最大步数")

    parser.add_argument("--ent-coef", type=float, default=None,
                        help="熵正则化系数")
    parser.add_argument("--gamma", type=float, default=None,
                        help="折扣因子")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="GAE λ 参数")
    parser.add_argument("--clip-eps", type=float, default=None,
                        help="PPO 裁剪参数 ε")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="PPO 更新轮数")

    parser.add_argument("--min-opponent-games", type=int, default=None,
                        help="对手最少参与对战次数")
    parser.add_argument("--exploit-prob", type=float, default=None,
                        help="利用型对手被选中的概率")
    parser.add_argument("--baseline-agents", nargs='+', type=str, default=None,
                        help="Baseline agents 列表 (如 BasicTowerAI BasicRandomAI)")

    return parser.parse_args()


def main():
    """训练入口函数

    流程：
    1. 解析 CLI 参数
    2. 加载配置
    3. 初始化路径管理
    4. 创建环境工厂
    5. 构建网络
    6. 创建 PPOTrainer
    7. 创建 SelfPlayTrainer
    8. 创建回调
    9. 启动训练主循环
    """
    args = parse_args()

    cli_overrides = _build_cli_overrides(args)
    config = create_ppo_config(
        yaml_path=args.config,
        cli_overrides=cli_overrides,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_config = PathConfig(base_output_dir=args.output_dir or "")

    log_dir = path_config.get_training_dir(run_id)
    app_logger = AppLogger(str(log_dir), run_id)
    time_tracker = TimeTracker()
    time_tracker.start()

    device = torch.device(config["system"]["device"])
    if device.type == "cpu":
        logger.warning("Training on CPU. GPU acceleration is not available.")

    def env_factory():
        env_cfg = config["env"]
        return AntWarEnv(
            player_id=env_cfg["player_id"],
            backend_type=env_cfg["backend_type"],
            prefer_native=env_cfg["prefer_native"],
        )

    battle_cfg = BaselineBattleConfig(
        max_rounds=config["training"]["max_steps_per_episode"],
        n_battles=config["training"]["n_battles"],
        baseline_agents=config.get("selfplay", {}).get("baseline_agents", []),
        device=device.type,
    )

    callbacks = create_ppo_callbacks(
        save_interval=config["training"]["save_interval"],
        log_interval=10,
        path_config=path_config,
        enable_tensorboard=True,
        tensorboard_dir=str(path_config.get_run_dir(run_id) / "tensorboard"),
        tensorboard_interval=1,
    )

    trainer = PPOTrainer(
        env_factory=env_factory,
        config=config,
        device=device,
        base_dir=str(path_config.get_run_dir(run_id)),
        callbacks=callbacks,
    )

    trainer.log_config_summary()

    selfplay_trainer = SelfPlayTrainer(
        trainer=trainer,
        config=config,
        path_config=path_config,
        run_id=run_id,
    )

    if callbacks is not None:
        callbacks.init_callback(trainer)
        callbacks.on_training_start()

    total_episodes = config["training"]["total_episodes"]
    try:
        selfplay_trainer.train(
            num_episodes=total_episodes,
            env_factory=env_factory,
            callbacks=callbacks,
        )
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise
    finally:
        if callbacks is not None:
            callbacks.on_training_end()
        app_logger.flush()


def _build_cli_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.learning_rate is not None:
        overrides.setdefault("ppo", {})["lr"] = args.learning_rate
    if args.batch_size is not None:
        overrides.setdefault("ppo", {})["batch_size"] = args.batch_size
    if args.episodes is not None:
        overrides.setdefault("training", {})["total_episodes"] = args.episodes
    if args.n_envs is not None:
        overrides.setdefault("training", {})["n_envs"] = args.n_envs
    if args.opponent_pool_size is not None:
        overrides.setdefault("selfplay", {})["opponent_pool_size"] = args.opponent_pool_size
    if args.save_interval is not None:
        overrides.setdefault("training", {})["save_interval"] = args.save_interval

    if args.battle_interval is not None:
        overrides.setdefault("training", {})["battle_interval"] = args.battle_interval
    if args.opponent_update_interval is not None:
        overrides.setdefault("training", {})["opponent_update_interval"] = args.opponent_update_interval
    if args.n_battles is not None:
        overrides.setdefault("training", {})["n_battles"] = args.n_battles
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps_per_episode"] = args.max_steps
    if args.ent_coef is not None:
        overrides.setdefault("ppo", {})["ent_coef"] = args.ent_coef
    if args.gamma is not None:
        overrides.setdefault("ppo", {})["gamma"] = args.gamma
    if args.gae_lambda is not None:
        overrides.setdefault("ppo", {})["gae_lambda"] = args.gae_lambda
    if args.clip_eps is not None:
        overrides.setdefault("ppo", {})["clip_eps"] = args.clip_eps
    if args.ppo_epochs is not None:
        overrides.setdefault("ppo", {})["ppo_epochs"] = args.ppo_epochs
    if args.min_opponent_games is not None:
        overrides.setdefault("selfplay", {})["min_opponent_games"] = args.min_opponent_games
    if args.exploit_prob is not None:
        overrides.setdefault("selfplay", {})["exploit_prob"] = args.exploit_prob
    if args.baseline_agents is not None:
        overrides.setdefault("selfplay", {})["baseline_agents"] = args.baseline_agents

    return overrides if overrides else None


if __name__ == "__main__":
    main()
