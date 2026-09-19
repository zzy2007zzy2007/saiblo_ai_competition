from typing import Optional, Tuple

import numpy as np
from loguru import logger

from ppo_antwar.monitor.constants import HISTORICAL_METRICS_WINDOW, RECENT_METRICS_WINDOW


def compute_and_save_training_status(trainer, episode: int, battle_result: Optional[Tuple[int, int, int]]) -> None:
    if not trainer.logger or not trainer.time_tracker:
        return

    try:
        if trainer.time_tracker.training_episodes_count > 0:
            training_speed = (
                trainer.time_tracker.training_episodes_count
                / (trainer.time_tracker.training_time_total / 60 if trainer.time_tracker.training_time_total > 0 else 0)
            )
        else:
            training_speed = 0

        history = trainer.training_metrics_cache.get("history", [])
        recent_100 = history[-HISTORICAL_METRICS_WINDOW:] if len(history) >= HISTORICAL_METRICS_WINDOW else history
        recent_10 = history[-RECENT_METRICS_WINDOW:] if len(history) >= RECENT_METRICS_WINDOW else history

        avg_reward_last_100 = np.mean([m["avg_reward"] for m in recent_100]) if recent_100 else 0
        avg_reward_last_10 = np.mean([m["avg_reward"] for m in recent_10]) if recent_10 else 0

        loss_values = [
            m["avg_loss"]
            for m in recent_100
            if m.get("avg_loss") is not None and not np.isnan(m["avg_loss"])
        ]
        avg_loss_last_100 = np.mean(loss_values) if loss_values else 0

        avg_rounds = np.mean(trainer.episode_lengths[-HISTORICAL_METRICS_WINDOW:]) if trainer.episode_lengths else 0

        win_rate = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        if battle_result:
            wins, losses, draws = battle_result
            total = wins + losses + draws
            win_rate = wins / total if total > 0 else 0
            total_wins = wins
            total_losses = losses
            total_draws = draws

        reward_std = np.std([m["avg_reward"] for m in recent_100]) if len(recent_100) > 1 else 0
        loss_std = np.std(loss_values) if len(loss_values) > 1 else 0
        training_stability = {"reward_std": float(reward_std), "loss_std": float(loss_std)}

        battle_data = {}

        trainer.logger.save_training_status(
            episode=episode,
            total_steps=trainer.total_steps,
            best_avg_reward=trainer.best_avg_reward,
            avg_reward_last_100=avg_reward_last_100,
            avg_reward_last_10=avg_reward_last_10,
            avg_loss_last_100=avg_loss_last_100,
            avg_rounds=avg_rounds,
            total_wins=total_wins,
            total_losses=total_losses,
            total_draws=total_draws,
            win_rate=win_rate,
            training_speed=training_speed,
            ppo_metrics={},
            training_stability=training_stability,
            battle_data=battle_data,
            recent_battle_data=battle_data,
            start_time=trainer.training_start_time,
            current_time=trainer.time_tracker.last_training_end_time if trainer.time_tracker else None,
        )

        if battle_data:
            trainer.logger.save_battle_stats(battle_data)

        trainer.time_tracker.save_time_statistics()

    except Exception as e:
        logger.warning(f"⚠ 保存训练状态异常: {e}")
