"""
训练日志记录器 - 封装 Logger 提供训练专用接口
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from .constants import RECENT_METRICS_WINDOW, HISTORICAL_METRICS_WINDOW


class TrainingLogger:
    """训练日志记录器 - 负责训练过程中的日志记录"""

    def __init__(self, log_dir: str, time_tracker, opponent_selector,
                 logger=None):
        """
        Args:
            log_dir: 日志目录
            time_tracker: 时间追踪器
            opponent_selector: 对手选择器
            logger: Logger实例，可选
        """
        self.log_dir = log_dir
        self.time_tracker = time_tracker
        self.opponent_selector = opponent_selector
        self.logger = logger or self._create_logger()

    def _create_logger(self):
        """创建Logger实例"""
        from .logger import Logger
        return Logger(self.log_dir)

    def _get_logger(self):
        """获取Logger实例"""
        return self.logger

    def log_training_start(self, config, device) -> None:
        """记录训练开始"""
        self._get_logger().log_training_start(config, device)

    def log_training_details(self, episode: int, avg_reward: float,
                            avg_loss: float, avg_rounds: float,
                            total_steps: int) -> None:
        """记录训练详情"""
        self._get_logger().log_training_details(
            episode, avg_reward, avg_loss, avg_rounds, total_steps
        )

    def log_training_metrics(self, episode: int, metrics_cache,
                            avg_rounds: float = None, steps: int = None,
                            context: Dict = None) -> None:
        """记录训练指标"""
        self._get_logger().save_training_metrics(
            episode,
            metrics_cache.get('avg_reward', 0),
            metrics_cache.get('avg_loss', 0),
            avg_rounds if avg_rounds is not None else metrics_cache.get('avg_rounds', 0),
            steps if steps is not None else metrics_cache.get('total_steps', 0),
            context,
            metrics_cache
        )

    def log_training_status(self, episode: int, config, battle_result: Tuple[int, int, int],
                           ppo_metrics: Dict, avg_reward: float, avg_loss: float,
                           training_time: float, training_metrics_cache,
                           best_avg_reward: float, episode_lengths: list,
                           training_start_time: str) -> None:
        """保存训练状态

        Args:
            episode: 当前训练轮次
            config: 训练配置
            battle_result: 对战结果 (wins, losses, draws)
            ppo_metrics: PPO指标
            avg_reward: 平均奖励
            avg_loss: 平均损失
            training_time: 训练时间
            training_metrics_cache: 训练指标缓存
            best_avg_reward: 最佳平均奖励
            episode_lengths: 训练回合长度列表
            training_start_time: 训练开始时间
        """
        if self.time_tracker.training_episodes_count > 0:
            training_speed = (
                self.time_tracker.training_episodes_count /
                (self.time_tracker.training_time_total / 60)
                if self.time_tracker.training_time_total > 0 else 0
            )
        else:
            training_speed = 0

        history = training_metrics_cache.get('history', [])
        recent_100 = history[-HISTORICAL_METRICS_WINDOW:] if len(history) >= HISTORICAL_METRICS_WINDOW else history
        recent_10 = history[-RECENT_METRICS_WINDOW:] if len(history) >= RECENT_METRICS_WINDOW else history

        avg_reward_last_100 = np.mean([m['avg_reward'] for m in recent_100]) if recent_100 else 0
        avg_reward_last_10 = np.mean([m['avg_reward'] for m in recent_10]) if recent_10 else 0

        loss_values = [
            m['avg_loss'] for m in recent_100
            if m.get('avg_loss') is not None and not np.isnan(m['avg_loss'])
        ]
        avg_loss_last_100 = np.mean(loss_values) if loss_values else 0

        avg_rounds = np.mean(episode_lengths[-HISTORICAL_METRICS_WINDOW:]) if episode_lengths else 0

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

        reward_std = np.std([m['avg_reward'] for m in recent_100]) if len(recent_100) > 1 else 0
        loss_std = np.std(loss_values) if len(loss_values) > 1 else 0
        training_stability = {'reward_std': float(reward_std), 'loss_std': float(loss_std)}

        battle_data = {}
        recent_battle_data = {}
        if hasattr(self.opponent_selector, 'get_battle_stats'):
            battle_data = self.opponent_selector.get_battle_stats()
        if hasattr(self.opponent_selector, 'get_recent_battle_stats'):
            recent_battle_data = self.opponent_selector.get_recent_battle_stats()

        self._get_logger().save_training_status(
            episode=episode,
            total_steps=episode * getattr(config, 'n_envs', 1),
            best_avg_reward=best_avg_reward,
            avg_reward_last_100=avg_reward_last_100,
            avg_reward_last_10=avg_reward_last_10,
            avg_loss_last_100=avg_loss_last_100,
            avg_rounds=avg_rounds,
            total_wins=total_wins,
            total_losses=total_losses,
            total_draws=total_draws,
            win_rate=win_rate,
            training_speed=training_speed,
            ppo_metrics=ppo_metrics,
            training_stability=training_stability,
            battle_data=battle_data,
            recent_battle_data=recent_battle_data,
            start_time=training_start_time,
            current_time=self.time_tracker.last_training_end_time
        )

        if battle_data:
            self._get_logger().save_battle_stats(battle_data)

        self.time_tracker.save_time_statistics()

    def log_training_complete(self, total_episodes: int, total_steps: int,
                             best_avg_reward: float, total_training_time: float,
                             win_rate: float) -> None:
        """记录训练完成"""
        self._get_logger().log_training_complete(
            total_episodes=total_episodes,
            total_steps=total_steps,
            best_avg_reward=best_avg_reward,
            total_training_time=total_training_time,
            win_rate=win_rate
        )

    def log_model_save(self, model_path: str) -> None:
        """记录模型保存"""
        self._get_logger().log_model_save(model_path)

    def log_battle_result(self, episode: int, opponent: str,
                         result: str, reward: float) -> None:
        """记录对战结果"""
        self._get_logger().log_battle_result(episode, opponent, result, reward)