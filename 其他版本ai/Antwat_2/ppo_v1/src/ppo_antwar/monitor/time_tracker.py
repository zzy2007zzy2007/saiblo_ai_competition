"""
时间统计管理模块
"""

import json
import os
import time
from typing import Optional

from loguru import logger
from ppo_antwar.config.path_config import PathConfig


class TimeTracker:
    """时间统计管理类"""

    def __init__(self, log_dir: Optional[str] = None):
        self._log_dir = log_dir or str(PathConfig().training_dir)
        self.training_time_total = 0.0
        self.battle_time_total = 0.0
        self.training_episodes_count = 0
        self.battle_evaluations_count = 0
        self.last_training_start_time: Optional[str] = None
        self.last_training_end_time: Optional[str] = None
        self.last_battle_start_time: Optional[str] = None
        self.last_battle_end_time: Optional[str] = None
        self.last_battle_duration = 0.0
        self._training_start: Optional[float] = None
        self._battle_start: Optional[float] = None

    def start_training(self) -> None:
        """标记训练开始"""
        self._training_start = time.time()
        self.last_training_start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    def end_training(self) -> None:
        """标记训练结束"""
        if self._training_start is not None:
            elapsed = time.time() - self._training_start
            self.training_time_total += elapsed
            self.training_episodes_count += 1
            self.last_training_end_time = time.strftime('%Y-%m-%d %H:%M:%S')
            self._training_start = None

    def start_battle(self) -> None:
        """标记对战开始"""
        self._battle_start = time.time()
        self.last_battle_start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    def end_battle(self) -> None:
        """标记对战结束"""
        if self._battle_start is not None:
            self.last_battle_duration = time.time() - self._battle_start
            self.battle_time_total += self.last_battle_duration
            self.battle_evaluations_count += 1
            self.last_battle_end_time = time.strftime('%Y-%m-%d %H:%M:%S')
            self._battle_start = None

    def print_time_statistics(self) -> None:
        """输出时间统计到控制台"""
        logger.info("\n==========================================")
        logger.info("   时间统计汇总")
        logger.info("==========================================")

        if self.training_episodes_count > 0:
            avg_training_time = self.training_time_total / self.training_episodes_count
            logger.info(f"训练统计:")
            logger.info(f"  总训练轮次: {self.training_episodes_count}")
            logger.info(f"  总训练时间: {self.training_time_total:.2f}s ({self.training_time_total/3600:.2f}h)")
            logger.info(f"  平均每轮时间: {avg_training_time:.2f}s")
            logger.info(f"  每小时训练轮次: {3600/avg_training_time:.2f}")

        if self.battle_evaluations_count > 0:
            avg_battle_time = self.battle_time_total / self.battle_evaluations_count
            logger.info(f"\n对战评估统计:")
            logger.info(f"  总对战评估次数: {self.battle_evaluations_count}")
            logger.info(f"  总对战时间: {self.battle_time_total:.2f}s ({self.battle_time_total/3600:.2f}h)")
            logger.info(f"  平均每次评估时间: {avg_battle_time:.2f}s")

        if self.training_time_total > 0 and self.battle_time_total > 0:
            total_time = self.training_time_total + self.battle_time_total
            training_ratio = self.training_time_total / total_time * 100
            battle_ratio = self.battle_time_total / total_time * 100
            logger.info(f"\n时间分配:")
            logger.info(f"  训练时间占比: {training_ratio:.2f}%")
            logger.info(f"  对战时间占比: {battle_ratio:.2f}%")
            logger.info(f"  总时间: {total_time:.2f}s ({total_time/3600:.2f}h)")

        if (self.last_training_start_time and self.last_training_end_time and
            self.last_battle_start_time and self.last_battle_end_time):
            logger.info(f"\n对战评估详情:")
            logger.info(f"  训练开始时间: {self.last_training_start_time}")
            logger.info(f"  训练结束时间: {self.last_training_end_time}")
            logger.info(f"  最后一次对战开始时间: {self.last_battle_start_time}")
            logger.info(f"  最后一次对战结束时间: {self.last_battle_end_time}")
            logger.info(f"  最后一次对战持续时间: {self.last_battle_duration:.2f}秒")
        logger.info("==========================================\n")

    def save_time_statistics(self) -> None:
        """保存时间统计到文件"""
        stats = {
            'training_episodes_count': self.training_episodes_count,
            'training_time_total': self.training_time_total,
            'battle_evaluations_count': self.battle_evaluations_count,
            'battle_time_total': self.battle_time_total,
            'last_training_start_time': self.last_training_start_time,
            'last_training_end_time': self.last_training_end_time,
            'last_battle_start_time': self.last_battle_start_time,
            'last_battle_end_time': self.last_battle_end_time,
            'last_battle_duration': self.last_battle_duration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        time_stats_file = os.path.join(self._log_dir, "time_statistics.json")
        os.makedirs(os.path.dirname(time_stats_file), exist_ok=True)
        with open(time_stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    @property
    def current_training_time(self) -> float:
        """获取当前训练阶段已用时间"""
        if self._training_start is not None:
            return time.time() - self._training_start
        return 0.0

    @property
    def current_battle_time(self) -> float:
        """获取当前对战阶段已用时间"""
        if self._battle_start is not None:
            return time.time() - self._battle_start
        return 0.0