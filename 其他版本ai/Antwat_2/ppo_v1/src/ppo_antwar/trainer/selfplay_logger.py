from __future__ import annotations

import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from loguru import logger


class SelfPlayLogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    @classmethod
    def from_string(cls, level_str: str) -> 'SelfPlayLogLevel':
        level_str = level_str.upper()
        if level_str in cls.__members__:
            return cls[level_str]
        return cls.INFO


class SelfPlayLogger:
    """
    SelfPlay训练过程的对战日志记录器
    参考battle_single的BattleLogger实现，提供分层日志输出
    """

    def __init__(
        self,
        agent_name: str = "current_agent",
        log_level: SelfPlayLogLevel = SelfPlayLogLevel.INFO,
        log_dir: Optional[str] = None,
        enable_timing: bool = True
    ):
        self.agent_name = agent_name
        self.log_level = log_level
        self.log_dir = log_dir
        self.enable_timing = enable_timing

        self._total_battles = 0
        self._wins = 0
        self._losses = 0
        self._draws = 0
        self._total_duration = 0.0
        self._total_rounds = 0
        self._overall_start_time = datetime.now()
        self._timings = {}

    def _get_log_level_str(self) -> str:
        level_map = {
            SelfPlayLogLevel.DEBUG: "DEBUG",
            SelfPlayLogLevel.INFO: "INFO",
            SelfPlayLogLevel.WARNING: "WARNING",
            SelfPlayLogLevel.ERROR: "ERROR"
        }
        return level_map.get(self.log_level, "INFO")

    def debug(self, category: str, message: str):
        if self.log_level.value <= SelfPlayLogLevel.DEBUG.value:
            logger.bind(category=category).debug(message)

    def info(self, category: str, message: str):
        if self.log_level.value <= SelfPlayLogLevel.INFO.value:
            logger.bind(category=category).info(message)

    def warning(self, category: str, message: str):
        if self.log_level.value <= SelfPlayLogLevel.WARNING.value:
            logger.bind(category=category).warning(message)

    def error(self, category: str, message: str):
        logger.bind(category=category).error(message)

    def log_battle_start(self, episode: int, opponent_id: str, player_position: int):
        position_str = "先手" if player_position == 0 else "后手"
        self.info('selfplay', f"START #{episode}: Agent({position_str}) vs {opponent_id}")

    def log_battle_end(
        self,
        episode: int,
        result: str,
        total_rounds: int,
        duration: float,
        final_hp: Dict[str, int]
    ):
        self.info('selfplay', f"END #{episode}: Result: {result}, Rounds: {total_rounds}, Time: {duration:.2f}s")
        self.info('selfplay', f"  Final HP - Our: {final_hp.get('our_hp', 0)}, Enemy: {final_hp.get('enemy_hp', 0)}")

    def log_round(
        self,
        episode: int,
        round_num: int,
        our_hp: int,
        enemy_hp: int,
        our_coins: int = 0,
        enemy_coins: int = 0
    ):
        self.debug('round', f"#{episode}-{round_num}: HP={our_hp}/{enemy_hp}, Coins={our_coins}/{enemy_coins}")

    def log_timing(self, operation: str, duration: float):
        if not self.enable_timing:
            return
        self.debug('timing', f"[TIMING] {operation}: {duration:.3f}s")

    def record_battle_result(self, result: str, duration: float, total_rounds: int):
        self._total_battles += 1
        self._total_duration += duration
        self._total_rounds += total_rounds

        if result == 'win':
            self._wins += 1
        elif result == 'loss':
            self._losses += 1
        else:
            self._draws += 1

    def print_summary(self, episode: Optional[int] = None):
        total = self._wins + self._losses + self._draws
        if total == 0:
            return

        win_rate = (self._wins / total * 100) if total > 0 else 0.0

        summary_lines = []
        summary_lines.append("")
        summary_lines.append("=" * 60)
        summary_lines.append("[SelfPlay] 对战汇总:")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Total Battles: {self._total_battles}")
        summary_lines.append(f"Wins: {self._wins} ({self._wins / total * 100:.1f}%)")
        summary_lines.append(f"Losses: {self._losses} ({self._losses / total * 100:.1f}%)")
        summary_lines.append(f"Draws: {self._draws} ({self._draws / total * 100:.1f}%)")
        if self._total_battles > 0:
            summary_lines.append(f"Average Rounds: {self._total_rounds / self._total_battles:.1f}")
            summary_lines.append(f"Average Duration: {self._total_duration / self._total_battles:.3f}s")
        summary_lines.append("=" * 60)
        summary_lines.append("")

        for line in summary_lines:
            self.info('summary', line)

    def print_performance_report(self):
        if not self._timings:
            return

        self.info('perf_report', "=== SelfPlay Performance Report ===")
        for operation, total_time in sorted(self._timings.items(), key=lambda x: -x[1]):
            self.info('perf_report', f"{operation}: {total_time:.3f}s")

    def record_timing(self, operation: str, duration: float):
        self._timings[operation] = self._timings.get(operation, 0.0) + duration
        self.log_timing(operation, duration)

    def get_statistics(self) -> Dict:
        total = self._wins + self._losses + self._draws
        return {
            'total_battles': self._total_battles,
            'wins': self._wins,
            'losses': self._losses,
            'draws': self._draws,
            'win_rate': (self._wins / total * 100) if total > 0 else 0.0,
            'avg_rounds': (self._total_rounds / self._total_battles) if self._total_battles > 0 else 0.0,
            'avg_duration': (self._total_duration / self._total_battles) if self._total_battles > 0 else 0.0
        }

    def reset_statistics(self):
        self._total_battles = 0
        self._wins = 0
        self._losses = 0
        self._draws = 0
        self._total_duration = 0.0
        self._total_rounds = 0
        self._timings = {}