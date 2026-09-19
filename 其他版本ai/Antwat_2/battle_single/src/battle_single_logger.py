import os
import sys
import atexit
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from loguru import logger


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        level_str = level_str.upper()
        if level_str in cls.__members__:
            return cls[level_str]
        return cls.INFO


WARNING_THRESHOLD = 2.0


class BattleLogger:
    """
    基于 loguru enqueue=True 的 BattleLogger

    架构说明：
    - 主进程：配置控制台 handler（无 enqueue）+ 文件 handlers（enqueue=True）
    - 子进程：通过 fork 继承 logger 配置，仅使用文件 handlers
    - 所有日志通过队列汇入后台 writer 线程，实现均匀输出
    """

    def __init__(self, agent1_name: str, agent2_name: str,
                 log_level: LogLevel = LogLevel.INFO, log_to_console: bool = True,
                 enable_timing: bool = True):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.log_level = log_level
        self.log_to_console = log_to_console
        self.enable_timing = enable_timing

        self._log_dir = self._create_log_dir()
        self._configure_logger()

        self._total_battles = 0
        self._completed_battles = 0
        self._error_battles = 0
        self._agent1_wins = 0
        self._agent2_wins = 0
        self._draws = 0
        self._total_duration = 0.0
        self._total_rounds = 0
        self._overall_start_time = datetime.now()
        self._overall_end_time = None
        self._timing_starts = {}
        self._timings = {}
        
        self.agent1_total_time = 0.0
        self.agent2_total_time = 0.0
        self.resolve_total_time = 0.0

        atexit.register(self._cleanup)

    def _create_log_dir(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.agent1_name}_vs_{self.agent2_name}_{timestamp}"
        dir_path = os.path.join('/tmp/battle_single/logs', dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _get_log_level_str(self) -> str:
        level_map = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR"
        }
        return level_map.get(self.log_level, "INFO")

    def _configure_logger(self):
        """配置 loguru handlers"""
        logger.remove()

        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>[{extra[category]}]</cyan> | "
            "<level>{message}</level>"
        )

        if self.log_to_console:
            logger.add(
                sys.stderr,
                format=log_format,
                level=self._get_log_level_str(),
                enqueue=False,
                colorize=True
            )

        logger.add(
            os.path.join(self._log_dir, 'debug.log'),
            format=log_format,
            level="DEBUG",
            enqueue=True,
            serialize=False
        )

        logger.add(
            os.path.join(self._log_dir, 'info.log'),
            format=log_format,
            level="INFO",
            enqueue=True,
            serialize=False
        )

        logger.add(
            os.path.join(self._log_dir, 'warning.log'),
            format=log_format,
            level="WARNING",
            enqueue=True,
            serialize=False
        )

        logger.add(
            os.path.join(self._log_dir, 'error.log'),
            format=log_format,
            level="ERROR",
            enqueue=True,
            serialize=False
        )

        logger.add(
            os.path.join(self._log_dir, 'all.log'),
            format=log_format,
            level="DEBUG",
            enqueue=True,
            serialize=False
        )

    def _cleanup(self):
        """清理资源"""
        self.info('battle', "Logger cleanup: waiting for queue to flush")
        logger.complete()
        import time
        time.sleep(0.1)

    def debug(self, category: str, message: str):
        if self.log_level.value <= LogLevel.DEBUG.value:
            logger.bind(category=category).debug(message)

    def info(self, category: str, message: str):
        if self.log_level.value <= LogLevel.INFO.value:
            logger.bind(category=category).info(message)

    def warning(self, category: str, message: str):
        if self.log_level.value <= LogLevel.WARNING.value:
            logger.bind(category=category).warning(message)

    def error(self, category: str, message: str):
        logger.bind(category=category).error(message)

    def log_battle_start(self, episode: int):
        self.info('battle', f"START #{episode}: {self.agent1_name} vs {self.agent2_name}")

    def log_battle_end(self, result: Dict):
        agent1 = result['agent1_name']
        agent2 = result['agent2_name']
        result_str = result['result']
        rounds = result['total_rounds']
        duration = result['duration']
        self.info('battle', f"END #{result['episode']}: {agent1} vs {agent2}")
        self.info('battle', f"  Result: {result_str}, Rounds: {rounds}, Time: {duration:.2f}s")

    def log_round(self, episode: int, round_num: int, hp1: int, hp2: int, coins1: int = 0, coins2: int = 0,
                  player0_ops: str = "", player1_ops: str = "", first_player: int = 0):
        agent1_is_player0 = (first_player == 0)
        player0_name = self.agent1_name if agent1_is_player0 else self.agent2_name
        player1_name = self.agent2_name if agent1_is_player0 else self.agent1_name

        self.debug('round', f"#{episode}-{round_num}: HP={hp1}/{hp2}, Coins={coins1}/{coins2}")
        self.debug('round', f"#{episode}-{round_num}: [{player0_name}(先手)] ops={player0_ops}")
        self.debug('round', f"#{episode}-{round_num}: [{player1_name}(后手)] ops={player1_ops}")

    def log_error(self, episode: int, error: Exception):
        self.error('battle', f"Battle #{episode} failed: {type(error).__name__}: {error}")

    def log_agent_load(self, agent_name: str, success: bool, error_msg: str = None):
        if success:
            self.info('load', f"✓ Loaded agent: {agent_name}")
        else:
            self.error('load', f"✗ Failed to load agent: {agent_name} - {error_msg}")

    def log_result(self, result: Dict):
        results_json_path = os.path.join(self._log_dir, 'results.json')
        with open(results_json_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

        self._total_battles += 1
        self._total_duration += result['duration']
        self._total_rounds += result.get('total_rounds', 0)

        if result['result'] == 'error':
            self._error_battles += 1
        else:
            self._completed_battles += 1
            if result['result'] == 'agent1_win':
                self._agent1_wins += 1
            elif result['result'] == 'agent2_win':
                self._agent2_wins += 1
            else:
                self._draws += 1

    def log_timing(self, operation: str, duration: float):
        if not self.enable_timing:
            return
        self.debug('timing', f"[TIMING] {operation}: {duration:.3f}s")

    def timing_start(self, operation: str):
        if not self.enable_timing:
            return
        self._timing_starts[operation] = datetime.now()

    def timing_stop(self, operation: str, duration: float = None) -> float:
        if not self.enable_timing:
            return 0.0

        if duration is None:
            if operation not in self._timing_starts:
                return 0.0
            duration = (datetime.now() - self._timing_starts[operation]).total_seconds()

        self._timings[operation] = self._timings.get(operation, 0.0) + duration
        self.log_timing(operation, duration)

        if 'agent1' in operation.lower():
            self.agent1_total_time += duration
        elif 'agent2' in operation.lower():
            self.agent2_total_time += duration
        elif 'resolve' in operation.lower():
            self.resolve_total_time += duration

        if duration > WARNING_THRESHOLD:
            self.warning('timeout', f"{operation} took {duration:.2f}s (exceeds {WARNING_THRESHOLD}s)")

        return duration

    def log_timing_summary(self, battle_total_time: float):
        if not self.enable_timing:
            return
        self.debug('timing', f"[TIMING SUMMARY] battle_total: {battle_total_time:.3f}s")
        self.debug('timing', f"[TIMING SUMMARY] agent1_total_time: {self.agent1_total_time:.3f}s")
        self.debug('timing', f"[TIMING SUMMARY] agent2_total_time: {self.agent2_total_time:.3f}s")
        self.debug('timing', f"[TIMING SUMMARY] resolve_total_time: {self.resolve_total_time:.3f}s")

    def reset_timing_summary(self):
        self.agent1_total_time = 0.0
        self.agent2_total_time = 0.0
        self.resolve_total_time = 0.0
        self._timings = {}

    def get_timing_total(self, operation: str) -> float:
        return self._timings.get(operation, 0.0)

    def print_performance_report(self):
        self.info('perf_report', "=== Performance Report ===")
        for operation, total_time in sorted(self._timings.items(), key=lambda x: -x[1]):
            self.info('perf_report', f"{operation}: {total_time:.3f}s")

    def get_log_dir(self) -> str:
        return self._log_dir

    def write_summary(self):
        self._overall_end_time = datetime.now()

        total_wins = self._agent1_wins + self._agent2_wins + self._draws
        if total_wins > 0:
            agent1_win_rate = (self._agent1_wins / total_wins * 100)
            agent2_win_rate = (self._agent2_wins / total_wins * 100)
            draw_rate = (self._draws / total_wins * 100)
        else:
            agent1_win_rate = 0.0
            agent2_win_rate = 0.0
            draw_rate = 0.0

        summary_lines = []
        summary_lines.append("")
        summary_lines.append("=" * 60)
        summary_lines.append("Summary:")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Total Battles: {self._total_battles}")
        summary_lines.append(f"Completed Battles: {self._completed_battles}")
        summary_lines.append(f"Error Battles: {self._error_battles}")

        summary_lines.append("")
        summary_lines.append("【对战结果】")
        summary_lines.append(f"{self.agent1_name} Wins: {self._agent1_wins} ({agent1_win_rate:.1f}%)")
        summary_lines.append(f"{self.agent2_name} Wins: {self._agent2_wins} ({agent2_win_rate:.1f}%)")
        summary_lines.append(f"Draws: {self._draws} ({draw_rate:.1f}%)")

        summary_lines.append("")
        summary_lines.append("【时间统计】")
        summary_lines.append(f"Overall Start Time: {self._overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Overall End Time: {self._overall_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total Duration: {self._total_duration:.3f}s")
        if self._completed_battles > 0:
            avg_rounds = self._total_rounds / self._completed_battles
            avg_duration = self._total_duration / self._completed_battles
            summary_lines.append(f"Average Rounds: {avg_rounds:.1f}")
            summary_lines.append(f"Average Duration: {avg_duration:.3f}s")
        summary_lines.append("=" * 60)
        summary_lines.append("")

        for line in summary_lines:
            self.info('summary', line)

    def close(self):
        self.write_summary()
        self._cleanup()


class SubProcessLogger:
    """
    子进程中使用的轻量级日志记录器
    利用 loguru 的 enqueue=True 机制实现多进程安全的实时日志输出
    """
    
    def __init__(self, agent1_name: str, agent2_name: str, log_level: LogLevel = LogLevel.DEBUG):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.log_level = log_level
    
    def debug(self, category: str, message: str):
        if self.log_level.value <= LogLevel.DEBUG.value:
            logger.bind(category=category).debug(message)
    
    def info(self, category: str, message: str):
        if self.log_level.value <= LogLevel.INFO.value:
            logger.bind(category=category).info(message)
    
    def warning(self, category: str, message: str):
        if self.log_level.value <= LogLevel.WARNING.value:
            logger.bind(category=category).warning(message)
    
    def error(self, category: str, message: str):
        logger.bind(category=category).error(message)
    
    def log_battle_start(self, episode: int):
        self.info('battle', f"START #{episode}: {self.agent1_name} vs {self.agent2_name}")
    
    def log_round(self, episode: int, round_num: int, hp1: int, hp2: int, coins1: int = 0, coins2: int = 0,
                  player0_ops: str = "", player1_ops: str = "", first_player: int = 0):
        agent1_is_player0 = (first_player == 0)
        player0_name = self.agent1_name if agent1_is_player0 else self.agent2_name
        player1_name = self.agent2_name if agent1_is_player0 else self.agent1_name

        self.debug('round', f"#{episode}-{round_num}: HP={hp1}/{hp2}, Coins={coins1}/{coins2}")
        self.debug('round', f"#{episode}-{round_num}: [{player0_name}(先手)] ops={player0_ops}")
        self.debug('round', f"#{episode}-{round_num}: [{player1_name}(后手)] ops={player1_ops}")
    
    def log_battle_end(self, result: Dict):
        agent1 = result['agent1_name']
        agent2 = result['agent2_name']
        result_str = result['result']
        rounds = result['total_rounds']
        duration = result['duration']
        self.info('battle', f"END #{result['episode']}: {agent1} vs {agent2}")
        self.info('battle', f"  Result: {result_str}, Rounds: {rounds}, Time: {duration:.2f}s")
