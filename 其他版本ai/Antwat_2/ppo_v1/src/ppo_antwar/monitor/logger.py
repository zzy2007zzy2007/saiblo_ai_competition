"""
日志管理模块 - 基于 loguru 实现
"""

import os
import json
import time
import sys
from loguru import logger as _loguru_logger
from typing import Optional, Dict, Any, List

from ppo_antwar.config.path_config import PathConfig


class MetricsCache:
    """训练指标缓存，使用环形缓冲区实现"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._buffer: List[Optional[Dict]] = [None] * max_size
        self._head = 0
        self._count = 0
        self.latest: Optional[Dict] = None

    def append(self, metrics: Dict[str, Any]) -> None:
        """添加新的指标

        Args:
            metrics: 指标字典
        """
        self.latest = metrics
        self._buffer[self._head] = metrics
        self._head = (self._head + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)

    def get_history(self, n: Optional[int] = None) -> List[Dict]:
        """获取历史记录

        Args:
            n: 获取最近n条记录，None表示全部

        Returns:
            历史记录列表
        """
        if self._count == 0:
            return []

        if n is None or n >= self._count:
            n = self._count

        result = []
        start = (self._head - n) % self.max_size
        for i in range(n):
            idx = (start + i) % self.max_size
            if self._buffer[idx] is not None:
                result.append(self._buffer[idx])
        return result

    def __getitem__(self, key: str) -> Any:
        """支持字典风格访问 latest 指标"""
        if self.latest is None:
            return None
        return self.latest.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """获取 latest 指标中的值"""
        if self.latest is None:
            return default
        return self.latest.get(key, default)


class LoggerOperationError(Exception):
    """日志操作异常，用于包装日志操作中的错误"""

    def __init__(self, operation: str, original_error: Exception, context: Optional[Dict] = None):
        self.operation = operation
        self.original_error = original_error
        self.context = context or {}
        super().__init__(f"Logger operation '{operation}' failed: {original_error}")


class Logger:
    """日志管理类 - 基于 loguru 实现"""

    _initialized = False

    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
        self.log_dir = str(path_config.base_dir)
        self._error_count = 0
        self._metrics_cache = MetricsCache(max_size=100)

        if not Logger._initialized:
            self._configure_loguru()
            Logger._initialized = True

    def _configure_loguru(self) -> None:
        """配置 loguru"""
        _loguru_logger.remove()

        _loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )

        training_log_path = str(self.path_config.training_dir / "training.log")
        _loguru_logger.add(
            training_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="30 days"
        )

    def _get_logger(self):
        """获取 loguru logger 实例"""
        return _loguru_logger

    def _ensure_dir(self, file_path: str) -> None:
        """确保目录存在"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def _write_file_safe(self, file_path: str, content: str, mode: str = 'a') -> None:
        """安全写入文件"""
        self._ensure_dir(file_path)
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)

    def _write_json_safe(self, file_path: str, data: Dict, mode: str = 'w') -> None:
        """安全写入JSON文件"""
        self._ensure_dir(file_path)
        with open(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def log_json_state(self, json_path: str, data: Dict, context: Optional[Dict] = None) -> None:
        """统一记录JSON状态到日志文件"""

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_path = json_path + ".json.log"

        entry = {
            "timestamp": timestamp,
            "data": data,
            "context": context
        }

        try:
            content = json.dumps(entry, ensure_ascii=False) + "\n"
            self._write_file_safe(log_path, content)
            _loguru_logger.debug(f"JSON state logged to {log_path}")
        except Exception as e:
            _loguru_logger.warning(f"Failed to write JSON state log: {e}")

    def log_exception(self, e: Exception, context: Optional[Dict] = None) -> Exception:
        """统一记录异常到error.log和training.log"""

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        error_log_path = str(self.path_config.training_dir / "error.log")
        training_log_path = str(self.path_config.training_dir / "training.log")

        error_entry = {
            "timestamp": timestamp,
            "exception_type": type(e).__name__,
            "message": str(e),
            "context": context
        }

        error_line = json.dumps(error_entry, ensure_ascii=False)

        try:
            self._ensure_dir(error_log_path)
            with open(error_log_path, "a", encoding='utf-8') as f:
                f.write(error_line + "\n")
            self._error_count += 1
            _loguru_logger.exception(f"Exception logged: {type(e).__name__}: {e}")
        except Exception as write_error:
            _loguru_logger.error(f"CRITICAL: Failed to write error log: {write_error}")

        try:
            self._ensure_dir(training_log_path)
            with open(training_log_path, "a", encoding='utf-8') as f:
                f.write(f"[ERROR] {error_line}\n")
        except Exception as write_error:
            _loguru_logger.warning(f"Failed to write training log: {write_error}")

        return e

    def log_warning(self, message: str, context: Optional[Dict] = None) -> None:
        """记录警告信息到warning.log和training.log"""

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        warning_log_path = str(self.path_config.training_dir / "warning.log")
        training_log_path = str(self.path_config.training_dir / "training.log")

        warning_entry = {
            "timestamp": timestamp,
            "message": message,
            "context": context
        }

        warning_line = json.dumps(warning_entry, ensure_ascii=False)

        try:
            self._ensure_dir(warning_log_path)
            with open(warning_log_path, "a", encoding='utf-8') as f:
                f.write(warning_line + "\n")
            _loguru_logger.warning(f"Warning logged: {message}")
        except Exception as write_error:
            _loguru_logger.warning(f"Failed to write warning log: {write_error}")

        try:
            self._ensure_dir(training_log_path)
            with open(training_log_path, "a", encoding='utf-8') as f:
                f.write(f"[WARNING] {warning_line}\n")
        except Exception as write_error:
            _loguru_logger.warning(f"Failed to write training log: {write_error}")

    def save_training_metrics(self, episode: int, avg_reward: float,
                              avg_loss: float, avg_rounds: float,
                              steps: int, context: Optional[Dict] = None,
                              training_metrics_cache=None) -> None:
        """保存训练指标到 training_metrics.json (latest+stats) 和 training_metrics_history.jsonl (append-only history)"""

        if training_metrics_cache is None:
            training_metrics_cache = self._metrics_cache

        ctx = context or {}
        metrics = {
            'episode': episode,
            'avg_reward': float(avg_reward),
            'avg_loss': float(avg_loss) if avg_loss is not None else 0,
            'avg_rounds': float(avg_rounds),
            'steps': steps,
            'policy_loss': float(ctx.get('policy_loss', 0)),
            'value_loss': float(ctx.get('value_loss', 0)),
            'aux_tower_loss': float(ctx.get('aux_tower_loss', 0)),
            'aux_gold_loss': float(ctx.get('aux_gold_loss', 0)),
            'aux_enemy_tower_loss': float(ctx.get('aux_enemy_tower_loss', 0)),
            'aux_enemy_gold_loss': float(ctx.get('aux_enemy_gold_loss', 0)),
            'entropy': float(ctx.get('entropy', 0)),
            'entropy_type': float(ctx.get('entropy_type', 0)),
            'entropy_target': float(ctx.get('entropy_target', 0)),
            'entropy_coef': float(ctx.get('entropy_coef', 0)),
            'learning_rate': float(ctx.get('learning_rate', 0)),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reward_mean': float(ctx.get('reward_mean', 0)),
            'reward_std': float(ctx.get('reward_std', 0)),
            'reward_min': float(ctx.get('reward_min', 0)),
            'reward_max': float(ctx.get('reward_max', 0)),
            'advantages_std': float(ctx.get('advantages_std', 0)),
            'return_mean': float(ctx.get('return_mean', 0)),
            'return_std': float(ctx.get('return_std', 0)),
            'clip_fraction': float(ctx.get('clip_fraction', 0)),
            'gradient_norm': float(ctx.get('gradient_norm', 0)),
            'value_input_mean': float(ctx.get('value_input_mean', 0)),
            'value_input_std': float(ctx.get('value_input_std', 0)),
            'value_pred_mean': float(ctx.get('value_pred_mean', 0)),
            'value_pred_std': float(ctx.get('value_pred_std', 0)),
            'ratio_mean': float(ctx.get('ratio_mean', 0)),
            'ratio_std': float(ctx.get('ratio_std', 0)),
            'grad_norm_policy': float(ctx.get('grad_norm_policy', 0)),
            'grad_norm_value': float(ctx.get('grad_norm_value', 0)),
            'grad_norm_max_layer': float(ctx.get('grad_norm_max_layer', 0)),
            'nan_skip_count': int(ctx.get('nan_skip_count', 0)),
            'valid_actions_mean': float(ctx.get('valid_actions_mean', 0)),
            'valid_actions_min': float(ctx.get('valid_actions_min', 0)),
            'td_error_mean': float(ctx.get('td_error_mean', 0)),
            'td_error_std': float(ctx.get('td_error_std', 0)),
            'batch_size': int(ctx.get('batch_size', 0)),
            'num_collected': int(ctx.get('num_collected', 0)),
            'avg_our_coins': float(ctx.get('avg_our_coins', 0)),
            'avg_enemy_coins': float(ctx.get('avg_enemy_coins', 0)),
            'avg_our_cumulative_coins': float(ctx.get('avg_our_cumulative_coins', 50)),
            'avg_enemy_cumulative_coins': float(ctx.get('avg_enemy_cumulative_coins', 50)),
            'rw_hp_attack_base': float(ctx.get('rw_hp_attack_base', 0)),
            'rw_hp_attack_tower': float(ctx.get('rw_hp_attack_tower', 0)),
            'rw_coin_gain': float(ctx.get('rw_coin_gain', 0)),
            'rw_tower_survival': float(ctx.get('rw_tower_survival', 0)),
            'rw_balance': float(ctx.get('rw_balance', 0)),
            'rw_tech_bonus': float(ctx.get('rw_tech_bonus', 0)),
            'rw_die_penalty': float(ctx.get('rw_die_penalty', 0)),
            'rw_end_reward': float(ctx.get('rw_end_reward', 0)),
        }

        for k, v in ctx.items():
            if isinstance(v, (int, float)):
                if k.startswith(('type_', 'opponent_', 'logit_', 'prob_', 'our_coins_', 'enemy_coins_', 'actrw_', 'actcnt_')):
                    metrics[k] = float(v)

        if hasattr(training_metrics_cache, 'append'):
            training_metrics_cache.append(metrics)
            history = training_metrics_cache.get_history()
        elif isinstance(training_metrics_cache, dict) and 'history' in training_metrics_cache:
            training_metrics_cache['history'].append(metrics)
            training_metrics_cache['latest'] = metrics
            history = training_metrics_cache['history']
        else:
            self._metrics_cache.append(metrics)
            history = self._metrics_cache.get_history()

        stats = self._calculate_stats(history)

        training_dir = self.path_config.training_dir

        # 1) 写入 training_metrics.json: 仅保留 latest + stats, 大小固定
        metrics_file = str(training_dir / "training_metrics.json")
        try:
            self._write_json_safe(metrics_file, {
                'latest': metrics,
                'stats': stats
            })
            self.log_json_state(metrics_file, metrics, context or {'event': 'training_metrics_update'})
        except Exception as e:
            _loguru_logger.warning(f"Failed to save training metrics json: {e}")

        # 2) 追加一行到 training_metrics_history.jsonl: append-only, 不截断
        history_file = str(training_dir / "training_metrics_history.jsonl")
        try:
            self._ensure_dir(history_file)
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            _loguru_logger.warning(f"Failed to append training metrics history: {e}")

        _loguru_logger.debug(f"Training metrics saved for episode {episode}")

    def _calculate_stats(self, history: List[Dict]) -> Dict:
        """计算历史奖励统计"""
        if not history:
            return {
                'reward_mean': 0.0,
                'reward_std': 0.0,
                'reward_max': 0.0,
                'reward_min': 0.0
            }

        rewards = [h.get('avg_reward', 0) for h in history]
        if not rewards:
            return {
                'reward_mean': 0.0,
                'reward_std': 0.0,
                'reward_max': 0.0,
                'reward_min': 0.0
            }

        import numpy as np
        return {
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_max': float(np.max(rewards)),
            'reward_min': float(np.min(rewards))
        }

    def log_training_start(self, config, device) -> None:
        """记录训练启动信息到 training_start.log"""
        import platform
        import torch

        start_info = f"""Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Episodes: {getattr(config, 'episodes', 'N/A')}
Batch size: {getattr(config, 'batch_size', 'N/A')}
Learning rate: {getattr(config, 'learning_rate', 'N/A')}
Parallel envs: {getattr(config, 'n_envs', 'N/A')}
Device: {device}
PyTorch version: {torch.__version__}
CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
System: {platform.platform()}
CPU cores: {os.cpu_count()}
PID: {os.getpid()}
"""
        start_file = str(self.path_config.training_dir / "training_start.log")
        self._ensure_dir(start_file)
        with open(start_file, "w") as f:
            f.write(start_info)

        _loguru_logger.info(f"Training started on device: {device}")

    def log_training_details(self, episode: int, avg_reward: float,
                              avg_loss: float, avg_rounds: float,
                              steps: int) -> None:
        """记录每轮训练详细指标到 training_details.log"""
        details_file = str(self.path_config.training_dir / "training_details.log")
        self._ensure_dir(details_file)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(details_file, "a") as f:
            f.write(f"{timestamp} | Episode: {episode} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Avg Rounds: {avg_rounds} | Steps: {steps}\n")

    def log_training_complete(self, total_episodes: int, total_steps: int,
                              best_avg_reward: float, total_training_time: float,
                              win_rate: float) -> None:
        """记录训练完成信息到 training_complete.log"""
        complete_file = str(self.path_config.training_dir / "training_complete.log")
        self._ensure_dir(complete_file)
        with open(complete_file, "w") as f:
            f.write(f"""Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total episodes: {total_episodes}
Total steps: {total_steps}
Best average reward: {best_avg_reward}
Total training time: {total_training_time} seconds
Final win rate: {win_rate}%
""")
        _loguru_logger.info(f"Training completed: {total_episodes} episodes, {total_steps} steps")

    def log_model_save(self, model_path: str) -> None:
        """记录模型保存到 model_saves.log"""
        save_file = str(self.path_config.training_dir / "model_saves.log")
        self._ensure_dir(save_file)
        with open(save_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Model saved: {model_path}\n")
        _loguru_logger.info(f"Model saved: {model_path}")

    def log_battle_result(self, episode: int, opponent: str,
                         result: str, reward: float) -> None:
        """记录对战结果到 battle_results.log"""
        battle_file = str(self.path_config.training_dir / "battle_results.log")
        self._ensure_dir(battle_file)
        with open(battle_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Episode: {episode} | Opponent: {opponent} | Result: {result} | Reward: {reward}\n")

    def log_opponent_selection(self, episode: int, opponent_name: str,
                               random_prob: float) -> None:
        """记录对手选择到 opponent_selection.log"""
        selection_file = str(self.path_config.training_dir / "opponent_selection.log")
        self._ensure_dir(selection_file)
        with open(selection_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Episode: {episode} | "
                    f"Opponent: {opponent_name} | Random prob: {random_prob}\n")
        _loguru_logger.debug(f"Opponent selection logged: {opponent_name} for episode {episode}")

    def save_training_status(self, **kwargs) -> None:
        """保存训练状态到 training_status.json"""
        status_file = str(self.path_config.training_dir / "training_status.json")
        self._ensure_dir(status_file)
        with open(status_file, "w", encoding='utf-8') as f:
            json.dump(kwargs, f, indent=2, ensure_ascii=False)
        self.log_json_state(status_file, kwargs, {'event': 'training_status_update'})

    def save_battle_stats(self, battle_stats: Dict) -> None:
        """保存对战统计到 battle_stats.json"""
        battle_file = str(self.path_config.battle_dir / "battle_stats.json")
        with open(battle_file, "w", encoding='utf-8') as f:
            json.dump(battle_stats, f, indent=2, ensure_ascii=False)
        self.log_json_state(battle_file, battle_stats, {'event': 'battle_stats_update'})

    @property
    def error_count(self) -> int:
        """获取错误计数"""
        return self._error_count

    @property
    def metrics_cache(self) -> MetricsCache:
        """获取指标缓存"""
        return self._metrics_cache


# 导出 loguru logger 实例供其他模块直接使用
loguru_logger = _loguru_logger