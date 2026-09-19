import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class Logger:
    """训练主日志管理器 - 负责所有结构化日志文件的写入。

    职责：
    - 训练配置快照（training_start.log）
    - 训练完成汇总（training_complete.log）
    - 对战统计（battle_stats.json）
    - JSON 状态变化日志（*.json.log）
    """

    def __init__(self, training_dir: str, run_id: str):
        self._training_dir = Path(training_dir)
        self._training_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id

    # ── 训练启动/完成 ─────────────────────────────────────────────────────

    def log_training_start(self, config: Dict[str, Any], device: Any) -> None:
        ppo_cfg = config.get("ppo", {})
        net_cfg = config.get("network", {})
        train_cfg = config.get("training", {})
        system_info = {
            "Training started at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Run ID": self._run_id,
            "Episodes": train_cfg.get("total_episodes", "N/A"),
            "Batch size": ppo_cfg.get("batch_size", "N/A"),
            "Learning rate": ppo_cfg.get("lr", "N/A"),
            "Entropy coef": ppo_cfg.get("ent_coef", "N/A"),
            "Gamma": ppo_cfg.get("gamma", "N/A"),
            "GAE lambda": ppo_cfg.get("gae_lambda", "N/A"),
            "Clip epsilon": ppo_cfg.get("clip_eps", "N/A"),
            "n_envs": train_cfg.get("n_envs", "N/A"),
            "Save interval": train_cfg.get("save_interval", "N/A"),
            "Opponent update interval": train_cfg.get(
                "opponent_update_interval", "N/A"
            ),
            "Network hidden_size": net_cfg.get("hidden_dim", "N/A"),
            "Network num_layers": net_cfg.get("num_layers", "N/A"),
            "Device": str(device),
            "PyTorch version": _get_torch_version(),
            "System": platform.platform(),
            "CPU cores": _get_cpu_count(),
            "PID": _get_pid(),
        }
        lines = [f"{k}: {v}" for k, v in system_info.items()]
        content = "\n".join(lines) + "\n"
        with open(self._training_dir / "training_start.log", "w") as f:
            f.write(content)

    def log_training_complete(self, summary: Dict[str, Any]) -> None:
        fields = [
            ("Total episodes", summary.get("total_episodes")),
            ("Total steps", summary.get("total_steps")),
            ("Total time", summary.get("total_time")),
            ("Best avg reward", summary.get("best_avg_reward")),
            ("Best avg reward episode", summary.get("best_avg_reward_episode")),
            ("Final win rate", summary.get("win_rate")),
            ("Final avg reward", summary.get("avg_reward")),
            ("Final avg loss", summary.get("avg_loss")),
        ]
        opponent_details = summary.get("opponent_win_rates", {})
        if opponent_details:
            for opp, rate in opponent_details.items():
                fields.append((f"Win rate vs {opp}", rate))
        fields.append(("Completed at", time.strftime("%Y-%m-%d %H:%M:%S")))
        lines = [f"{k}: {v}" for k, v in fields if v is not None]
        content = "\n".join(lines) + "\n"
        with open(self._training_dir / "training_complete.log", "w") as f:
            f.write(content)

    def flush(self) -> None:
        logger.complete()

    # ── 对战统计 ─────────────────────────────────────────────────────────

    def save_battle_stats(self, stats_data: Dict[str, Any]) -> None:
        with open(self._training_dir / "battle_stats.json", "w") as f:
            json.dump(stats_data, f, indent=2, default=str)

    # ── JSON 状态变化 ────────────────────────────────────────────────────

    def log_json_state(
        self,
        filename: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
            "context": context,
        }
        log_path = self._training_dir / f"{filename}.json.log"
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ── 辅助函数 ───────────────────────────────────────────────────────────────


def _get_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "N/A"


def _get_cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 0
    except Exception:
        return 0


def _get_pid() -> int:
    try:
        import os

        return os.getpid()
    except Exception:
        return 0
