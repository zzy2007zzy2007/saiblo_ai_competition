"""Baseline Battle 评估日志写入器

写入以下文件：
- eval_battle_log.jsonl：每局结构化数据
- eval_battle_round_log_{timestamp}.log：回合级步日志
- episode_batch_baseline_winrates.jsonl：EpisodeBatch 窗口 baseline 胜率
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalBattleWriter:
    """Baseline Battle 评估日志写入器。

    职责：
    - 每局评估结束后写入 eval_battle_log.jsonl
    - 每步写入回合级日志 eval_battle_round_log_{timestamp}.log
    - EpisodeBatch 窗口结束时写入 episode_batch_baseline_winrates.jsonl
    """

    def __init__(self, eval_dir: str, window_size: int = 8):
        self._eval_dir = Path(eval_dir)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

        self._window_size = window_size
        self._battle_count = 0

        self._battle_log_path = self._eval_dir / "eval_battle_log.jsonl"
        self._baseline_winrates_path = (
            self._eval_dir / "episode_batch_baseline_winrates.jsonl"
        )

        self._round_handler_id = None
        self._eval_buffer: list = []
        self._init_round_logger()

    def _init_round_logger(self) -> None:
        """初始化回合级 loguru logger"""
        from loguru import logger as _logger

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        round_log_path = self._eval_dir / f"eval_battle_round_log_{timestamp}.log"

        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [EVAL] | {message}"
        self._round_handler_id = _logger.add(
            str(round_log_path),
            level="INFO",
            format=fmt,
            enqueue=True,
            filter=lambda record: "ppo_trainer" not in record.get("name", ""),
        )

    def write_battle(
        self,
        episode: int,
        baseline_agent_name: str,
        battle_mode: str = "selfplay_eval",
        swap: bool = False,
        opponent_id: str = "",
        result: str = "",
        rounds: int = 0,
        reward: float = 0.0,
        length: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        **extra,
    ) -> None:
        now = time.time()
        record = {
            "episode": episode,
            "baseline_agent_name": baseline_agent_name,
            "battle_mode": battle_mode,
            "swap": swap,
            "opponent_id": opponent_id,
            "result": result,
            "rounds": rounds,
            "reward": round(reward, 4),
            "length": length,
            "start_time": start_time if start_time is not None else now,
            "end_time": end_time if end_time is not None else now,
            "duration": round(duration if duration is not None else 0.0, 3),
        }
        if extra:
            record.update(extra)

        with open(self._battle_log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._eval_buffer.append(record)
        self._battle_count += 1

    def write_round(self, episode: int, round_num: int, **fields) -> None:
        """写入回合级日志。

        字段示例：
        - action_self, action_opponent
        - reward_self
        - agent1_round_start_coin, agent1_round_end_coin 等
        """
        from loguru import logger as _logger

        fields_str = " ".join(f"{k}={v}" for k, v in fields.items())
        _logger.bind(category="eval_round").info(
            f"[EVAL] episode={episode} round={round_num} {fields_str}"
        )

    def on_episode_batch_complete(
        self, episode_start: int, episode_end: int
    ) -> Dict[str, Any]:
        """EpisodeBatch 窗口结束时，从 eval_battle_log.jsonl 读取并聚合 baseline 胜率。

        返回聚合结果字典（也写入文件）。
        """
        records = [r for r in self._eval_buffer
                   if episode_start <= r.get("episode", -1) <= episode_end]

        if not records:
            return {}

        # 清理已消费的旧缓冲区条目
        self._eval_buffer = [r for r in self._eval_buffer
                             if r.get("episode", -1) > episode_end]

        baseline_stats: Dict[str, Dict] = {}
        for r in records:
            agent_name = r.get("baseline_agent_name", "unknown")
            if agent_name not in baseline_stats:
                baseline_stats[agent_name] = {
                    "battles": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "total_rounds": 0,
                    "total_reward": 0.0,
                }
            s = baseline_stats[agent_name]
            s["battles"] += 1
            res = r.get("result", "")
            if res == "win":
                s["wins"] += 1
            elif res == "loss":
                s["losses"] += 1
            else:
                s["draws"] += 1
            s["total_rounds"] += r.get("rounds", 0)
            s["total_reward"] += r.get("reward", 0.0)

        baseline_winrates = {}
        for name, s in baseline_stats.items():
            b = max(s["battles"], 1)
            baseline_winrates[name] = {
                "battles": s["battles"],
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_rate": round(s["wins"] / b, 4),
                "avg_rounds": round(s["total_rounds"] / b, 1),
                "avg_reward": round(s["total_reward"] / b, 4),
            }

        record = {
            "episode_start": episode_start,
            "episode_end": episode_end,
            "num_episodes": len(records),
            "baseline_winrates": baseline_winrates,
        }

        with open(self._baseline_winrates_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def _read_eval_records(self, ep_start: int, ep_end: int) -> List[Dict]:
        if not self._battle_log_path.exists():
            return []
        records = []
        with open(self._battle_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ep = rec.get("episode", -1)
                    if ep_start <= ep <= ep_end:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
        return records

    def close(self) -> None:
        from loguru import logger as _logger

        if self._round_handler_id is not None:
            try:
                _logger.remove(self._round_handler_id)
            except (ValueError, AttributeError):
                pass
            self._round_handler_id = None
