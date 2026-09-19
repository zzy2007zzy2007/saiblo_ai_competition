"""EpisodeBatch 聚合日志写入器

负责在窗口满时从内存缓冲区读取数据并写入聚合结果：
- episode_batch_battle_stats.jsonl：对战数据聚合
- episode_batch_train_stats.jsonl：训练指标聚合

使用内存环形缓冲区替代 JSONL 文件回读，避免性能随训练时间退化。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class EpisodeBatchWriter:
    """EpisodeBatch 聚合日志写入器。

    职责：
    - 维护 EpisodeBatch 窗口计数器
    - 接收每局对战数据存入内存缓冲
    - 窗口满时从内存缓冲聚合对战数据和训练指标
    """

    def __init__(
        self,
        battle_log_path: str,
        batch_metrics_path: str,
        battle_stats_path: str,
        train_stats_path: str,
        window_size: int = 8,
    ):
        self._battle_log_path = Path(battle_log_path)
        self._batch_metrics_path = Path(batch_metrics_path)
        self._battle_stats_path = Path(battle_stats_path)
        self._train_stats_path = Path(train_stats_path)

        for p in [self._battle_stats_path, self._train_stats_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        self._window_size = window_size
        self._episode_count_in_window = 0
        self._current_window_start: Optional[int] = None
        self._batch_count_in_window = 0

        # 内存缓冲，替代从 JSONL 文件回读
        self._battle_buffer: List[Dict] = []
        self._batch_buffer: List[Dict] = []

    def on_episode_complete(
        self, episode: int, battle_record: Optional[Dict] = None
    ) -> None:
        """每局结束后调用。仅计数，等 on_batch_complete 统一处理。"""
        if self._current_window_start is None:
            self._current_window_start = episode
        self._episode_count_in_window += 1
        if battle_record is not None:
            self._battle_buffer.append(battle_record)

    def on_batch_complete(self, batch_metrics: Optional[Dict] = None) -> None:
        """每次 PPO update 后调用。窗口满时同时 flush battle stats 和 train stats。"""
        self._batch_count_in_window += 1
        if batch_metrics is not None:
            self._batch_buffer.append(batch_metrics)
        if self._episode_count_in_window >= self._window_size:
            self._flush_battle_window()
            self._flush_train_window()
            self._reset_window()

    def flush_now(self) -> None:
        """强制刷新当前窗口（训练结束时调用）。"""
        if self._episode_count_in_window > 0:
            self._flush_battle_window()
            self._flush_train_window()
            self._reset_window()

    def _flush_battle_window(self) -> None:
        if self._current_window_start is None:
            return

        episode_end = self._current_window_start + self._episode_count_in_window - 1

        if not self._battle_buffer:
            return

        aggregated = self._aggregate_battle(self._battle_buffer, episode_end)
        with open(self._battle_stats_path, "a") as f:
            f.write(json.dumps(aggregated, ensure_ascii=False) + "\n")

    def _flush_train_window(self) -> None:
        if self._current_window_start is None:
            return

        episode_end = self._current_window_start + self._episode_count_in_window - 1

        if not self._batch_buffer:
            return

        aggregated = self._aggregate_train(self._batch_buffer, episode_end)
        with open(self._train_stats_path, "a") as f:
            f.write(json.dumps(aggregated, ensure_ascii=False) + "\n")

    @staticmethod
    def _aggregate_battle(records: List[Dict], episode_end: int) -> Dict:
        n = len(records)
        if n == 0:
            return {}

        def mean(key: str) -> float:
            return sum(r.get(key, 0.0) for r in records) / n

        result = {
            "episode_start": records[0].get("episode", 0),
            "episode_end": episode_end,
            "num_episodes": n,
            "win_rate": sum(1 for r in records if r.get("result") == "win") / n,
            "draw_rate": sum(1 for r in records if r.get("result") == "draw") / n,
            "avg_reward": mean("reward"),
            "avg_rounds": mean("rounds"),
            "avg_end_own_hp": mean("end_own_hp"),
            "avg_end_enemy_hp": mean("end_enemy_hp"),
            "avg_end_own_coins": mean("end_own_coins"),
            "avg_max_own_coins": mean("max_own_coins"),
            "avg_total_own_coin_income": mean("total_own_coin_income"),
            "avg_end_enemy_coins": mean("end_enemy_coins"),
            "avg_max_enemy_coins": mean("max_enemy_coins"),
            "avg_total_enemy_coin_income": mean("total_enemy_coin_income"),
        }

        action_counts_all = {}
        for r in records:
            for k, v in r.get("action_counts", {}).items():
                action_counts_all[k] = action_counts_all.get(k, 0) + v
        result["avg_action_counts"] = {
            k: round(v / n, 2) for k, v in action_counts_all.items()
        }

        action_rewards_all = {}
        for r in records:
            for k, v in r.get("action_rewards", {}).items():
                action_rewards_all[k] = action_rewards_all.get(k, 0.0) + v
        result["avg_action_rewards"] = {
            k: round(v / n, 4) for k, v in action_rewards_all.items()
        }

        reward_source_keys = [
            "rw_hp_attack_base",
            "rw_hp_attack_tower",
            "rw_coin_gain",
            "rw_tower_survival",
            "rw_balance",
            "rw_tech_bonus",
            "rw_die_penalty",
            "rw_end_reward",
        ]
        for key in reward_source_keys:
            # reward_source 字段嵌套在 reward_sources dict 中
            vals = [r.get("reward_sources", {}).get(key, 0.0) for r in records]
            result[f"avg_{key}"] = sum(vals) / max(n, 1)

        return result

    @staticmethod
    def _aggregate_train(records: List[Dict], episode_end: int) -> Dict:
        n = len(records)
        if n == 0:
            return {}

        def mean(key: str) -> float:
            return sum(r.get(key, 0.0) for r in records) / n

        episode_start = records[0].get("episode", 0)
        result = {
            "episode_start": episode_start,
            "episode_end": episode_end,
            "num_batches": n,
            "avg_policy_loss": mean("policy_loss"),
            "avg_value_loss": mean("value_loss"),
            "avg_aux_tower_loss": mean("aux_tower_loss"),
            "avg_aux_gold_loss": mean("aux_gold_loss"),
            "avg_aux_enemy_tower_loss": mean("aux_enemy_tower_loss"),
            "avg_aux_enemy_gold_loss": mean("aux_enemy_gold_loss"),
            "avg_aux_base_loss": mean("aux_base_loss"),
            "avg_aux_enemy_base_loss": mean("aux_enemy_base_loss"),
            "avg_entropy": mean("entropy"),
            "avg_clip_fraction": mean("clip_fraction"),
            "avg_grad_norm": mean("grad_norm"),
            "avg_learning_rate": mean("learning_rate"),
            "avg_reward_mean": mean("reward_mean"),
            "avg_return_mean": mean("return_mean"),
        }
        return result

    def _reset_window(self) -> None:
        self._current_window_start = None
        self._episode_count_in_window = 0
        self._batch_count_in_window = 0
        self._battle_buffer = []
        self._batch_buffer = []

    def close(self) -> None:
        self.flush_now()
