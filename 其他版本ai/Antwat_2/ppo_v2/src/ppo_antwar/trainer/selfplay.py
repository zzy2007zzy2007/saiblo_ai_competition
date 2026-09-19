import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from ..trainer.ppo_trainer import PPOTrainer
from ..trainer.batch import EpisodeBatch
from ..battle.opponent_agent import OpponentAgent
from ..league.selfplay_manager import SelfPlayManager
from ..battle.battle_coordinator import BattleCoordinator as V2BattleCoordinator
from ..battle.battle_config import BaselineBattleConfig as V2BaselineBattleConfig
from ..config.path_config import PathConfig
from ..env.antwar_env import AntWarEnv
from ..monitor.logger import Logger
from ..monitor.time_tracker import TimeTracker
from ..monitor.system_metrics_sampler import SystemMetricsSampler
from ..monitor.selfplay_battle_writer import SelfPlayBattleWriter
from ..monitor.batch_metrics_writer import BatchMetricsWriter
from ..monitor.episode_batch_writer import EpisodeBatchWriter
from ..monitor.eval_battle_writer import EvalBattleWriter
from ..monitor.episode_record import EpisodeRecord
from .logging_subsystem import LoggingSubsystem
from .selfplay_logger import SelfPlayLogger
from .episode_collector import EpisodeCollector
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork


class SelfPlayTrainer:
    """自对弈训练器 - 驱动完整的训练循环。

    职责：
    - 驱动训练主循环
    - 管理数据收集（对战 + 先手/后手交替）
    - 协调对手选择和评分更新
    - 触发 PPO 更新
    - 触发基线对战评估
    - 管理训练状态持久化
    """

    def __init__(
        self,
        trainer: PPOTrainer,
        config: Dict[str, Any],
        path_config: PathConfig,
        run_id: str = "default",
    ):
        self._trainer = trainer
        self._config = config
        self._path_config = path_config
        self._run_id = run_id

        selfplay_cfg = config.get("selfplay", {})
        self._selfplay_manager = SelfPlayManager(
            opponent_pool_size=selfplay_cfg.get("opponent_pool_size"),
            min_opponent_games=selfplay_cfg.get("min_opponent_games"),
            exploit_prob=selfplay_cfg.get("exploit_prob"),
        )

        self._batch_pool: List[EpisodeBatch] = []
        self._episode_count: int = 0
        self._total_episodes: int = self._config.get("training", {}).get(
            "total_episodes"
        )
        self._last_battle_result: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}
        self._callbacks = None
        self._cache_clean_counter = 0

        self._episode_rewards: List[float] = []
        self._episode_lengths: List[float] = []
        self._max_episode_stats = self._config.get("training", {}).get(
            "max_episode_stats", 1000
        )

        # ── 创建 NeuralAgent 和 EpisodeCollector ──
        self._self_agent = self._trainer.create_agent(deterministic=False)
        self._episode_collector = EpisodeCollector(
            config=self._config,
            episode_count_ref=lambda: self._episode_count,
            self_agent=self._self_agent,
        )

        self._init_logging_subsystem()

        self._load_league_state()

    CUDA_CACHE_CLEAN_INTERVAL = 3
    LEAGUE_SAVE_INTERVAL_MULTIPLIER = 5
    INITIAL_OPPONENT_COUNT = 3

    @staticmethod
    def _determine_battle_result(
        battle_details: Dict[str, Any],
        self_first: bool,
    ) -> str:
        """基于引擎权威返回值判定胜负。

        判定逻辑（与引擎 _judge_timeout_winner 对齐）：
        1. 正常结束（terminated=True）：使用引擎 state.winner
        2. 截断（truncated=True）：基于基地 HP 判定，与引擎超时判定一致
        """
        winner = battle_details.get("winner")
        terminated = battle_details.get("terminated", False)
        truncated = battle_details.get("truncated", False)

        if terminated and winner is not None:
            self_player = 0 if self_first else 1
            if winner == self_player:
                return "win"
            else:
                return "loss"

        if truncated:
            # 与引擎 _judge_timeout_winner 一致：基地 HP 高者胜
            snapshots = battle_details.get("snapshots", [])
            last_snapshot = snapshots[-1] if snapshots else {}
            own_base_hp = last_snapshot.get("own_base_hp", 0)
            enemy_base_hp = last_snapshot.get("enemy_base_hp", 0)
            if own_base_hp > enemy_base_hp:
                return "win"
            elif own_base_hp < enemy_base_hp:
                return "loss"
            else:
                return "draw"

        # 兜底：terminated 但无 winner，视为平局
        return "draw"

    PPO_METRIC_KEYS = [
        "policy_loss",
        "value_loss",
        "entropy",
        "entropy_type",
        "entropy_target",
        "aux_tower_loss",
        "aux_gold_loss",
        "aux_enemy_tower_loss",
        "aux_enemy_gold_loss",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "advantages_std",
        "return_mean",
        "return_std",
        "clip_fraction",
        "approx_kl",
        "grad_norm",
        "value_input_mean",
        "value_input_std",
        "value_pred_mean",
        "value_pred_std",
        "ratio_mean",
        "ratio_std",
        "grad_norm_policy",
        "grad_norm_value",
        "grad_norm_max_layer",
        "td_error_mean",
        "td_error_std",
        "valid_actions_mean",
        "valid_actions_min",
        "batch_size",
        "num_collected",
        "nan_skip_count",
    ]

    SOURCE_KEYS = [
        "rw_hp_attack_base",
        "rw_hp_attack_tower",
        "rw_coin_gain",
        "rw_tower_survival",
        "rw_balance",
        "rw_tech_bonus",
        "rw_die_penalty",
        "rw_end_reward",
    ]

    def _init_logging_subsystem(self) -> None:
        self._logging = self._create_logging_subsystem(
            self._path_config, self._run_id, self._config
        )
        self._logging.system_sampler.start()

    @staticmethod
    def _create_logging_subsystem(
        path_config: PathConfig, run_id: str, config: Dict
    ) -> LoggingSubsystem:
        """创建日志子系统，返回 LoggingSubsystem 实例"""
        training_dir = str(path_config.get_training_dir(run_id))
        evaluation_dir = str(path_config.get_evaluation_dir(run_id))
        system_dir = str(path_config.get_system_dir(run_id))
        selfplay_dir = str(path_config.get_selfplay_dir(run_id))

        loguru_handler_id = logger.add(
            training_dir + "/training_{time}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
        )
        logger.add(
            training_dir + "/training_error_{time}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="100 MB",
            retention="90 days",
        )

        time_tracker = TimeTracker()
        _logger = Logger(training_dir=training_dir, run_id=run_id)
        selfplay_logger = SelfPlayLogger(log_dir=selfplay_dir)

        from pathlib import Path

        battle_logger = SelfPlayBattleWriter(
            filepath=str(Path(training_dir) / "selfplay_battle_log.jsonl")
        )
        batch_logger = BatchMetricsWriter(
            filepath=str(Path(training_dir) / "batch_metrics.jsonl")
        )
        batch_window = config.get("training", {}).get(
            "episode_batch_window",
            config.get("training", {}).get("n_envs", 8),
        )
        ep_batch_logger = EpisodeBatchWriter(
            battle_log_path=str(Path(training_dir) / "selfplay_battle_log.jsonl"),
            batch_metrics_path=str(Path(training_dir) / "batch_metrics.jsonl"),
            battle_stats_path=str(
                Path(training_dir) / "episode_batch_battle_stats.jsonl"
            ),
            train_stats_path=str(
                Path(training_dir) / "episode_batch_train_stats.jsonl"
            ),
            window_size=batch_window,
        )
        eval_logger = EvalBattleWriter(
            eval_dir=evaluation_dir,
            window_size=batch_window,
        )

        system_sampler = SystemMetricsSampler(output_dir=system_dir)

        return LoggingSubsystem(
            loguru_handler_id=loguru_handler_id,
            time_tracker=time_tracker,
            logger=_logger,
            selfplay_logger=selfplay_logger,
            battle_logger=battle_logger,
            batch_logger=batch_logger,
            ep_batch_logger=ep_batch_logger,
            eval_logger=eval_logger,
            system_sampler=system_sampler,
        )

    def _close_logging(self) -> None:
        self._logging.close_all()

    # ── 上下文构建管道 ───────────────────────────────────────────────────────

    def _build_training_context(
        self, ppo_metrics: Dict, pending_action_stats: List[Dict] = None
    ) -> Dict[str, Any]:
        context = {}

        for key in self.PPO_METRIC_KEYS:
            if key in ppo_metrics:
                context[key] = ppo_metrics[key]

        ppo_cfg = self._config.get("ppo", {})
        context["entropy_coef"] = ppo_cfg.get("ent_coef")
        policy_lr, value_lr = self._trainer.get_learning_rate()
        context["learning_rate"] = policy_lr

        context["num_collected"] = len(self._batch_pool)
        context["last_reward"] = (
            self._episode_rewards[-1] if self._episode_rewards else 0.0
        )

        reward_sources = self._aggregate_reward_sources(pending_action_stats or [])
        context.update(reward_sources)

        return context

    def _aggregate_reward_sources(self, pending_stats: List[Dict]) -> Dict[str, float]:
        result = {k: 0.0 for k in self.SOURCE_KEYS}
        for stats in pending_stats:
            rd = stats.get("reward_sources", {})
            for k in self.SOURCE_KEYS:
                result[k] = result.get(k, 0.0) + rd.get(k, 0.0)
        return result

    def _aggregate_action_reward_stats(
        self, pending_stats: List[Dict]
    ) -> Dict[str, Dict]:
        """聚合一批对战的动作计数和动作奖励

        Returns:
            {'action_counts': Dict[str, int], 'action_rewards': Dict[str, float]}
        """
        total_action_counts: Dict[str, int] = {}
        total_action_rewards: Dict[str, float] = {}
        for stats in pending_stats:
            for action_type, count in stats.get("action_counts", {}).items():
                total_action_counts[action_type] = (
                    total_action_counts.get(action_type, 0) + count
                )
            for action_type, reward in stats.get("action_rewards", {}).items():
                total_action_rewards[action_type] = (
                    total_action_rewards.get(action_type, 0.0) + reward
                )
        return {
            "action_counts": total_action_counts,
            "action_rewards": total_action_rewards,
        }

    # ── 训练主循环 ─────────────────────────────────────────────────────────

    def _select_opponent_agent(self) -> Tuple[Optional[str], Optional[OpponentAgent]]:
        opponent_id = None
        opponent_agent = None
        try:
            opponent_id = self._selfplay_manager.select_opponent()
            checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(
                opponent_id
            )
            opponent_agent = OpponentAgent.from_checkpoint(
                checkpoint_path,
                device=self._trainer.device,
                hidden_dim=self._config.get("network", {}).get("hidden_dim"),
                enable_auxiliary=self._config.get("ppo", {}).get("enable_auxiliary"),
            )
        except Exception as e:
            logger.error(
                f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}"
            )
            opponent_id = None

        return opponent_id, opponent_agent

    def _collect_and_update_payoff(
        self,
        opponent_agent: Optional[OpponentAgent],
        opponent_id: Optional[str],
        self_first: bool,
        env_factory: Callable[[], AntWarEnv],
        episode_start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """收集对战数据并写入日志。

        返回集包含 action_counts、action_rewards、reward_sources、reward、length，
        供训练上下文构建和主循环使用。
        """
        env = env_factory()
        updated_info = {}
        try:
            batch, battle_details = self._collect_episode_with_swap(
                env,
                opponent_agent,
                self_first,
                opponent_id or "none",
            )
            self._batch_pool.append(batch)

            reward = battle_details.get("reward", 0)
            length = battle_details.get("length", 0)

            self._episode_rewards.append(reward)
            self._episode_lengths.append(length)
            if len(self._episode_rewards) > self._max_episode_stats:
                self._episode_rewards = self._episode_rewards[
                    -self._max_episode_stats :
                ]
            if len(self._episode_lengths) > self._max_episode_stats:
                self._episode_lengths = self._episode_lengths[
                    -self._max_episode_stats :
                ]

            result_str = self._determine_battle_result(battle_details, self_first)

            action_counts = battle_details.get("action_counts", {})
            action_rewards = battle_details.get("action_rewards", {})
            reward_sources = battle_details.get("reward_sources", {})
            snapshots = battle_details.get("snapshots", [])

            # 从步快照提取终局数据，消除对 env 的重复访问
            last_snapshot = snapshots[-1] if snapshots else {}
            end_own_hp = sum(last_snapshot.get("own_tower_hp", {}).values())
            end_enemy_hp = sum(last_snapshot.get("enemy_tower_hp", {}).values())
            end_own_coins = last_snapshot.get("own_coins", 0.0)
            end_enemy_coins = last_snapshot.get("enemy_coins", 0.0)
            max_own_coins = EpisodeCollector.compute_max_coin(snapshots, player="self")
            max_enemy_coins = EpisodeCollector.compute_max_coin(
                snapshots, player="enemy"
            )
            total_own_coin_income = EpisodeCollector.compute_total_coin_income(
                snapshots, player="self"
            )
            total_enemy_coin_income = EpisodeCollector.compute_total_coin_income(
                snapshots, player="enemy"
            )

            if self._logging.battle_logger:
                self._logging.battle_logger.write_episode(
                    EpisodeRecord(
                        episode=self._episode_count,
                        swap=self_first,
                        opponent_id=opponent_id or "none",
                        result=result_str,
                        rounds=battle_details.get("rounds", length),
                        reward=reward,
                        length=length,
                        start_time=episode_start_time,
                        end_time=time.time(),
                        duration=time.time() - episode_start_time,
                        end_own_hp=end_own_hp,
                        end_own_coins=end_own_coins,
                        max_own_coins=max_own_coins,
                        total_own_coin_income=total_own_coin_income,
                        end_enemy_hp=end_enemy_hp,
                        end_enemy_coins=end_enemy_coins,
                        max_enemy_coins=max_enemy_coins,
                        total_enemy_coin_income=total_enemy_coin_income,
                        action_counts=action_counts,
                        action_rewards=action_rewards,
                        reward_sources=reward_sources,
                    )
                )

            if self._logging.ep_batch_logger:
                # 构造对战记录 dict 传入内存缓冲（含 episode_batch_writer 聚合所需字段）
                battle_record = {
                    "episode": self._episode_count,
                    "swap": self_first,
                    "opponent_id": opponent_id or "none",
                    "result": result_str,
                    "rounds": battle_details.get("rounds", length),
                    "reward": reward,
                    "length": length,
                    "start_time": episode_start_time,
                    "end_time": time.time(),
                    "duration": time.time() - episode_start_time,
                    "end_own_hp": end_own_hp,
                    "end_own_coins": end_own_coins,
                    "max_own_coins": max_own_coins,
                    "total_own_coin_income": total_own_coin_income,
                    "end_enemy_hp": end_enemy_hp,
                    "end_enemy_coins": end_enemy_coins,
                    "max_enemy_coins": max_enemy_coins,
                    "total_enemy_coin_income": total_enemy_coin_income,
                    "action_counts": action_counts,
                    "action_rewards": action_rewards,
                    "reward_sources": reward_sources,
                }
                self._logging.ep_batch_logger.on_episode_complete(
                    self._episode_count, battle_record
                )

            if opponent_id is not None and self._selfplay_manager is not None:
                result = 1 if result_str == "win" else (-1 if result_str == "loss" else 0)
                self._selfplay_manager.update_payoff(
                    opponent_id,
                    result,
                )

            updated_info = {
                "action_counts": action_counts,
                "action_rewards": action_rewards,
                "reward_sources": reward_sources,
                "reward": reward,
                "length": length,
                "winner": battle_details.get("winner"),
                "terminated": battle_details.get("terminated", False),
                "truncated": battle_details.get("truncated", False),
                "snapshots": battle_details.get("snapshots", []),
            }
        finally:
            env.close()

        return updated_info

    def _flush_batch_to_update(self, pending_action_stats: List[Dict] = None) -> None:
        batch_size = self._config.get("ppo", {}).get("batch_size")
        n_envs = self._config.get("training", {}).get("n_envs")
        total_steps = sum(len(b) for b in self._batch_pool)
        if len(self._batch_pool) < n_envs or total_steps < batch_size:
            return

        merged = EpisodeBatch.merge(self._batch_pool)
        is_valid, issues = self._trainer.check_batch_quality(merged)
        if not is_valid:
            has_critical = any("NaN" in issue or "Inf" in issue for issue in issues)
            log_level = logger.error if has_critical else logger.warning
            for issue in issues:
                log_level(f"Batch validation failed: {issue}")
            if has_critical:
                self._trainer.handle_nan_recovery(self._episode_count)
            self._batch_pool = []
            return

        metrics = self._trainer.ppo_update(merged)

        context = self._build_training_context(metrics, pending_action_stats)
        metrics.update(context)

        self._trainer.store_last_metrics(metrics)
        if self._callbacks is not None:
            self._callbacks.on_step()

        aggregated = self._aggregate_action_reward_stats(pending_action_stats or [])
        metrics["action_counts"] = aggregated["action_counts"]
        metrics["action_rewards"] = aggregated["action_rewards"]

        metrics["episode"] = self._episode_count
        if self._logging.batch_logger:
            self._logging.batch_logger.write_batch(metrics)

        if self._logging.ep_batch_logger:
            self._logging.ep_batch_logger.on_batch_complete(metrics)

        # 在 EpisodeBatch 窗口边界上同时聚合评估日志
        if self._logging.eval_logger:
            batch_window = self._config.get("training", {}).get(
                "episode_batch_window",
                self._config.get("training", {}).get("n_envs", 10),
            )
            if self._episode_count % batch_window == 0:
                episode_start = self._episode_count - batch_window + 1
                self._logging.eval_logger.on_episode_batch_complete(
                    episode_start, self._episode_count
                )

        logger.info(
            f"Episode {self._episode_count}/{self._total_episodes} | "
            f"loss={metrics.get('training_loss', 0):.4f}, "
            f"reward_mean={metrics.get('reward_mean', 0):.2f}, "
            f"entropy={metrics.get('entropy', 0):.4f}"
        )

        self._batch_pool = []
        self._update_lr_schedule()

        # 周期性清空 CUDA 缓存，缓解显存碎片化
        self._cache_clean_counter += 1
        if self._cache_clean_counter % self.CUDA_CACHE_CLEAN_INTERVAL == 0:
            torch.cuda.empty_cache()

    def train(
        self,
        num_episodes: int,
        env_factory: Callable[[], AntWarEnv],
        callbacks=None,
    ) -> None:
        logger.info(f"Starting self-play training for {num_episodes} episodes")

        self._total_episodes = num_episodes
        self._callbacks = callbacks

        device = getattr(self._trainer, "device", "cpu")
        self._logging.logger.log_training_start(self._config, device)
        self._logging.time_tracker.start()

        self._episode_count = 0
        self._last_battle_result = {"wins": 0, "losses": 0, "draws": 0}

        # 创建初始对手，避免前 100 个 episode 无有效对手
        self._create_initial_opponents()

        opponent_win_stats: Dict[str, Dict[str, int]] = {}
        total_steps = 0

        try:
            while self._episode_count < num_episodes:
                self._episode_count += 1

                pending_info, reward, length, battle_result = self._run_single_episode(
                    env_factory
                )
                total_steps += length

                self._update_opponent_stats(
                    opponent_win_stats,
                    battle_result,
                    pending_info.get("opponent_id", "none"),
                )

                self._flush_batch_to_update(
                    pending_action_stats=[pending_info] if pending_info else None
                )

                self._run_periodic_tasks()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error at episode {self._episode_count}: {e}")
            raise
        finally:
            self._finalize_training(total_steps, opponent_win_stats)

    def _run_single_episode(
        self,
        env_factory: Callable[[], AntWarEnv],
    ) -> Tuple[Dict[str, Any], float, int, str]:
        """运行单个 episode：更新进度、选择对手、收集对战数据并记录"""
        self._logging.time_tracker.start_training()
        episode_start_time = time.time()

        if self._selfplay_manager is not None:
            self._selfplay_manager.update_progress(self._episode_count)

        self_first = self._episode_count % 2 == 1
        opponent_id, opponent_agent = self._select_opponent_agent()

        self._logging.selfplay_logger.log_battle_start(
            self._episode_count, opponent_id or "none", self_first
        )

        pending_info = self._collect_and_update_payoff(
            opponent_agent,
            opponent_id,
            self_first,
            env_factory,
            episode_start_time=episode_start_time,
        )
        pending_info["opponent_id"] = opponent_id or "none"

        episode_duration = time.time() - episode_start_time
        self._logging.time_tracker.end_training()
        self._logging.time_tracker.record_episode(episode_duration)

        reward = pending_info.get("reward", 0.0)
        length = pending_info.get("length", 0)
        battle_result = self._determine_battle_result(pending_info, self_first)
        self._logging.selfplay_logger.log_battle_end(
            self._episode_count,
            opponent_id or "none",
            battle_result,
            reward,
            length,
        )

        return pending_info, reward, length, battle_result

    def _update_opponent_stats(
        self,
        opponent_win_stats: Dict[str, Dict[str, int]],
        battle_result: str,
        opp_key: str,
    ) -> None:
        """累计对手对战统计"""
        if opp_key not in opponent_win_stats:
            opponent_win_stats[opp_key] = {"wins": 0, "losses": 0, "draws": 0}
        if battle_result == "win":
            opponent_win_stats[opp_key]["wins"] += 1
            self._last_battle_result["wins"] += 1
        elif battle_result == "loss":
            opponent_win_stats[opp_key]["losses"] += 1
            self._last_battle_result["losses"] += 1
        else:
            opponent_win_stats[opp_key]["draws"] += 1
            self._last_battle_result["draws"] += 1

    def _run_periodic_tasks(self) -> None:
        """执行周期性任务：对手更新、基线评估、联赛持久化、状态保存"""
        save_frequency = self._config.get("training", {}).get("save_interval")
        opponent_update_frequency = self._config.get("training", {}).get(
            "opponent_update_interval"
        )
        eval_frequency = self._config.get("training", {}).get("battle_interval")
        status_save_interval = self._config.get("training", {}).get(
            "status_save_interval", 50
        )

        if self._episode_count % opponent_update_frequency == 0:
            self._update_opponent(self._episode_count)
        if self._episode_count % eval_frequency == 0:
            self._run_battle_evaluation()
        if (
            self._episode_count
            % (save_frequency * self.LEAGUE_SAVE_INTERVAL_MULTIPLIER)
            == 0
        ):
            self._save_league_state()
        if self._episode_count % status_save_interval == 0:
            system_dir = str(self._path_config.get_system_dir(self._run_id))
            time_stats_path = Path(system_dir) / "time_statistics.json"
            self._logging.time_tracker.save_time_statistics(str(time_stats_path))
            self._logging.logger.save_battle_stats(self._last_battle_result)

    def _finalize_training(
        self,
        total_steps: int,
        opponent_win_stats: Dict[str, Dict[str, int]],
    ) -> None:
        """训练结束清理：flush 残留、保存最终状态、训练摘要"""
        if self._logging.ep_batch_logger:
            self._logging.ep_batch_logger.flush_now()
        self._save_league_state()

        total_time = self._logging.time_tracker.elapsed_seconds

        opponent_win_rates = {}
        for opp, stats in opponent_win_stats.items():
            total_battles = sum(stats.values())
            if total_battles > 0:
                opponent_win_rates[opp] = round(stats["wins"] / total_battles, 4)

        summary = {
            "total_episodes": self._episode_count,
            "total_steps": total_steps,
            "best_avg_reward": max(self._episode_rewards)
            if self._episode_rewards
            else 0,
            "best_avg_reward_episode": self._episode_rewards.index(
                max(self._episode_rewards)
            )
            + 1
            if self._episode_rewards
            else 0,
            "total_time": f"{total_time:.2f}s ({total_time / 3600:.2f}h)",
            "win_rate": (
                self._last_battle_result["wins"]
                / max(sum(self._last_battle_result.values()), 1)
            ),
            "avg_reward": float(np.mean(self._episode_rewards))
            if self._episode_rewards
            else 0,
            "avg_loss": 0,
            "opponent_win_rates": opponent_win_rates,
        }
        self._logging.logger.log_training_complete(summary)

        system_dir = str(self._path_config.get_system_dir(self._run_id))
        time_stats_path = Path(system_dir) / "time_statistics.json"
        self._logging.time_tracker.save_time_statistics(str(time_stats_path))

        logger.info(f"Training completed at episode {self._episode_count}")
        self._close_logging()

    # ── 数据收集 ───────────────────────────────────────────────────────────

    def _collect_episode_with_swap(
        self,
        env: AntWarEnv,
        opponent_agent,
        self_first: bool,
        opponent_id: str,
    ) -> Tuple[EpisodeBatch, Dict[str, Any]]:
        """委托 EpisodeCollector 收集对战数据。"""
        return self._episode_collector.collect(
            env=env,
            opponent_agent=opponent_agent,
            opponent_id=opponent_id,
            self_first=self_first,
        )

    # ── 对手管理 ───────────────────────────────────────────────────────────

    def _create_initial_opponents(self) -> None:
        """创建初始对手池，避免前 100 个 episode 无有效对手训练。"""
        if self._selfplay_manager is None:
            return

        opponent_count = self._selfplay_manager.get_opponent_count()
        if opponent_count > 0:
            logger.info(
                f"Opponent pool already has {opponent_count} opponents, skipping initial creation"
            )
            return

        logger.info(
            f"Creating {self.INITIAL_OPPONENT_COUNT} initial random opponents..."
        )
        checkpoint_dir = (
            self._path_config.get_selfplay_dir(self._run_id) / "opponent_checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.INITIAL_OPPONENT_COUNT):
            try:
                random_policy = AntWarPolicyValueNetwork(
                    hidden_dim=self._config.get("network", {}).get("hidden_dim"),
                    enable_auxiliary=self._config.get("ppo", {}).get(
                        "enable_auxiliary"
                    ),
                )
                random_policy.to(self._trainer.device)

                checkpoint = {
                    "policy_state_dict": random_policy.state_dict(),
                    "config": self._config,
                }
                filename = f"initial_opponent_{uuid.uuid4().hex[:8]}.pt"
                checkpoint_path = str(checkpoint_dir / filename)
                torch.save(checkpoint, checkpoint_path)

                self._selfplay_manager.add_opponent(checkpoint_path, -1)
                logger.info(
                    f"Created initial opponent {i + 1}/{self.INITIAL_OPPONENT_COUNT}: {filename}"
                )
            except Exception as e:
                logger.error(f"Failed to create initial opponent {i + 1}: {e}")

    def _update_opponent(self, episode: int) -> None:
        if self._trainer is None:
            return

        checkpoint_dir = (
            self._path_config.get_selfplay_dir(self._run_id) / "opponent_checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_dir / f"opponent_ep{episode}.pt")

        self._trainer.save_checkpoint(checkpoint_path, episode, {})

        if self._selfplay_manager is not None:
            self._selfplay_manager.add_opponent(checkpoint_path, episode)
            opponent_id = f"opponent_ep{episode}"
            pool_size = self._selfplay_manager.get_opponent_count()
            self._logging.selfplay_logger.log_opponent_added(
                episode, opponent_id, pool_size
            )
            logger.info(f"Opponent snapshot saved at episode {episode}")

    # ── 对战评估 ───────────────────────────────────────────────────────────

    def _run_battle_evaluation(self) -> None:
        ppo_agent = self._trainer.create_agent(deterministic=True)
        baseline_agents = self._config.get("selfplay", {}).get(
            "baseline_agents", ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"]
        )

        try:
            from ..env.observation import ObservationEncoder
            from ..env.action_mask import ActionMaskHandler
            from ..battle.ppo_war_agent import PPOWarAgent

            observation_encoder = ObservationEncoder()
            action_mask_handler = ActionMaskHandler()
            ppo_war_agent = PPOWarAgent(
                player_id=0,
                ppo_agent=ppo_agent,
                observation_encoder=observation_encoder,
                action_mask_handler=action_mask_handler,
            )
            torch.cuda.empty_cache()

            v2_cfg = V2BaselineBattleConfig(
                max_rounds=self._config["training"]["max_steps_per_episode"],
                n_battles=self._config["training"]["n_battles"],
                baseline_agents=baseline_agents,
                device=self._config.get("device", "cpu"),
            )
            v2_coordinator = V2BattleCoordinator(
                v2_cfg, self._path_config, eval_logger=self._logging.eval_logger
            )
            results = v2_coordinator.run_ppo_evaluation_with_agent(
                ppo_war_agent, baseline_agents, episode=self._episode_count
            )
            self._log_battle_evaluation_results(results, prefix="battle_v2")
        except Exception as e:
            logger.error(f"Battle evaluation failed: {e}")

    def _log_battle_evaluation_results(
        self, results: Dict[str, Dict[str, Any]], prefix: str = ""
    ) -> None:
        """记录基线对战评估结果到日志和 eval_logger。

        Args:
            results: BattleCoordinator.run_ppo_evaluation_with_agent 的返回值，
                     格式为 {baseline_name: {agent1_wins, agent2_wins, draws, ...}}
        """
        for baseline, stats in results.items():
            wins = stats.get("agent1_wins", 0)
            losses = stats.get("agent2_wins", 0)
            draws = stats.get("draws", 0)
            total = wins + losses + draws
            win_rate = wins / max(total, 1)
            logger.info(
                f"[{prefix}] Episode {self._episode_count} | vs {baseline}: win_rate={win_rate:.2%}",
            )

            if self._logging.eval_logger:
                n_battles = stats.get("total_battles", 0)
                for i in range(n_battles):
                    self._logging.eval_logger.write_battle(
                        episode=self._episode_count,
                        baseline_agent_name=baseline,
                        battle_mode="selfplay_eval",
                        result="win"
                        if i < wins
                        else ("loss" if i < (wins + losses) else "draw"),
                        rounds=int(stats.get("avg_rounds", 0)),
                        reward=stats.get("avg_reward", 0),
                        length=int(stats.get("avg_rounds", 0)),
                    )

    # ── 学习率调度 ─────────────────────────────────────────────────────────

    def _update_lr_schedule(self) -> None:
        if self._trainer is not None:
            self._trainer.update_lr_schedule(self._episode_count, self._total_episodes)

    def get_last_metrics(self) -> Optional[Dict]:
        """公开最近一次 PPO update 的指标"""
        return getattr(self, "_last_metrics", None)

    # ── 联赛状态持久化 ─────────────────────────────────────────────────────

    def _save_league_state(self) -> None:
        if self._selfplay_manager is None:
            return

        league_dir = self._path_config.get_selfplay_dir(self._run_id) / "league"
        league_dir.mkdir(parents=True, exist_ok=True)

        self._selfplay_manager.save_state(
            str(league_dir / "pool.json"),
            str(league_dir / "payoff.json"),
        )
        logger.info("League state saved")

    def _load_league_state(self) -> None:
        if self._selfplay_manager is None:
            return

        league_dir = self._path_config.get_selfplay_dir(self._run_id) / "league"
        pool_path = league_dir / "pool.json"
        payoff_path = league_dir / "payoff.json"

        if pool_path.exists() and payoff_path.exists():
            self._selfplay_manager.load_state(
                str(pool_path),
                str(payoff_path),
            )
            logger.info("League state loaded")
