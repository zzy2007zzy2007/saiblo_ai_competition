"""Episode 数据收集器 - 将对战数据收集从 SelfPlayTrainer 中解耦出来。

职责：
- 运行单回合对战循环（时间步推进）
- 管理观测视角切换（先手/后手）
- 采集步快照（塔 HP、金币）
- 计算辅助任务标签
- 构建对战详情
"""

import time as time_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..battle.opponent_agent import OpponentAgent
from ..env.antwar_env import AntWarEnv
from ..utils.aux_labels import compute_aux_labels_from_trajectory
from .._agent.protocol import RecordingAgent
from .batch import EpisodeBatch


class EpisodeCollector:
    """Episode 数据收集器。

    封装完整的单回合对战数据收集流程，与训练编排解耦。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        episode_count_ref: Callable[[], int],
        self_agent: RecordingAgent,
    ):
        self._config = config
        self._episode_count_ref = episode_count_ref
        self._self_agent = self_agent
        self._max_steps = config.get("training", {}).get("max_steps_per_episode", 512)

        self._source_keys = [
            "rw_hp_attack_base",
            "rw_hp_attack_tower",
            "rw_coin_gain",
            "rw_tower_survival",
            "rw_balance",
            "rw_tech_bonus",
            "rw_die_penalty",
            "rw_end_reward",
        ]

    # ═══════════════════════════════════════════════════════════════
    # 公开入口
    # ═══════════════════════════════════════════════════════════════

    def collect(
        self,
        env: AntWarEnv,
        opponent_agent: Optional[OpponentAgent],
        opponent_id: str,
        self_first: bool,
    ) -> Tuple[EpisodeBatch, Dict[str, Any]]:
        """收集一局对战数据。

        Args:
            env: 游戏环境（需已 reset）
            opponent_agent: 对手 Agent
            opponent_id: 对手 ID（仅用于日志）
            self_first: 当前 agent 是否为先手

        Returns:
            (batch, battle_details)
        """
        batch = EpisodeBatch()

        observations, _ = env.reset()
        obs_self, obs_opponent = self._setup_observations(observations)

        done = False
        episode_reward = 0.0
        episode_length = 0
        action_counts: Dict[str, int] = {}
        action_rewards: Dict[str, float] = {}
        reward_sources = {k: 0.0 for k in self._source_keys}
        step_snapshots = []
        step_aux_data = []  # per-round: own_twr_dmg, enemy_twr_dmg, own_gold, enemy_gold

        while not done:
            step_snapshots.append(
                self._capture_snapshot(env, self_first, episode_length)
            )

            action_self, log_prob, value = self._self_agent.act_record(obs_self)
            action_opponent = opponent_agent.act(obs_opponent)

            (
                observations_new,
                (reward_self, reward_opponent),
                terminated,
                truncated,
                info,
            ) = self._resolve_turn_and_extract_reward(
                env, action_self, action_opponent, self_first
            )
            done = terminated or truncated

            batch.add_step(obs_self, action_self, reward_self, value, log_prob, done)

            episode_reward += reward_self
            episode_length += 1

            # 采集 per-round 辅助标签原始值（mid → after 纯值）
            rd = info.get("reward_detail", {})
            coin_gain = rd.get("coin_gain_raw", (0.0, 0.0))
            hp_dmg_raw = rd.get("hp_dmg_raw", (0.0, 0.0))
            step_aux_data.append({
                "own_twr_dmg": rd.get("self_tower_dmg", 0.0),
                "enemy_twr_dmg": rd.get("opponent_tower_dmg", 0.0),
                "own_gold": coin_gain[0],
                "enemy_gold": coin_gain[1],
                "own_base_dmg": hp_dmg_raw[1],
                "enemy_base_dmg": hp_dmg_raw[0],
            })

            self._update_reward_sources(reward_sources, info)
            self._update_action_stats(
                action_counts, action_rewards, action_self, env, self_first
            )

            rd = info.get("reward_detail", {})
            coin_gain_self = rd.get("coin_gain_raw", (0.0, 0.0))[0]
            twr_hp_begin = rd.get("self_tower_hp_before", 0)
            twr_hp_end = rd.get("self_tower_hp_after", 0)
            twr_hp_delta = twr_hp_end - twr_hp_begin
            twr_hp_dmg = rd.get("self_tower_dmg", 0)
            base_hp_begin = rd.get("self_hp_before", 0)
            base_hp_end = rd.get("self_hp_after", 0)
            base_hp_dmg = max(0.0, base_hp_begin - base_hp_end)
            non_action_rewards = (
                f"hp_dmg={rd.get('hp_dmg_reward', 0):.3f} "
                f"twr_dmg={rd.get('tower_dmg_reward', 0):.3f} "
                f"coin_gain={rd.get('coin_gain_reward', 0):.3f} "
                f"coin_pnl={rd.get('coin_penalty_reward', 0):.3f} "
                f"twr_surv={rd.get('tower_survival', 0):.3f} "
                f"tech={rd.get('tech_bonus', 0):.3f} "
                f"bal={rd.get('balance', 0):.3f} "
                f"die={rd.get('die_penalty', 0):.3f} "
                f"end={rd.get('end_reward', 0):.3f}"
            )
            logger.info(
                f"[ROUND] episode={self._episode_count_ref()} round={episode_length} "
                f"action_self={action_self} action_opp={action_opponent} "
                f"time={time_module.time():.2f} "
                f"act_rw={rd.get('action_reward', 0):.3f} "
                f"coin_spent={rd.get('self_expense', 0):.1f} "
                f"coin_begin={rd.get('self_coins_before', 0):.1f} "
                f"coin_earn={coin_gain_self:.1f} "
                f"coin_end={rd.get('self_coins_after', 0):.1f} "
                f"twr_hp_begin={twr_hp_begin:.1f} "
                f"twr_hp_delta={twr_hp_delta:.1f} "
                f"twr_hp_dmg={twr_hp_dmg:.1f} "
                f"twr_hp_end={twr_hp_end:.1f} "
                f"base_hp_begin={base_hp_begin:.1f} "
                f"base_hp_dmg={base_hp_dmg:.1f} "
                f"base_hp_end={base_hp_end:.1f} "
                f"{non_action_rewards} "
                f"r_total={reward_self:.4f} r_opp={reward_opponent:.4f} "
                f"done={done} opponent_id={opponent_id}"
            )

            if done and episode_length == 1:
                logger.warning(
                    f"[TERMINAL_1ST_ROUND] episode={self._episode_count_ref()} "
                    f"action_self={action_self} action_opp={action_opponent} "
                    f"reward={reward_self:.4f} done={done}"
                )

            if not done:
                obs_self, obs_opponent = self._update_observations(
                    observations_new,
                    obs_self,
                    obs_opponent,
                )

            if episode_length >= self._max_steps:
                done = True

        self._attach_aux_labels(batch, step_aux_data)

        # 检测截断：达到 max_steps 但 env 未返回 terminated/truncated
        # 此时 obs_self 已在第 129-134 行更新为 s_{t+1}（因为 not done 为 True）
        was_truncated = (episode_length >= self._max_steps) and not (terminated or truncated)
        if was_truncated:
            batch.final_value = self._self_agent.get_value(obs_self)
            # 修正末步 done=True，确保 GAE 在 episode 边界正确重置
            batch.dones[-1] = True
        else:
            batch.final_value = 0.0
        # final_values 列表：单 episode batch 仅一个元素，merge 时会收集所有 episode 的值
        batch.final_values = [batch.final_value]

        battle_details = self._build_battle_details(
            episode_reward,
            episode_length,
            env,
            action_counts,
            action_rewards,
            reward_sources,
            step_snapshots,
        )
        # 从最后一步的 info 中传递引擎的权威胜负信息
        battle_details["winner"] = info.get("winner")
        battle_details["terminated"] = info.get("terminated", False)
        # EpisodeCollector 自身截断也算 truncated
        battle_details["truncated"] = info.get("truncated", False) or was_truncated

        return batch, battle_details

    # ═══════════════════════════════════════════════════════════════
    # 观测视角切换
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _setup_observations(observations) -> Tuple[Dict, Dict]:
        """从 observations 中提取 self/opponent 观测。"""
        obs_self = observations.get("self", {})
        obs_opponent = observations.get("opponent", {})
        return obs_self, obs_opponent

    @staticmethod
    def _update_observations(observations_new, obs_self, obs_opponent):
        """更新双方观测。"""
        obs_self = observations_new.get("self", obs_self)
        obs_opponent = observations_new.get("opponent", obs_opponent)
        return obs_self, obs_opponent

    # ═══════════════════════════════════════════════════════════════
    # 对手动作 & 回合推进（内化到 Collector，消除外部回调）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _get_opponent_move(opponent_agent, obs: Dict) -> int:
        """获取对手动作。若 opponent_agent 为 None，随机选择合法动作。"""
        if opponent_agent is None:
            valid_actions = np.where(obs["action_mask"] > 0.5)[0]
            return int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0
        return opponent_agent.act(obs)

    @staticmethod
    def _resolve_turn_and_extract_reward(
        env: AntWarEnv,
        action: int,
        opponent_action: int,
        self_first: bool,
    ) -> Tuple[Dict, Tuple[float, float], bool, bool, Dict]:
        """执行回合推进，使用新 env.step() 接口（action_self, action_opponent, self_first）。"""
        observations, (reward_self, reward_opponent), terminated, truncated, info = (
            env.step(action, opponent_action, self_first)
        )
        return observations, (reward_self, reward_opponent), terminated, truncated, info

    # ═══════════════════════════════════════════════════════════════
    # 状态快照
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _capture_snapshot(env: AntWarEnv, self_first: bool, round_idx: int) -> Dict:
        """采集单步状态快照（基地 HP、塔 HP、金币）。

        使用 AntWarEnv 的公共方法 get_state_snapshot()（阶段一已添加）。
        """
        self_player = 0 if self_first else 1
        snapshot = env.get_state_snapshot(self_player)
        return {
            "round_idx": snapshot.round_index,
            "own_base_hp": snapshot.self_state.hp,
            "enemy_base_hp": snapshot.opponent_state.hp,
            "own_tower_hp": snapshot.self_state.tower_hp,
            "enemy_tower_hp": snapshot.opponent_state.tower_hp,
            "own_coins": snapshot.self_state.coins,
            "enemy_coins": snapshot.opponent_state.coins,
        }

    # ═══════════════════════════════════════════════════════════════
    # 奖励 & 动作统计
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _update_reward_sources(reward_sources: Dict, info: Dict):
        """从 info 中更新奖励来源统计。"""
        reward_detail = info.get("reward_detail", {})
        if not reward_detail:
            return
        reward_sources["rw_hp_attack_base"] += reward_detail.get("hp_dmg_reward", 0.0)
        reward_sources["rw_hp_attack_tower"] += reward_detail.get(
            "tower_dmg_reward", 0.0
        )
        reward_sources["rw_coin_gain"] += reward_detail.get("coin_gain_reward", 0.0)
        reward_sources["rw_tower_survival"] += reward_detail.get("tower_survival", 0.0)
        reward_sources["rw_balance"] += reward_detail.get("balance", 0.0)
        reward_sources["rw_tech_bonus"] += reward_detail.get("tech_bonus", 0.0)
        reward_sources["rw_die_penalty"] += reward_detail.get("die_penalty", 0.0)
        reward_sources["rw_end_reward"] += reward_detail.get("end_reward", 0.0)

    @staticmethod
    def _update_action_stats(
        action_counts: Dict,
        action_rewards: Dict,
        action_self: int,
        env: AntWarEnv,
        self_first: bool,
    ):
        """更新动作计数和动作奖励统计。

        使用 AntWarEnv 的公共方法 resolve_action_type() 和 compute_action_reward()（阶段一已添加）。
        """
        from ..env.antwar_env import AntWarEnv as EnvCls

        action_type_name, _ = EnvCls.resolve_action_type(action_self)
        if action_type_name is None:
            action_type_name = "unknown"
        action_counts[action_type_name] = action_counts.get(action_type_name, 0) + 1
        action_rewards[action_type_name] = action_rewards.get(
            action_type_name, 0.0
        ) + env.compute_action_reward(action_self, 0 if self_first else 1)

    # ═══════════════════════════════════════════════════════════════
    # 辅助标签
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _attach_aux_labels(batch: EpisodeBatch, step_aux_data: List[Dict]):
        """从 per-round 辅助数据计算累积标签并附加到 batch。"""
        tower_labels, gold_labels, base_labels = compute_aux_labels_from_trajectory(step_aux_data)
        if tower_labels.shape[0] > 0:
            num_steps = len(batch)
            valid_count = tower_labels.shape[0]
            if valid_count < num_steps:
                padded_tower = np.zeros((num_steps, 10), dtype=np.float32)
                padded_gold = np.zeros((num_steps, 10), dtype=np.float32)
                padded_base = np.zeros((num_steps, 10), dtype=np.float32)
                padded_tower[:valid_count] = tower_labels
                padded_gold[:valid_count] = gold_labels
                padded_base[:valid_count] = base_labels
                batch.aux_tower_damage = padded_tower
                batch.aux_gold_income = padded_gold
                batch.aux_base_damage = padded_base
            else:
                valid_count = num_steps
                batch.aux_tower_damage = tower_labels
                batch.aux_gold_income = gold_labels
                batch.aux_base_damage = base_labels
            mask = np.zeros(num_steps, dtype=bool)
            mask[:valid_count] = True
            batch.aux_valid_mask = mask

    # ═══════════════════════════════════════════════════════════════
    # 对战详情
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_battle_details(
        episode_reward: float,
        episode_length: int,
        env: AntWarEnv,
        action_counts: Dict,
        action_rewards: Dict,
        reward_sources: Dict,
        step_snapshots: List[Dict],
    ) -> Dict:
        """构建对战详情字典（精简版，不含训练上下文元数据）。"""
        return {
            "reward": episode_reward,
            "length": episode_length,
            "rounds": getattr(env, "round_count", episode_length),
            "action_counts": action_counts,
            "action_rewards": action_rewards,
            "reward_sources": reward_sources,
            "snapshots": step_snapshots,
        }

    # ═══════════════════════════════════════════════════════════════
    # HP/Coin 统计（供外部调用）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_max_coin(step_snapshots: List[Dict], player: str = "self") -> float:
        """从步快照中计算最高 Coin 余额。"""
        key = "own_coins" if player == "self" else "enemy_coins"
        if not step_snapshots:
            return 0.0
        return max(s.get(key, 0.0) for s in step_snapshots)

    @staticmethod
    def compute_total_coin_income(
        step_snapshots: List[Dict], player: str = "self"
    ) -> float:
        """从步快照中计算总金币收入（各步正向增量和）。"""
        key = "own_coins" if player == "self" else "enemy_coins"
        if len(step_snapshots) < 2:
            return 0.0
        total = 0.0
        prev = step_snapshots[0].get(key, 0.0)
        for s in step_snapshots[1:]:
            curr = s.get(key, 0.0)
            delta = curr - prev
            if delta > 0:
                total += delta
            prev = curr
        return total
