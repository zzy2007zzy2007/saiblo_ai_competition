from __future__ import annotations

import json
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from loguru import logger

import numpy as np
import torch

from ..league.payoff import BattleSharedPayoff
from ..league.opponent_selector import OpponentSelector, OpponentPool
from ..battle import BattleCoordinator, BaselineBattleConfig
from .ppo_trainer import PPOAgent, EpisodeBatch
from .selfplay_logger import SelfPlayLogger, SelfPlayLogLevel
from ppo_antwar.utils.action_constants import ACTION_DIM, TOWER_POSITIONS, ACTION_SPACE_CONFIG
from ppo_antwar.monitor.constants import BATTLE_PRINT_INTERVAL
from ppo_antwar.trainer.training_status import compute_and_save_training_status
from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork


class SelfPlayManager:
    def __init__(
        self,
        agent: PPOAgent,
        opponent_pool: OpponentPool,
        payoff: BattleSharedPayoff,
        exploit_prob: float = 0.7,
        adaptive_explore: bool = True,
        explore_schedule: str = "linear",
    ) -> None:
        self.agent = agent
        self.opponent_pool = opponent_pool
        self.payoff = payoff
        self.selector = OpponentSelector(
            payoff=payoff,
            exploit_prob=exploit_prob,
            adaptive=adaptive_explore,
            schedule=explore_schedule,
        )
        self.current_player_id = "current_agent"
        self.games_against_opponents = {}
        self._current_episode = 0
        self._total_rounds = 0
        self._total_battles = 0

    def update_progress(self, episode: int) -> None:
        """更新训练进度，用于动态探索/利用调整"""
        self._current_episode = episode
        self.selector.update_progress(episode)

    def select_opponent(self) -> Optional[str]:
        if len(self.opponent_pool) == 0:
            return None

        opponent_id = self.selector.select(
            player_id=self.current_player_id,
            opponent_candidates=self.opponent_pool.get_all(),
        )
        return opponent_id

    def add_opponent(self, checkpoint_path: str, episode: int = None) -> str:
        """
        添加新对手到对手池。

        Args:
            checkpoint_path: 对手的模型 checkpoint 路径
            episode: 添加时的训练 episode 数

        Returns:
            str: 新对手的 ID
        """
        opponent_id = f"opp_{uuid.uuid4().hex[:8]}"
        self.opponent_pool.add(opponent_id, checkpoint_path, episode=episode)
        self.payoff.add_player(opponent_id)
        return opponent_id

    def update_payoff(self, opponent_id: str, result: int, rounds: int = 0) -> None:
        """
        更新对战记录和评分。

        Args:
            opponent_id: 对手 ID
            result: 对战结果 (1=win, 0=draw, -1=loss)
            rounds: 对战回合数
        """
        self.payoff.update(
            home=self.current_player_id,
            away=opponent_id,
            result=result,
        )

        # 更新对手的对战场次
        self.opponent_pool.increment_games(opponent_id)

        # 更新内部统计字典（可保留用于监控）
        if opponent_id not in self.games_against_opponents:
            self.games_against_opponents[opponent_id] = {'wins': 0, 'losses': 0, 'draws': 0, 'total_rounds': 0}
        if result == 1:
            self.games_against_opponents[opponent_id]['wins'] += 1
        elif result == 0:
            self.games_against_opponents[opponent_id]['draws'] += 1
        else:
            self.games_against_opponents[opponent_id]['losses'] += 1

        self.games_against_opponents[opponent_id]['total_rounds'] += rounds
        self._total_rounds += rounds
        self._total_battles += 1

    def get_exploitability(self) -> float:
        """
        使用 TrueSkill 评分计算可利用性。

        可利用性解释：
        - = 1.0: 当前智能体远弱于对手池
        - = 0.5: 当前智能体与对手池平均水平相当
        - = 0.0: 当前智能体远强于对手池

        Returns:
            float: 可利用性评分，范围 [0.0, 1.0]
        """
        return self.payoff.get_exploitability()

    def get_statistics(self) -> dict:
        stats = {
            'num_opponents': len(self.opponent_pool),
            'games_against_opponents': dict(self.games_against_opponents),
            'exploitability': self.get_exploitability(),
            'avg_rounds': (self._total_rounds / self._total_battles) if self._total_battles > 0 else 0.0,
        }

        if len(self.opponent_pool) > 0:
            win_rates = self.payoff.get_all_win_rates(self.current_player_id)
            stats['win_rates'] = win_rates

        return stats


class SelfPlayTrainer:
    PPO_METRIC_KEYS = [
        'policy_loss', 'value_loss', 'entropy',
        'aux_tower_loss', 'aux_gold_loss',
        'reward_mean', 'reward_std', 'reward_min', 'reward_max',
        'advantages_std', 'return_mean', 'return_std',
        'clip_fraction', 'gradient_norm',
        'value_input_mean', 'value_input_std',
        'ratio_mean', 'ratio_std',
        'grad_norm_policy', 'grad_norm_value', 'grad_norm_max_layer',
        'nan_skip_count', 'valid_actions_mean', 'valid_actions_min',
        'td_error_mean', 'td_error_std', 'batch_size',
    ]

    PPO_METRIC_PREFIX_KEYS = ('type_', 'logit_', 'prob_')

    SOURCE_KEYS = [
        'rw_hp_attack_base', 'rw_hp_attack_tower', 'rw_coin_gain', 'rw_tower_survival',
        'rw_balance',
        'rw_tech_bonus', 'rw_die_penalty', 'rw_end_reward',
    ]

    def __init__(
        self,
        trainer,
        opponent_pool_size: int = 10,
        min_opponent_games: int = 8,
        exploit_prob: float = 0.7,
        n_battles: int = 5,
        battle_interval: int = 1000,
        selfplay_log_level: str = "INFO",
        decay_interval: int = 1000,
        initial_opponents: int = 3,
        baseline_agents: Optional[List[str]] = None,
    ) -> None:
        self.decay_interval = decay_interval
        self.initial_opponents = initial_opponents
        self.baseline_agents = baseline_agents or ['BasicTowerAI']
        self.trainer = trainer
        self.checkpoint_dir = trainer.path_config.checkpoint_dir
        self.log_dir = str(trainer.path_config.base_dir)
        self.n_battles = n_battles
        self.battle_interval = battle_interval
        self.training_status_save_interval = 100

        self.selfplay_logger = SelfPlayLogger(
            agent_name="current_agent",
            log_level=SelfPlayLogLevel.from_string(selfplay_log_level),
            log_dir=self.log_dir,
            enable_timing=True
        )

        # 初始化持久化目录（在 OpponentPool 之前）
        self.persist_dir = self.checkpoint_dir / "league_state"
        self.persist_dir.mkdir(exist_ok=True)

        self.payoff = BattleSharedPayoff(
            decay=0.99,
            min_win_rate_games=min_opponent_games,
        )

        self.opponent_pool = OpponentPool(
            max_size=opponent_pool_size,
            min_games_threshold=min_opponent_games,
            eviction_log_dir=str(self.persist_dir),
        )
        self.opponent_pool.set_payoff_reference(self.payoff)
        # 由于 checkpoint_path 现在由 OpponentPool 内部管理，_opponent_checkpoints 不再需要
        # self._opponent_checkpoints: Dict[str, str] = {}

        self.selfplay_manager = SelfPlayManager(
            agent=trainer.get_agent(),
            opponent_pool=self.opponent_pool,
            payoff=self.payoff,
            exploit_prob=exploit_prob,
        )

        # 初始化battle缓存（在_load_league_state之前）
        self.selfplay_battle_data_cache: Dict[str, Dict] = {}
        self.selfplay_battle_details_cache: List[Dict] = []
        self._selfplay_battle_save_interval = 100
        self._load_league_state()

        # 只有当没有加载到任何对手时，才创建初始对手
        if len(self.opponent_pool) == 0:
            self._create_initial_opponents()

        self.battle_coordinator: Optional[BattleCoordinator] = None

        if self.log_dir is not None:
            battle_config = BaselineBattleConfig()
            battle_config.baseline_agents = self.baseline_agents
            battle_config.n_battles = self.n_battles
            self.battle_coordinator = BattleCoordinator(battle_config, path_config=self.trainer.path_config)

        self.best_checkpoint = None
        self.best_win_rate = 0.0
        self.battle_count = 0
        self._last_battle_result = None

    def _load_opponent_agent(self, opponent_id: str) -> Optional[PPOAgent]:
        """
        从对手池加载对手的 agent。

        Args:
            opponent_id: 对手 ID

        Returns:
            PPOAgent 实例或 None（如果加载失败）
        """
        checkpoint_path = self.opponent_pool.get_checkpoint_path(opponent_id)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for opponent {opponent_id}: {checkpoint_path}")
            return None

        # Fix for PyTorch 2.6: weights_only default to True, need to allow easydict
        import torch.serialization
        from easydict import EasyDict
        with torch.serialization.safe_globals([EasyDict]):
            checkpoint = torch.load(checkpoint_path, map_location=self.trainer.device)

        # 从checkpoint的config读取网络配置（而不是当前trainer的配置）
        # 这样可以确保加载旧版本checkpoint时也能正常工作
        net_config = checkpoint.get('config', {}).get('network', self.trainer.network_config)

        policy = self.trainer.policy.__class__(
            board_shape=net_config.board_shape,
            global_dim=net_config.global_dim,
            action_dim=ACTION_DIM,
            hidden_dim=net_config.hidden_dim,
        ).to(self.trainer.device)
        policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        return PPOAgent(policy, self.trainer.device)

    def _save_selfplay_battle_details(self, battle_details: Dict) -> None:
        """保存单场SelfPlay对战详情到缓存"""
        opponent_id = battle_details.get('opponent_id', 'unknown')
        
        if opponent_id not in self.selfplay_battle_data_cache:
            self.selfplay_battle_data_cache[opponent_id] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'total_battles': 0,
                'total_reward': 0.0,
                    'avg_reward': 0.0,
                    'avg_rounds': 0.0,
                }
        
        self.selfplay_battle_data_cache[opponent_id]['total_battles'] += 1
        self.selfplay_battle_data_cache[opponent_id]['total_reward'] += battle_details.get('total_reward', 0)
        
        result = battle_details.get('result', 'draw')
        if result == 'win':
            self.selfplay_battle_data_cache[opponent_id]['wins'] += 1
        elif result == 'loss':
            self.selfplay_battle_data_cache[opponent_id]['losses'] += 1
        else:
            self.selfplay_battle_data_cache[opponent_id]['draws'] += 1
        
        total = self.selfplay_battle_data_cache[opponent_id]['total_battles']
        self.selfplay_battle_data_cache[opponent_id]['avg_reward'] = (
            self.selfplay_battle_data_cache[opponent_id]['total_reward'] / total
        )
        self.selfplay_battle_data_cache[opponent_id]['avg_rounds'] = (
            (self.selfplay_battle_data_cache[opponent_id]['avg_rounds'] * (total - 1) + 
             battle_details.get('rounds', 0)) / total
        )
        
        self.selfplay_battle_details_cache.append(battle_details)

    def _flush_selfplay_battle_cache(self) -> None:
        """将SelfPlay对战缓存刷新到文件"""
        if not self.log_dir:
            return
        
        if not self.selfplay_battle_data_cache and not self.selfplay_battle_details_cache:
            return
        
        battle_dir = os.path.join(self.log_dir, "selfplay", "selfplay_battles")
        detailed_dir = os.path.join(battle_dir, "detailed_battles")
        os.makedirs(detailed_dir, exist_ok=True)
        
        stats_file = os.path.join(battle_dir, "selfplay_battle_stats.json")
        try:
            with open(stats_file, "w", encoding='utf-8') as f:
                json.dump(self.selfplay_battle_data_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"SelfPlay对战统计已保存到 {stats_file}")
        except Exception as e:
            logger.warning(f"保存SelfPlay对战统计失败: {e}")
        
        for battle_detail in self.selfplay_battle_details_cache:
            opponent_safe = str(battle_detail.get('opponent_id', 'unknown')).replace(' ', '_').replace('/', '_')
            episode = battle_detail.get('episode', 0)
            battle_id = f"selfplay_{episode}_{opponent_safe}_{int(time.time() * 1000)}"
            battle_log_path = os.path.join(detailed_dir, f"{battle_id}.json")
            try:
                with open(battle_log_path, "w", encoding='utf-8') as f:
                    json.dump(battle_detail, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"保存SelfPlay对战详情失败: {e}")
        
        self.selfplay_battle_details_cache = []
        self.selfplay_battle_data_cache = {}
        logger.debug(f"SelfPlay对战缓存已保存并清空")

    def _save_training_status(self, episode: int, battle_result: Optional[Tuple[int, int, int]]) -> None:
        logger.info(f"[_save_training_status] 开始执行: episode={episode}, battle_result={battle_result}")
        try:
            compute_and_save_training_status(self.trainer, episode, battle_result)
        except Exception as e:
            logger.error(f"⚠ 保存训练状态异常: {e}", exc_info=True)
            if self.trainer.logger:
                self.trainer.logger.log_exception(e, {"event": "save_training_status_error", "episode": episode})
        else:
            logger.info(f"[_save_training_status] 执行完成")

    def _collect_episode_with_swap(
        self,
        env,
        opponent_agent: Optional[PPOAgent],
        swap_positions: bool,
        opponent_id: Optional[str] = None,
        episode: int = 0,
    ):
        batch = EpisodeBatch(
            observations_board=[],
            observations_global=[],
            observations_mask=[],
            actions=[],
            rewards=[],
            values=[],
            log_probs=[],
            dones=[],
        )

        player = 1 if swap_positions else 0

        battle_start_time = time.time()
        battle_details = {
            'episode': episode,
            'opponent_id': opponent_id or 'random',
            'player_position': player,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(battle_start_time)),
            'round_details': [],
            'actions': [],
            'action_counts': {},
            'final_hp': {},
            'result': '',
            'total_reward': 0.0,
            'rounds': 0,
            'first_move_wins': 0,
            'first_move_losses': 0,
            'second_move_wins': 0,
            'second_move_losses': 0,
        }

        self.selfplay_logger.log_battle_start(episode, opponent_id or 'random', player)

        obs, _ = env.reset()
        done = False
        max_steps = self.trainer.training_config.get('max_steps_per_episode', 512)
        steps = 0
        current_round = 0
        total_reward = 0.0
        our_cumulative = 50
        enemy_cumulative = 50
        action_counts = {}
        step_snapshots = []
        enemy_step_snapshots = []

        while not done and steps < max_steps:
            round_record = self._snapshot_round_state(
                env, player, steps, current_round, our_cumulative, enemy_cumulative
            )

            our_tower_hp = 0
            enemy_tower_hp = 0
            try:
                game_state = env._runtime.state
                for tower in game_state.towers:
                    if tower.player == player:
                        our_tower_hp += tower.hp
                    else:
                        enemy_tower_hp += tower.hp
            except Exception:
                pass
            step_snapshots.append((our_tower_hp, our_cumulative))
            enemy_step_snapshots.append((enemy_tower_hp, enemy_cumulative))

            if steps % BATTLE_PRINT_INTERVAL == 0:
                self.selfplay_logger.log_round(
                    episode, current_round,
                    round_record['our_hp'], round_record['enemy_hp'],
                    round_record['our_coins'], round_record['enemy_coins']
                )

            battle_details['round_details'].append(round_record)

            if not swap_positions:
                player_obs = obs["player_0"]
            else:
                player_obs = obs["player_1"]

            action, log_prob, value, type_probs = self.trainer._select_action(player_obs)

            opponent_action_id, opponent_ops = self._get_opponent_move(
                opponent_agent, env, obs, swap_positions
            )

            batch.observations_board.append(player_obs['board'])
            batch.observations_global.append(player_obs['global'])
            batch.observations_mask.append(player_obs['action_mask'])
            batch.actions.append(action)
            batch.values.append(value)
            batch.log_probs.append(log_prob)
            batch.dones.append(0.0)

            action_counts[action] = action_counts.get(action, 0) + 1

            mask_valid = int(player_obs['action_mask'].sum())

            obs, reward, reward_detail, terminated, truncated, info = self._resolve_turn_and_extract_reward(
                env, obs, action, opponent_action_id, opponent_ops, swap_positions
            )

            batch.rewards.append(reward)
            total_reward += reward

            our_cumulative += reward_detail.get('own_income', 0)
            enemy_cumulative += reward_detail.get('enemy_income', 0)

            action_record = self._build_action_record(
                steps, action, opponent_action_id,
                float(log_prob), float(value), float(reward),
                reward_detail, mask_valid, type_probs,
            )
            battle_details['actions'].append(action_record)

            done = terminated or truncated
            if done:
                batch.dones[-1] = 1.0

            if info and 'round_index' in info:
                current_round = info['round_index']
            elif hasattr(env, '_runtime') and env._runtime and hasattr(env._runtime, 'state'):
                current_round = env._runtime.state.round_index

            steps += 1

        battle_details['final_hp'] = self._read_final_hp(env, player)
        battle_details = self._finalize_battle_details(
            battle_details, total_reward, our_cumulative, enemy_cumulative,
            steps, action_counts, swap_positions, battle_start_time,
        )

        self.selfplay_logger.log_battle_end(
            episode,
            battle_details['result'],
            steps,
            battle_details['duration'],
            battle_details['final_hp']
        )
        self.selfplay_logger.record_battle_result(battle_details['result'], battle_details['duration'], steps)

        is_valid, issues = self.trainer._validate_batch(batch)
        if not is_valid:
            logger.warning(f"[警告] 批次数据无效: {', '.join(issues)}")

        if getattr(self.trainer, 'enable_auxiliary', False):
            from ppo_antwar.utils.aux_labels import compute_aux_labels_from_trajectory
            our_tower_labels, our_gold_labels, enemy_tower_labels, enemy_gold_labels = \
                compute_aux_labels_from_trajectory(step_snapshots, enemy_step_snapshots)
            if batch.aux_tower_damage is None:
                batch.aux_tower_damage = []
            if batch.aux_gold_income is None:
                batch.aux_gold_income = []
            if batch.aux_enemy_tower_damage is None:
                batch.aux_enemy_tower_damage = []
            if batch.aux_enemy_gold_income is None:
                batch.aux_enemy_gold_income = []
            batch.aux_tower_damage.extend(our_tower_labels)
            batch.aux_gold_income.extend(our_gold_labels)
            batch.aux_enemy_tower_damage.extend(enemy_tower_labels)
            batch.aux_enemy_gold_income.extend(enemy_gold_labels)

        return batch, battle_details

    def _get_training_phase(self, episode: int) -> str:
        return "selfplay"

    @staticmethod
    def _get_towers_from_state(state, player: int):
        our_towers = []
        enemy_towers = []
        try:
            for tower in state.towers:
                tt_name = tower.tower_type.name if hasattr(tower.tower_type, 'name') else str(tower.tower_type)
                tv = int(tower.tower_type)
                if tv == 0:
                    level = 0
                elif tv < 10:
                    level = 1
                else:
                    level = 2
                info = {
                    'id': tower.tower_id,
                    'type': tt_name,
                    'level': level,
                    'position': (tower.x, tower.y)
                }
                if tower.player == player:
                    our_towers.append(info)
                else:
                    enemy_towers.append(info)
        except Exception:
            pass
        return our_towers, enemy_towers

    @staticmethod
    def _build_action_record(
        steps: int,
        action: int,
        opponent_action_id,
        log_prob: float,
        value: float,
        reward: float,
        reward_detail: Dict,
        mask_valid: int,
        type_probs,
    ) -> Dict:
        return {
            'step': steps,
            'action': action,
            'opponent_action': opponent_action_id,
            'log_prob': float(log_prob),
            'value': float(value),
            'reward': float(reward),
            'reward_detail': reward_detail,
            'mask_valid': mask_valid,
            'type_probs': [round(p, 4) for p in type_probs],
        }

    @staticmethod
    def _determine_battle_result(final_hp: Dict[str, int]) -> str:
        our_hp = final_hp.get('our_hp', 0)
        enemy_hp = final_hp.get('enemy_hp', 0)
        if enemy_hp <= 0 and our_hp > 0:
            return 'win'
        elif our_hp <= 0 and enemy_hp > 0:
            return 'loss'
        return 'draw'

    def _read_final_hp(self, env, player: int) -> Dict[str, int]:
        try:
            if hasattr(env, '_runtime') and env._runtime and hasattr(env._runtime, 'state'):
                game_state = env._runtime.state
                if hasattr(game_state, 'bases') and len(game_state.bases) >= 2:
                    return {
                        'our_hp': game_state.bases[player].hp,
                        'enemy_hp': game_state.bases[1 - player].hp,
                    }
        except Exception:
            pass
        return {'our_hp': 0, 'enemy_hp': 0}

    def _resolve_turn_and_extract_reward(
        self,
        env,
        obs: Dict,
        action: int,
        opponent_action_id,
        opponent_ops,
        swap_positions: bool,
    ):
        if not swap_positions:
            obs, rewards, terminated, truncated, info = env._resolve_turn_from_ids_or_ops(
                player_0_action_id=action,
                player_1_ops=opponent_ops,
                player_1_action_id=opponent_action_id,
            )
            reward = rewards.get("player_0", 0.0)
            agent_key = "player_0"
        else:
            obs, rewards, terminated, truncated, info = env._resolve_turn_from_ids_or_ops(
                player_0_ops=opponent_ops,
                player_0_action_id=opponent_action_id,
                player_1_action_id=action,
            )
            reward = rewards.get("player_1", 0.0)
            agent_key = "player_1"

        reward_detail = env._last_reward_detail.get(agent_key, {})
        return obs, reward, reward_detail, terminated, truncated, info

    def _snapshot_round_state(
        self,
        env,
        player: int,
        steps: int,
        current_round: int,
        our_cumulative: int,
        enemy_cumulative: int,
    ) -> Dict:
        record = {
            'step': steps,
            'round': current_round,
            'our_hp': 50,
            'enemy_hp': 50,
            'our_coins': 0,
            'enemy_coins': 0,
            'our_cumulative_coins': our_cumulative,
            'enemy_cumulative_coins': enemy_cumulative,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        try:
            if not (hasattr(env, '_runtime') and env._runtime and hasattr(env._runtime, 'state')):
                return record

            game_state = env._runtime.state

            if hasattr(game_state, 'bases') and len(game_state.bases) >= 2:
                record['our_hp'] = game_state.bases[player].hp
                record['enemy_hp'] = game_state.bases[1 - player].hp

            if hasattr(game_state, 'coins') and len(game_state.coins) >= 2:
                record['our_coins'] = game_state.coins[player]
                record['enemy_coins'] = game_state.coins[1 - player]

            if hasattr(game_state, 'towers'):
                our_towers, enemy_towers = self._get_towers_from_state(game_state, player)
                record['our_towers'] = our_towers
                record['enemy_towers'] = enemy_towers

            if hasattr(game_state, 'ants_of'):
                our_ants = game_state.ants_of(player)
                enemy_ants = game_state.ants_of(1 - player)
                record['our_ants'] = len(our_ants)
                record['enemy_ants'] = len(enemy_ants)
                our_kinds = {}
                enemy_kinds = {}
                for a in our_ants:
                    k = a.kind.name if hasattr(a.kind, 'name') else str(a.kind)
                    our_kinds[k] = our_kinds.get(k, 0) + 1
                for a in enemy_ants:
                    k = a.kind.name if hasattr(a.kind, 'name') else str(a.kind)
                    enemy_kinds[k] = enemy_kinds.get(k, 0) + 1
                record['our_ant_kinds'] = our_kinds
                record['enemy_ant_kinds'] = enemy_kinds

            if hasattr(game_state, 'bases') and len(game_state.bases) >= 2:
                record['our_tech'] = {
                    'gen_speed': int(game_state.bases[player].generation_level),
                    'ant_strength': int(game_state.bases[player].ant_level),
                }
                record['enemy_tech'] = {
                    'gen_speed': int(game_state.bases[1 - player].generation_level),
                    'ant_strength': int(game_state.bases[1 - player].ant_level),
                }

        except Exception:
            pass

        return record

    def _finalize_battle_details(
        self,
        battle_details: Dict,
        total_reward: float,
        our_cumulative: int,
        enemy_cumulative: int,
        steps: int,
        action_counts: Dict,
        swap_positions: bool,
        battle_start_time: float,
    ) -> Dict:
        final_hp = battle_details.get('final_hp', {'our_hp': 0, 'enemy_hp': 0})
        result = self._determine_battle_result(final_hp)
        battle_details['result'] = result

        if result == 'win':
            if not swap_positions:
                battle_details['first_move_wins'] += 1
            else:
                battle_details['second_move_wins'] += 1
        elif result == 'loss':
            if not swap_positions:
                battle_details['first_move_losses'] += 1
            else:
                battle_details['second_move_losses'] += 1

        battle_end_time = time.time()
        battle_details['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(battle_end_time))
        battle_details['duration'] = battle_end_time - battle_start_time
        battle_details['total_reward'] = total_reward
        battle_details['our_cumulative_coins'] = our_cumulative
        battle_details['enemy_cumulative_coins'] = enemy_cumulative
        battle_details['rounds'] = steps
        battle_details['action_counts'] = action_counts

        return battle_details

    @staticmethod
    def _compute_coin_buckets(coin_list) -> Dict[str, float]:
        buckets = {'lt_30': 0, 'ge_30_lt_60': 0, 'ge_60_lt_100': 0, 'ge_100': 0}
        for c in coin_list:
            if c < 30:
                buckets['lt_30'] += 1
            elif c < 60:
                buckets['ge_30_lt_60'] += 1
            elif c < 100:
                buckets['ge_60_lt_100'] += 1
            else:
                buckets['ge_100'] += 1
        total = len(coin_list)
        if total == 0:
            return {f"coins_{k}": 0.0 for k in buckets}
        return {f"coins_{k}": v / total * 100.0 for k, v in buckets.items()}

    def _aggregate_coin_stats(self, details_pool) -> Dict[str, float]:
        all_our_coins = []
        all_enemy_coins = []
        all_our_cumulative = []
        all_enemy_cumulative = []

        for d in details_pool:
            for r in d.get('round_details', []):
                all_our_coins.append(r.get('our_coins', 0))
                all_enemy_coins.append(r.get('enemy_coins', 0))
            all_our_cumulative.append(d.get('our_cumulative_coins', 50))
            all_enemy_cumulative.append(d.get('enemy_cumulative_coins', 50))

        result = {
            'avg_our_coins': sum(all_our_coins) / len(all_our_coins) if all_our_coins else 0,
            'avg_enemy_coins': sum(all_enemy_coins) / len(all_enemy_coins) if all_enemy_coins else 0,
            'avg_our_cumulative_coins': sum(all_our_cumulative) / len(all_our_cumulative) if all_our_cumulative else 50,
            'avg_enemy_cumulative_coins': sum(all_enemy_cumulative) / len(all_enemy_cumulative) if all_enemy_cumulative else 50,
        }

        our_buckets = self._compute_coin_buckets(all_our_coins)
        for k, v in our_buckets.items():
            result[f'our_{k}'] = v
        enemy_buckets = self._compute_coin_buckets(all_enemy_coins)
        for k, v in enemy_buckets.items():
            result[f'enemy_{k}'] = v

        return result

    def _aggregate_reward_sources(self, details_pool) -> Dict[str, float]:
        result = {k: 0.0 for k in self.SOURCE_KEYS}

        for d in details_pool:
            for rd_key in ('round_details', 'actions'):
                step_list = d.get(rd_key, [])
                if not isinstance(step_list, list):
                    continue
                for step in step_list:
                    rd = step.get('reward_detail', {})
                    if not isinstance(rd, dict):
                        continue
                    for src_key in self.SOURCE_KEYS:
                        bare_key = src_key[3:]
                        result[src_key] += rd.get(bare_key, 0.0)

        return result

    @staticmethod
    def _resolve_action_type(action_id: int) -> str:
        for tname, tcfg in ACTION_SPACE_CONFIG.items():
            if tcfg['start'] <= action_id < tcfg['end']:
                return tname
        return 'noop'

    def _aggregate_action_reward_stats(self, details_pool) -> Dict[str, float]:
        action_rewards = {}
        action_counts = {}

        for d in details_pool:
            for action_rec in d.get('actions', []):
                action_id = action_rec.get('action', 0)
                rd = action_rec.get('reward_detail', {})
                reward_val = rd.get('action_reward', 0.0) if isinstance(rd, dict) else 0.0
                type_name = self._resolve_action_type(action_id)
                action_rewards[type_name] = action_rewards.get(type_name, 0.0) + reward_val
                action_counts[type_name] = action_counts.get(type_name, 0) + 1

        result = {}
        for type_name in set(action_rewards) | set(action_counts):
            result[f'actrw_{type_name}'] = action_rewards.get(type_name, 0.0)
            result[f'actcnt_{type_name}'] = action_counts.get(type_name, 0)
        return result

    def _aggregate_opponent_stats(self, per_episode_buffer) -> Dict[str, float]:
        opponent_stats = {}
        for s in per_episode_buffer:
            oid = s['opponent_id']
            if oid not in opponent_stats:
                opponent_stats[oid] = {'rewards': [], 'wins': 0, 'losses': 0, 'draws': 0}
            opponent_stats[oid]['rewards'].append(s['reward'])
            if s['result'] == 'win':
                opponent_stats[oid]['wins'] += 1
            elif s['result'] == 'loss':
                opponent_stats[oid]['losses'] += 1
            elif s['result'] == 'draw':
                opponent_stats[oid]['draws'] += 1

        result = {}
        for oid, stats in opponent_stats.items():
            total = max(stats['wins'] + stats['losses'] + stats['draws'], 1)
            result[f'opponent_win_rate_{oid}'] = stats['wins'] / total
            result[f'opponent_reward_{oid}'] = sum(stats['rewards']) / max(len(stats['rewards']), 1)
        return result

    def _extract_ppo_metric_fields(self, ppo_metrics) -> Dict[str, float]:
        if not ppo_metrics:
            logger.warning("[警告] ppo_metrics 为空，PPO 更新可能失败")
            return {k: float('nan') for k in self.PPO_METRIC_KEYS}

        result = {k: ppo_metrics.get(k, 0) for k in self.PPO_METRIC_KEYS}

        for k, v in ppo_metrics.items():
            if k.startswith(self.PPO_METRIC_PREFIX_KEYS) and isinstance(v, (int, float)):
                result[k] = v

        return result

    def _update_lr_schedule(self, episode: int) -> None:
        self.trainer.current_ent_coef = self.trainer.get_entropy_coef(episode)
        self.trainer.current_lr = self.trainer.get_learning_rate(episode)
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = self.trainer.current_lr
        for param_group in self.trainer.value_optimizer.param_groups:
            param_group['lr'] = self.trainer.ppo_config.get(
                'lr_vf', self.trainer.ppo_config.lr * 0.1
            )

    def _build_training_context(
        self,
        ppo_metrics,
        details_pool,
        per_episode_buffer,
    ) -> Dict[str, float]:
        context = {}

        context.update(self._extract_ppo_metric_fields(ppo_metrics))

        context['entropy_coef'] = self.trainer.current_ent_coef
        context['learning_rate'] = self.trainer.current_lr

        context['num_collected'] = len(details_pool)

        context.update(self._aggregate_coin_stats(details_pool))

        context.update(self._aggregate_reward_sources(details_pool))

        context.update(self._aggregate_action_reward_stats(details_pool))

        context.update(self._aggregate_opponent_stats(per_episode_buffer))

        return context

    def _get_epsilon(self, episode: int, phase: str) -> float:
        return 0.0

    def _get_opponent_move(self, opponent_agent, env, obs, swap_positions: bool):
        opponent_action_id = None
        opponent_ops = None

        if opponent_agent is None:
            opponent_action_id = self._random_action_from_mask(obs, swap_positions)
        elif hasattr(opponent_agent, 'act'):
            opponent_obs = obs["player_1"] if not swap_positions else obs["player_0"]
            opponent_action_id, _, _ = opponent_agent.act(opponent_obs)
        elif hasattr(opponent_agent, 'choose_operations'):
            state = env._runtime.state
            opponent_player = 0 if swap_positions else 1
            try:
                ops = opponent_agent.choose_operations(state, opponent_player)
                opponent_ops = ops if ops else []
            except Exception as e:
                logger.warning(f"[WARMUP] opponent choose_operations failed: {type(e).__name__}: {e}")
                opponent_ops = []
        elif hasattr(opponent_agent, 'choose_bundle'):
            state = env._runtime.state
            opponent_player = 0 if swap_positions else 1
            try:
                result = opponent_agent.choose_bundle(state, opponent_player)
                if hasattr(result, 'operations'):
                    opponent_ops = list(result.operations)
                else:
                    opponent_ops = result if result else []
            except Exception as e:
                logger.warning(f"[WARMUP] opponent choose_bundle failed: {type(e).__name__}: {e}")
                opponent_ops = []
        else:
            opponent_action_id = self._random_action_from_mask(obs, swap_positions)

        return opponent_action_id, opponent_ops

    def _random_action_from_mask(self, obs, swap_positions: bool) -> int:
        opponent_obs = obs["player_1"] if not swap_positions else obs["player_0"]
        opponent_mask = torch.FloatTensor(opponent_obs["action_mask"]).to(self.trainer.device)
        valid_actions = torch.where(opponent_mask > 0)[0]
        if len(valid_actions) > 0:
            return int(np.random.choice(valid_actions.cpu().numpy()))
        return 0

    def train(
        self,
        num_episodes: int = 100000,
        opponent_update_interval: int = 500,
        checkpoint_callback=None,
    ) -> None:
        logger.info(f"Starting self-play training for {num_episodes} episodes")

        if checkpoint_callback is not None:
            checkpoint_callback.init_callback(self.trainer)

        env = self.trainer.env_factory()

        N = self.trainer.training_config.n_envs
        batch_pool = []
        details_pool = []
        per_episode_buffer = []

        for episode in range(num_episodes):
            self.opponent_pool.set_current_episode(episode)
            logger.debug(f"[Phase:SELFPLAY] Episode {episode}: ε=0")

            # 标记训练开始
            if self.trainer.time_tracker:
                self.trainer.time_tracker.start_training()

            # 更新训练进度，用于动态探索/利用调整
            self.selfplay_manager.update_progress(episode)

            swap_positions = (self.battle_count % 2 == 1)

            opponent_id = self.selfplay_manager.select_opponent()
            opponent_agent = self._load_opponent_agent(opponent_id) if opponent_id else None
            batch, battle_details = self._collect_episode_with_swap(
                env, opponent_agent, swap_positions, opponent_id, episode
            )

            self.battle_count += 1
            
            try:
                self._save_selfplay_battle_details(battle_details)
            except Exception as e:
                logger.warning(f"保存SelfPlay对战详情失败: {e}")

            if opponent_id is not None:
                result_map = {'win': 1, 'loss': -1, 'draw': 0}
                rounds = battle_details.get('rounds', 0)
                self.selfplay_manager.update_payoff(opponent_id, result_map.get(battle_details['result'], 0), rounds)

            self._flush_selfplay_battle_cache()

            batch_pool.append(batch)
            details_pool.append(battle_details)

            episode_reward_this = sum(batch.rewards)
            episode_length_this = len(batch)
            per_episode_buffer.append({
                'episode': episode,
                'reward': float(episode_reward_this),
                'length': episode_length_this,
                'opponent_id': opponent_id or 'none',
                'result': battle_details.get('result', 'unknown'),
                'rounds': battle_details.get('rounds', 0),
                'swap': swap_positions,
            })

            # 标记训练结束
            if self.trainer.time_tracker:
                self.trainer.time_tracker.end_training()

            # 每 N 个 episode 做一次 PPO update
            if (episode + 1) % N == 0 or episode == num_episodes - 1:
                if len(batch_pool) > 0 and any(len(b) > 0 for b in batch_pool):
                    combined_batch = batch_pool[0].merge(batch_pool[1:]) if len(batch_pool) > 1 else batch_pool[0]

                    ppo_metrics = self.trainer._ppo_update(combined_batch)
                    self.trainer.episode_count += 1

                    self._update_lr_schedule(episode + 1)

                    episode_reward = sum(combined_batch.rewards)
                    episode_length_total = len(combined_batch)
                    avg_episode_length = episode_length_total // N
                    self.trainer.episode_rewards.append(episode_reward)
                    self.trainer.episode_lengths.append(avg_episode_length)
                    self.trainer.total_steps += episode_length_total

                    if self.trainer.logger:
                        avg_loss = ppo_metrics.get('loss', 0) if ppo_metrics else 0
                        context = self._build_training_context(ppo_metrics, details_pool, per_episode_buffer)
                        self._flush_per_episode_stats(per_episode_buffer)
                        per_episode_buffer = []
                        self.trainer._log_training_metrics(
                            episode + 1,
                            episode_reward,
                            avg_loss,
                            avg_episode_length,
                            context
                        )

                    if checkpoint_callback is not None:
                        checkpoint_callback.on_step()

                batch_pool = []
                details_pool = []

            if episode % opponent_update_interval == 0 and episode > 0:
                self._save_weight_stats(episode)
                checkpoint_path = self.trainer.save_checkpoint(
                    str(self.checkpoint_dir / f"model_{episode}.pt")
                )
                new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path, episode=episode)
                logger.info(f"Added new opponent checkpoint: {checkpoint_path}")
                
                # 同步保存league状态
                self._save_league_state()

            # Baseline Battle评估
            if self.battle_coordinator is not None and episode > 0 and episode % self.battle_interval == 0:
                try:
                    logger.info(f"Episode {episode}: 开始Baseline对战评估")
                    
                    # 使用新的 battle coordinator 进行对战评估
                    results = self.battle_coordinator.run_ppo_evaluation_with_agent(
                        ppo_agent=self.trainer.get_ppo_agent_for_battle(),
                        baseline_agents=self.baseline_agents,
                        num_episodes=self.n_battles // 2
                    )
                    
                    if results:
                        logger.info(f"Episode {episode}: Baseline对战评估完成")
                        # 从 results 中提取统计信息用于保存训练状态
                        total_wins = 0
                        total_losses = 0
                        total_draws = 0
                        for agent_name, result in results.items():
                            total_wins += result.get('agent1_wins', 0)
                            total_losses += result.get('agent2_wins', 0)
                            total_draws += result.get('draws', 0)
                        self._last_battle_result = (total_wins, total_losses, total_draws)
                        logger.info(f"Episode {episode}: Baseline - Win:{total_wins}, Loss:{total_losses}, Draw:{total_draws}")
                    else:
                        logger.warning(f"Episode {episode}: 对战评估返回None")
                except Exception as e:
                    logger.error(f"Episode {episode}: 对战评估失败 - {e}")
                    logger.error(f"Episode {episode}: {traceback.format_exc()}")

            self._maybe_decay(episode)

            # 定期保存训练状态
            if episode > 0 and episode % self.training_status_save_interval == 0:
                self._save_training_status(episode, self._last_battle_result)

        env.close()

    def _save_weight_stats(self, episode: int) -> None:
        stats = {}
        for name, param in self.trainer.policy.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': round(float(param.data.mean().item()), 6),
                    'std': round(float(param.data.std().item()), 6),
                    'min': round(float(param.data.min().item()), 6),
                    'max': round(float(param.data.max().item()), 6),
                    'grad_mean': round(float(param.grad.mean().item()), 6) if param.grad is not None else 0,
                    'grad_std': round(float(param.grad.std().item()), 6) if param.grad is not None else 0,
                    'grad_norm': round(float(param.grad.norm().item()), 6) if param.grad is not None else 0,
                }
        wf = os.path.join(self.trainer.path_config.training_dir, "weight_stats.jsonl")
        record = {'episode': episode, 'weights': stats}
        with open(wf, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _flush_per_episode_stats(self, buffer: List[Dict]) -> None:
        ep_file = os.path.join(self.trainer.path_config.training_dir, "per_episode_stats.jsonl")
        with open(ep_file, "a") as f:
            for record in buffer:
                f.write(json.dumps(record) + "\n")

    def _maybe_decay(self, episode: int) -> None:
        """检查是否需要执行历史记录衰减"""
        if episode > 0 and episode % self.decay_interval == 0:
            self.payoff.decay_all()
            logger.debug(f"Applied history decay at episode {episode}")

    def _create_initial_opponents(self) -> None:
        """创建初始对手池"""
        if self.initial_opponents <= 0:
            return

        logger.info(f"Creating {self.initial_opponents} initial opponents...")
        for i in range(self.initial_opponents):
            checkpoint_path = self._create_random_agent_checkpoint()
            if checkpoint_path:
                new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
                logger.info(f"Created initial opponent: {new_opponent_id}")

    def _create_random_agent_checkpoint(self) -> Optional[str]:
        """创建随机策略的agent checkpoint"""
        try:
            policy = AntWarPolicyValueNetwork(
                board_shape=self.trainer.network_config.get('board_shape', (27, 15, 15)),
                global_dim=self.trainer.network_config.get('global_dim', 64),
                action_dim=self.trainer.network_config.get('action_dim', 96),
                hidden_dim=self.trainer.network_config.get('hidden_dim', 256),
            )

            checkpoint = {
                'policy_state_dict': policy.state_dict(),
                'config': {'network': self.trainer.network_config}
            }

            checkpoint_path = str(self.checkpoint_dir / f"initial_opponent_{uuid.uuid4().hex[:8]}.pt")
            torch.save(checkpoint, checkpoint_path)
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to create random agent checkpoint: {e}")
            return None

    def _save_league_state(self) -> None:
        """
        保存整个 league 状态。
        由于 checkpoint_path 现在由 OpponentPool 内部管理，不需要单独保存 _opponent_checkpoints。
        """
        pool_path = self.persist_dir / "opponent_pool.json"
        self.opponent_pool.save_state(str(pool_path))

        payoff_path = self.persist_dir / "payoff.json"
        self.payoff.save_state(str(payoff_path))

        # 不需要单独保存 checkpoint_map，因为已包含在 opponent_pool 的状态中

    def _load_league_state(self) -> None:
        """
        加载之前保存的 league 状态。

        加载顺序必须为：先 payoff（评分数据），再 opponent_pool（对手池）。
        确保 opponent_pool 加载后能正确引用到已恢复的 payoff 对象。
        """
        pool_path = self.persist_dir / "opponent_pool.json"
        payoff_path = self.persist_dir / "payoff.json"

        if payoff_path.exists():
            try:
                self.payoff = BattleSharedPayoff.load_state(str(payoff_path))
                logger.info(f"Loaded payoff state from {payoff_path}")
            except Exception as e:
                logger.warning(f"Failed to load payoff state: {e}")

        if pool_path.exists():
            try:
                self.opponent_pool = OpponentPool.load_state(str(pool_path))
                self.opponent_pool.set_payoff_reference(self.payoff)
                logger.info(f"Loaded opponent pool state from {pool_path}")
            except Exception as e:
                logger.warning(f"Failed to load opponent pool state: {e}")

    def evaluate(self, num_games: int = 100) -> dict:
        if len(self.opponent_pool) == 0:
            return {'win_rate': 0.0, 'num_games': 0}

        total_wins = 0
        total_draws = 0
        total_losses = 0

        env = self.trainer.env_factory()
        eval_battle_count = 0

        for _ in range(num_games):
            opponent_id = self.selfplay_manager.select_opponent()
            if opponent_id is None:
                break

            swap_positions = (eval_battle_count % 2 == 1)
            result, rounds = self._play_evaluation_game(env, opponent_id, swap_positions)
            eval_battle_count += 1

            if result == 1:
                total_wins += 1
            elif result == 0:
                total_draws += 1
            else:
                total_losses += 1

            self.selfplay_manager.update_payoff(opponent_id, result, rounds)

        env.close()

        total_games = total_wins + total_draws + total_losses
        win_rate = (total_wins + total_draws * 0.5) / total_games if total_games > 0 else 0.0

        return {
            'win_rate': win_rate,
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_losses,
            'num_games': total_games,
        }

    def _play_evaluation_game(self, env, opponent_id: str, swap_positions: bool = False) -> Tuple[int, int]:
        self.trainer.reset_rnn_state()
        opponent_agent = self._load_opponent_agent(opponent_id)

        obs, _ = env.reset()
        done = False
        max_steps = self.trainer.training_config.get('max_steps_per_episode', 512)
        steps = 0

        game_winner = None

        while not done and steps < max_steps:
            if not swap_positions:
                player_obs = obs["player_0"]
                action, _, _, _ = self.trainer.get_agent().act(player_obs)

                opponent_action = 0
                if opponent_agent is not None:
                    opponent_obs = obs["player_1"]
                    opponent_action, _, _, _ = opponent_agent.act(opponent_obs)

                obs, _, _, _, _ = env.step(action)
                obs, rewards, terminated, truncated, info = env.step(opponent_action)

                result_reward = rewards.get("player_0", 0)
                opponent_reward = rewards.get("player_1", 0)
            else:
                player_obs = obs["player_1"]
                action, _, _, _ = self.trainer.get_agent().act(player_obs)

                opponent_action = 0
                if opponent_agent is not None:
                    opponent_obs = obs["player_0"]
                    opponent_action, _, _, _ = opponent_agent.act(opponent_obs)

                obs, _, _, _, _ = env.step(opponent_action)
                obs, rewards, terminated, truncated, info = env.step(action)

                result_reward = rewards.get("player_1", 0)
                opponent_reward = rewards.get("player_0", 0)

            done = terminated or truncated
            steps += 1

            if done and info is not None:
                game_winner = info.get("winner")

        env.reset()

        if game_winner is not None:
            if swap_positions:
                result = -1 if game_winner == 1 else (1 if game_winner == 0 else 0)
            else:
                result = 1 if game_winner == 0 else (-1 if game_winner == 1 else 0)
            return result, steps

        if result_reward > opponent_reward:
            return 1, steps
        elif result_reward < opponent_reward:
            return -1, steps
        else:
            return 0, steps
