from __future__ import annotations

from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass
import os
import time
import math
import random
import traceback
from pathlib import Path

from loguru import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict

from ..network.antwar_net import AntWarPolicyValueNetwork, count_parameters
from ..env.antwar_env import AntWarEnv
from ..monitor.logger import Logger
from ..monitor.time_tracker import TimeTracker
from ..monitor.system_metrics_sampler import SystemMetricsSampler
from ..monitor.constants import RECENT_METRICS_WINDOW, HISTORICAL_METRICS_WINDOW
from ..config.path_config import PathConfig
from ppo_antwar.utils.action_constants import ACTION_DIM, TYPE_CONFIG

# PPO 数值安全边界从配置文件读取



@dataclass
class EpisodeBatch:
    observations_board: List[np.ndarray]
    observations_global: List[np.ndarray]
    observations_mask: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[np.ndarray]
    values: List[np.ndarray]
    log_probs: List[np.ndarray]
    dones: List[np.ndarray]
    aux_tower_damage: Optional[List[np.ndarray]] = None
    aux_gold_income: Optional[List[np.ndarray]] = None
    aux_enemy_tower_damage: Optional[List[np.ndarray]] = None
    aux_enemy_gold_income: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.actions)

    def merge(self, others: List[EpisodeBatch]) -> EpisodeBatch:
        merged = EpisodeBatch(
            observations_board=[],
            observations_global=[],
            observations_mask=[],
            actions=[],
            rewards=[],
            values=[],
            log_probs=[],
            dones=[],
        )
        for attr in ['observations_board', 'observations_global',
                     'observations_mask', 'actions', 'rewards',
                     'values', 'log_probs', 'dones']:
            lst = getattr(merged, attr)
            lst.extend(getattr(self, attr))
            for b in others:
                lst.extend(getattr(b, attr))
        for attr in ['aux_tower_damage', 'aux_gold_income',
                     'aux_enemy_tower_damage', 'aux_enemy_gold_income']:
            lst = getattr(merged, attr)
            if lst is None:
                lst = []
            my_attr = getattr(self, attr)
            if my_attr is not None:
                lst.extend(my_attr)
            for b in others:
                other_attr = getattr(b, attr)
                if other_attr is not None:
                    lst.extend(other_attr)
            setattr(merged, attr, lst if lst else None)
        return merged

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        result = {
            'board': torch.FloatTensor(np.array(self.observations_board)).to(device),
            'global': torch.FloatTensor(np.array(self.observations_global)).to(device),
            'action_mask': torch.FloatTensor(np.array(self.observations_mask)).to(device),
            'actions': torch.LongTensor(np.array(self.actions)).to(device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'dones': torch.FloatTensor(np.array(self.dones)).to(device),
        }
        for key in ['aux_tower_damage', 'aux_gold_income',
                     'aux_enemy_tower_damage', 'aux_enemy_gold_income']:
            data = getattr(self, key)
            if data is not None and len(data) > 0:
                result[key] = torch.FloatTensor(np.array(data)).to(device)
        return result


class PPOTrainer:
    def __init__(
        self,
        env_factory: Callable,
        config: EasyDict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        base_dir: str = None,
        callbacks=None,
    ) -> None:
        try:
            self.env_factory = env_factory
            self.config = config
            self.device = torch.device(device)
            self.path_config = PathConfig(base_dir)
            self.log_dir = str(self.path_config.base_dir)
            self.callbacks = callbacks

            logger.info("Initializing PPOTrainer...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Log directory: {self.log_dir}")

            self.ppo_config = config.ppo
            self.training_config = config.training
            self.network_config = config.network

            logger.info("Creating policy network...")
            enable_auxiliary = self.ppo_config.get('enable_auxiliary', False)
            self.policy = AntWarPolicyValueNetwork(
                board_shape=tuple(self.network_config.board_shape),
                global_dim=self.network_config.global_dim,
                action_dim=self.network_config.action_dim,
                hidden_dim=self.network_config.hidden_dim,
                enable_auxiliary=enable_auxiliary,
            ).to(self.device)

            self.policy.exploration_epsilon = self.ppo_config.get('exploration_epsilon', 0.0)
            self.policy.logit_noise_std = self.ppo_config.get('logit_noise_std', 0.0)

            self.enable_auxiliary = enable_auxiliary
            self.aux_tower_coef = self.ppo_config.get('aux_tower_coef', 0.1)
            self.aux_gold_coef = self.ppo_config.get('aux_gold_coef', 0.1)
            self.aux_enemy_tower_coef = self.ppo_config.get('aux_enemy_tower_coef', 0.02)
            self.aux_enemy_gold_coef = self.ppo_config.get('aux_enemy_gold_coef', 0.002)

            from ..env.observation import ObservationEncoder
            from ..env.action_mask import ActionMaskHandler
            self.observation_encoder = ObservationEncoder()
            self.action_mask_handler = ActionMaskHandler()

            logger.info("Creating optimizer...")
            policy_params = [p for n, p in self.policy.named_parameters() if 'value_head' not in n]
            value_params = [p for n, p in self.policy.named_parameters() if 'value_head' in n]
            self.optimizer = optim.Adam(
                policy_params,
                lr=self.ppo_config.lr,
            )
            self.value_optimizer = optim.Adam(
                value_params,
                lr=self.ppo_config.get('lr_vf', self.ppo_config.lr * 0.1),
            )

            self.total_steps = 0
            self.episode_count = 0

            self.current_ent_coef = self.ppo_config.ent_coef
            self.current_lr = self.ppo_config.lr
            self.base_lr = self.ppo_config.lr

            self.total_episodes = self.training_config.total_episodes

            self.episode_rewards = []
            self.episode_lengths = []

            # NaN 鲁棒性相关初始化 - 从配置读取数值边界
            self.nan_count = 0
            clip_config = self.ppo_config.get('clip_values', {})
            
            # 确保从配置读取的值是数字类型（YAML可能将科学计数法解析为字符串）
            def to_float(value, default):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return float(default)
            
            self.LOGITS_MIN = to_float(clip_config.get('logits_min', -20), -20)
            self.LOGITS_MAX = to_float(clip_config.get('logits_max', 20), 20)
            self.VALUES_MIN = to_float(clip_config.get('values_min', -10), -10)
            self.VALUES_MAX = to_float(clip_config.get('values_max', 10), 10)
            self.RETURNS_MIN = to_float(clip_config.get('returns_min', -1e6), -1e6)
            self.RETURNS_MAX = to_float(clip_config.get('returns_max', 1e6), 1e6)
            self.ADVANTAGES_MIN = to_float(clip_config.get('advantages_min', -1e4), -1e4)
            self.ADVANTAGES_MAX = to_float(clip_config.get('advantages_max', 1e4), 1e4)
            self.PROB_MIN = to_float(clip_config.get('prob_min', 1e-10), 1e-10)
            self.RATIO_MIN = to_float(clip_config.get('ratio_min', 1e-5), 1e-5)
            self.RATIO_MAX = to_float(clip_config.get('ratio_max', 10.0), 10.0)
            self.LOSS_MAX = to_float(clip_config.get('loss_max', 1e6), 1e6)
            self.NAN_THRESHOLD = int(clip_config.get('nan_threshold', 5))

            # 初始化日志系统（必须在battle_manager之前，因为battle_manager需要time_tracker）
            logger.info("Initializing logging system...")
            self._init_logging()



            # 初始化回调系统
            if self.callbacks:
                logger.info("Initializing callbacks...")
                self.callbacks.init_callback(self)

            logger.info("PPOTrainer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PPOTrainer: {str(e)}")
            logger.error(traceback.format_exc())
            if self.log_dir:
                try:
                    from ..monitor.logger import Logger
                    temp_logger = Logger(self.log_dir)
                    temp_logger.log_exception(e, {'phase': 'initialization'})
                except:
                    pass
            raise

    def get_entropy_coef(self, episode: int) -> float:
        return self.ppo_config.ent_coef

    def get_learning_rate(self, episode: int) -> float:
        """计算当前学习率（可选余弦退火）"""
        use_cosine_decay = self.ppo_config.get('lr_cosine_decay', False)

        if use_cosine_decay:
            progress = episode / max(1, self.total_episodes)
            progress = min(1.0, max(0.0, progress))
            return self.base_lr * (1 + math.cos(math.pi * progress)) / 2
        else:
            return self.base_lr

    def _init_logging(self) -> None:
        """初始化日志系统"""
        self.logger = Logger(self.path_config)
        self.time_tracker = TimeTracker(str(self.path_config.system_dir))
        self.system_metrics_sampler = SystemMetricsSampler(str(self.path_config.system_dir), sample_interval=10)
        if self.system_metrics_sampler:
            self.system_metrics_sampler.start(phase="training")
        self.training_metrics_cache = {'history': [], 'latest': None}
        self.best_avg_reward = float('-inf')
        self.training_start_time = time.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def _check_tensor_nan(name: str, tensor: torch.Tensor) -> bool:
        """检测张量是否包含 NaN/Inf"""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            logger.warning(f"[警告] {name} 包含异常值: NaN={has_nan}, Inf={has_inf}")
            return True
        return False

    @staticmethod
    def _clean_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """清理张量中的 NaN/Inf 值"""
        if tensor.numel() == 1:
            if torch.isnan(tensor).item() or torch.isinf(tensor).item():
                logger.warning(f"[警告] 清理 {name}: 异常值={tensor.item()}")
                return torch.tensor(0.0, device=tensor.device)
        else:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                logger.warning(f"[警告] 清理 {name}: NaN={nan_count}, Inf={inf_count}")
                return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        return tensor

    def _on_nan_detected(self) -> bool:
        """检测到 NaN 时的处理

        Returns:
            True: 可以继续训练
            False: 应该停止训练
        """
        self.nan_count += 1
        logger.warning(f"[警告] 检测到第 {self.nan_count} 次 NaN")

        if self.nan_count >= 2 and hasattr(self, 'last_good_checkpoint') and self.last_good_checkpoint:
            logger.warning(f"[恢复] 尝试从 {self.last_good_checkpoint} 恢复权重")
            try:
                import torch.serialization
                from easydict import EasyDict
                with torch.serialization.safe_globals([EasyDict]):
                    checkpoint = torch.load(self.last_good_checkpoint, map_location=self.device)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'value_optimizer_state_dict' in checkpoint:
                    self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
                self.nan_count = 0
                logger.warning(f"[恢复] 成功从 episode {checkpoint.get('episode_count', '?')} 恢复")
                return True
            except Exception as e:
                logger.warning(f"[恢复] 加载 checkpoint 失败: {e}")

        if self.nan_count >= self.NAN_THRESHOLD:
            logger.error(f"[错误] 连续 {self.nan_count} 次检测到 NaN，训练异常")
            return False
        return True

    def _validate_batch(self, batch: EpisodeBatch) -> Tuple[bool, List[str]]:
        """验证批次数据的有效性

        Returns:
            (is_valid, issues): 数据是否有效，以及问题列表
        """
        issues = []

        if len(batch) < 1:
            issues.append(f"批次为空，长度={len(batch)}")

        if any(np.isnan(r) or np.isinf(r) for r in batch.rewards):
            issues.append("rewards 包含 NaN/Inf")

        if any(np.isnan(v) or np.isinf(v) for v in batch.values):
            issues.append("values 包含 NaN/Inf")

        non_zero_rewards = [r for r in batch.rewards if r != 0]
        if len(non_zero_rewards) == 0 and len(batch.rewards) > 0:
            issues.append("所有奖励都为0，可能存在奖励函数问题")

        if len(batch.rewards) > 0:
            reward_array = np.array(batch.rewards)
            reward_mean = np.mean(reward_array)
            reward_std = np.std(reward_array)
            if reward_std < 1e-6 and abs(reward_mean) < 1e-6:
                issues.append(f"奖励统计异常: mean={reward_mean}, std={reward_std}")

        return len(issues) == 0, issues

    def _validate_config(self) -> bool:
        """验证配置参数的有效性"""
        issues = []

        if self.training_config.total_episodes is None:
            issues.append("total_episodes 未设置")
        elif self.training_config.total_episodes < 10:
            issues.append(f"total_episodes ({self.training_config.total_episodes}) 过小")

        if self.ppo_config.batch_size is None:
            issues.append("batch_size 未设置")
        elif self.ppo_config.batch_size < 1:
            issues.append(f"batch_size ({self.ppo_config.batch_size}) 无效")

        if self.ppo_config.lr is None:
            issues.append("learning_rate 未设置")
        elif self.ppo_config.lr <= 0 or self.ppo_config.lr > 1:
            issues.append(f"learning_rate ({self.ppo_config.lr}) 超出合理范围")

        if self.training_config.n_envs is None:
            issues.append("n_envs 未设置")
        elif self.training_config.n_envs < 1:
            issues.append(f"n_envs ({self.training_config.n_envs}) 无效")

        if issues:
            logger.error(f"[错误] 配置验证失败:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        return True

    # train() 方法已删除。
    # 理由：PPOTrainer 是纯算子层，不持有训练循环。
    # 实际训练循环由 SelfPlayTrainer.train() 驱动，
    # 它复用 PPOTrainer 的 _ppo_update / collect_episode 等核心方法。

    def _log_training_metrics(self, episode: int, avg_reward: float, avg_loss: float,
                              avg_rounds: float, context: Dict = None) -> None:
        """记录训练指标到日志系统"""
        if not self.logger:
            return

        # 更新指标缓存
        metrics = {
            'episode': episode,
            'avg_reward': float(avg_reward),
            'avg_loss': float(avg_loss),
            'avg_rounds': float(avg_rounds),
            'total_steps': self.total_steps,
            **(context or {})
        }
        self.training_metrics_cache['history'].append(metrics)
        self.training_metrics_cache['latest'] = metrics

        # 保存训练指标
        self.logger.save_training_metrics(
            episode=episode,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            avg_rounds=avg_rounds,
            steps=self.total_steps,
            context=context,
            training_metrics_cache=self.training_metrics_cache
        )

        # 记录训练详情
        self.logger.log_training_details(
            episode=episode,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            avg_rounds=avg_rounds,
            steps=self.total_steps
        )

    def collect_episode(
        self, env: AntWarEnv, opponent: Optional["PPOAgent"] = None
    ) -> EpisodeBatch:
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

        obs, _ = env.reset()

        done = False
        max_steps = self.training_config.get('max_steps_per_episode', 512)

        while not done and len(batch) < max_steps:
            player_obs = obs[f"player_0"]

            action, log_prob, value, _ = self._select_action(player_obs)

            opponent_action = 0
            if opponent is not None:
                opponent_obs = obs[f"player_1"]
                opponent_action, _, _ = opponent.act(opponent_obs)
            else:
                # 改进：当没有对手时，使用随机动作而不是 NO_OP
                # 这保证了训练初期也有对抗性
                opponent_obs = obs["player_1"]
                opponent_mask = torch.FloatTensor(opponent_obs["action_mask"]).to(self.device)
                valid_actions = torch.where(opponent_mask > 0)[0]
                if len(valid_actions) > 0:
                    opponent_action = int(np.random.choice(valid_actions.cpu().numpy()))
                else:
                    opponent_action = 0  # fallback to NO_OP

            batch.observations_board.append(player_obs['board'])
            batch.observations_global.append(player_obs['global'])
            batch.observations_mask.append(player_obs['action_mask'])
            batch.actions.append(action)
            batch.values.append(value)
            batch.log_probs.append(log_prob)
            batch.dones.append(0.0)

            actions_dict = {"player_0": action, "player_1": opponent_action}
            obs, rewards, terminated, truncated, info = env.step(actions_dict)

            reward = rewards.get("player_0", 0.0)
            batch.rewards.append(reward)

            done = terminated or truncated
            if done:
                batch.dones[-1] = 1.0

        is_valid, issues = self._validate_batch(batch)
        if not is_valid:
            logger.warning(f"[警告] 批次数据无效: {', '.join(issues)}")

        return batch

    def _select_action(self, observation: Dict[str, np.ndarray]) -> tuple:
        board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

        if torch.isnan(board).any() or torch.isnan(global_obs).any():
            logger.warning(f"[警告] 观察输入包含 NaN: board_nan={torch.isnan(board).sum().item()}, global_nan={torch.isnan(global_obs).sum().item()}")
            board = torch.nan_to_num(board, nan=0.0)
            global_obs = torch.nan_to_num(global_obs, nan=0.0)

        with torch.no_grad():
            action, log_prob, value, type_probs = self.policy.get_action(
                board, global_obs, action_mask, deterministic=False
            )

        raw_action = action.item()
        raw_log_prob = log_prob.item()
        raw_value = value.item()

        if torch.isnan(action) or torch.isnan(log_prob) or torch.isnan(value):
            logger.warning(f"[警告] 网络输出包含 NaN: action={raw_action}, log_prob={raw_log_prob}, value={raw_value}")
            action = torch.tensor(0, device=self.device)
            log_prob = torch.tensor(0.0, device=self.device)
            value = torch.tensor(0.0, device=self.device)

        action = action.cpu().item()
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()
        type_probs = type_probs.cpu().numpy().tolist()

        return action, log_prob, value, type_probs

    def _select_action_with_exploration(self, observation, epsilon, action_mask_handler) -> tuple:
        action, log_prob, value, _ = self._select_action(observation)

        if random.random() < epsilon:
            if not hasattr(self, '_explore_count'):
                self._explore_count = 0
                self._exploit_count = 0
            self._explore_count += 1

            board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
            global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
            action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value_tensor = self.policy(board, global_obs, action_mask)
                masked_logits = logits.clone()
                masked_logits[action_mask < 0.5] = -1e9
                probs = torch.softmax(masked_logits, dim=-1)

            valid_indices = torch.where(action_mask[0] > 0)[0]
            if len(valid_indices) > 0:
                action = int(np.random.choice(valid_indices.cpu().numpy()))
            else:
                action = 0

            if self._explore_count <= 3 or self._explore_count % 50 == 0:
                nonzero_indices = valid_indices.cpu().numpy().tolist()
                logger.info(
                    f"[EXPLORE-DEBUG] explore#{self._explore_count}: "
                    f"mask_valid={len(valid_indices)} sel_action={action} "
                    f"all_valid={nonzero_indices if len(nonzero_indices) <= 15 else f'total_{len(nonzero_indices)}'}"
                )

            prob = probs[0, action].clamp(min=1e-10)
            log_prob = float(torch.log(prob).item())
            value = float(value_tensor.item())
        else:
            if not hasattr(self, '_explore_count'):
                self._explore_count = 0
                self._exploit_count = 0
            self._exploit_count += 1

        return action, log_prob, value

    def _ppo_update(self, batch: EpisodeBatch) -> Dict[str, float]:
        """执行 PPO 更新并返回训练指标（带 NaN 鲁棒性）"""
        if len(batch) < 1:
            return {}

        tensors = batch.to_tensors(self.device)
        batch_size = len(batch)

        reward_tensor = tensors['rewards']
        reward_mean = reward_tensor.mean().item()
        reward_std = reward_tensor.std().item()
        reward_min = reward_tensor.min().item()
        reward_max = reward_tensor.max().item()

        if self.episode_count % 10 == 0:
            logger.info(f"[奖励统计] mean={reward_mean:.4f}, std={reward_std:.4f}, "
                  f"min={reward_min:.4f}, max={reward_max:.4f}")

        if reward_std < 1e-6 and abs(reward_mean) < 1e-6:
            logger.warning(f"[警告] 奖励统计异常: mean={reward_mean}, std={reward_std}")

        # Pre-compute GAE on full (temporally ordered) trajectory
        cleaned_values = self._clean_tensor(tensors['values'], "old_values")
        cleaned_values = torch.clamp(cleaned_values, min=self.VALUES_MIN, max=self.VALUES_MAX)
        full_returns = self._compute_returns(tensors['rewards'], cleaned_values, tensors['dones'])
        full_advantages = full_returns - cleaned_values.detach()

        if self._check_tensor_nan("returns", full_returns):
            full_returns = self._clean_tensor(full_returns, "returns")
        if self._check_tensor_nan("advantages", full_advantages):
            full_advantages = self._clean_tensor(full_advantages, "advantages")

        full_returns = torch.clamp(full_returns, min=self.RETURNS_MIN, max=self.RETURNS_MAX)
        full_advantages = torch.clamp(full_advantages, min=self.ADVANTAGES_MIN, max=self.ADVANTAGES_MAX)

        td_error = full_returns - cleaned_values.detach()
        td_error_mean = td_error.mean().item()
        td_error_std = td_error.std().item()

        # 归一化 returns 到零均值单位方差，防止 value_loss 尺度爆炸
        return_mean = full_returns.mean()
        return_std = full_returns.std() + 1e-8
        full_returns = (full_returns - return_mean) / return_std

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_type_entropy = 0.0
        total_target_entropy = 0.0
        total_loss = 0.0
        total_aux_tower_loss = 0.0
        total_aux_gold_loss = 0.0
        total_aux_enemy_tower_loss = 0.0
        total_aux_enemy_gold_loss = 0.0
        update_count = 0
        total_clip_fraction = 0.0
        total_value_pred_mean = 0.0
        total_value_pred_std = 0.0
        action_type_counts = [0] * len(TYPE_CONFIG)
        total_ratio_mean = 0.0
        total_ratio_std = 0.0
        nan_skip_count = 0
        total_valid_actions = 0.0
        valid_action_min = float('inf')
        type_valid_counts = [0.0] * len(TYPE_CONFIG)
        type_logit_sums = [0.0] * len(TYPE_CONFIG)
        type_prob_sums = [0.0] * len(TYPE_CONFIG)
        type_logit_counts = [0] * len(TYPE_CONFIG)

        for _ in range(self.ppo_config.ppo_epochs):
            for start in range(0, batch_size, self.ppo_config.batch_size):
                end = min(start + self.ppo_config.batch_size, batch_size)
                batch_indices = indices[start:end]

                board = tensors['board'][batch_indices]
                global_obs = tensors['global'][batch_indices]
                action_mask = tensors['action_mask'][batch_indices]
                actions = tensors['actions'][batch_indices]
                old_log_probs = tensors['log_probs'][batch_indices]

                valid_counts = action_mask.sum(dim=-1)
                total_valid_actions += valid_counts.mean().item()
                valid_action_min = min(valid_action_min, valid_counts.min().item())

                for tid, tc in enumerate(TYPE_CONFIG):
                    type_mask = action_mask[:, tc['flat_start']:tc['flat_end']]
                    type_valid = type_mask.sum(dim=-1).mean().item()
                    type_valid_counts[tid] += type_valid

                action_ids = actions.cpu().numpy()
                for aid in action_ids:
                    for tid, type_config in enumerate(TYPE_CONFIG):
                        if type_config['flat_start'] <= aid < type_config['flat_end']:
                            action_type_counts[tid] += 1
                            break
                old_values = tensors['values'][batch_indices]
                advantages = full_advantages[batch_indices]
                returns = full_returns[batch_indices]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                action_log_probs, values, entropy, action_logits, type_entropy, target_entropy = self.policy.evaluate_actions(
                    board, global_obs, actions, action_mask
                )

                if self._check_tensor_nan("action_log_probs", action_log_probs):
                    action_log_probs = self._clean_tensor(action_log_probs, "action_log_probs")
                if self._check_tensor_nan("values", values):
                    values = self._clean_tensor(values, "values")
                if self._check_tensor_nan("entropy", entropy):
                    entropy = self._clean_tensor(entropy, "entropy")

                values = torch.clamp(values, min=self.VALUES_MIN, max=self.VALUES_MAX)

                total_value_pred_mean += values.mean().item()
                total_value_pred_std += values.std().item()

                with torch.no_grad():
                    type_logits = action_logits.detach()
                    type_probs = torch.softmax(type_logits, dim=-1)
                    for tid, tc in enumerate(TYPE_CONFIG):
                        s, e = tc['flat_start'], tc['flat_end']
                        type_logit_sums[tid] += type_logits[:, s:e].mean().item()
                        type_prob_sums[tid] += type_probs[:, s:e].sum().item()
                        type_logit_counts[tid] += 1

                log_ratio = action_log_probs - old_log_probs
                log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=80.0, neginf=-80.0)
                log_ratio = torch.clamp(log_ratio, min=-80.0, max=80.0)
                ratio = torch.exp(log_ratio)
                ratio = torch.clamp(ratio, min=self.RATIO_MIN, max=self.RATIO_MAX)
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(
                    ratio, 1 - self.ppo_config.clip_eps, 1 + self.ppo_config.clip_eps
                ) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()

                total_ratio_mean += ratio.mean().item()
                total_ratio_std += ratio.std().item()
                clip_mask = (ratio < 1 - self.ppo_config.clip_eps) | (ratio > 1 + self.ppo_config.clip_eps)
                total_clip_fraction += clip_mask.float().mean().item()

                values_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.ppo_config.clip_eps_vf,
                    self.ppo_config.clip_eps_vf
                )
                value_loss_unclipped = F.mse_loss(values, returns.detach(), reduction='none')
                value_loss_clipped = F.mse_loss(values_clipped, returns.detach(), reduction='none')
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                value_loss = torch.clamp(value_loss, max=5000.0)
                
                entropy_loss = -entropy.mean()
                total_type_entropy += type_entropy.mean().item()
                total_target_entropy += target_entropy.mean().item()

                policy_loss = self._clean_tensor(policy_loss, "policy_loss")
                value_loss = self._clean_tensor(value_loss, "value_loss")
                entropy_loss = self._clean_tensor(entropy_loss, "entropy_loss")

                loss = (
                    policy_loss
                    + self.ppo_config.vf_coef * value_loss
                    + self.current_ent_coef * entropy_loss
                )

                if self.enable_auxiliary and \
                   'aux_tower_damage' in tensors and 'aux_gold_income' in tensors and \
                   'aux_enemy_tower_damage' in tensors and 'aux_enemy_gold_income' in tensors:
                    merged = self.policy._encode(board, global_obs)
                    our_tower_loss, enemy_tower_loss, our_gold_loss, enemy_gold_loss = \
                        self.policy._compute_aux_loss(
                            merged,
                            tensors['aux_tower_damage'][batch_indices],
                            tensors['aux_gold_income'][batch_indices],
                            tensors['aux_enemy_tower_damage'][batch_indices],
                            tensors['aux_enemy_gold_income'][batch_indices])
                    loss = loss \
                        + self.aux_tower_coef * our_tower_loss \
                        + self.aux_gold_coef * our_gold_loss \
                        + self.aux_enemy_tower_coef * enemy_tower_loss \
                        + self.aux_enemy_gold_coef * enemy_gold_loss
                    total_aux_tower_loss += our_tower_loss.item()
                    total_aux_gold_loss += our_gold_loss.item()
                    total_aux_enemy_tower_loss += enemy_tower_loss.item()
                    total_aux_enemy_gold_loss += enemy_gold_loss.item()

                loss = self._clean_tensor(loss, "total_loss")

                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > self.LOSS_MAX:
                    logger.warning(f"[警告] loss 异常: {loss.item()}，跳过此批次")
                    nan_skip_count += 1
                    if self._on_nan_detected():
                        continue
                    else:
                        return {'nan_skip_count': nan_skip_count}

                self.nan_count = 0

                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                total_grad_norm = 0.0
                total_grad_norm_policy = 0.0
                total_grad_norm_value = 0.0
                max_layer_grad_norm = 0.0
                policy_params = []
                value_params = []
                
                for name, param in self.policy.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"[权重监控] 梯度 NaN/INF 首次出现: {name}, "
                                          f"grad_norm={param.grad.norm().item()}")
                            param.grad = torch.zeros_like(param.grad)
                            continue

                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        max_layer_grad_norm = max(max_layer_grad_norm, grad_norm)
                        
                        if 'value_head' in name:
                            total_grad_norm_value += grad_norm ** 2
                            value_params.append(param)
                            if grad_norm > 2.0:
                                logger.warning(f"[警告] 价值网络参数 {name} 梯度范数过大: {grad_norm:.2f}")
                        else:
                            total_grad_norm_policy += grad_norm ** 2
                            policy_params.append(param)
                            if grad_norm > 10.0:
                                logger.warning(f"[警告] 策略网络参数 {name} 梯度范数过大: {grad_norm:.2f}")

                total_grad_norm = total_grad_norm ** 0.5
                total_grad_norm_policy = total_grad_norm_policy ** 0.5
                total_grad_norm_value = total_grad_norm_value ** 0.5
                if total_grad_norm > self.ppo_config.max_grad_norm * 10:
                    logger.warning(f"[警告] 梯度范数 {total_grad_norm:.2f} 远超过阈值 {self.ppo_config.max_grad_norm}")

                # 分别对策略网络和价值网络进行梯度裁剪
                nn.utils.clip_grad_norm_(policy_params, self.ppo_config.max_grad_norm)
                nn.utils.clip_grad_norm_(value_params, self.ppo_config.max_grad_norm_vf)
                self.optimizer.step()
                self.value_optimizer.step()

                for name, param in self.policy.named_parameters():
                    if torch.isnan(param.data).any():
                        logger.warning(f"[权重监控] 权重 NaN 已写入: {name}, episode={self.episode_count}")
                        break

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                update_count += 1

                if self.episode_count % 20 == 0 and update_count == 1:
                    checkpoint_path = self.save_checkpoint(f"recovery_checkpoint_ep{self.episode_count}.pt")
                    self.last_good_checkpoint = checkpoint_path

        if update_count > 0:
            type_dist = {f"type_{t['name']}": c / sum(action_type_counts)
                         for t, c in zip(TYPE_CONFIG, action_type_counts)} if sum(action_type_counts) > 0 else {}
            type_valid_dist = {f"type_valid_{t['name']}": c / update_count
                               for t, c in zip(TYPE_CONFIG, type_valid_counts)} if update_count > 0 else {}

            return {
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy': total_entropy / update_count,
                'entropy_type': total_type_entropy / update_count,
                'entropy_target': total_target_entropy / update_count,
                'loss': total_loss / update_count,
                'aux_tower_loss': total_aux_tower_loss / update_count,
                'aux_gold_loss': total_aux_gold_loss / update_count,
                'aux_enemy_tower_loss': total_aux_enemy_tower_loss / update_count,
                'aux_enemy_gold_loss': total_aux_enemy_gold_loss / update_count,
                'reward_mean': reward_mean,
                'reward_std': reward_std,
                'reward_min': reward_min,
                'reward_max': reward_max,
                'advantages_std': full_advantages.std().item(),
                'return_mean': return_mean.item(),
                'return_std': return_std.item(),
                'clip_fraction': total_clip_fraction / update_count,
                'gradient_norm': total_grad_norm,
                'value_input_mean': cleaned_values.mean().item(),
                'value_input_std': cleaned_values.std().item(),
                'value_pred_mean': total_value_pred_mean / update_count,
                'value_pred_std': total_value_pred_std / update_count,
                'ratio_mean': total_ratio_mean / update_count,
                'ratio_std': total_ratio_std / update_count,
                'grad_norm_policy': total_grad_norm_policy,
                'grad_norm_value': total_grad_norm_value,
                'grad_norm_max_layer': max_layer_grad_norm,
                'nan_skip_count': nan_skip_count,
                'valid_actions_mean': total_valid_actions / update_count,
                'valid_actions_min': valid_action_min,
                'td_error_mean': td_error_mean,
                'td_error_std': td_error_std,
                'batch_size': batch_size,
                'update_count': update_count,
                **type_dist,
                **type_valid_dist,
                **{f"logit_{t['name']}": s / c
                   for t, s, c in zip(TYPE_CONFIG, type_logit_sums, type_logit_counts) if c > 0},
                **{f"prob_{t['name']}": s / c
                   for t, s, c in zip(TYPE_CONFIG, type_prob_sums, type_logit_counts) if c > 0},
            }
        return {'nan_skip_count': nan_skip_count}

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        running_advantage = 0.0
        gamma = self.ppo_config.gamma
        gae_lambda = self.ppo_config.gae_lambda

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            running_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * running_advantage
            advantages[t] = running_advantage

        returns = advantages + values.detach()
        return advantages, returns

    def _compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        _, returns = self._compute_gae(rewards, values, dones)
        return returns

    def save_checkpoint(self, filename: str) -> str:
        checkpoint_path = self.path_config.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'config': self.config,
        }
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        # Fix for PyTorch 2.6: weights_only default to True, need to allow easydict
        import torch.serialization
        from easydict import EasyDict
        with torch.serialization.safe_globals([EasyDict]):
            checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'value_optimizer_state_dict' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)

    def get_agent(self) -> "PPOAgent":
        return PPOAgent(self.policy, self.device)
    
    def get_ppo_agent_for_battle(self) -> "PPOAgent":
        return PPOAgent(
            self.policy, self.device,
            observation_encoder=self.observation_encoder,
            action_mask_handler=self.action_mask_handler,
        )
    
    def reset_rnn_state(self) -> None:
        # No RNN in this implementation
        pass


class PPOAgent:
    def __init__(
        self,
        policy: nn.Module,
        device: torch.device,
        observation_encoder=None,
        action_mask_handler=None,
    ) -> None:
        self.policy = policy
        self.device = device
        self.observation_encoder = observation_encoder
        self.action_mask_handler = action_mask_handler

    def act(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> tuple:
        board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.policy.get_action(
                board, global_obs, action_mask, deterministic=deterministic
            )

        return (
            action.cpu().item(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def select_action(self, obs: Dict[str, Any]) -> int:
        player_obs = obs.get('player_0', obs)
        board = torch.FloatTensor(player_obs['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(player_obs['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(player_obs['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.policy.get_action(
                board, global_obs, action_mask, deterministic=True
            )

        return action.cpu().item()

    def evaluate(
        self,
        observation: Dict[str, np.ndarray],
    ) -> tuple:
        board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.policy(
                board, global_obs, action_mask
            )

        probs = torch.softmax(action_logits, dim=-1)
        return probs.cpu().numpy(), value.cpu().item()

    def choose_operations(self, state, player: int):
        if self.observation_encoder is None or self.action_mask_handler is None:
            return []

        obs = self.observation_encoder.encode(state, player)
        obs['action_mask'] = self.action_mask_handler.get_action_mask(state, player)

        action_id, _, _ = self.act(obs, deterministic=True)

        operation = self.action_mask_handler.action_id_to_operation(
            int(action_id), state, player
        )
        if operation is not None:
            return [operation]
        return []


def train_ppo(
    env_factory: Callable,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    callbacks=None,
    battle_interval: Optional[int] = None,
    n_battles: Optional[int] = None,
) -> str:
    from ..env.antwar_env import AntWarEnv
    from .selfplay import SelfPlayTrainer
    from ..callbacks.callback_factory import create_ppo_callbacks
    from ..config.config_parser import create_ppo_config
    from ..config.path_config import PathConfig
    import time

    # 使用新的配置解析器，支持命令行参数优先
    config = create_ppo_config(
        yaml_path=config_path,
        cli_overrides={
            "training": {
                "battle_interval": battle_interval,
                "n_battles": n_battles,
            },
        },
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if 'system' in config and 'device' in config.system:
        device = config.system.device

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    timestamp_dir = Path(output_dir) / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    path_config = PathConfig(str(timestamp_dir))
    
    log_dir = str(path_config.training_dir)
    checkpoint_dir = str(path_config.checkpoint_dir)
    
    final_battle_interval = config.training.battle_interval
    final_n_battles = config.training.n_battles
    save_interval = config.training.get('save_interval', 5000)

    logger.info("=" * 60)
    logger.info("训练配置摘要")
    logger.info("=" * 60)
    logger.info(f"  battle_interval: {final_battle_interval}")
    logger.info(f"  save_interval: {save_interval}")
    logger.info(f"  对战与保存完全独立: {final_battle_interval != save_interval}")
    logger.info("=" * 60)

    if callbacks is None:
        callbacks = create_ppo_callbacks(
            save_interval=save_interval,
            log_interval=1,
            path_config=path_config,
        )

    trainer = PPOTrainer(
        env_factory=env_factory,
        config=config,
        device=device,
        base_dir=str(path_config.base_dir),
        callbacks=callbacks,
    )

    selfplay_trainer = SelfPlayTrainer(
        trainer=trainer,
        opponent_pool_size=config.selfplay.opponent_pool_size,
        min_opponent_games=config.selfplay.min_opponent_games,
        exploit_prob=config.selfplay.exploit_prob,
        n_battles=final_n_battles,
        battle_interval=final_battle_interval,
        baseline_agents=config.get('baseline_agents'),
    )

    selfplay_trainer.train(
        num_episodes=config.training.total_episodes,
        opponent_update_interval=config.training.opponent_update_interval,
    )

    final_model_path = str(path_config.checkpoint_dir / "final_model.pt")
    return final_model_path


class AntWarAgent:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        # Fix for PyTorch 2.6: weights_only default to True, need to allow easydict
        import torch.serialization
        from easydict import EasyDict
        with torch.serialization.safe_globals([EasyDict]):
            checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint.get('config', None)
        if config is None:
            raise ValueError(f"Model checkpoint does not contain config: {model_path}")

        self.policy = AntWarPolicyValueNetwork(
            board_shape=tuple(config.network.board_shape),
            global_dim=config.network.global_dim,
            action_dim=config.network.action_dim,
            hidden_dim=config.network.hidden_dim,
        ).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        self.policy.eval()

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.policy.get_action(
                board, global_obs, action_mask, deterministic=True
            )

        return action.cpu().item()

    def evaluate(self, observation: Dict[str, np.ndarray]) -> tuple:
        board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
        global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.policy(
                board, global_obs, action_mask
            )

        probs = torch.softmax(action_logits, dim=-1)
        return probs.cpu().numpy(), value.cpu().item()
