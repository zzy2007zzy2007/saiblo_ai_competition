import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from loguru import logger

from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..utils.action_constants import TYPE_CONFIG
from ..utils.gae import compute_gae
from .batch import EpisodeBatch
from .neural_agent import NeuralAgent
from .safety import check_tensor_nan, clean_tensor, NanRecoveryHandler
from .checkpoint_manager import CheckpointManager
from .lr_scheduler import LRScheduler


# ─── PPOTrainer ────────────────────────────────────────────────────────────


class PPOTrainer:
    """PPO 训练器 - 核心 PPO 算法实现。

    提供以下功能：
    - PPO-Clip 策略优化
    - GAE 优势估计
    - 辅助任务损失（塔伤害、金币收入）
    - NaN 鲁棒性保护
    - 结构化动作头的特殊处理
    """

    # 价值网络参数前缀（用于优化器分组和梯度裁剪分组）
    VALUE_PREFIXES = (
        "value_cnn.",
        "value_mlp.",
        "value_proj.",
        "value_layer_norm.",
        "value_head.",
        "value_tower_damage_head.",
        "value_gold_income_head.",
        "value_base_damage_head.",
    )

    # 梯度警告阈值常量
    LARGE_GRAD_VALUE_WARN = 2.0
    LARGE_GRAD_POLICY_WARN = 10.0
    GRAD_NORM_ABNORMAL_FACTOR = 10
    LR_LOG_INTERVAL = 500

    def __init__(
        self,
        env_factory: Callable,
        config: Dict[str, Any],
        device: torch.device,
        base_dir: str,
        callbacks: Optional[Any] = None,
    ):
        self._env_factory = env_factory
        self.config = config
        self.device = device
        self.base_dir = base_dir
        self.callbacks = callbacks

        self._build_network()
        self._build_optimizers()
        self._init_loggers()

        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        nan_threshold = (
            self.config.get("ppo", {}).get("clip_values", {}).get("nan_threshold")
        )
        self._nan_handler = NanRecoveryHandler(nan_threshold=nan_threshold)
        self._checkpoint_manager = CheckpointManager(device=self.device)
        self._lr_scheduler = LRScheduler.from_ppo_config(self.config.get("ppo", {}))

        if not self._validate_config():
            raise ValueError("Configuration validation failed. See error logs above.")

    # ── 网络和优化器初始化 ────────────────────────────────────────────────

    def _build_network(self):
        """构建策略-价值网络"""
        self.policy = AntWarPolicyValueNetwork(
            hidden_dim=self.config.get("network", {}).get("hidden_dim"),
            enable_auxiliary=self.config.get("ppo", {}).get("enable_auxiliary"),
        ).to(self.device)

        ppo_cfg = self.config.get("ppo", {})
        self.policy.logit_noise_std = ppo_cfg.get("logit_noise_std", 0.0)

    def _build_optimizers(self):
        """构建双优化器：策略网络和价值网络使用独立优化器"""
        ppo_cfg = self.config.get("ppo", {})
        lr = ppo_cfg.get("lr")
        lr_vf = ppo_cfg.get("lr_vf")

        policy_params = []
        value_params = []
        for name, param in self.policy.named_parameters():
            if name.startswith(self.VALUE_PREFIXES):
                value_params.append(param)
            else:
                policy_params.append(param)

        self.optimizer = optim.Adam(policy_params, lr=lr)
        self.value_optimizer = optim.Adam(value_params, lr=lr_vf)

    # ── PPO 更新 ───────────────────────────────────────────────────────────

    def _ppo_update(self, batch: EpisodeBatch) -> Dict[str, Any]:
        """执行完整的 PPO 更新，返回训练指标字典"""
        ppo_cfg = self.config.get("ppo", {})

        tensors = batch.to_tensors(self.device)
        advantages, returns = self._compute_gae(
            tensors["rewards"].tolist(),
            tensors["values"].tolist(),
            tensors["dones"].tolist(),
            final_value=batch.final_value,
            final_values=batch.final_values if batch.final_values else None,
        )

        td_errors = returns - tensors["values"]
        td_error_mean = td_errors.mean().item()
        td_error_std = td_errors.std().item()

        # Per-batch advantage normalization（一次性，使用全批量统计量）
        # 所有 PPO epoch 和 minibatch 共享同一组 (μ, σ)，避免 per-minibatch
        # 归一化引入的尺度不一致和训练方差
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std

        reward_stats = self._compute_reward_stats(tensors)
        type_metrics = self._compute_type_metrics(tensors)
        metrics = self._train_ppo_epochs(
            tensors, advantages_norm, returns, ppo_cfg
        )

        self._nan_handler.save_good_state(
            policy_state=self.policy.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            value_optimizer_state=self.value_optimizer.state_dict(),
        )

        result = self._aggregate_metrics(
            metrics,
            reward_stats,
            type_metrics,
            advantages,
            returns,
            td_error_mean,
            td_error_std,
            len(tensors["rewards"]),
        )
        ppo_cfg = self.config.get("ppo", {})
        result["entropy_coef"] = ppo_cfg.get("ent_coef")
        policy_lr = self.optimizer.param_groups[0]["lr"]
        result["learning_rate"] = policy_lr
        return result

    @staticmethod
    def _compute_reward_stats(tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        rewards_raw = tensors["rewards"]
        return {
            "reward_mean": rewards_raw.mean().item(),
            "reward_std": rewards_raw.std().item(),
            "reward_min": rewards_raw.min().item(),
            "reward_max": rewards_raw.max().item(),
            "batch_size": len(rewards_raw),
        }

    def _compute_type_metrics(
        self, tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        actions_t = tensors["actions"]
        action_mask_t = tensors["action_mask"]
        total_actions = actions_t.size(0)
        metrics = {}

        for tc in TYPE_CONFIG:
            name = tc["name"]
            flat_start = tc["flat_start"]
            flat_end = tc["flat_end"]
            type_mask = (actions_t >= flat_start) & (actions_t < flat_end)
            type_count = type_mask.sum().item()
            metrics[f"type_{name}_ratio"] = type_count / max(total_actions, 1)

            slice_mask = action_mask_t[:, flat_start:flat_end]
            valid_count = slice_mask.sum(dim=1)
            metrics[f"type_{name}_valid"] = valid_count.float().mean().item()

        with torch.no_grad():
            try:
                type_logits_all, target_logits_per_type, _ = self.policy(
                    tensors["board"], tensors["global"], tensors["action_mask"]
                )
                type_logits_all = type_logits_all.clamp(-50, 50)

                # 记录类型选择 logit（未经目标偏移）
                for i, tc in enumerate(TYPE_CONFIG):
                    name = tc["name"]
                    metrics[f"type_{name}_type_logit"] = type_logits_all[:, i].mean().item()

                for i, tc in enumerate(TYPE_CONFIG):
                    name = tc["name"]
                    flat_start = tc["flat_start"]
                    flat_end = tc["flat_end"]
                    n_valid = flat_end - flat_start

                    t_logits = target_logits_per_type[i].clamp(-50, 50)
                    type_mask_slice = action_mask_t[:, flat_start : flat_start + n_valid]

                    masked_logits = t_logits.clone()
                    masked_logits[type_mask_slice < 0.5] = float("-inf")

                    target_logit_mean = (
                        masked_logits[masked_logits > -1e30].mean().item()
                        if (masked_logits > -1e30).any()
                        else 0.0
                    )
                    metrics[f"type_{name}_target_logit"] = target_logit_mean

                    probs = torch.softmax(masked_logits, dim=1)
                    probs = torch.nan_to_num(probs, nan=0.0)
                    prob_total = probs.sum(dim=1).mean().item()
                    metrics[f"type_{name}_prob"] = prob_total
            except Exception as e:
                logger.warning(f"Failed to compute action type logit/prob stats: {e}")

        return metrics

    def _train_ppo_epochs(
        self,
        tensors: Dict[str, torch.Tensor],
        advantages_norm: torch.Tensor,
        returns: torch.Tensor,
        ppo_cfg: Dict[str, Any],
    ) -> Dict[str, List]:
        """多 Epoch PPO 训练 + minibatch 迭代

        Returns:
            {metric_name: [values_per_step]} 格式的原始指标列表
        """
        clip_eps = ppo_cfg.get("clip_eps")
        clip_eps_vf = ppo_cfg.get("clip_eps_vf")
        ent_coef = ppo_cfg.get("ent_coef")
        vf_coef = ppo_cfg.get("vf_coef")
        ppo_epochs = ppo_cfg.get("ppo_epochs")
        batch_size = ppo_cfg.get("batch_size")
        clip_values = ppo_cfg.get("clip_values")

        metrics = {
            "grad_norm": [],
            "grad_norm_policy": [],
            "grad_norm_value": [],
            "grad_norm_max_layer": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "skipped_minibatches": 0,
        }
        total_samples = len(tensors["rewards"])

        target_kl = ppo_cfg.get("target_kl")

        for epoch in range(ppo_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            epoch_kl_list = []
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                mb_indices = indices[start:end]

                mb_data = self._prepare_minibatch(
                    tensors, advantages_norm, returns, mb_indices
                )

                step_metrics = self._process_minibatch(
                    mb_data,
                    ppo_cfg,
                    clip_eps,
                    clip_eps_vf,
                    ent_coef,
                    vf_coef,
                    clip_values,
                )
                if step_metrics is None:
                    metrics["skipped_minibatches"] += 1
                    continue

                for k, v in step_metrics.items():
                    metrics.setdefault(k, []).append(v)

                if "approx_kl" in step_metrics:
                    epoch_kl_list.append(step_metrics["approx_kl"])

            # 基于近似 KL 散度的 epoch 早停：如果策略偏移过大，停止后续 epoch
            if epoch_kl_list:
                epoch_approx_kl = sum(epoch_kl_list) / len(epoch_kl_list)
                if epoch_approx_kl > target_kl:
                    logger.info(
                        f"Early stopping at PPO epoch {epoch + 1}/{ppo_epochs}, "
                        f"approx_kl={epoch_approx_kl:.6f} > target_kl={target_kl}"
                    )
                    break

        return metrics

    @staticmethod
    def _prepare_minibatch(tensors, advantages_norm, returns, mb_indices):
        mb_board = tensors["board"][mb_indices]
        mb_global = tensors["global"][mb_indices]
        mb_mask = tensors["action_mask"][mb_indices]
        mb_actions = tensors["actions"][mb_indices]
        mb_old_log_probs = tensors["log_probs"][mb_indices]
        mb_advantages_norm = advantages_norm[mb_indices]
        mb_returns = returns[mb_indices]

        return {
            "board": mb_board,
            "global": mb_global,
            "mask": mb_mask,
            "actions": mb_actions,
            "old_log_probs": mb_old_log_probs,
            "advantages_norm": mb_advantages_norm,
            "returns": mb_returns,
            "returns_raw": mb_returns,
            "advantages_raw": mb_advantages_norm,
            "old_values": tensors["values"][mb_indices],
            "aux_tower_damage": tensors["aux_tower_damage"][mb_indices],
            "aux_gold_income": tensors["aux_gold_income"][mb_indices],
            "aux_base_damage": tensors["aux_base_damage"][mb_indices],
            "aux_valid_mask": tensors["aux_valid_mask"][mb_indices],
        }

    def _process_minibatch(
        self,
        mb_data: Dict[str, torch.Tensor],
        ppo_cfg: Dict[str, Any],
        clip_eps: float,
        clip_eps_vf: float,
        ent_coef: float,
        vf_coef: float,
        clip_values: Dict,
    ) -> Optional[Dict[str, float]]:
        """处理单个 minibatch：前向 → 损失 → 反传 → 梯度裁剪"""
        enable_auxiliary = ppo_cfg.get("enable_auxiliary")
        max_grad_norm = ppo_cfg.get("max_grad_norm")
        max_grad_norm_vf = ppo_cfg.get("max_grad_norm_vf", 50.0)

        (
            new_log_probs,
            new_values,
            entropy,
            type_ent,
            target_ent,
            aux_preds,
            features,
            value_features,
        ) = self.policy.evaluate_actions(
            mb_data["board"], mb_data["global"], mb_data["mask"], mb_data["actions"]
        )

        new_log_probs = clean_tensor(new_log_probs)
        new_values = clean_tensor(new_values)

        # 计算 loss
        loss_result = self._compute_losses(
            mb_data,
            new_log_probs,
            new_values,
            entropy,
            aux_preds,
            features,
            value_features,
            ppo_cfg,
            enable_auxiliary,
            clip_eps,
            clip_eps_vf,
            ent_coef,
            vf_coef,
            clip_values,
        )
        if loss_result is None:
            return None
        total_loss, aux_metrics, clip_fraction, ratio, log_ratio, policy_loss_scalar, value_loss_scalar = loss_result

        # 反传 + 梯度裁剪
        skip_update, grad_metrics = self._backward_and_clip(
            total_loss, max_grad_norm, max_grad_norm_vf
        )
        if skip_update:
            return None

        # 收集指标
        return self._collect_step_metrics(
            mb_data,
            new_values,
            type_ent,
            target_ent,
            ratio,
            log_ratio,
            clip_fraction,
            aux_metrics,
            grad_metrics,
            entropy,
            clip_eps,
            clip_eps_vf,
            policy_loss_scalar,
            value_loss_scalar,
        )

    def _compute_losses(
        self,
        mb_data,
        new_log_probs,
        new_values,
        entropy,
        aux_preds,
        features,
        value_features,
        ppo_cfg,
        enable_auxiliary,
        clip_eps,
        clip_eps_vf,
        ent_coef,
        vf_coef,
        clip_values,
    ) -> Optional[tuple]:
        """计算所有损失项：policy_loss + value_loss + entropy + aux_loss"""
        log_ratio = new_log_probs - mb_data["old_log_probs"]
        log_ratio = torch.clamp(
            log_ratio,
            math.log(float(clip_values.get("ratio_min", 1e-5))),
            math.log(float(clip_values.get("ratio_max", 10.0))),
        )
        ratio = torch.exp(log_ratio)

        surr1 = ratio * mb_data["advantages_norm"]
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            * mb_data["advantages_norm"]
        )
        policy_loss = -torch.min(surr1, surr2).mean()
        clip_fraction = (
            ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean().item()
        )

        new_values = new_values.clamp(
            float(clip_values.get("values_min", -500)),
            float(clip_values.get("values_max", 500)),
        )
        # 禁用 value clipping：torch.max(unclipped, clipped) 的梯度方向
        # 在正确更新方向上为 0（阻止学习），错误方向上正常（允许犯错），
        # 是有害的梯度遮蔽。双优化器 + Adam 自适应梯度已足够约束 value 更新。
        # 使用 Huber Loss 替代 MSE，对大 TD-error 天然鲁棒
        # 避免 return 范围大（±500）导致 value_loss 量级过大（4000+）淹没策略梯度
        value_loss = nn.HuberLoss(delta=10.0)(new_values, mb_data["returns"])

        entropy_loss = -ent_coef * entropy.mean()
        total_loss = policy_loss + vf_coef * value_loss + entropy_loss

        total_loss = self._add_auxiliary_loss(
            total_loss, mb_data, aux_preds, ppo_cfg, features, value_features, enable_auxiliary
        )
        if isinstance(total_loss, tuple):
            total_loss, aux_metrics = total_loss
        else:
            aux_metrics = {}

        if check_tensor_nan(total_loss, "total_loss"):
            return None
        loss_val = total_loss.item()
        if abs(loss_val) > float(clip_values.get("loss_max", 1e6)):
            logger.warning(f"Loss exceeds max: {loss_val}, skipping minibatch")
            return None

        policy_loss_scalar = policy_loss.detach().item()
        value_loss_scalar = value_loss.detach().item()

        return total_loss, aux_metrics, clip_fraction, ratio, log_ratio, policy_loss_scalar, value_loss_scalar

    def _backward_and_clip(
        self,
        total_loss,
        max_grad_norm: float,
        max_grad_norm_vf: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """反向传播 + 梯度裁剪"""
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        skip_update, grad_metrics = self._check_and_clip_gradients(
            max_grad_norm,
            max_grad_norm_vf,
        )
        if skip_update:
            return True, grad_metrics

        self.optimizer.step()
        self.value_optimizer.step()

        for name, param in self.policy.named_parameters():
            if torch.isnan(param.data).any():
                logger.warning(f"[权重异常] NaN 已写入参数={name}")
                break

        return False, grad_metrics

    def _collect_step_metrics(
        self,
        mb_data,
        new_values,
        type_ent,
        target_ent,
        ratio,
        log_ratio,
        clip_fraction,
        aux_metrics,
        grad_metrics,
        entropy,
        clip_eps,
        clip_eps_vf,
        policy_loss_scalar,
        value_loss_scalar,
    ) -> Dict[str, float]:
        """收集 minibatch 训练指标"""
        valid_counts = mb_data["mask"].sum(dim=-1)

        # 近似 KL 散度：KL(pi_old || pi_new) ≈ (exp(log_ratio) - 1 - log_ratio).mean()
        approx_kl = (ratio - 1 - log_ratio).mean().item()

        return {
            "grad_norm": grad_metrics["total_grad_norm"],
            "grad_norm_policy": grad_metrics["grad_norm_policy"],
            "grad_norm_value": grad_metrics["grad_norm_value"],
            "grad_norm_max_layer": grad_metrics["max_layer_grad_norm"],
            "policy_loss": policy_loss_scalar,
            "value_loss": value_loss_scalar,
            "entropy": entropy.mean().item(),
            "clip_fraction": clip_fraction,
            "value_input_mean": mb_data["old_values"].mean().item(),
            "value_input_std": mb_data["old_values"].std().item(),
            "value_pred_mean": new_values.mean().item(),
            "value_pred_std": new_values.std().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "entropy_type": type_ent.mean().item(),
            "entropy_target": target_ent.mean().item(),
            "valid_actions_mean": valid_counts.float().mean().item(),
            "valid_actions_min": valid_counts.float().min().item(),
            "approx_kl": approx_kl,
            **aux_metrics,
        }

    def _check_and_clip_gradients(
        self,
        max_grad_norm: float,
        max_grad_norm_vf: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """梯度检测 + 裁剪

        Returns:
            (skip_update, grad_metrics)
            skip_update=True 表示应跳过此 minibatch
        """
        skip_update = False
        total_grad_norm_sq = 0.0
        grad_norm_policy_sq = 0.0
        grad_norm_value_sq = 0.0
        max_layer_grad_norm = 0.0

        for name, param in self.policy.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                grad_n = param.grad.norm().item()
                n_nan = torch.isnan(param.grad).sum().item()
                n_inf = torch.isinf(param.grad).sum().item()
                logger.warning(
                    f"[梯度异常] 参数={name} | "
                    f"grad_norm={grad_n:.2f} | "
                    f"NaN={n_nan}, Inf={n_inf} | "
                    f"跳过当前 minibatch 更新"
                )
                skip_update = True
                break
            g_norm = param.grad.norm().item()
            total_grad_norm_sq += g_norm**2
            max_layer_grad_norm = max(max_layer_grad_norm, g_norm)
            if name.startswith(self.VALUE_PREFIXES):
                grad_norm_value_sq += g_norm**2
                if g_norm > self.LARGE_GRAD_VALUE_WARN:
                    logger.warning(f"[大梯度] 价值参数={name} | grad_norm={g_norm:.2f}")
            else:
                grad_norm_policy_sq += g_norm**2
                if g_norm > self.LARGE_GRAD_POLICY_WARN:
                    logger.warning(f"[大梯度] 策略参数={name} | grad_norm={g_norm:.2f}")

        if skip_update:
            return True, {}

        total_grad_norm = total_grad_norm_sq**0.5
        grad_norm_policy = grad_norm_policy_sq**0.5
        grad_norm_value = grad_norm_value_sq**0.5

        if grad_norm_policy > max_grad_norm * self.GRAD_NORM_ABNORMAL_FACTOR:
            logger.warning(
                f"[总梯度异常] 策略梯度范数={grad_norm_policy:.2f} 远超裁剪阈值 "
                f"{max_grad_norm} ({self.GRAD_NORM_ABNORMAL_FACTOR}×)"
            )

        policy_params = [
            p
            for n, p in self.policy.named_parameters()
            if not n.startswith(self.VALUE_PREFIXES) and p.requires_grad
        ]
        value_params = [
            p
            for n, p in self.policy.named_parameters()
            if n.startswith(self.VALUE_PREFIXES) and p.requires_grad
        ]
        nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
        nn.utils.clip_grad_norm_(value_params, max_grad_norm_vf)

        return False, {
            "total_grad_norm": total_grad_norm,
            "grad_norm_policy": grad_norm_policy,
            "grad_norm_value": grad_norm_value,
            "max_layer_grad_norm": max_layer_grad_norm,
        }

    def _add_auxiliary_loss(
        self,
        total_loss: torch.Tensor,
        mb_data: Dict[str, torch.Tensor],
        aux_preds: Optional[Dict],
        ppo_cfg: Dict[str, Any],
        features: torch.Tensor,
        value_features: Optional[torch.Tensor],
        enable_auxiliary: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not enable_auxiliary or aux_preds is None:
            return total_loss, {}
        aux_base_damage = mb_data.get("aux_base_damage")
        aux_valid_mask = mb_data.get("aux_valid_mask")
        aux_losses = self.policy.compute_auxiliary_loss(
            features,
            mb_data["aux_tower_damage"],
            mb_data["aux_gold_income"],
            aux_base_damage=aux_base_damage,
            value_features=value_features,
            aux_valid_mask=aux_valid_mask,
        )
        aux_tower_coef = ppo_cfg.get("aux_tower_coef")
        aux_gold_coef = ppo_cfg.get("aux_gold_coef")
        aux_enemy_tower_coef = ppo_cfg.get("aux_enemy_tower_coef")
        aux_enemy_gold_coef = ppo_cfg.get("aux_enemy_gold_coef")
        aux_base_coef = ppo_cfg.get("aux_base_dmg_coef")
        aux_enemy_base_coef = ppo_cfg.get("aux_enemy_base_dmg_coef")

        total_loss = total_loss + aux_tower_coef * aux_losses["aux_tower_loss"]
        total_loss = total_loss + aux_gold_coef * aux_losses["aux_gold_loss"]
        total_loss = (
            total_loss + aux_enemy_tower_coef * aux_losses["aux_enemy_tower_loss"]
        )
        total_loss = (
            total_loss + aux_enemy_gold_coef * aux_losses["aux_enemy_gold_loss"]
        )
        if "aux_base_loss" in aux_losses:
            total_loss = total_loss + aux_base_coef * aux_losses["aux_base_loss"]
        if "aux_enemy_base_loss" in aux_losses:
            total_loss = (
                total_loss + aux_enemy_base_coef * aux_losses["aux_enemy_base_loss"]
            )

        # ── value encoder 辅助损失（使用独立系数，保持与 policy encoder 相对影响一致）──
        if value_features is not None:
            vf_aux_tower_coef = ppo_cfg.get("vf_aux_tower_coef", aux_tower_coef * 0.05)
            vf_aux_gold_coef = ppo_cfg.get("vf_aux_gold_coef", aux_gold_coef * 0.05)
            vf_aux_enemy_tower_coef = ppo_cfg.get("vf_aux_enemy_tower_coef", aux_enemy_tower_coef * 0.05)
            vf_aux_enemy_gold_coef = ppo_cfg.get("vf_aux_enemy_gold_coef", aux_enemy_gold_coef * 0.05)
            vf_aux_base_coef = ppo_cfg.get("vf_aux_base_dmg_coef", aux_base_coef * 0.05)
            vf_aux_enemy_base_coef = ppo_cfg.get("vf_aux_enemy_base_dmg_coef", aux_enemy_base_coef * 0.05)

            total_loss = total_loss + vf_aux_tower_coef * aux_losses["vf_aux_tower_loss"]
            total_loss = total_loss + vf_aux_gold_coef * aux_losses["vf_aux_gold_loss"]
            total_loss = (
                total_loss
                + vf_aux_enemy_tower_coef * aux_losses["vf_aux_enemy_tower_loss"]
            )
            total_loss = (
                total_loss
                + vf_aux_enemy_gold_coef * aux_losses["vf_aux_enemy_gold_loss"]
            )
            if "vf_aux_base_loss" in aux_losses:
                total_loss = (
                    total_loss + vf_aux_base_coef * aux_losses["vf_aux_base_loss"]
                )
            if "vf_aux_enemy_base_loss" in aux_losses:
                total_loss = (
                    total_loss
                    + vf_aux_enemy_base_coef * aux_losses["vf_aux_enemy_base_loss"]
                )

        raw_aux = {k: v.item() for k, v in aux_losses.items()}
        return total_loss, raw_aux

    def _aggregate_metrics(
        self,
        metrics: Dict[str, List],
        reward_stats: Dict[str, float],
        type_metrics: Dict[str, float],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        td_error_mean: Optional[float] = None,
        td_error_std: Optional[float] = None,
        num_collected: Optional[int] = None,
    ) -> Dict[str, Any]:
        import numpy as np

        ppo_cfg = self.config.get("ppo", {})
        vf_coef = ppo_cfg.get("vf_coef")
        ent_coef = ppo_cfg.get("ent_coef")

        # 新增：精确反映优化目标的 total loss
        training_loss = (
            np.mean(metrics.get("policy_loss", [0]))
            + vf_coef * np.mean(metrics.get("value_loss", [0]))
            - ent_coef * np.mean(metrics.get("entropy", [0]))
        )

        result = {
            "training_loss": training_loss,
            "policy_loss": np.mean(metrics.get("policy_loss", [0])),
            "value_loss": np.mean(metrics.get("value_loss", [0])),
            "entropy": np.mean(metrics.get("entropy", [0])),
            "loss": np.mean(metrics.get("policy_loss", [0]))
            + np.mean(metrics.get("value_loss", [0])),
            "clip_fraction": np.mean(metrics.get("clip_fraction", [0])),
            "approx_kl": np.mean(metrics.get("approx_kl", [0])),
            "reward_mean": reward_stats["reward_mean"],
            "reward_std": reward_stats["reward_std"],
            "reward_min": reward_stats["reward_min"],
            "reward_max": reward_stats["reward_max"],
            "advantages_std": advantages.std().item(),
            "return_mean": returns.mean().item(),
            "return_std": returns.std().item(),
            "batch_size": reward_stats["batch_size"],
            "grad_norm": np.mean(metrics.get("grad_norm", [0])),
            "grad_norm_policy": np.mean(metrics.get("grad_norm_policy", [0])),
            "grad_norm_value": np.mean(metrics.get("grad_norm_value", [0])),
            "grad_norm_max_layer": np.max(metrics.get("grad_norm_max_layer", [0.0]))
            if metrics.get("grad_norm_max_layer")
            else 0.0,
            "skipped_minibatches": metrics.get("skipped_minibatches", 0),
            "nan_skip_count": metrics.get("skipped_minibatches", 0),
            "aux_tower_loss": np.mean(metrics.get("aux_tower_loss", [0])),
            "aux_gold_loss": np.mean(metrics.get("aux_gold_loss", [0])),
            "aux_enemy_tower_loss": np.mean(metrics.get("aux_enemy_tower_loss", [0])),
            "aux_enemy_gold_loss": np.mean(metrics.get("aux_enemy_gold_loss", [0])),
            "aux_base_loss": np.mean(metrics.get("aux_base_loss", [0])),
            "aux_enemy_base_loss": np.mean(metrics.get("aux_enemy_base_loss", [0])),
            **type_metrics,
        }

        if td_error_mean is not None:
            result["td_error_mean"] = td_error_mean
        if td_error_std is not None:
            result["td_error_std"] = td_error_std
        if num_collected is not None:
            result["num_collected"] = num_collected

        return result

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        final_value: float = 0.0,
        final_values: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 优势估计和折扣回报"""
        ppo_cfg = self.config.get("ppo", {})
        clip_values = ppo_cfg.get("clip_values")

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=ppo_cfg.get("gamma"),
            gae_lambda=ppo_cfg.get("gae_lambda"),
            device=self.device,
            final_value=final_value,
            final_values=final_values,
        )

        advantages = torch.clamp(
            advantages,
            float(clip_values.get("advantages_min", -1e4)),
            float(clip_values.get("advantages_max", 1e4)),
        )
        returns = torch.clamp(
            returns,
            float(clip_values.get("returns_min", -1e6)),
            float(clip_values.get("returns_max", 1e6)),
        )

        return advantages, returns

    def _compute_aux_loss(
        self,
        features: torch.Tensor,
        aux_tower_damage: torch.Tensor,
        aux_gold_income: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算辅助任务损失"""
        return self.policy.compute_auxiliary_loss(
            features, aux_tower_damage, aux_gold_income
        )

    # ── 数值安全（委托 NanRecoveryHandler）─────────────────────────────────

    # ── 检查点管理 ─────────────────────────────────────────────────────────

    def save_checkpoint(self, filepath: str, episode: int, metrics: Dict) -> None:
        """保存模型检查点"""
        self._checkpoint_manager.save(
            filepath,
            episode,
            policy_state=self.policy.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            value_optimizer_state=self.value_optimizer.state_dict(),
            metrics=metrics,
            config=self.config,
        )

    def load_checkpoint(self, filepath: str) -> int:
        """加载模型检查点"""
        return self._checkpoint_manager.load(
            filepath, self.policy, self.optimizer, self.value_optimizer
        )

    # ── 智能体获取 ─────────────────────────────────────────────────────────

    def create_agent(self, deterministic: bool = False) -> NeuralAgent:
        """从当前策略创建 NeuralAgent。"""
        return NeuralAgent(
            policy=self.policy,
            device=self.device,
            exploration_epsilon=self.config.get("ppo", {}).get("exploration_epsilon")
            if not deterministic
            else 0.0,
        )

    def ppo_update(self, batch: EpisodeBatch) -> Dict[str, Any]:
        """公共接口：执行 PPO 更新。委托给 _ppo_update。"""
        return self._ppo_update(batch)

    def store_last_metrics(self, metrics: Dict[str, Any]) -> None:
        """公共接口：保存最近一次 PPO update 的指标。替代直接写入 _last_metrics。"""
        self._last_metrics = metrics

    def handle_nan_recovery(self, episode: int) -> bool:
        """公共接口：手动触发 NaN 恢复流程。封装 _nan_handler 内部对象。"""
        return self._nan_handler.on_nan_detected(
            episode,
            self.policy,
            self.optimizer,
            self.value_optimizer,
        )

    def log_config_summary(self):
        ppo_cfg = self.config.get("ppo", {})
        training_cfg = self.config.get("training", {})
        selfplay_cfg = self.config.get("selfplay", {})
        network_cfg = self.config.get("network", {})

        policy_lr = ppo_cfg.get("lr")
        value_lr = ppo_cfg.get("lr_vf")

        lines = []
        lines.append("=" * 70)
        lines.append("  PPO Training Configuration Summary")
        lines.append("=" * 70)

        lines.append("  [Environment]")
        device_str = str(self.device)
        lines.append(f"    Device:             {device_str}")
        lines.append(f"    Base Dir:           {self.base_dir}")

        lines.append("")
        lines.append("  [Network]")
        lines.append(f"    Hidden Dim:         {network_cfg.get('hidden_dim')}")
        aux_enabled = ppo_cfg.get("enable_auxiliary")
        lines.append(
            f"    Auxiliary Tasks:    {'enabled' if aux_enabled else 'disabled'}"
        )

        lines.append("")
        lines.append("  [PPO Algorithm]")
        lines.append(
            f"    Learning Rate:      policy={policy_lr:.2e}  value={value_lr:.2e}"
        )
        lines.append(f"    Batch Size:         {ppo_cfg.get('batch_size')}")
        lines.append(f"    PPO Epochs:         {ppo_cfg.get('ppo_epochs')}")
        lines.append(f"    Clip Epsilon:       {ppo_cfg.get('clip_eps')}")
        lines.append(f"    Gamma:              {ppo_cfg.get('gamma')}")
        lines.append(f"    GAE Lambda:         {ppo_cfg.get('gae_lambda')}")
        lines.append(f"    Entropy Coef:       {ppo_cfg.get('ent_coef', 0.1)}")

        lines.append("")
        lines.append("  [Training Schedule]")
        lines.append(f"    Total Episodes:     {training_cfg.get('total_episodes')}")
        lines.append(f"    Save Interval:      {training_cfg.get('save_interval', 20)}")
        lines.append(
            f"    Opponent Update:    {training_cfg.get('opponent_update_interval')}"
        )
        lines.append(
            f"    Max Steps/Episode:  {training_cfg.get('max_steps_per_episode')}"
        )

        lines.append("")
        lines.append("  [Battle Evaluation]")
        lines.append(f"    Battle Interval:    {training_cfg.get('battle_interval')}")
        lines.append(f"    Battles per Eval:   {training_cfg.get('n_battles')}")
        baseline_list = selfplay_cfg.get("baseline_agents", [])
        baseline_str = ", ".join(baseline_list) if baseline_list else "none"
        lines.append(f"    Baseline Agents:    {baseline_str}")

        lines.append("")
        lines.append("  [Self-Play]")
        lines.append(
            f"    Opponent Pool Size: {selfplay_cfg.get('opponent_pool_size')}"
        )
        lines.append(f"    Exploit Prob:       {selfplay_cfg.get('exploit_prob')}")
        lines.append("=" * 70)

        logger.info("\n" + "\n".join(lines))

    def check_batch_quality(self, batch: EpisodeBatch) -> Tuple[bool, List[str]]:
        issues = []
        batch_len = len(batch)

        if batch_len == 0:
            issues.append("Batch is empty: length=0")
            return False, issues

        rewards = np.array(batch.rewards, dtype=np.float64)
        values = np.array(batch.values, dtype=np.float64)

        reward_nan_mask = np.isnan(rewards)
        if reward_nan_mask.any():
            nan_indices = np.where(reward_nan_mask)[0]
            issues.append(f"rewards contain NaN at indices: {nan_indices.tolist()}")

        reward_inf_mask = np.isinf(rewards)
        if reward_inf_mask.any():
            inf_indices = np.where(reward_inf_mask)[0]
            issues.append(f"rewards contain Inf at indices: {inf_indices.tolist()}")

        value_nan_mask = np.isnan(values)
        if value_nan_mask.any():
            nan_indices = np.where(value_nan_mask)[0]
            issues.append(f"values contain NaN at indices: {nan_indices.tolist()}")

        if np.all(rewards == 0):
            issues.append(
                f"All rewards are zero (len={batch_len}), possible reward function issue"
            )

        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if abs(reward_mean) < 1e-6 and abs(reward_std) < 1e-6:
            issues.append(
                f"Reward distribution nearly flat: mean={reward_mean:.1e}, std={reward_std:.1e}"
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_learning_rate(self) -> Tuple[float, float]:
        """获取当前学习率（策略网络，价值网络）"""
        policy_lr = self.optimizer.param_groups[0]["lr"]
        value_lr = self.value_optimizer.param_groups[0]["lr"]
        return policy_lr, value_lr

    def get_last_metrics(self) -> Optional[Dict]:
        """公开最近一次 PPO update 的指标"""
        return getattr(self, "_last_metrics", None)

    # ── 配置验证 ───────────────────────────────────────────────────────────

    def _validate_config(self) -> bool:
        """验证配置参数有效性"""
        required_sections = ["ppo", "training"]
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required config section: {section}")
                return False

        ppo_cfg = self.config["ppo"]
        required_params = [
            "lr",
            "lr_vf",
            "gamma",
            "gae_lambda",
            "clip_eps",
            "batch_size",
        ]
        for param in required_params:
            if param not in ppo_cfg:
                logger.error(f"Missing required PPO parameter: {param}")
                return False

        return True

    # ── 日志初始化 ─────────────────────────────────────────────────────────

    def _init_loggers(self):
        """初始化日志记录器"""
        self._episode_rewards = []
        self._episode_lengths = []

    # ── 学习率调度 ─────────────────────────────────────────────────────────

    def update_lr_schedule(self, episode: int, total_episodes: int):
        """更新学习率"""
        lr, lr_vf, schedule_type = self._lr_scheduler.compute(episode, total_episodes)
        LRScheduler.apply(self.optimizer, self.value_optimizer, lr, lr_vf)

        if (
            episode == 1
            or episode % self.LR_LOG_INTERVAL == 0
            or episode >= total_episodes
        ):
            logger.info(
                f"[LR] episode={episode}/{total_episodes} "
                f"schedule={schedule_type} "
                f"lr_policy={lr:.2e}, lr_value={lr_vf:.2e}"
            )
