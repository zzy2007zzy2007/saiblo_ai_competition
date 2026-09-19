from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.action_constants import (
    TYPE_CONFIG,
    action_id_to_type_and_target,
)
from .hex_conv import orthogonal_init
from .hex_cnn_encoder import HexCNNEncoder
from .mlp_encoder import MLPEncoder, MLP_OUTPUT_DIM
from .heads import StructuredActionHead, ValueHead, TowerDamageHead, GoldIncomeHead, BaseDamageHead


class AntWarPolicyValueNetwork(nn.Module):
    """策略-价值网络 - 独立 Policy/Value 编码器架构。

    策略和价值使用各自的 CNN+MLP 编码器，避免 value 梯度淹没 policy 梯度。

    输入:
        board:        (batch, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        global_vec:   (batch, GLOBAL_FEATURE_DIM)
        action_mask:  (batch, 119)

    输出:
        type_logits:            (batch, 9)  各动作类型的 logits
        target_logits_per_type: List[(batch, n_i)]  每个类型的目标 logits
        value:                  (batch, 1)  状态价值
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.logit_noise_std = 0.0  # 由 trainer 在训练时设置

        # ── 策略编码器（policy 专用） ──
        self.policy_cnn = HexCNNEncoder(hidden_dim)
        self.policy_mlp = MLPEncoder()
        cnn_out_channels = HexCNNEncoder.CHANNEL_COMPRESS  # 128
        cnn_out_dim = (
            cnn_out_channels * HexCNNEncoder.POOL_OUTPUT_SIZE * HexCNNEncoder.POOL_OUTPUT_SIZE
        )
        mlp_out_dim = MLP_OUTPUT_DIM
        fused_dim = cnn_out_dim + mlp_out_dim
        self.policy_proj = nn.Linear(fused_dim, hidden_dim)
        self.policy_layer_norm = nn.LayerNorm(hidden_dim)

        # ── 价值编码器（value 专用） ──
        self.value_cnn = HexCNNEncoder(hidden_dim)
        self.value_mlp = MLPEncoder()
        self.value_proj = nn.Linear(fused_dim, hidden_dim)
        self.value_layer_norm = nn.LayerNorm(hidden_dim)

        # ── 输出头 ──
        self.action_head = StructuredActionHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)

        self.enable_auxiliary = enable_auxiliary
        if enable_auxiliary:
            # ── 策略编码器辅助头 ──
            self.tower_damage_head = TowerDamageHead(hidden_dim)
            self.gold_income_head = GoldIncomeHead(hidden_dim)
            self.base_damage_head = BaseDamageHead(hidden_dim)
            # ── 价值编码器辅助头（独立实例，让辅助任务也训练 value encoder）──
            self.value_tower_damage_head = TowerDamageHead(hidden_dim)
            self.value_gold_income_head = GoldIncomeHead(hidden_dim)
            self.value_base_damage_head = BaseDamageHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        orthogonal_init(self.policy_proj, gain=np.sqrt(2))
        orthogonal_init(self.value_proj, gain=np.sqrt(2))

    def _encode_policy(
        self, board: torch.Tensor, global_vec: torch.Tensor
    ) -> torch.Tensor:
        """策略编码器：棋盘 CNN + 全局 MLP → hidden_dim 向量"""
        cnn_out = self.policy_cnn(board)
        mlp_out = self.policy_mlp(global_vec)
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        projected = self.policy_proj(fused)
        return self.policy_layer_norm(projected)

    def _encode_value(
        self, board: torch.Tensor, global_vec: torch.Tensor
    ) -> torch.Tensor:
        """价值编码器：棋盘 CNN + 全局 MLP → hidden_dim 向量"""
        cnn_out = self.value_cnn(board)
        mlp_out = self.value_mlp(global_vec)
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        projected = self.value_proj(fused)
        return self.value_layer_norm(projected)

    def forward(
        self,
        board: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """标准前向传播 —— 返回结构化动作 logits + 价值。

        Returns:
            type_logits:            (batch, 9)  各动作类型的 logits
            target_logits_per_type: List[(batch, n_i)]  每个类型的目标 logits（9 个）
            value:                  (batch, 1)  状态价值
        """
        policy_features = self._encode_policy(board, global_vec)
        type_logits, target_logits_per_type = self.action_head.calc_all_logits(policy_features)

        value = self.value_head(self._encode_value(board, global_vec))
        return type_logits, target_logits_per_type, value

    @torch.no_grad()
    def get_value_only(
        self, board: torch.Tensor, global_vec: torch.Tensor
    ) -> torch.Tensor:
        """仅计算状态价值，跳过策略编码（用于无需动作的估值场景）。"""
        return self.value_head(self._encode_value(board, global_vec))

    @torch.no_grad()
    def get_action(
        self,
        board: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
        exploration_epsilon: float = 0.05,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:

        policy_features = self._encode_policy(board, global_vec)
        value = self.value_head(self._encode_value(board, global_vec))

        type_logits = self.action_head.calc_type_logits(policy_features)
        type_logits = torch.clamp(type_logits, min=-20.0, max=20.0)

        # logit noise：训练时添加高斯噪声，增强探索，防止过早确定性收敛
        if not deterministic and self.logit_noise_std > 0:
            type_logits = (
                type_logits + torch.randn_like(type_logits) * self.logit_noise_std
            )

        if action_mask is not None:
            type_mask = self.action_head.flat_mask_to_type_mask(action_mask)
            type_logits = type_logits.masked_fill(type_mask == 0, float("-inf"))

        explore = (
            not deterministic
            and exploration_epsilon > 0
            and torch.rand(1).item() < exploration_epsilon
        )

        target_logits = None

        if explore:
            valid = torch.where(action_mask[0] > 0.5)[0]
            if len(valid) > 0:
                action_id = valid[torch.randint(len(valid), (1,)).item()].item()
            else:
                action_id = 0
        else:
            if deterministic:
                type_id = torch.argmax(type_logits, dim=-1, keepdim=True)
            else:
                type_probs = F.softmax(type_logits, dim=-1)
                type_probs = torch.nan_to_num(type_probs, nan=0.0)
                type_probs = type_probs / (type_probs.sum(dim=-1, keepdim=True) + 1e-8)

                if type_probs.sum() < 0.5 or torch.isnan(type_probs).any():
                    type_id = torch.zeros(1, dtype=torch.long, device=type_probs.device)
                else:
                    type_id = torch.multinomial(type_probs, 1)

            type_id_val = type_id.item()

            target_logits = self.action_head.calc_target_logits(policy_features, type_id_val)
            if action_mask is not None:
                self.action_head.mask_target_logits(target_logits, action_mask, type_id_val)

            if deterministic:
                target_k = torch.argmax(target_logits, dim=-1, keepdim=True)
            else:
                target_probs_t = F.softmax(target_logits, dim=-1)
                target_probs_t = torch.nan_to_num(target_probs_t, nan=0.0)
                target_probs_t = target_probs_t / target_probs_t.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-8)
                if target_probs_t.sum() < 1e-8:
                    target_k = torch.zeros(
                        1, dtype=torch.long, device=target_probs_t.device
                    )
                else:
                    target_k = torch.multinomial(target_probs_t, 1)

            action_id = TYPE_CONFIG[type_id_val]["flat_start"] + target_k.item()

            # -----------------------------------------------------------------
            # 防御性校验：当所有目标 logits 均为 -inf 时，argmax 返回 0，
            # 可能产生无效 action_id。此时回退到随机合法动作或 noop。
            # -----------------------------------------------------------------
            if action_mask is not None and action_mask[0, action_id] < 0.5:
                valid = torch.where(action_mask[0] > 0.5)[0]
                if len(valid) > 0:
                    action_id = valid[torch.randint(len(valid), (1,))].item()
                else:
                    action_id = 0  # fallback to noop

        action_t = torch.tensor(
            [action_id], device=policy_features.device, dtype=torch.long
        )
        type_id_act, target_k_act = action_id_to_type_and_target(action_t)
        type_id_0 = type_id_act.item()
        target_k_0 = target_k_act.item()

        type_logp = F.log_softmax(type_logits, dim=-1)
        type_logp_selected = type_logp[:, type_id_0 : type_id_0 + 1]

        if target_logits is None:
            target_logits = self.action_head.calc_target_logits(policy_features, type_id_0)
            if action_mask is not None:
                self.action_head.mask_target_logits(target_logits, action_mask, type_id_0)

        target_logp = F.log_softmax(target_logits, dim=-1)
        target_logp_selected = target_logp[:, target_k_0 : target_k_0 + 1]

        log_prob = (type_logp_selected + target_logp_selected).squeeze(-1)
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=20.0, neginf=-20.0)

        type_probs_out = F.softmax(type_logits, dim=-1)
        type_probs_out = torch.nan_to_num(type_probs_out, nan=0.0)
        type_probs_out = type_probs_out / type_probs_out.sum(dim=-1, keepdim=True)

        return action_id, log_prob, value.squeeze(0), type_probs_out.squeeze(0)

    def evaluate_actions(
        self,
        board: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Dict],
        torch.Tensor,
        torch.Tensor,
    ]:

        policy_features = self._encode_policy(board, global_vec)
        value_features = self._encode_value(board, global_vec)
        values = self.value_head(value_features)

        type_ids, target_ks = action_id_to_type_and_target(action)

        type_logits = self.action_head.calc_type_logits(policy_features)

        if action_mask is not None:
            type_mask = self.action_head.flat_mask_to_type_mask(action_mask)
            type_logits = type_logits.masked_fill(type_mask == 0, float("-inf"))

        type_logp = F.log_softmax(type_logits, dim=-1)
        type_logp_selected = type_logp.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

        target_logp_selected = torch.zeros_like(type_logp_selected)
        type_probs = torch.exp(type_logp)
        type_probs = torch.nan_to_num(type_probs, nan=0.0)
        type_logp_safe = torch.log(type_probs.clamp(min=1e-10))
        type_entropy = -(type_probs * type_logp_safe).sum(dim=-1)
        target_entropy = torch.zeros_like(type_entropy)

        _, all_target_logits = self.action_head.calc_all_logits(policy_features)

        for type_id_val, t_logits in enumerate(all_target_logits):
            sample_mask = type_ids == type_id_val

            if action_mask is not None:
                self.action_head.mask_target_logits(t_logits, action_mask, type_id_val)

            t_logp = F.log_softmax(t_logits, dim=-1)

            # log_prob — 仅当有样本属于此 type 时才计算
            if sample_mask.any():
                target_logp_selected[sample_mask] = t_logp[sample_mask].gather(
                    1, target_ks[sample_mask].unsqueeze(-1)
                ).squeeze(-1)

            # entropy — 始终计算
            t_probs = torch.exp(t_logp)
            t_probs = torch.nan_to_num(t_probs, nan=0.0)
            t_logp_safe = torch.log(t_probs.clamp(min=1e-10))
            h_target_t = -(t_probs * t_logp_safe).sum(dim=-1)
            target_entropy = target_entropy + type_probs[:, type_id_val] * h_target_t

        action_log_probs = type_logp_selected + target_logp_selected
        action_log_probs = torch.nan_to_num(
            action_log_probs, nan=0.0, posinf=20.0, neginf=-20.0
        )

        entropy = type_entropy + target_entropy

        aux_predictions = None
        if self.enable_auxiliary:
            aux_predictions = {
                # policy encoder 辅助预测
                "tower_damage": self.tower_damage_head(policy_features),
                "gold_income": self.gold_income_head(policy_features),
                "base_damage": self.base_damage_head(policy_features),
                # value encoder 辅助预测（独立 head，梯度同时训练 value encoder）
                "vf_tower_damage": self.value_tower_damage_head(value_features),
                "vf_gold_income": self.value_gold_income_head(value_features),
                "vf_base_damage": self.value_base_damage_head(value_features),
            }

        return (
            action_log_probs,
            values.squeeze(-1),
            entropy,
            type_entropy,
            target_entropy,
            aux_predictions,
            policy_features,
            value_features,
        )

    @staticmethod
    def _masked_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """带掩码的 MSE loss，仅对 mask=True 的样本计算"""
        if mask is None:
            return F.mse_loss(pred, target)
        valid_mask = mask.unsqueeze(1).expand_as(pred)
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(valid_pred, valid_target)

    def compute_auxiliary_loss(
        self,
        features: torch.Tensor,
        aux_tower_damage: torch.Tensor,
        aux_gold_income: torch.Tensor,
        aux_base_damage: Optional[torch.Tensor] = None,
        value_features: Optional[torch.Tensor] = None,
        aux_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算辅助任务损失

        Args:
            features: (batch, hidden_dim) policy 编码特征
            aux_tower_damage: (batch, 10) 塔伤害标签 [own_5h, enemy_5h]
            aux_gold_income: (batch, 10) 金币收入标签 [own_5h, enemy_5h]
            aux_base_damage: (batch, 10) 基地伤害标签 [own_5h, enemy_5h]，None 表示不计算
            value_features: (batch, hidden_dim) value 编码特征，不为 None 时同时计算
                            value encoder 的辅助损失

        Returns:
            包含 aux_tower_loss, aux_gold_loss, aux_enemy_tower_loss,
            aux_enemy_gold_loss, aux_base_loss, aux_enemy_base_loss 的字典。
            当 value_features 不为 None 时，额外包含 vf_aux_tower_loss,
            vf_aux_gold_loss, vf_aux_enemy_tower_loss, vf_aux_enemy_gold_loss,
            vf_aux_base_loss, vf_aux_enemy_base_loss。
        """
        tower_pred = self.tower_damage_head(features)  # (batch, 10)
        gold_pred = self.gold_income_head(features)  # (batch, 10)

        mid = tower_pred.size(1) // 2  # 5

        aux_tower_loss = self._masked_mse_loss(tower_pred[:, :mid], aux_tower_damage[:, :mid], aux_valid_mask)
        aux_enemy_tower_loss = self._masked_mse_loss(
            tower_pred[:, mid:], aux_tower_damage[:, mid:], aux_valid_mask
        )
        aux_gold_loss = self._masked_mse_loss(gold_pred[:, :mid], aux_gold_income[:, :mid], aux_valid_mask)
        aux_enemy_gold_loss = self._masked_mse_loss(gold_pred[:, mid:], aux_gold_income[:, mid:], aux_valid_mask)

        result = {
            "aux_tower_loss": aux_tower_loss,
            "aux_enemy_tower_loss": aux_enemy_tower_loss,
            "aux_gold_loss": aux_gold_loss,
            "aux_enemy_gold_loss": aux_enemy_gold_loss,
        }

        if aux_base_damage is not None:
            base_pred = self.base_damage_head(features)  # (batch, 10)
            result["aux_base_loss"] = self._masked_mse_loss(
                base_pred[:, :mid], aux_base_damage[:, :mid], aux_valid_mask
            )
            result["aux_enemy_base_loss"] = self._masked_mse_loss(
                base_pred[:, mid:], aux_base_damage[:, mid:], aux_valid_mask
            )

        # ── value encoder 的辅助损失（使用独立 head，梯度流入 value encoder）──
        if value_features is not None:
            vf_tower_pred = self.value_tower_damage_head(value_features)
            vf_gold_pred = self.value_gold_income_head(value_features)

            result["vf_aux_tower_loss"] = self._masked_mse_loss(
                vf_tower_pred[:, :mid], aux_tower_damage[:, :mid], aux_valid_mask
            )
            result["vf_aux_enemy_tower_loss"] = self._masked_mse_loss(
                vf_tower_pred[:, mid:], aux_tower_damage[:, mid:], aux_valid_mask
            )
            result["vf_aux_gold_loss"] = self._masked_mse_loss(
                vf_gold_pred[:, :mid], aux_gold_income[:, :mid], aux_valid_mask
            )
            result["vf_aux_enemy_gold_loss"] = self._masked_mse_loss(
                vf_gold_pred[:, mid:], aux_gold_income[:, mid:], aux_valid_mask
            )

            if aux_base_damage is not None:
                vf_base_pred = self.value_base_damage_head(value_features)
                result["vf_aux_base_loss"] = self._masked_mse_loss(
                    vf_base_pred[:, :mid], aux_base_damage[:, :mid], aux_valid_mask
                )
                result["vf_aux_enemy_base_loss"] = self._masked_mse_loss(
                    vf_base_pred[:, mid:], aux_base_damage[:, mid:], aux_valid_mask
                )

        return result


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["AntWarPolicyValueNetwork", "count_parameters"]
