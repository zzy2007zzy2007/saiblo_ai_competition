from __future__ import annotations

import math
import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ppo_antwar.utils.action_constants import MAX_ACTIONS, TYPE_CONFIG


class HexConv(nn.Module):
    HEX_DIRECTIONS = [
        (+1, 0), (+1, -1), (0, -1),
        (-1, 0), (-1, +1), (0, +1),
    ]

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 6))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight.view(-1, self.in_channels * 6), a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * 6
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, neighbor_sum: torch.Tensor) -> torch.Tensor:
        if neighbor_sum.device != self.weight.device:
            neighbor_sum = neighbor_sum.to(self.weight.device)

        batch, channels, height, width = x.shape
        neighbor_sum = neighbor_sum.view(batch, channels, 6, height, width)
        out = torch.einsum('ocn,bc nhw->bohw', self.weight, neighbor_sum)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class HexToCubeConverter(nn.Module):
    def __init__(self, map_size: int = 19) -> None:
        super().__init__()
        self.map_size = map_size
        self.register_buffer('cube_coords', self._compute_cube_coords(map_size))

    def _compute_cube_coords(self, map_size: int) -> torch.Tensor:
        coords = torch.zeros(3, map_size, map_size, dtype=torch.long)
        for x in range(map_size):
            for y in range(map_size):
                col = x
                row = y
                cube_q = col
                cube_r = row - (col - (col & 1)) // 2
                cube_s = -cube_q - cube_r
                coords[0, x, y] = cube_q
                coords[1, x, y] = cube_r
                coords[2, x, y] = cube_s
        return coords

    def offset_to_cube(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = x.float()
        r = y.float() - (x.float() - (x.float() % 2)) / 2
        s = -q - r
        return q, r, s

    def forward(self) -> torch.Tensor:
        return self.cube_coords


class HexCNNEncoder(nn.Module):
    HEX_DIRECTIONS = [
        (+1, 0), (+1, -1), (0, -1),
        (-1, 0), (-1, +1), (0, +1),
    ]

    def __init__(self, input_channels: int = 28, hidden_channels: int = 128, map_size: int = 19) -> None:
        super().__init__()
        self.map_size = map_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        c1, c2, c3 = hidden_channels // 2, hidden_channels, hidden_channels * 2

        self.to_hex = HexToCubeConverter(map_size)
        self.register_buffer('neighbor_offsets', self._compute_neighbor_offsets())

        self.conv1 = nn.Conv2d(input_channels, c1, kernel_size=1)
        self.hex_conv1 = HexConv(c1, c1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.hex_conv2 = HexConv(c2, c2)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.hex_conv3 = HexConv(c3, c3)
        self.bn3 = nn.BatchNorm2d(c3)

        self.pool = nn.AdaptiveAvgPool2d((5, 5))

    def _compute_neighbor_offsets(self) -> torch.Tensor:
        offsets = torch.tensor([
            [1, 0], [1, -1], [0, -1],
            [-1, 0], [-1, +1], [0, +1],
        ], dtype=torch.long)
        return offsets

    def get_hex_neighbor_sum(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        neighbor_sums = []

        for dx, dy in self.HEX_DIRECTIONS:
            pad_h = abs(dx)
            pad_w = abs(dy)

            if dx > 0:
                pad_top = pad_h
                pad_bottom = 0
            else:
                pad_top = 0
                pad_bottom = pad_h

            if dy > 0:
                pad_left = pad_w
                pad_right = 0
            else:
                pad_left = 0
                pad_right = pad_w

            padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

            if dx > 0:
                h_start = 0
                h_end = height
            else:
                h_start = pad_h
                h_end = height + pad_h

            if dy > 0:
                w_start = 0
                w_end = width
            else:
                w_start = pad_w
                w_end = width + pad_w

            shifted = padded[:, :, h_start:h_end, w_start:w_end]
            neighbor_sums.append(shifted)

        neighbor_tensor = torch.stack(neighbor_sums, dim=2)
        return neighbor_tensor.view(batch, channels * 6, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        hex_neighbor = self.get_hex_neighbor_sum(x)
        x = x + F.relu(self.hex_conv1(x, hex_neighbor))

        x = F.relu(self.bn2(self.conv2(x)))
        hex_neighbor = self.get_hex_neighbor_sum(x)
        x = x + F.relu(self.hex_conv2(x, hex_neighbor))

        x = F.relu(self.bn3(self.conv3(x)))
        hex_neighbor = self.get_hex_neighbor_sum(x)
        x = x + F.relu(self.hex_conv3(x, hex_neighbor))

        x = self.pool(x)
        x = x.flatten(1)
        return x

    def get_output_dim(self) -> int:
        return self.hidden_channels * 2 * 5 * 5


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StructuredActionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.num_types = len(TYPE_CONFIG)

        self.type_head = nn.Linear(hidden_dim, self.num_types)

        target_dims = [config['max_targets'] for config in TYPE_CONFIG]
        self.target_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in target_dims
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.type_head.weight, gain=0.5)
        nn.init.zeros_(self.type_head.bias)
        for head in self.target_heads:
            if head.out_features <= 1:
                nn.init.orthogonal_(head.weight, gain=0.01)
            else:
                nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type_logits = self.type_head(x)

        B = x.size(0)
        device = x.device
        flat_logits = torch.full((B, MAX_ACTIONS), fill_value=-1e9, device=device)

        for type_id, config in enumerate(TYPE_CONFIG):
            start = config['flat_start']
            end = config['flat_end']
            n_valid = end - start
            target_logits = self.target_heads[type_id](x)
            target_logits_valid = target_logits[:, :n_valid]
            flat_logits[:, start:end] = type_logits[:, type_id:type_id+1] + target_logits_valid

        return flat_logits

    @staticmethod
    def flat_mask_to_type_mask(flat_mask: torch.Tensor) -> torch.Tensor:
        B = flat_mask.size(0)
        device = flat_mask.device
        type_mask = torch.zeros(B, len(TYPE_CONFIG), device=device)
        for type_id, config in enumerate(TYPE_CONFIG):
            start = config['flat_start']
            end = config['flat_end']
            has_legal = flat_mask[:, start:end].any(dim=1)
            type_mask[:, type_id] = has_legal.float()
        return type_mask

    @staticmethod
    def flat_mask_to_target_mask(flat_mask: torch.Tensor, type_id: int) -> torch.Tensor:
        config = TYPE_CONFIG[type_id]
        start = config['flat_start']
        end = config['flat_end']
        return flat_mask[:, start:end]


def action_id_to_type_and_target(action_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    type_ids = torch.zeros_like(action_ids)
    target_ks = torch.zeros_like(action_ids)
    for type_id, config in enumerate(TYPE_CONFIG):
        start = config['flat_start']
        end = config['flat_end']
        in_range = (action_ids >= start) & (action_ids < end)
        type_ids[in_range] = type_id
        target_ks[in_range] = action_ids[in_range] - start
    return type_ids, target_ks


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


NUM_HORIZONS = 5


class TowerDamageHead(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_HORIZONS * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoldIncomeHead(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_HORIZONS * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AntWarPolicyValueNetwork(nn.Module):
    def __init__(
        self,
        board_shape: Tuple[int, int, int] = (28, 19, 19),
        global_dim: int = 30,
        action_dim: int = 96,
        hidden_dim: int = 256,
        enable_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.board_shape = board_shape
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary

        self.cnn_encoder = HexCNNEncoder(board_shape[0], hidden_dim, board_shape[1])
        self.mlp_encoder = MLPEncoder(global_dim, hidden_dim // 2)

        cnn_output_dim = self.cnn_encoder.get_output_dim()
        mlp_output_dim = hidden_dim // 2
        self.projection = nn.Linear(cnn_output_dim + mlp_output_dim, hidden_dim)
        self.merged_norm = nn.LayerNorm(hidden_dim)

        self.policy_head = StructuredActionHead(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        if enable_auxiliary:
            self.tower_damage_head = TowerDamageHead(hidden_dim)
            self.gold_income_head = GoldIncomeHead(hidden_dim)

        self.apply(orthogonal_init)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self.policy_head._init_weights()

    def _compute_aux_loss(
        self,
        merged: torch.Tensor,
        our_tower_labels: torch.Tensor,
        our_gold_labels: torch.Tensor,
        enemy_tower_labels: torch.Tensor,
        enemy_gold_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tower_pred = self.tower_damage_head(merged)
        gold_pred = self.gold_income_head(merged)
        our_tower_pred = tower_pred[:, :NUM_HORIZONS]
        enemy_tower_pred = tower_pred[:, NUM_HORIZONS:]
        our_gold_pred = gold_pred[:, :NUM_HORIZONS]
        enemy_gold_pred = gold_pred[:, NUM_HORIZONS:]
        our_tower_loss = F.mse_loss(our_tower_pred, our_tower_labels)
        enemy_tower_loss = F.mse_loss(enemy_tower_pred, enemy_tower_labels)
        our_gold_loss = F.mse_loss(our_gold_pred, our_gold_labels)
        enemy_gold_loss = F.mse_loss(enemy_gold_pred, enemy_gold_labels)
        return our_tower_loss, enemy_tower_loss, our_gold_loss, enemy_gold_loss

    def _encode(self, board: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        board_encoded = self.cnn_encoder(board)
        global_encoded = self.mlp_encoder(global_features)
        merged = torch.cat([board_encoded, global_encoded], dim=1)
        merged = self.projection(merged)
        merged = self.merged_norm(merged)
        return merged

    def forward(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        merged = self._encode(board, global_features)

        action_logits = self.policy_head(merged)
        value = self.value_head(merged)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        return action_logits, value

    def get_action(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        merged = self._encode(board, global_features)

        action_logits = self.policy_head(merged)
        value = self.value_head(merged)

        type_logits = self.policy_head.type_head(merged)

        noise_std = getattr(self, 'logit_noise_std', 0.0)
        if not deterministic and noise_std > 0:
            type_logits = type_logits + torch.randn_like(type_logits) * noise_std

        if action_mask is not None:
            type_mask = self.policy_head.flat_mask_to_type_mask(action_mask)
            type_logits = type_logits.masked_fill(type_mask == 0, float('-inf'))

        eps = getattr(self, 'exploration_epsilon', 0.0)
        if not deterministic and eps > 0 and random.random() < eps:
            if action_mask is not None:
                valid_indices = torch.where(action_mask[0] > 0)[0]
                action = valid_indices[torch.randint(len(valid_indices), (1,))].unsqueeze(0)
                action = action.squeeze(0)
            else:
                action = torch.randint(0, action_logits.size(-1), (1,), device=action_logits.device)
        else:
            if deterministic:
                type_id = torch.argmax(type_logits, dim=-1, keepdim=True)
            else:
                type_probs = F.softmax(type_logits, dim=-1)
                type_probs = torch.nan_to_num(type_probs, nan=0.0)
                type_probs = type_probs.clamp(min=1e-10)
                type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
                type_id = torch.multinomial(type_probs, 1)

            type_id_scalar = type_id.item()
            config = TYPE_CONFIG[type_id_scalar]
            n_valid = config['flat_end'] - config['flat_start']

            target_logits = self.policy_head.target_heads[type_id_scalar](merged)
            target_logits_valid = target_logits[:, :n_valid]

            if not deterministic and noise_std > 0:
                target_logits_valid = target_logits_valid + torch.randn_like(target_logits_valid) * noise_std

            if action_mask is not None:
                target_mask = action_mask[:, config['flat_start']:config['flat_end']]
                target_logits_valid = target_logits_valid.masked_fill(target_mask == 0, float('-inf'))

            if deterministic:
                target_k = torch.argmax(target_logits_valid, dim=-1, keepdim=True)
            else:
                target_probs = F.softmax(target_logits_valid, dim=-1)
                target_probs = torch.nan_to_num(target_probs, nan=0.0)
                target_probs = target_probs.clamp(min=1e-10)
                target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
                target_k = torch.multinomial(target_probs, 1)

            action = torch.tensor(
                config['flat_start'] + target_k.item(),
                device=action_logits.device,
            ).unsqueeze(0)

        type_ids, target_ks = action_id_to_type_and_target(action)
        type_id_0 = type_ids.item()
        target_k_0 = target_ks.item()

        type_log_prob = F.log_softmax(type_logits, dim=-1)
        type_log_prob_selected = type_log_prob[:, type_id_0:type_id_0+1]

        config = TYPE_CONFIG[type_id_0]
        n_valid = config['flat_end'] - config['flat_start']
        target_logits_0 = self.policy_head.target_heads[type_id_0](merged)
        target_logits_0 = target_logits_0[:, :n_valid]
        if not deterministic and noise_std > 0:
            target_logits_0 = target_logits_0 + torch.randn_like(target_logits_0) * noise_std
        if action_mask is not None:
            target_mask_0 = action_mask[:, config['flat_start']:config['flat_end']]
            target_logits_0 = target_logits_0.masked_fill(target_mask_0 == 0, float('-inf'))
        target_log_prob = F.log_softmax(target_logits_0, dim=-1)
        target_log_prob_selected = target_log_prob[:, target_k_0:target_k_0+1]

        action_log_prob = (type_log_prob_selected + target_log_prob_selected).squeeze(-1)
        action_log_prob = torch.nan_to_num(action_log_prob, nan=0.0, posinf=20.0, neginf=-20.0)

        type_probs = F.softmax(type_logits, dim=-1)
        type_probs = torch.nan_to_num(type_probs, nan=0.0)
        type_probs = type_probs.clamp(min=1e-10)
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)

        return action, action_log_prob, value, type_probs.squeeze(0)

    def evaluate_actions(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        merged = self._encode(board, global_features)
        action_logits = self.policy_head(merged)
        value = self.value_head(merged)

        noise_std = getattr(self, 'logit_noise_std', 0.0)

        type_ids, target_ks = action_id_to_type_and_target(actions.flatten())

        type_logits = self.policy_head.type_head(merged)
        if noise_std > 0:
            type_logits = type_logits + torch.randn_like(type_logits) * noise_std
        if action_mask is not None:
            type_mask = self.policy_head.flat_mask_to_type_mask(action_mask)
            type_logits = type_logits.masked_fill(type_mask == 0, float('-inf'))

        type_log_prob = F.log_softmax(type_logits, dim=-1)
        type_log_prob_selected = type_log_prob.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

        target_log_prob_selected = torch.zeros_like(type_log_prob_selected)
        for type_id_val in range(len(TYPE_CONFIG)):
            mask = (type_ids == type_id_val)
            if not mask.any():
                continue
            config = TYPE_CONFIG[type_id_val]
            n_valid = config['flat_end'] - config['flat_start']
            target_logits_t = self.policy_head.target_heads[type_id_val](merged)
            target_logits_t = target_logits_t[:, :n_valid]
            if noise_std > 0:
                target_logits_t = target_logits_t + torch.randn_like(target_logits_t) * noise_std
            if action_mask is not None:
                target_mask = action_mask[:, config['flat_start']:config['flat_end']]
                target_logits_t = target_logits_t.masked_fill(target_mask == 0, float('-inf'))
            t_log_prob = F.log_softmax(target_logits_t, dim=-1)
            target_log_prob_selected[mask] = t_log_prob.gather(1, target_ks[mask].unsqueeze(-1)).squeeze(-1)

        action_log_prob = type_log_prob_selected + target_log_prob_selected
        action_log_prob = torch.nan_to_num(action_log_prob, nan=0.0, posinf=20.0, neginf=-20.0)

        type_probs = torch.exp(type_log_prob)
        type_probs = torch.nan_to_num(type_probs, nan=0.0)
        type_probs = type_probs.clamp(min=1e-10)
        type_log_prob = torch.log(type_probs)
        type_entropy = -(type_probs * type_log_prob).sum(dim=-1)

        target_entropy_sum = torch.zeros_like(type_entropy)

        for type_id, config in enumerate(TYPE_CONFIG):
            start = config['flat_start']
            n_valid = config['flat_end'] - start

            target_logits_t = self.policy_head.target_heads[type_id](merged)
            target_logits_t = target_logits_t[:, :n_valid]

            if noise_std > 0:
                target_logits_t = target_logits_t + torch.randn_like(target_logits_t) * noise_std

            if action_mask is not None:
                target_mask = action_mask[:, start:config['flat_end']]
                target_logits_t = target_logits_t.masked_fill(target_mask == 0, float('-inf'))

            target_log_prob_t = F.log_softmax(target_logits_t, dim=-1)
            target_probs_t = torch.exp(target_log_prob_t)

            target_probs_t = torch.nan_to_num(target_probs_t, nan=0.0)
            target_probs_t = target_probs_t.clamp(min=1e-10)
            target_log_prob_t = torch.log(target_probs_t)

            target_entropy_t = -(target_probs_t * target_log_prob_t).sum(dim=-1)
            target_entropy_sum = target_entropy_sum + type_probs[:, type_id] * target_entropy_t

        dist_entropy = type_entropy + target_entropy_sum

        return action_log_prob, value.squeeze(-1), dist_entropy, action_logits, type_entropy, target_entropy_sum


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
