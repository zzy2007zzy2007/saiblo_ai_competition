from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass
class LayerBoundary:
    """层边界信息，用于层级杂交时定位切点"""
    start_idx: int          # 在扁平向量中的起始位置
    end_idx: int            # 在扁平向量中的结束位置（不含）
    layer_type: str         # conv | linear | norm | bias | head
    layer_name: str         # 层名，如 policy_cnn.stage1.0


@dataclass
class Genome:
    """基因组 —— 策略网络所有可训练参数的扁平化表示"""
    weights: np.ndarray                              # 一维 float32 权重向量
    shapes: List[Tuple[int, ...]]                    # 各参数的原始形状
    layer_boundaries: List[LayerBoundary]             # 层边界索引
    param_count: int                                  # 总参数量
    param_names: List[str]                            # 各参数名（与 shapes 一一对应）


def extract_genome(network: nn.Module) -> Genome:
    """从网络实例提取基因组。

    遍历 network.named_parameters()，将每个参数展平拼接为一维向量，
    同时记录形状和层边界信息。

    Args:
        network: AntWarPolicyValueNetwork 实例

    Returns:
        Genome 实例
    """
    weights_list = []
    shapes = []
    param_names = []
    layer_boundaries = []

    offset = 0
    for name, param in network.named_parameters():
        flat = param.data.cpu().numpy().flatten()
        shapes.append(tuple(param.shape))
        param_names.append(name)

        # 判定层类型
        layer_type = _classify_layer(name, param)

        boundary = LayerBoundary(
            start_idx=offset,
            end_idx=offset + len(flat),
            layer_type=layer_type,
            layer_name=name,
        )
        layer_boundaries.append(boundary)

        weights_list.append(flat)
        offset += len(flat)

    all_weights = np.concatenate(weights_list).astype(np.float32)

    return Genome(
        weights=all_weights,
        shapes=shapes,
        layer_boundaries=layer_boundaries,
        param_count=len(all_weights),
        param_names=param_names,
    )


def genome_to_state_dict(genome: Genome) -> Dict[str, torch.Tensor]:
    """将基因组转换回 state_dict 格式，用于加载到网络。

    按记录的 shapes 将一维权重向量切分并 reshape 为原始形状。

    Args:
        genome: Genome 实例

    Returns:
        {param_name: Tensor} 字典
    """
    state_dict = {}
    offset = 0
    for name, shape in zip(genome.param_names, genome.shapes):
        size = 1
        for s in shape:
            size *= s
        tensor = torch.from_numpy(
            genome.weights[offset:offset + size].copy()
        ).reshape(shape)
        state_dict[name] = tensor
        offset += size
    return state_dict


def load_genome_to_network(network: nn.Module, genome: Genome) -> None:
    """将基因组加载到网络实例（in-place 修改网络参数）。

    Args:
        network: AntWarPolicyValueNetwork 实例
        genome: Genome 实例
    """
    state_dict = genome_to_state_dict(genome)
    # strict=False: Genome 仅包含 named_parameters()，不含 BatchNorm running_mean/var 等 buffer
    network.load_state_dict(state_dict, strict=False)


def cosine_similarity(weights_a: np.ndarray, weights_b: np.ndarray) -> float:
    """计算两个扁平权重向量的余弦相似度。

    Args:
        weights_a: shape (N,)
        weights_b: shape (N,)

    Returns:
        cos(a, b) = (a·b) / (|a|·|b|)，范围 [-1, 1]
    """
    dot = np.dot(weights_a, weights_b)
    norm_a = np.linalg.norm(weights_a)
    norm_b = np.linalg.norm(weights_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _classify_layer(name: str, param: nn.Parameter) -> str:
    """根据参数名和形状判定层类型。"""
    if 'weight' in name:
        if len(param.shape) >= 3:
            return 'conv'       # Conv2d 权重 (out, in, kH, kW) 或 HexConv 权重 (out, in, 6)
        elif len(param.shape) == 2:
            return 'linear'     # Linear 权重 (out, in)
        elif len(param.shape) == 1:
            return 'norm'       # BatchNorm/LayerNorm weight
    elif 'bias' in name:
        return 'bias'
    else:
        return 'other'
