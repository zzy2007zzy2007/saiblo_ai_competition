import math
import torch
import torch.nn as nn
import numpy as np


def random_init_network(network: nn.Module) -> None:
    """对网络的所有可训练参数执行完全随机初始化（均匀分布）。

    初始化策略：U(-k, k)，其中 k = 1 / sqrt(fan_in)
    - Conv2d 权重：fan_in = in_channels * kernel_height * kernel_width
    - Linear 权重：fan_in = in_features
    - Conv2d/Linear 偏置：0
    - LayerNorm/BatchNorm weight：1
    - LayerNorm/BatchNorm bias：0

    注意：不使用正交初始化，因为正交初始化引入先验偏向，与"完全随机"初衷不符。

    Args:
        network: AntWarPolicyValueNetwork 实例
    """
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue

        if 'weight' in name:
            if isinstance(param, (nn.Conv2d, nn.Linear)) or _is_conv_or_linear_weight(name, param):
                fan_in = _compute_fan_in(param)
                k = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.uniform_(param, -k, k)
            elif _is_norm_weight(name):
                nn.init.ones_(param)
            else:
                # HexConv.weight 等自定义参数
                fan_in = _compute_fan_in(param)
                k = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.uniform_(param, -k, k)
        elif 'bias' in name:
            nn.init.zeros_(param)
        else:
            pass


def _compute_fan_in(param: torch.Tensor) -> int:
    """计算 fan_in（输入维度大小）。

    对于 Conv2d 权重 (out, in, kH, kW)：fan_in = in * kH * kW
    对于 Linear 权重 (out, in)：fan_in = in
    对于 HexConv 权重 (out, in, 6)：fan_in = in * 6
    """
    shape = param.shape
    if len(shape) >= 3:
        # Conv2d 或 HexConv: fan_in = dim[1] * prod(dim[2:])
        fan_in = shape[1]
        for d in shape[2:]:
            fan_in *= d
        return fan_in
    elif len(shape) == 2:
        # Linear: fan_in = shape[1]
        return shape[1]
    else:
        return shape[0]


def _is_norm_weight(name: str) -> bool:
    """判断是否为归一化层的 weight 参数"""
    return any(k in name for k in ['LayerNorm', 'BatchNorm', 'layer_norm', 'batch_norm'])


def _is_norm_bias(name: str) -> bool:
    """判断是否为归一化层的 bias 参数"""
    return _is_norm_weight(name)


def _is_conv_or_linear_weight(name: str, param: torch.Tensor) -> bool:
    """判断是否为 Conv2d 或 Linear 的 weight 参数（通过形状推断）"""
    return len(param.shape) in (2, 4)
