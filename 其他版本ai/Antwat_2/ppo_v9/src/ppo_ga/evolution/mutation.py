import numpy as np
from ..genome.genome import Genome


def gaussian_mutation(
    genome: Genome,
    rate: float = 0.1,
    scale: float = 0.03,
) -> Genome:
    """全局高斯变异。

    对扁平权重向量的每个元素，以概率 rate 决定是否变异，
    被选中的元素加上高斯噪声 N(0, scale²)。

    Args:
        genome: 待变异的基因组（会被 in-place 修改）
        rate: 变异率（每个参数被变异的概率，默认 0.1）
        scale: 变异幅度（高斯噪声标准差，默认 0.03）

    Returns:
        变异后的基因组（与输入为同一对象）
    """
    # 1. 生成变异掩码
    mask = np.random.random(genome.weights.shape) < rate

    # 2. 生成高斯噪声
    noise = np.random.randn(*genome.weights.shape).astype(np.float32) * scale

    # 3. 施加变异（仅 mask=True 的位置）
    genome.weights += noise * mask

    return genome
