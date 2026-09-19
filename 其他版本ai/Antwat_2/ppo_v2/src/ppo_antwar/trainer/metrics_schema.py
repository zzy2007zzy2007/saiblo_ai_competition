from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricDef:
    """单个训练指标的定义元信息"""

    name: str  # 指标名称（dict key）
    dtype: str  # 'float' | 'int' | 'str'
    precision: Optional[int] = None  # 浮点精度（如 6 表示 round_6）；None 表示原始值
    log_jsonl: bool = True  # 是否写入 batch_metrics.jsonl
    log_tensorboard: bool = True  # 是否写入 TensorBoard
    description: str = ""  # 可读说明


class MetricsSchema:
    """训练指标 Schema - 所有训练指标的单一事实来源"""

    # ── 训练损失类 ──────────────────────────────────────────────────────
    POLICY_LOSS = MetricDef(
        "policy_loss", "float", precision=6, description="PPO policy clip loss"
    )
    VALUE_LOSS = MetricDef(
        "value_loss", "float", precision=6, description="PPO value loss (unweighted)"
    )
    ENTROPY = MetricDef(
        "entropy", "float", precision=6, description="Policy entropy (type + target)"
    )
    ENTROPY_TYPE = MetricDef(
        "entropy_type", "float", precision=6, log_tensorboard=False
    )
    ENTROPY_TARGET = MetricDef(
        "entropy_target", "float", precision=6, log_tensorboard=False
    )
    CLIP_FRACTION = MetricDef("clip_fraction", "float", precision=6)
    APPROX_KL = MetricDef("approx_kl", "float", precision=6, log_tensorboard=False)

    # ── 辅助任务损失 ─────────────────────────────────────────────────────
    AUX_TOWER_LOSS = MetricDef("aux_tower_loss", "float", precision=6)
    AUX_GOLD_LOSS = MetricDef("aux_gold_loss", "float", precision=6)
    AUX_ENEMY_TOWER_LOSS = MetricDef("aux_enemy_tower_loss", "float", precision=6)
    AUX_ENEMY_GOLD_LOSS = MetricDef("aux_enemy_gold_loss", "float", precision=6)
    AUX_BASE_LOSS = MetricDef("aux_base_loss", "float", precision=6)
    AUX_ENEMY_BASE_LOSS = MetricDef("aux_enemy_base_loss", "float", precision=6)

    # ── 梯度类 ──────────────────────────────────────────────────────────
    GRAD_NORM = MetricDef("grad_norm", "float", precision=6)
    GRAD_NORM_POLICY = MetricDef("grad_norm_policy", "float", precision=6)
    GRAD_NORM_VALUE = MetricDef("grad_norm_value", "float", precision=6)
    GRAD_NORM_MAX_LAYER = MetricDef(
        "grad_norm_max_layer", "float", precision=6, log_tensorboard=False
    )

    # ── 奖励/价值统计 ───────────────────────────────────────────────────
    REWARD_MEAN = MetricDef("reward_mean", "float", precision=4)
    REWARD_STD = MetricDef("reward_std", "float", precision=4)
    REWARD_MIN = MetricDef("reward_min", "float", precision=4, log_tensorboard=False)
    REWARD_MAX = MetricDef("reward_max", "float", precision=4, log_tensorboard=False)
    RETURN_MEAN = MetricDef("return_mean", "float", precision=4)
    RETURN_STD = MetricDef("return_std", "float", precision=4)
    ADVANTAGES_STD = MetricDef("advantages_std", "float", precision=4)

    # ── Value 统计 ──────────────────────────────────────────────────────
    VALUE_INPUT_MEAN = MetricDef(
        "value_input_mean", "float", precision=6, log_tensorboard=False
    )
    VALUE_INPUT_STD = MetricDef(
        "value_input_std", "float", precision=6, log_tensorboard=False
    )
    VALUE_PRED_MEAN = MetricDef(
        "value_pred_mean", "float", precision=6, log_tensorboard=False
    )
    VALUE_PRED_STD = MetricDef(
        "value_pred_std", "float", precision=6, log_tensorboard=False
    )
    RATIO_MEAN = MetricDef("ratio_mean", "float", precision=6, log_tensorboard=False)
    RATIO_STD = MetricDef("ratio_std", "float", precision=6, log_tensorboard=False)
    TD_ERROR_MEAN = MetricDef(
        "td_error_mean", "float", precision=6, log_tensorboard=False
    )
    TD_ERROR_STD = MetricDef(
        "td_error_std", "float", precision=6, log_tensorboard=False
    )

    # ── Action 统计 ─────────────────────────────────────────────────────
    VALID_ACTIONS_MEAN = MetricDef(
        "valid_actions_mean", "float", precision=6, log_tensorboard=False
    )
    VALID_ACTIONS_MIN = MetricDef(
        "valid_actions_min", "float", precision=6, log_tensorboard=False
    )
    BATCH_SIZE = MetricDef("batch_size", "int", log_tensorboard=False)
    NUM_COLLECTED = MetricDef("num_collected", "int", log_tensorboard=False)
    NAN_SKIP_COUNT = MetricDef("nan_skip_count", "int", log_tensorboard=False)

    # ── 超参类（不含 precision，直接记录原始值）────────────────────────
    LEARNING_RATE = MetricDef("learning_rate", "float", log_jsonl=True)
    ENTROPY_COEF = MetricDef("entropy_coef", "float", log_jsonl=True)

    # ── 修正后的 loss 指标 ──────────────────────────────────────────────
    TRAINING_LOSS = MetricDef(
        "training_loss",
        "float",
        precision=6,
        description="Total training loss = policy_loss + vf_coef*value_loss - ent_coef*entropy",
    )
    LEGACY_LOSS = MetricDef(
        "loss", "float", precision=6, log_tensorboard=False, log_jsonl=True
    )

    # ── 全局注册表 ──────────────────────────────────────────────────────
    _REGISTRY: Dict[str, MetricDef] = {}

    @classmethod
    def _build_registry(cls):
        if cls._REGISTRY:
            return
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, MetricDef):
                cls._REGISTRY[attr.name] = attr

    @classmethod
    def get(cls, name: str) -> Optional[MetricDef]:
        cls._build_registry()
        return cls._REGISTRY.get(name)

    @classmethod
    def get_jsonl_keys(cls) -> List[str]:
        cls._build_registry()
        return [d.name for d in cls._REGISTRY.values() if d.log_jsonl]

    @classmethod
    def get_tensorboard_keys(cls) -> List[str]:
        cls._build_registry()
        return [d.name for d in cls._REGISTRY.values() if d.log_tensorboard]

    @classmethod
    def format_value(cls, name: str, value) -> object:
        """按 schema 定义格式化单个指标值"""
        metric_def = cls.get(name)
        if metric_def is None:
            return value
        if metric_def.dtype == "int":
            return int(value)
        if metric_def.dtype == "float" and metric_def.precision is not None:
            return round(float(value), metric_def.precision)
        return value
