from datetime import datetime
from typing import Dict, Optional, Callable
from functools import wraps


class Timer:
    """
    计时器类，支持 timing_start(), timing_stop(), 阈值警告等功能。
    """

    def __init__(self, warning_threshold: float = 2.0):
        """
        初始化计时器。

        Args:
            warning_threshold: 警告阈值（秒），超过该阈值会触发警告
        """
        self.warning_threshold = warning_threshold
        self._timing_starts: Dict[str, datetime] = {}
        self._timings: Dict[str, float] = {}
        self._timing_warnings: Dict[str, int] = {}

    def timing_start(self, operation: str):
        """
        开始计时。

        Args:
            operation: 操作名称
        """
        self._timing_starts[operation] = datetime.now()

    def timing_stop(self, operation: str, duration: float = None) -> float:
        """
        停止计时。

        Args:
            operation: 操作名称
            duration: 可选的持续时间（如果不提供，则自动计算

        Returns:
            持续时间（秒）
        """
        if duration is None:
            if operation not in self._timing_starts:
                return 0.0
            duration = (datetime.now() - self._timing_starts[operation]).total_seconds()

        self._timings[operation] = self._timings.get(operation, 0.0) + duration

        if duration > self.warning_threshold:
            self._timing_warnings[operation] = self._timing_warnings.get(operation, 0) + 1

        return duration

    def timing_warning(self, operation: str) -> bool:
        """
        检查某个操作是否有超时警告。

        Args:
            operation: 操作名称

        Returns:
            是否有超时警告
        """
        return self._timing_warnings.get(operation, 0) > 0

    def get_timing_total(self, operation: str) -> float:
        """
        获取某个操作的总时间。

        Args:
            operation: 操作名称

        Returns:
            总时间（秒）
        """
        return self._timings.get(operation, 0.0)

    def get_all_timings(self) -> Dict[str, float]:
        """
        获取所有操作的计时记录。

        Returns:
            操作名称到总时间的映射
        """
        return self._timings.copy()

    def get_warning_count(self, operation: str) -> int:
        """
        获取某个操作的警告次数。

        Args:
            operation: 操作名称

        Returns:
            警告次数
        """
        return self._timing_warnings.get(operation, 0)

    def reset(self):
        """
        重置所有计时记录。
        """
        self._timing_starts.clear()
        self._timings.clear()
        self._timing_warnings.clear()


def timer_decorator(
    timer: Optional[Timer] = None,
    operation: Optional[str] = None,
    warning_threshold: float = 2.0
):
    """
    计时器装饰器。

    Args:
        timer: 可选的 Timer 实例，如果不提供则创建新的
        operation: 操作名称，如果不提供则使用函数名
        warning_threshold: 警告阈值（秒）

    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal timer
            if timer is None:
                timer = Timer(warning_threshold=warning_threshold)

            op_name = operation if operation else func.__name__
            timer.timing_start(op_name)
            try:
                return func(*args, **kwargs)
            finally:
                timer.timing_stop(op_name)

        return wrapper
    return decorator
