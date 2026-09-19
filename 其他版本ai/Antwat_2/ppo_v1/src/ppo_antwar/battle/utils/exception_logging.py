"""
统一异常处理和日志记录工具模块

遵循设计原则：
1. 异常应合理捕获并记录
2. 异常日志包含必要的环境参数信息
3. 异常应抛向上层，除非有明确的容错需求
"""

import traceback
import os
import sys
from typing import Optional, Any, Dict
from loguru import logger


def log_exception(
    logger_obj: Optional[Any],
    category: str,
    context: Dict[str, Any],
    exception: Exception
) -> None:
    """
    统一的异常日志记录函数
    
    Args:
        logger_obj: 日志记录器实例（可为None，此时使用loguru）
        category: 日志分类
        context: 上下文信息字典
        exception: 异常对象
    """
    error_msg = f"Exception in {category}"
    error_msg += f", context: {context}"
    error_msg += f", exception_type: {type(exception).__name__}"
    error_msg += f", exception_msg: {str(exception)}"
    
    stack_trace = traceback.format_exc()
    error_msg += f"\nStack trace:\n{stack_trace}"
    
    if logger_obj:
        logger_obj.error(category, error_msg)
    else:
        logger.error(error_msg)


def get_error_context(**kwargs) -> Dict[str, Any]:
    """
    获取标准错误上下文信息
    
    Returns:
        包含环境信息的字典
    """
    context = {
        'pid': os.getpid(),
        'cwd': os.getcwd(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'sys_path_length': len(sys.path),
    }
    context.update(kwargs)
    return context


def safe_execute(
    func,
    default_return=None,
    log_level: str = "error",
    context: Optional[Dict[str, Any]] = None,
    logger_obj: Optional[Any] = None,
    reraise: bool = True
):
    """
    安全执行函数的装饰器或包装函数
    
    Args:
        func: 要执行的函数
        default_return: 异常时的默认返回值
        log_level: 日志级别 ('error', 'warning', 'info')
        context: 上下文信息
        logger_obj: 日志记录器实例
        reraise: 是否重新抛出异常
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return func()
    except Exception as e:
        error_context = get_error_context(**(context or {}))
        log_exception(logger_obj, func.__name__, error_context, e)
        if reraise:
            raise
        return default_return
