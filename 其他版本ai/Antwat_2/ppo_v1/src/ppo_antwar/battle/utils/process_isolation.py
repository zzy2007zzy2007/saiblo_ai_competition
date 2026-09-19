import sys
import os
from contextlib import contextmanager
from typing import List, Set


ISOLATED_MODULE_PREFIXES = ['ai', 'common', 'SDK', 'AI']


@contextmanager
def process_isolation(
    extra_isolated_modules: List[str] = None,
    preserve_modules: Set[str] = None
):
    """
    进程隔离的上下文管理器，用于隔离特定模块的加载。

    Args:
        extra_isolated_modules: 额外需要隔离的模块列表
        preserve_modules: 需要保留的模块集合（不会被删除）
    """
    original_sys_path = sys.path.copy()
    original_modules = {}
    original_cwd = os.getcwd()

    if extra_isolated_modules is None:
        extra_isolated_modules = []

    if preserve_modules is None:
        preserve_modules = set()

    modules_to_remove = []
    for key in list(sys.modules.keys()):
        should_remove = False

        if key in ISOLATED_MODULE_PREFIXES or key in extra_isolated_modules:
            should_remove = True
        else:
            for prefix in ISOLATED_MODULE_PREFIXES:
                if key.startswith(f'{prefix}.'):
                    should_remove = True
                    break
            for prefix in extra_isolated_modules:
                if key.startswith(f'{prefix}.'):
                    should_remove = True
                    break

        if should_remove and key not in preserve_modules:
            modules_to_remove.append(key)

    for mod in modules_to_remove:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
            del sys.modules[mod]

    try:
        yield
    finally:
        os.chdir(original_cwd)
        sys.path = original_sys_path
        for mod, module in original_modules.items():
            sys.modules[mod] = module
