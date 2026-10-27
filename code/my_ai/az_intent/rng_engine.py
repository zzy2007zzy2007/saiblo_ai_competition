"""加载"RNG 可控"的引擎变体（`code/cpp_engine/rng_build/native_game.pyd`）。

## 为什么要单独一个模块

`native_game` 是 pybind11 模块，**模块名必须等于 .pyd 文件名**，所以两套引擎不能
叫同一个名字放在同一个目录里。做法是：RNG 变体放在**子目录** `rng_build/`，文件名仍
叫 `native_game.pyd`，靠**导入路径顺序**选择 —— 现役 `code/cpp_engine/native_game.pyd`
一个字节都不动。

RNG 变体与现役 pyd 的**唯一**差别：`game.hpp` 多了 3 个 header-inline 方法
（`get_rng_state / set_rng_state / consume_rng`），`game.cpp` 与官方源码 MD5 相同。
⇒ 不调用这 3 个方法时，两者逐位一致（`probe_engine_rng.py parity` 在 6 条真实对局、
2730 回合上验证过：逐回合强摘要 + rng_state 序列全等）。

## 用法

```python
from my_ai.az_intent import rng_engine
if rng_engine.enable():          # 必须在**第一次建局之前**
    ...                          # 之后 GameStateFacade 拿到的就是 RNG 变体
rng_engine.available(sim)        # 这个 sim 能不能 consume_rng
```

⚠️ 顺序很重要：`GameStateFacade.initial()` 里才 `import native_game`，
所以 `enable()` 要赶在第一次建局之前调用。官方链路里 `az_bridge_ai.main()` 是
`build_engine()`（构造搜索器）→ `GameState.initial()`，顺序天然满足。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RNG_BUILD_DIR = REPO / "code" / "cpp_engine" / "rng_build"
_PYD = RNG_BUILD_DIR / "native_game.pyd"


def pyd_path() -> Path:
    return _PYD


def built() -> bool:
    return _PYD.exists()


def _is_rng(mod) -> bool:
    f = str(getattr(mod, "__file__", "") or "")
    return "rng_build" in f.replace("\\", "/")


def enable(verbose: bool = True) -> bool:
    """把 `rng_build/` 插到 sys.path 最前面，使后续 `import native_game` 命中 RNG 变体。

    返回 True = 之后建出来的状态支持 `consume_rng`。若 `native_game` 已经被
    import 过（先建过局），则**无法**替换，返回 False（并说明原因）。
    """
    if not built():
        if verbose:
            print(f"[rng_engine] ✗ 找不到 {_PYD}；先跑 "
                  f"`python code/cpp_engine/build_rng_variant.py`", flush=True)
        return False
    mod = sys.modules.get("native_game")
    if mod is not None:
        ok = _is_rng(mod)
        if verbose:
            print(f"[rng_engine] {'✓' if ok else '✗'} native_game 已加载: "
                  f"{getattr(mod, '__file__', '?')}"
                  + ("" if ok else "（不是 rng_build 版本，无法替换 ⇒ 不会消费 RNG）"),
                  flush=True)
        return ok
    path = str(RNG_BUILD_DIR)
    if path not in sys.path or sys.path[0] != path:
        sys.path.insert(0, path)
    if verbose:
        print(f"[rng_engine] ✓ 已把 {path} 插到 sys.path[0]"
              f"（之后 import native_game 命中 RNG 变体）", flush=True)
    return True


def available(obj=None) -> bool:
    """`obj`（facade 或 NativeGame）是否支持 `consume_rng`。

    `obj=None` 时看当前已加载的 `native_game` 模块。
    """
    if obj is None:
        mod = sys.modules.get("native_game")
        if mod is None:
            return False
        try:
            cls = mod.NativeGame
        except AttributeError:
            return False
        return hasattr(cls, "consume_rng")
    g = getattr(obj, "_g", obj)
    return hasattr(g, "consume_rng")


def describe() -> str:
    mod = sys.modules.get("native_game")
    f = getattr(mod, "__file__", "") if mod is not None else ""
    return f"native_game={f or '(未加载)'} rng_build={'✓' if _is_rng(mod) else '✗'}"
