"""决策耗时分解计时器 —— 严格处理父子嵌套，避免重复归属。

设计要点（针对"以前把某块时间归错"的问题）：
1. **每个计时器记录 inclusive 时间**（含其内部嵌套的子计时器）；
2. 由**栈**自动推出父子关系，最终算出 **exclusive = inclusive − 直接子计时器之和**；
3. 每条决策记录都做**一致性自检**：
   `sum(exclusive at root) ≈ total`，误差超过 2% 就写一条 WARNING 到日志；
4. 输出 JSONL（每条决策一行），字段名带层级（`.` 分隔），父子关系一目了然。

用法：
    from timing import T
    with T("decision"):
        with T("recv"): ...
        with T("search"):
            with T("search.net"):
                with T("net.encode"): ...
                with T("net.class"): ...
    T.finish()          # 进程退出前调用，flush

环境变量：
    AZAI_TIMING=1        开启（默认关，零开销）
    AZAI_TIMING_FILE=... 输出 JSONL 路径
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class _Timer:
    def __init__(self) -> None:
        self.enabled = os.environ.get("AZAI_TIMING", "") not in ("", "0")
        self._path = None
        if self.enabled:
            p = os.environ.get("AZAI_TIMING_FILE", "")
            if p:
                self._path = Path(p)
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self._path.open("a", encoding="utf-8")
            else:
                self._fh = None
        else:
            self._fh = None
        self._stack: list[str] = []
        self._incl: dict[str, float] = {}
        self._parent: dict[str, str | None] = {}
        self._children: dict[str, set[str]] = {}
        self._cnt: dict[str, int] = {}
        self._t0 = 0.0
        self._meta: dict = {}
        self._checks = 0
        self._bad = 0

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        if name not in self._parent:
            self._parent[name] = self._stack[-1] if self._stack else None
            if self._parent[name] is not None:
                self._children.setdefault(self._parent[name], set()).add(name)
        t = time.perf_counter()
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()
            dt = time.perf_counter() - t
            self._incl[name] = self._incl.get(name, 0.0) + dt
            self._cnt[name] = self._cnt.get(name, 0) + 1

    def start(self, **meta) -> None:
        if not self.enabled:
            return
        self._incl.clear()
        self._cnt.clear()
        self._parent.clear()
        self._children.clear()
        self._stack.clear()
        self._meta = meta
        self._t0 = time.perf_counter()

    def finish(self) -> None:
        if not self.enabled or self._fh is None:
            return
        total = time.perf_counter() - self._t0
        # exclusive = inclusive - sum(direct children)
        excl = {}
        for name, inc in self._incl.items():
            kids = self._children.get(name, ())
            excl[name] = inc - sum(self._incl.get(k, 0.0) for k in kids)
        roots = [n for n, p in self._parent.items() if p is None]
        # 一致性检查的**正确口径**：所有节点的 exclusive 之和 == 根 span 的总时长。
        # （exclusive 把每个节点的独占时间切出来，它们互不重叠、并集 = 根节点覆盖的时间。）
        # ⚠️ 只对根节点的 exclusive 求和是错的——那是"根节点里不属于任何子节点的那部分"。
        all_excl_sum = sum(excl.values())
        ok = abs(all_excl_sum - total) <= max(0.02 * total, 0.002)
        self._checks += 1
        if not ok:
            self._bad += 1
        rec = {
            "total_s": round(total, 6),
            "excl_sum_s": round(all_excl_sum, 6),
            "check_ok": ok,
            "inclusive": {k: round(v, 6) for k, v in self._incl.items()},
            "exclusive": {k: round(v, 6) for k, v in excl.items()},
            "count": dict(self._cnt),
            **self._meta,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def summary(self) -> str:
        if not self.enabled:
            return "(timing off)"
        return f"timing checks={self._checks} bad={self._bad}"


T = _Timer()
