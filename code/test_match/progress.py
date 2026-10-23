"""测试进度输出（2026-10-23，用户要求：像 `az_progress_seed*.txt` 那样**逐回合实时**可看）。

输出到 `training_history/progress/`：

1. **逐局逐回合 trace**：`progress/<tag>_seed<seed>.txt`
   格式**完全照抄**用户给的参考文件（`az_bn_kgeo/.../az_progress_seed820000000.txt`）：
   ```
   round=0 HP0=50 HP1=50 G0=50 G1=50 P0=[]                               P1=[]
   round=2 HP0=50 HP1=50 G0=38 G1=53 P0=[BUILD(4,16)]                    P1=[]
   round=3 HP0=50 HP1=50 G0=54 G1=56 P0=[DOWNGRADE(id=0)]                P1=[]
   terminal winner=1 hp=[0, 2] rounds=441
   ```
   · `HP0/HP1` = 双方基地血量，`G0/G1` = 双方金币，取**本回合操作执行前**（= 决策时刻）的值；
   · `P0` 段补空格到 36 列再写 `P1`，与原文件对齐一致。

2. **整批进度**：`progress/<tag>_status.txt`（每完成一局覆盖重写）
   ```
   tag=VS_RU_newcfg  total=16  done=9  W=1 L=8  无效=0
   now=10:12:33  elapsed=36.0m  eta=28.0m
   inflight: 7 局（回合 210, 388, 120）
   最近完成: 9r 胜 9:5 (512 回合)
   ```

设计原则：**永不抛异常**（进度输出绝不能影响对局）；status 用 `.tmp` + `os.replace` 原子重写。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROGRESS_DIR = REPO / "training_history/progress"
P0_WIDTH = 36

# 操作类型 → 参考文件里的可读名字
_OP_NAME = {
    11: "BUILD", 12: "UPGRADE", 13: "DOWNGRADE",
    21: "LIGHTNING", 22: "EMP", 23: "DEFLECTOR", 24: "EVASION",
    31: "UP_SPEED", 32: "UP_ANT",
}


def _ensure_dir() -> Path:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return PROGRESS_DIR


def fmt_op(t: int, a0: int, a1: int) -> str:
    """单个操作 → `BUILD(4,16)` / `UPGRADE(id=3,tgt=2)` / `DOWNGRADE(id=0)` …"""
    t, a0, a1 = int(t), int(a0), int(a1)
    if t == 11:                      # BUILD_TOWER(x, y)
        return f"BUILD({a0},{a1})"
    if t == 12:                      # UPGRADE_TOWER(tower_id, target_type)
        return f"UPGRADE(id={a0},tgt={a1})"
    if t == 13:                      # DOWNGRADE_TOWER(tower_id)
        return f"DOWNGRADE(id={a0})"
    if t in (21, 22, 23, 24):        # 超级武器(x, y)
        return f"{_OP_NAME[t]}({a0},{a1})"
    if t in (31, 32):                # 基地升级，无位置
        return _OP_NAME[t]
    return f"OP{t}({a0},{a1})"


def fmt_ops(ops) -> str:
    if not ops:
        return "[]"
    try:
        return "[" + ", ".join(fmt_op(*o) for o in ops) + "]"
    except Exception:
        return "[]"


def _fmt_pair(v) -> str:
    if v is None:
        return "None"
    try:
        return f"[{int(v[0])}, {int(v[1])}]"
    except Exception:
        return "None"


class GameTrace:
    """逐回合 trace 写入器（追加 + 每行 flush，便于 `Get-Content -Tail` / `tail -f`）。"""

    def __init__(self, tag: str, seed):
        self.path = _ensure_dir() / f"{tag}_seed{seed}.txt"
        try:
            self._fh = self.path.open("w", encoding="utf-8")
        except Exception:
            self._fh = None

    def round(self, rnd: int, p0_ops, p1_ops, hp=None, coins=None) -> None:
        """写一回合。`hp`/`coins` = 本回合操作执行前（决策时刻）的双方值。"""
        if self._fh is None:
            return
        try:
            hp0 = int(hp[0]) if hp else -1
            hp1 = int(hp[1]) if hp else -1
            g0 = int(coins[0]) if coins else -1
            g1 = int(coins[1]) if coins else -1
            p0s = "P0=" + fmt_ops(p0_ops)
            p1s = "P1=" + fmt_ops(p1_ops)
            _head = f"round={rnd} HP0={hp0} HP1={hp1} G0={g0} G1={g1} "
            _pad = f"{p0s:<{P0_WIDTH}}" if len(p0s) < P0_WIDTH else p0s + " "
            self._fh.write(_head + _pad + p1s + "\n")
            self._fh.flush()
        except Exception:
            pass

    def terminal(self, winner, hp, rounds) -> None:
        if self._fh is None:
            return
        try:
            w = "None" if winner is None else int(winner)
            self._fh.write(f"terminal winner={w} hp={_fmt_pair(hp)} rounds={int(rounds)}\n")
            self._fh.flush()
        except Exception:
            pass
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def write_status(tag: str, *, total: int, done: int, wins: int, losses: int, invalid: int = 0,
                 started: float | None = None, inflight_rounds: list[int] | None = None,
                 recent: list[str] | None = None) -> None:
    """原子重写整批进度文件。任何异常都吞掉。"""
    try:
        d = _ensure_dir()
        now = time.time()
        head = f"tag={tag}  total={total}  done={done}  W={wins} L={losses}  无效={invalid}"
        stamp = f"now={time.strftime('%H:%M:%S', time.localtime(now))}"
        if started:
            elapsed = (now - started) / 60.0
            stamp += f"  elapsed={elapsed:.1f}m"
            if done > 0:
                stamp += f"  eta={elapsed / done * (total - done):.1f}m"
        lines = [head, stamp]
        if inflight_rounds:
            rs = sorted(inflight_rounds, reverse=True)
            lines.append(f"inflight: {len(rs)} 局（回合 " + ", ".join(str(r) for r in rs[:10]) + "）")
        for r in (recent or [])[:6]:
            lines.append(f"最近完成: {r}")
        tmp = d / f"{tag}_status.txt.tmp"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, d / f"{tag}_status.txt")
    except Exception:
        pass
