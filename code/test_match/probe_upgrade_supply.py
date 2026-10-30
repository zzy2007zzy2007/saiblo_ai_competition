"""离线测量：**升级在真局里到底有多少回合"买得起且有条目"**（2026-10-31）。

动机：`AZAI_CAND_DECODE=split` 的核心指标是"含升级的回合占比"。但这个占比有**硬上界**：
升级要 60–120 金币，而我方金币常年 13–56 ⇒ 某些回合**根本没有合法升级**（位置掩码为空）。
所以要先分清两种情况：
  (a) 大量回合"升级合法但生成器没提出来" ⇒ 旧 decode 确实漏了 ⇒ split 应该显著提高占比；
  (b) 绝大多数回合"升级就不合法（买不起/没有可升的塔）" ⇒ 占比上界本来就很低 ⇒
      split 提高不了多少，是**经济限制**，不是 decode 的问题。

本脚本回放一局真实对局（`--trace-ops`），对**每个决策点**算：
* `n_upg_arms` = split 语义下"有合法升级格"的通道数（通道 1-15 里 mask ∩ 我方塔格 非空）
* `n_build_cells` = split 语义下通道 0 的合法空格数
* `n_upg_ops` = 允许的**不同**升级 op 条数（(tower, step) 去重后）

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_upgrade_supply.py --tag UNIFIED_CAND_FULL --seed 8
```
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SDK.backend.model import Operation            # noqa: E402
from SDK.utils.constants import OperationType      # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.decoder import make_position_masks, upgrade_step, CHANNEL_TO_TOWER_TYPE  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pct(k: int, n: int) -> str:
    return f"{100.0*k/max(1,n):5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="UNIFIED_CAND_FULL")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--player", type=int, default=0)
    args = ap.parse_args()

    rsc = _load("_rsc_upg", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        print(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行")
        return 2
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    n_dec = 0
    n_upg_any = 0            # 至少有 1 个合法升级 op 的决策数
    n_upg_arms_any = 0
    n_build_any = 0
    hist_upg_ops: dict = {}
    hist_upg_arms: dict = {}
    coins_when_upg: list = []
    coins_all: list = []
    for r in range(0, max(rounds) + 1):
        me = int(args.player)
        pm = make_position_masks(st, me, intent_decoding=False)
        my_cell = np.zeros_like(pm[0], dtype=bool)
        for t in st.towers_of(me):
            my_cell[int(t.x), int(t.y)] = True
        n_arms = 0
        upg_ops = set()
        for ch in range(1, 16):
            cells = pm[ch] & my_cell
            if not cells.any():
                continue
            n_arms += 1
            for x, y in zip(*np.nonzero(cells)):
                tw = st.tower_at(int(x), int(y))
                if tw is None:
                    continue
                step = upgrade_step(tw.tower_type, CHANNEL_TO_TOWER_TYPE[ch])
                if step is None:
                    continue
                op = Operation(OperationType.UPGRADE_TOWER, int(tw.tower_id), int(step))
                try:
                    if st.can_apply_operation(me, op, ()):
                        upg_ops.add((int(tw.tower_id), int(step)))
                except Exception:      # noqa: BLE001
                    pass
        n_build = int((pm[0] & (~my_cell)).sum())
        n_dec += 1
        n_upg_any += int(len(upg_ops) > 0)
        n_upg_arms_any += int(n_arms > 0)
        n_build_any += int(n_build > 0)
        hist_upg_ops[len(upg_ops)] = hist_upg_ops.get(len(upg_ops), 0) + 1
        hist_upg_arms[n_arms] = hist_upg_arms.get(n_arms, 0) + 1
        coins_all.append(int(st.coins[me]))
        if upg_ops:
            coins_when_upg.append(int(st.coins[me]))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
        st.advance_round()

    print(f"[sup] 局面 = {args.tag}/seed{args.seed} player={args.player}  决策点 = {n_dec}")
    print(f"[sup] 金币：中位 {sorted(coins_all)[len(coins_all)//2]}  "
          f"min {min(coins_all)} max {max(coins_all)}")
    print("")
    print("## split 语义下『升级』的**供给上界**（= 位置掩码 + can_apply 之后真实可发的升级 op）")
    print(f"  至少 1 条合法升级 op 的决策：{n_upg_any}/{n_dec} = {_pct(n_upg_any, n_dec)}"
          f"   ← **含升级回合占比的理论上界**")
    print(f"  至少 1 个可升级通道(1-15)的决策：{n_upg_arms_any}/{n_dec} = {_pct(n_upg_arms_any, n_dec)}")
    print(f"  至少 1 个可建空格(通道 0)的决策：{n_build_any}/{n_dec} = {_pct(n_build_any, n_dec)}")
    print(f"  合法升级 op 条数分布（条数: 决策数）：{dict(sorted(hist_upg_ops.items()))}")
    print(f"  可升级通道数分布（通道数: 决策数）：{dict(sorted(hist_upg_arms.items()))}")
    if coins_when_upg:
        print(f"  有合法升级时的金币：中位 {sorted(coins_when_upg)[len(coins_when_upg)//2]}  "
              f"min {min(coins_when_upg)} max {max(coins_when_upg)}")
    print("")
    print("判读：")
    print("  * 若『至少 1 条合法升级 op』的占比 **明显高于** 旧 decode 冒烟实测的『含升级回合占比』")
    print("    ⇒ 旧 decode 确实在漏升级（生成器问题）⇒ split 应当显著提高该占比。")
    print("  * 若两者接近 ⇒ 升级本来就被**经济**卡住（买不起），split 提高不了占比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
