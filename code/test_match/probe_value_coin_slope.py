"""测 **价值网对金币的边际价值** `∂V/∂coins`（用于第 8 轮的"只给资源定价"版评分）。

为什么：`MC_SCORE=delta/value_only` 的冒烟显示"塔数增加会让 V 上升" ⇒ V 的**全差分**不是资源定价器。
本脚本用**直接干预**（给同一局面加/减金币，其余不动）测出干净的 `∂V/∂coins`，
这样"金币成本项"的权重就是**量出来的**，不是手设的。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.features import FeatureExtractor                      # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade        # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt        # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "M6cROOT_128"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/posnet_A_k5_m32.pt"
    txt = (REPO / "match_results" / "ladder_logs" / tag / f"{seed}.log").read_text(
        encoding="utf-8", errors="replace")
    rounds: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rounds.setdefault(int(m.group(1)), {})[int(m.group(2)[1])] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]

    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(ckpt, feat)
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    deltas, samples = [], 0
    for r in range(0, 320):
        for p in (0, 1):
            for (t, a0, a1) in rounds.get(r, {}).get(p, []):
                st.apply_operation_list(p, [Operation(OperationType(t), a0, a1)])
            if p == 1:
                st.advance_round()
        if st.terminal or r % 25:
            continue
        for delta in (+30, -30):
            c = st.clone()
            try:
                c.coins[0] = int(c.coins[0]) + delta
                c._cache.clear()          # ⚠️ 必须清缓存，否则读到旧视图（第 5 轮踩过）
                v1 = float(net_fn(c, 0)["value"])
            except Exception as e:
                print(f"  干预失败: {type(e).__name__}: {e}"); continue
            c0 = st.clone()
            v0 = float(net_fn(c0, 0)["value"])
            deltas.append((delta, v1 - v0, int(st.coins[0])))
            samples += 1
    if not deltas:
        print("没采到样本（干预不可行）"); return 1
    d30 = [d for dd, d, _ in deltas if dd == +30]
    dneg = [d for dd, d, _ in deltas if dd == -30]
    print(f"样本 {len(deltas)} 个干预点（±30 金币，其余状态不动）")
    print(f"  +30 金币 ⇒ ΔV 均值 {np.mean(d30):+.5f}（中位 {np.median(d30):+.5f}）")
    print(f"  −30 金币 ⇒ ΔV 均值 {np.mean(dneg):+.5f}（中位 {np.median(dneg):+.5f}）")
    slope = (np.mean(d30) - np.mean(dneg)) / 60.0
    print(f"  ⇒ ∂V/∂coins ≈ {slope:+.6f} /金币  （若 ≈0 ⇒ 价值网确实不把金币当资源）")
    print(f"  ⇒ 即：30 金币 ≈ {30*slope:+.5f} 个价值单位（HP 差单位下 = {30*slope*20:+.3f} HP）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
