"""P3：rule_v4 vs rule_v4 走「官方 SDK 引擎 + resolve_turn」这条路。

用途（2026-09-22，用户提出）：隔离并量化"引擎 / 回合结算 API"这两个变量。
三路径同 seed 7–14、两侧都是原版 rule_v4：

  P1  rule_v4_lightning_match.py --null                 C++ facade + apply_operation_list
  P2  同上 + --python-engine                            Python SDK + apply_operation_list
  P3  本脚本                                             Python SDK + resolve_turn

关键点：**故意不用 `code/test_match/run_match.py`**，因为它的 `GameState.initial()` 没传
`cold_handle_rule_illegal`，而 P1/P2 走的是 `cold=True` ⇒ 用它会把"引擎/API"和"非法操作处理"
两个变量一起改掉，结论就说不清了。这里照抄 `vs_rule_v4.py` 的循环（同样 `cold=True`）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ORIG = REPO / "其他版本ai" / "rule_v4"
for _p in (REPO / "Ant-Game", REPO / "code", str(ORIG)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SDK.backend.engine import GameState           # noqa: E402
from SDK.utils.constants import MAX_ROUND          # noqa: E402


def _orig_ai_class():
    """原封不动的 rule_v4 AI；只把 log() 换成空实现（否则写 GB 级 ai_decisions.log）。"""
    spec = importlib.util.spec_from_file_location("rv4_orig_ai_p3", ORIG / "ai.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rv4_orig_ai_p3"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.RuleBasedAI.log = staticmethod(lambda message: None)
    return mod.AI


def main() -> int:
    RV4 = _orig_ai_class()
    seeds = [int(a) for a in sys.argv[1:]] or list(range(7, 15))
    print("[engine] Python SDK（SDK.backend.engine.GameState）+ resolve_turn + cold=True", flush=True)
    for seed in seeds:
        a0, a1 = RV4(seed=seed), RV4(seed=seed)
        st = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        rnd = 0
        for rnd in range(MAX_ROUND):
            if st.terminal:
                break
            u = a0.choose_operations(st, 0)
            v = a1.choose_operations(st, 1)
            st.resolve_turn(u or [], v or [])
        h0, h1 = int(st.bases[0].hp), int(st.bases[1].hp)
        w = st.winner
        print("[game] seed=%d rounds=%d winner=%s hp=%d,%d coins=%d,%d"
              % (seed, rnd, w, h0, h1, int(st.coins[0]), int(st.coins[1])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
