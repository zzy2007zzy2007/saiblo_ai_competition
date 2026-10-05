"""离线复现 M6（mc_light）的崩溃：在"金币够放闪电"的局面直接跑 MC 比较，看异常是什么。"""
from __future__ import annotations

import sys
import traceback
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


def main() -> int:
    st = GameStateFacade.initial(seed=7, cold_handle_rule_illegal=True)
    # 空转到金币够
    r = 0
    while (not st.terminal) and st.coins[0] < 95 and r < 80:
        st.advance_round()
        r += 1
    print(f"准备局面: round={getattr(st, 'round_index', -1)} coins={st.coins} hp={[b.hp for b in st.bases]} terminal={st.terminal}")

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt("training_history/vprior/posnet_A_k5_m32.pt", feat)

    from my_ai.az_intent.mc_class import _decode_class_op, mc_choose_class
    from my_ai.decoder import make_class_mask, make_position_masks

    player = 0
    out = net_fn(st, player)
    hl = np.asarray(out["head_logits"][0], dtype=np.float32)
    am = np.asarray(out["action_map"], dtype=np.float32)
    pm = make_position_masks(st, player, intent_decoding=True)
    cm = make_class_mask(st, player, position_mask=pm, intent_decoding=True)
    print("可放闪电?", _decode_class_op(hl, am, cm, pm, st, player, 17) is not None,
          " 可建塔?", _decode_class_op(hl, am, cm, pm, st, player, 0) is not None)
    try:
        c, sc = mc_choose_class(net_fn, st, player, [17, 23], hl, am, cm, pm, max_rounds=100)
        print("MC 结果: class=", c, " scores=", sc)
    except Exception:  # noqa: BLE001
        print("!! MC 抛异常：")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
