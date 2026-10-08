"""官方 `ExampleAgent` 的 launcher（Phase 0，预注册 `docs/prereg_20261017_phase0_nolightning.md`）。

官方样例是 `class` 形式（`Ant-Game/AI/ai_example.py` 的 `ExampleAgent(BaseAgent)`），
用官方自己的 `AI/main.py:main(ai_cls=...)` 挂进我们的 ladder（= 与 rule_v4 同一套 `MatchSession`/`ProtocolSession` 协议）。

`AI_BAN_LIGHTNING=1` 时，在 `choose_bundle` 里**先滤掉含闪电的候选束**（用于对称的子博弈基准）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "Ant-Game", REPO / "Ant-Game" / "AI"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from AI.ai_example import ExampleAgent        # noqa: E402  官方样例
from AI.main import main as official_main     # noqa: E402  官方 launcher
from SDK.backend.model import OperationType   # noqa: E402

LIGHT = int(OperationType.USE_LIGHTNING_STORM)
BAN = os.environ.get("AI_BAN_LIGHTNING", "0") == "1"


def _has_light(bundle) -> bool:
    ops = getattr(bundle, "operations", ()) or ()
    return any(int(getattr(op, "op_type", -1)) == LIGHT for op in ops)


class MaybeNoLightningExample(ExampleAgent):
    def choose_bundle(self, state, player, bundles=None):
        if BAN:
            bs = bundles if bundles is not None else self.list_bundles(state, player)
            keep = [b for b in bs if not _has_light(b)]
            if keep:
                return super().choose_bundle(state, player, keep)
        return super().choose_bundle(state, player, bundles)


if __name__ == "__main__":
    official_main(MaybeNoLightningExample)
