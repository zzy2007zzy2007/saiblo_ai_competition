"""rule_v4 的**禁闪电副本**（Phase 0，预注册 `docs/prereg_20261017_phase0_nolightning.md`）。

⚠️ **不改动** `code/test_match/rv4_pkg/` 原文件：这里用**子类覆盖**闪电路径
（① `get_lightning_bundle` 直接返回 None；② `choose_bundle` 兜底：若选出的束里含闪电，改取"不含闪电的最佳束"）。

用途：把"闪电被禁用"的子博弈做成**对手也禁闪电**的对称基准（同时会暴露 rule_v4 的退化程度 —— 它的策略本身就是围绕闪电写的）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RV4 = REPO / "code" / "test_match" / "rv4_pkg"
if str(RV4) not in sys.path:
    sys.path.insert(0, str(RV4))

import main as rv4main                      # noqa: E402  (rv4_pkg/main.py)
from ai import AI as Rv4AI                  # noqa: E402  (rv4_pkg/ai.py 的类名就是 AI)
from SDK.backend.model import OperationType  # noqa: E402

LIGHT = int(OperationType.USE_LIGHTNING_STORM)


def _has_light(bundle) -> bool:
    ops = getattr(bundle, "operations", ()) or ()
    return any(int(getattr(op, "op_type", -1)) == LIGHT for op in ops)


class NoLightningAI(Rv4AI):
    """rule_v4 的规则不变，只是**永远不放闪电**。"""

    def get_lightning_bundle(self, bundles):
        return None

    def choose_bundle(self, state, player, bundles=None):
        b = super().choose_bundle(state, player, bundles)
        if b is not None and _has_light(b):
            bs = bundles or self.list_bundles(state, player)
            cand = [x for x in bs if not _has_light(x)]
            if cand:
                return max(cand, key=lambda x: getattr(x, "score", 0.0))
        return b


if __name__ == "__main__":
    rv4main.main(NoLightningAI)
