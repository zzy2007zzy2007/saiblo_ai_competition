"""**教师臂** launcher：直接用官方启发式打分选束（≈ 我们蒸馏用的那个教师），用于归因消融。

动机（章程②：手写启发式只能当**诊断臂**）：我们的蒸馏教师 = SDK `ActionCatalog` 的评分反投影，
而官方 example 用的**也是**这套评分——只是它的选择规则是"跳过第 0 个束，在前 1..7 里取最高分"
（见 `Ant-Game/AI/ai_example.py`）。所以必须测一臂"**忠实教师**"来回答：
**我们的 0.80 到底是"蒸馏把启发式搬了进来"，还是"我们的管线（类/位置网 + MC 仲裁 + 搜索）真的加了价值"？**

本臂策略（无搜索、无网络）：
    best = max(bundles, key=lambda b: (b.score, -len(b.operations)))     # = 启发式的 argmax
`HEUR_SKIP_FIRST=1` ⇒ 模仿 example 的"跳过第 0 个束"（用于把两个差异拆开）。
`AI_BAN_LIGHTNING=1` ⇒ 滤掉含闪电的束（子博弈设定）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "Ant-Game", REPO / "Ant-Game" / "AI"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from AI.ai_example import ExampleAgent     # noqa: E402  官方样例（已证能过官方 launcher 的类型检查）
from AI.main import main as official_main  # noqa: E402
from SDK.backend.model import OperationType  # noqa: E402

LIGHT = int(OperationType.USE_LIGHTNING_STORM)
BAN = os.environ.get("AI_BAN_LIGHTNING", "0") == "1"
SKIP_FIRST = os.environ.get("HEUR_SKIP_FIRST", "0") == "1"


def _has_light(bundle) -> bool:
    ops = getattr(bundle, "operations", ()) or ()
    return any(int(getattr(op, "op_type", -1)) == LIGHT for op in ops)


class HeuristicArgmax(ExampleAgent):
    """忠实教师：在（可选的闪电过滤后的）候选里取 `(score, -len(ops))` 最大者。"""

    def choose_bundle(self, state, player, bundles=None):
        bs = bundles or self.list_bundles(state, player)
        if not bs:
            return None
        if BAN:
            keep = [b for b in bs if not _has_light(b)]
            if keep:
                bs = keep
        if SKIP_FIRST and len(bs) > 1:
            bs = bs[1:]
        best = max(bs, key=lambda b: (b.score, -len(getattr(b, "operations", ()) or ())))
        return best


if __name__ == "__main__":
    official_main(HeuristicArgmax)
