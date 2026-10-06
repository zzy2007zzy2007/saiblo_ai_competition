"""给 mc_quiet 加**决策可观测性**：把每次 MC 的候选、分数、所选类写 JSONL。

用法（独立脚本，不改 bundle_mcts 主逻辑）：
  python -u code/my_ai/az_intent/patch_mc_dump.py     # 幂等地打补丁
补丁内容：
  * `bundle_mcts.py` 的 mc_quiet 分支里，选完类之后，若环境变量 `AZAI_MC_DUMP` 指向一个文件，
    就追加一行 JSON：{round, player, coins, hp, cands, scores, chosen}。
  * 这是**只读观测**（不改变任何决策）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P = REPO / "code" / "my_ai" / "az_intent" / "bundle_mcts.py"

OLD = """                        self.mc_last_scores = _sc
                        self.mc_taken += 1
                        pinned = [int(_c)] * len(head_logits_list)"""
NEW = """                        self.mc_last_scores = _sc
                        self.mc_taken += 1
                        pinned = [int(_c)] * len(head_logits_list)
                        # 观测（可选，只读）：把这次 MC 的候选/分数/选择写成 JSONL
                        _dump = _os.environ.get("AZAI_MC_DUMP", "")
                        if _dump:
                            try:
                                import json as _json
                                with open(_dump, "a", encoding="utf-8") as _fh:
                                    _fh.write(_json.dumps({
                                        "round": _rnd, "player": int(node.player),
                                        "coins": [int(x) for x in node.state.coins],
                                        "hp": [int(b.hp) for b in node.state.bases],
                                        "cands": [int(x) for x in _cands],
                                        "scores": {str(k): round(float(v), 5) for k, v in _sc.items()},
                                        "chosen": int(_c),
                                    }, ensure_ascii=False) + "\\n")
                            except Exception:  # noqa: BLE001
                                pass"""


def main() -> int:
    t = P.read_text(encoding="utf-8")
    if "AZAI_MC_DUMP" in t:
        print("已打过补丁（含 AZAI_MC_DUMP），跳过")
        return 0
    if OLD not in t:
        print("!! 锚点没找到，未改动")
        return 1
    t = t.replace(OLD, NEW, 1)
    if "\nimport os as _os\n" not in t:
        t = t.replace("\nimport numpy as np\n", "\nimport os as _os\n\nimport numpy as np\n", 1)
    P.write_text(t, encoding="utf-8")
    print("补丁完成：mc_quiet 现在支持 AZAI_MC_DUMP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
