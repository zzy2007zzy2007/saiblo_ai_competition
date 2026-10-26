"""把探针改成用 facade 的后端对象（`st._g`）生成局面文本。

依据：`game_state_facade.py:152` 里 `self._g = native`，
而 `state_to_text(state)` 需要的是后端 GameState（有 `round_index`/`towers` 等属性）。
"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

if "st._g" not in t:
    t = t.replace("        text = state_to_text(st)",
                  "        text = state_to_text(st._g)   # facade 内部的后端 state", 1)
    p.write_text(t, encoding="utf-8")
    print("  已改用 st._g")
else:
    print("  已用 st._g")
