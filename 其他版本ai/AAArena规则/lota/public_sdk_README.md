# LOTA public AI9 SDK

选择英雄，控制兵线、技能、成长与复活，摧毁对方防御。

## Files

- `ai.cpp`: editable starter policy.
- `sdk-src/`: public game SDK headers and implementation.
- `include/`: platform headers needed by the public sample.
- `Makefile`: local compile check that produces `player.so`.

Only `ai.cpp` is submitted as strategy; the evaluator rebuilds it against the
complete private game SDK.
