# 计划：亚军自对弈采集 + 桥接 raw-state dump

> 2026-10。目标：给"亚军 value 函数蒸馏"提供**部署分布**（亚军自己的对局）+ **精确标签**（亚军 `evaluate_terminal`）。

## 背景

- 现有数据源（`训练/评测`用的简单策略）分布偏：塔少、不用闪电、局长短。
- 亚军 exe（`ai_cpp_lure_v4.exe`）**能跑了**，但必须把 `C:\mingw64\bin` **前置到 PATH**（否则
  `D:\anaconda3\Library\mingw-w64\bin` 的 DLL 版本不符，退出码 `0xC0000139`）。
- 速度实测：**亚军自对弈 ~2.0–2.3 s/回合**，M6c-ROOT ~4.4 s/回合 ⇒ **亚军快约 2×**。

## 改动：桥接加 raw-state dump（默认关）

`code/test_match/match_via_sdk.py` 的 `run_match` 里，每回合开头（AI 决策前）加：

- 环境变量 `MVS_RAW_STATE_DIR=<dir>`（默认空 ⇒ 行为逐位不变）。
- 每回合、每个玩家写一条定长记录到 `raw_seed<seed>.bin`：
  ```
  board: 28*19*19 个 float16
  stats: 42 个 float16
  player: 1 个 uint8
  round: 1 个 int16
  value: 1 个 float32   # = 亚军 evaluate_terminal(state, player)
  ```
- `value` 用 `code/my_ai/distill_top2/lure_value_eval.py`（亚军的 Python 复刻），状态取桥接的
  `GameStateFacade`（权威 C++ 引擎影子）。

⚠️ 只在 `MVS_RAW_STATE_DIR` 非空时才 import/写盘 ⇒ **对现有所有实验零影响**。

## 流程

1. 跑亚军自对弈（`_tmp_ladder.py --ai0=ai_cpp_lure_v4.exe --ai1=...`，PATH 前置 mingw）。
2. 后处理 `postprocess_bridge_raw.py`：读 `.bin` → `(board, stats, value, seed, round, player)` → npz。
3. 用 `train_value_distill.py` 训练（train/val 按 seed 切）。

## 不做的

- 不改亚军代码；不动任何现有实验；不把亚军行为当"策略蒸馏"目标（只蒸它的**价值函数**）。