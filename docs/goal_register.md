# 结论寄存器（跨会话防绕圈用）

> 2026-09-23 建。**这是给长任务用的"短到能整篇读完"的结论表**——`docs/experiment_log.md` 是日志
> （越长越读不完），本文件只放**已定论**的东西，一行一条。
> **每轮开始先读本文件**；**每轮结束必须新增或更新至少一行**；**想重开一个"已排除"的方向，
> 必须先写明"为什么这次的证据会与当初不同"，否则跳过**。（规则见 `docs/goal_longrun_plan.md` §4）

## 当前方针（三行，随时更新）

- **在攻**：方法层假设 H1–H4（`goal_longrun_plan.md` §2.1）。短期目标由自己提，但**先写假设 + 先写判据**。
- **为什么**：`B0_step0_smoke` 显示我们在实战里几乎不建经济（368 回合只出招 16 次、建塔 6 座，对手 151 次/76 座）。
  **按用户定的立场，这是"训练方法学不会"，不是"要给它加建塔先验"** ⇒ 观测"建塔数"当**证据**，不当目标。
- **下一步**：填 B 基线（`B1/B2` 跑中）→ 在钉死协议下重测 `vs rule_v4` 当前基线 → 挑一个假设做便宜实验。

## 已证实

| 结论 | 支撑（run / 文件）|
|---|---|
| 我们的 C++ 引擎 **== 官方二进制**（精确字段逐回合一致）| `test_official_consistency.py --rounds 30`；`test_official_ops.py` 14/14 |
| `resolve_turn` ≡ `apply_operation_list`+`advance_round`（同引擎下逐字段相同）| 三路径对照 P2=P3 8/8（`P2_rv4_self_python_engine` / `P3_rv4_self_resolve_turn`）|
| **协议层不改变对局**（进程内直连 vs 协议 bridge 等价）| `ctrl_rv4_self_inprocess` vs `A1_rv4_self` 16/16 逐字段一致 |
| `rule_v4` **确定性**（镜像两局读数逐字段相同）| `A1_rv4_self` 8/8 对 |
| **冠军/亚军不可复现** ⇒ **只读聚合、不读逐 seed** | 四跑（`DET_seed11_run1`/`DET2_*`/`DET3_seed11_hexdump`）：三次同读数、第四次不同棋盘 |
| `rule_v4` vs 冠军 **0/16**、vs 亚军 **0/16**，且**是被打爆基地**（311–467 回合）| `A2a_rv4_vs_champion` / `A2b_rv4_vs_runnerup` |
| 历史读数**全部**在 C++ 引擎上（内部自洽）| `_tmp_audit_eval_cmds.py`：eval.py 14/14、svs 6/6 带 flag；两匹配工具默认 True |
| 判词是**官方 5 级级联**；**官方从不判平局**；异常截断的局现标 `INVALID` 不计分 | `cpp_engine/src/game.cpp: judge_winner()`；bridge RESULT 行（commit `57724c0`）|
| 我们的协议壳在**长局可靠**：367 回合 `mismatch=0`、`ill=0`、不提前退出 | `B0_step0_smoke` |
| `Ant-Game/game/output/main.exe` **不是**官方预编译产物，是**我们自己编的** | 子模块 `.gitignore:38` 忽略 `game/output/`；子模块仅 2 处改动 = 我们的二进制模式补丁 |

## 已排除（**别再试**）

| 结论 / 排除的理由 | 支撑 |
|---|---|
| **用 Python SDK 引擎测强度**：它漂移 —— 会**接受官方拒绝的操作**、地图合法性不同 | `test_consistency.py --rounds 120` → **0/6 seeds 一致**（最早第 16 回合分叉）|
| **跨引擎比较**：同 seed 只换引擎，`rule_v4` 自对弈 **4/8 个 seed 胜负翻转** | 三路径对照 P1 vs P2/P3 8/8 都不同 |
| **单次 / 单 seed / 单臂读数下结论** | 单臂排名在轮次间完全不可复现（项目史）；`DET*` 四跑 |
| **用短回合上限判断"AI 是否弃权"**（如 8 回合）| 08-09 那次得出过错误结论；`rule_v4` 前 ~30 回合本来就在攒钱（`ctrl_rv4_self_orig_runner` 首次出招在 r42）|
| **追"引擎版本差异"的根因**（三份 `game/src` 逐字节相同；Python 引擎是漂的那边，已定论）| `other_versions_ai_sources` 时期的比对 + 本轮 `test_consistency.py` |
| **把"建经济"当修复方向硬修**（那是症状；用户立场：应修训练方法/架构）| `goal_longrun_plan.md` §2；`B0_step0_smoke` 的机制读数 |

## 待验证

| 问题 | 计划 / 状态 |
|---|---|
| **我们 vs 冠军 / vs 亚军** 的基线（多少 / N） | `B1_us_vs_champion`、`B2_us_vs_runnerup`（跑中）|
| **我们 vs `rule_v4` 在当前钉死协议下的基线**（历史 46.9% 是旧协议，不能直接比） | 待跑（rule_v4 一局 1.5–2 分钟，很便宜）|
| `rule_v4` 每局**建塔数**（机制基线，用作"方法是否修好"的参照）| 待跑 |
| 方法层假设 **H1**（信用分配 / 96% HOLD 标签不平衡）| 待设计便宜判据 |
| 方法层假设 **H2**（价值标签视野 `tau`）| 已有先例 tau20→50 让 search-vs-search 43.75%→81.2% |
| 方法层假设 **H3**（策略表达不出多头 bundle）| 待设计 |
| 方法层假设 **H4**（类轴 masked/HOLD 结构）| 待设计 |
