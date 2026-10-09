# antwar2 — AntWar2 / 蚁洋陷役2

第 30 届 Saiblo 游戏，19×19 offset-hex 网格上的双人回合制塔防对抗。

本目录是**以游戏为组织维度**的完整 GamePack，也是新游戏接入的参考范式。

## 结构

| 路径 | 内容 |
| --- | --- |
| `game.yaml` | 游戏元信息 / 角色 / 别名 / 赛道索引 |
| `rules.md` | 完整规则 |
| `decision_space.yaml` | 决策空间 + 合法动作语义（IG/KL 前提） |
| `replay_format.md` | 回放字段说明（承 `docs/replay-format-spec.md`） |
| `evaluator/` | 可运行对战器（满足 `docs/evaluator-contract.md`，真跑验证 COMPLETE） |
| `plugin.py` | 暴露 `PLUGIN`，被框架注册表发现 |
| `players/pool/` | 完整 population pool + `manifest.tsv`（rank/Elo/split） |

## 对战器

`evaluator/` 从 `games/antwar2/backend/logic/` 编译官方 C++ 后端，与
`players/pool/` 中带 rank 的人类选手对战，解析回放后返回契约要求的 `EvaluateResult`。
B (HL) / C (RL) 通过 A 的统一入口 `evaluate("antwar2", ...)` 复用它，不各自维护裁决逻辑。

- 规则权威：内容哈希固定的官方 **C++ 后端**。
- 候选接口权威：公开 **Python SDK**（`AI.choose_operations` 入口，见 `decision_space.yaml`）。

## 选手池

- `players/pool/`：AntWar2 完整选手池（各提交版本为独立测例）。
- `players/manifest.tsv`：统一登记 rank/Elo/username/split。
- `players/interface_candidates/`：早期从后端目录清理迁入的一批接口候选代码，见其 `README.md`。
