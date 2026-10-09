# omegafantasy 公开的历届比赛 AI

> 整理时间：2026-10-10
> 来源：GitHub 账号 [`omegafantasy`](https://github.com/omegafantasy)（= 本目录 `saiblo-30th-AI` 的作者），通过 `https://api.github.com/users/omegafantasy/repos` 枚举其公开仓库后逐个克隆。

## 为什么这件事重要

AAArena（`其他版本ai/AAArena/`）的放出规则是 **`rank > 8 且名次为偶数`** —— 也就是说**冻结榜前 8 名一律扣留，一份源码都不给**（见 [`../docs/other_versions_ai_sources.md`](../docs/other_versions_ai_sources.md) 与 AAArena 的 `validation/player-publication.json`）。

而 omegafantasy 是这几届比赛里长期霸榜的选手（AAArena 的 antwar 回放里，人类一方就是 **`rank01__omegafantasy__Magica__v13`** —— 第 27 届第 1 名）。**他把自己在多个游戏里的 AI 直接开源在了 GitHub 上**，正好补上了 AAArena 扣掉的那一块。

→ 结果是：**AAArena 不给你最强对手，但最强对手自己公开了源码。**

## 已克隆的 9 个仓库（都放在 `其他版本ai/` 下）

| 目录 | 对应 AAArena 游戏 | 届次 | HEAD 日期 | 体积 | 文件特征（用于确认身份） |
|---|---|---|---|---|---|
| `Miracle-AI/` | Miracle | 24 | 2022-03-24 | 1,160 KB | `ai.cpp` / `calculator.cpp,h` / `ai_client.h` |
| `AquaWar-AI/` | AquaWar | 25 | 2022-03-24 | 529 KB | `Action.cpp,hpp` / `sdk/` |
| `LostSpace-AI/` | LostSpace | 25 | 2022-03-24 | 479 KB | `mapconf2.map`（就是第一轮 AST 报告里被误解析那份地图）/ `space.cpp` / `jsoncpp/` |
| `SnakeGo-AI/` | SnakeGo | 26 | 2022-03-24 | 190 KB | `adk.hpp`（AAArena 的 snakego 也是这个 SDK 头）/ `main.cpp` |
| `ANTWar-AI/` | AntWar | 27 | 2023-05-19 | 424 KB | `include/` / `main.cpp` / `Doxyfile` |
| `Generals-AI/` | Generals | 28 | 2024-05-15 | 1,435 KB | `include/` / `main.cpp` |
| `RollMan-AI/` | Rollman | 29 | 2025-05-10 | 1,389 KB | **`main_pac.cpp` + `main_ghost.cpp`**（非对称双角色，正好对上）/ `ai_to_judger.h` / `core/` / `utils/` |
| `SnowCup-AI/` | ⚠️ **不在 AAArena 的 12 个里** | ? | 2022-05-25 | 1,189 KB | `contestant_code.cpp` / `egg_sdk.cpp,h`（"雪杯"） |
| `stupid_car/` | ❌ **与比赛无关** | – | 2023-12-24 | 416 KB | `bmm150.c`（磁力计驱动）/ `Emakefun_MotorDriver.cpp` / `Buzzer.cpp` —— 是嵌入式小车代码 |

已有的第 30 届（AntWar2）在他这里没有独立仓库（就是本目录的 `saiblo-30th-AI`，里面含两个游戏）。

## 同账号下**未克隆**的仓库（判断为无关，供参考）

| 仓库 | 判断 |
|---|---|
| `OpenRDW2` / `ControllableLandscape` / `ControllableLandscape_UnityClient` / `Bypassing-Obstacles` | Unity/C# 或其他研究项目 |
| `ScapePhotographer_MM26` | 另一个 Python 项目 |
| `claude-code` | 大概是 Claude Code 的 fork |
| `Jack-everything` | Python 项目 |
| `omegafantasy.github.io` | 个人主页（173 MB，含 GitHub Pages 历史） |

## 怎么用

1. **当本地强对手池**：AAArena 本来只给你 91 个中段对手（第 10~190 名的偶数名次）；这 7 个补上了"顶"那一端。合起来就是一个**近似完整的梯子**（缺的是第 2~8 名的其他人）。
2. **当策略参考**：尤其是 `RollMan-AI` 的 `main_pac.cpp` / `main_ghost.cpp` 把"非对称双角色"拆成两份独立程序，`LostSpace-AI` 带着完整 `mapconf2.map`，`ANTWar-AI` / `Generals-AI` 是那两届冠军的实现。
3. **注意接口不同**：这些是**原始赛事接口**（C++ 编译成 `.so` / stdio 协议），不是 AAArena 的 Python SDK。要接进 AAArena 的 evaluator 需要适配层。

## ⚠️ 未验证 / 待办

- [ ] **没有验证这些仓库能否编译、能否跑起来**（只是克隆并核对了文件特征）
- [ ] 届次对应是根据"游戏名 + 文件特征 + 推送时间"推断的，**没有逐一比对 AAArena 的对局记录**
- [ ] 尚未按本仓库约定登记为**真子模块**（9 个仓库，需要你确认；参考 AAArena 那次是 `git submodule add`）
- [ ] `SnowCup-AI` 是哪个比赛的"雪杯"、`stupid_car` 是否要删 —— 待你决定

## 合规提醒（同 AAArena 那份）

这些代码是原作者**自己公开**的，但"公开"不等于"可以搬进你的参赛提交"。AAArena 给自家 AI Agent 定的协议里明确有 `✗ No reading rivals' code`。
**当对手池 / 读思路** ≠ **直接搬进 submission**。后者请先确认比赛规则。
