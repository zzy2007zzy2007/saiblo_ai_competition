# AAArena 12 个游戏的规则文档（本地副本）

> 建立时间：2026-10-09
> 来源：从 `其他版本ai/AAArena/assets/<game>.tar.gz` 里解出来的**副本**（AAArena 本身是子模块，这里的文件是为了不每次解包就能直接翻）

## 这里有什么

每个游戏一个子目录，对应 AAArena 里的 `games/<game>/`：

| 文件 | 来源路径 | 内容 |
|---|---|---|
| `rules.md` | `games/<game>/rules.md` | **规则正文** |
| `game.yaml` | `games/<game>/game.yaml` | 元信息：届次、别名、角色、赛道、规则/决策空间文件名 |
| `decision_space.yaml` | `games/<game>/decision_space.yaml` | 决策空间契约 + 合法动作语义（入口符号、原子动作码、参数） |
| `replay_format.md` | `games/<game>/replay_format.md` | 回放字段说明 |
| `replay_skill.md` | `games/<game>/replay_skill.md` | 怎么把回放读成可证伪的策略假设 |
| `README.md` | `games/<game>/README.md` | GamePack 目录说明 |
| `public_sdk_README.md` | `games/<game>/public_sdk/README.md` | 公开 SDK 说明 |
| `public_sdk_INTERFACE.md` | `games/<game>/public_sdk/INTERFACE.md` | 接口文档（legacy 游戏用） |
| `backend_RULES.md` | `games/<game>/backend/RULES.md` | 后端自带的规则（**只有 antwar 有**，30 KB，最详细） |
| `known_issues.md` | `games/<game>/known_issues.md` | 已知问题（**只有 antwar 有**） |

文件名里的下划线是路径扁平化：`public_sdk/README.md` → `public_sdk_README.md`。

## 完整度分两档

| 游戏 | 届次 | `rules.md` | `decision_space.yaml` | 完整度 |
|---|---|---:|---:|---|
| generals | 28 | 8,028 B / 135 行 | 6,012 B | ✅ 完整 |
| snakego | 26 | 6,425 B / 115 行 | 4,436 B | ✅ 完整 |
| lostspace | 25 | 5,876 B / 98 行 | 6,160 B | ✅ 完整 |
| miracle | 24 | 5,632 B / 98 行 | 6,201 B | ✅ 完整 |
| rollman | 29 | 5,360 B / 111 行 | 4,472 B | ✅ 完整 |
| aquawar | 25 | 5,188 B / 87 行 | 4,817 B | ✅ 完整 |
| antwar | 27 | 4,380 B / 82 行（+ `backend_RULES.md` 30 KB） | 4,387 B | ✅ 完整 |
| **antwar2** | **30** | **3,363 B / 49 行** | 3,868 B | ✅ 完整（精炼版） |
| monecraft | legacy_ai9 | 742 B / 19 行 | — | ⚠️ **仅摘要** |
| dorado | legacy_ai9 | 725 B / 18 行 | — | ⚠️ **仅摘要** |
| pacman | legacy_ai9 | 695 B / 19 行 | — | ⚠️ **仅摘要** |
| lota | legacy_ai9 | 676 B / 18 行 | — | ⚠️ **仅摘要** |

**4 个 `legacy_ai9` 游戏的完整规则拿不到**：它们的 `README.md` 只写"规则：`rules.md`"，那份就是官方唯一的规则文档；完整语义只能从 C++ 后端代码反推。另外这 4 个**不走 Saiblo stdio 协议**，而是 AI9 后端 + `ailoader` + 选手 `.so`，所以也没有 `decision_space.yaml`。

## 规则权威链（重要）

各游戏的 `rules.md` 自己就写明了权威顺序：

> 规则权威是**内容哈希固定的官方 C++ 后端**；公开 Python SDK 是**候选接口权威**。

所以：

1. **C++ 后端代码** = 最终裁判（`games/<game>/backend/`）
2. `rules.md` = 已经和后端核对过的说明（antwar2 那份明确写着"本文件已按本仓官方后端核对"）
3. `decision_space.yaml` = 候选接口契约（入口符号、动作码）
4. **上游赛事方自己的文档（如你本地 `Ant-Game/README.md`）可能与后端不一致** —— 已经发现一处，见下面的勘误

## ⚠️ 已知勘误

**造塔价格公式**：你本地 `Ant-Game/README.md` 写的是 `15 × 2^i`，**与 C++ 后端和 Python SDK 都不符**。正确公式是 `15 × 3^(n//2)`（n 为奇数再 ×2）。

详见 [`../../docs/antwar2规则勘误_造塔价格.md`](../../docs/antwar2规则勘误_造塔价格.md)。

## 相关文档

- AAArena 仓库总说明：[`../AAArena说明.md`](../AAArena说明.md)
- 其他版本 AI 索引：[`../CLAUDE.md`](../CLAUDE.md)
- 子模块来源与取舍：[`../../docs/other_versions_ai_sources.md`](../../docs/other_versions_ai_sources.md)

## 怎么更新

这些是解包出来的副本，AAArena 子模块更新后可以重解（脚本逻辑：按 `games/<game>/` 下的 `*.md` / `*.yaml` + `public_sdk/*.md` + `backend/RULES.md`，扁平化命名）：

```powershell
cd "D:\2026智能体大赛_新2\其他版本ai\AAArena"
tar -tzf assets/antwar2.tar.gz | Where-Object { $_ -match '^games/antwar2/[^/]+\.(md|yaml)$' }
```
