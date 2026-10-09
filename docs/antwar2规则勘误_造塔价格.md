# 勘误：AntWar2 造塔价格公式（本地 `Ant-Game/README.md` 写错了）

> 记录时间：2026-10-09
> 发现场景：核对 AAArena（`其他版本ai/AAArena/`）的 antwar2 `rules.md` 与本地 `Ant-Game/README.md` 时发现两者不一致，随后查 C++ 后端与 Python SDK 定案。

## 结论一句话

**本地 `Ant-Game/README.md` 里的造塔价格公式 `15 × 2^i` 是错的。**

正确公式（C++ 后端与 Python SDK 一致）：

```
cost(n) = 15 × 3^(n // 2) × (n 为奇数 ? 2 : 1)
```

`n` = 你**当前已有的防御塔数量**。

## 两处说法对照

| 来源 | 公式 |
|---|---|
| `Ant-Game/README.md` | 建造价格 = `15 × 2^i`，i 为当前存在的己方防御塔数量 |
| AAArena `games/antwar2/rules.md` | 下一座塔造价为 `15 * 3**(n//2)`，现有塔数 n 为奇数时再乘 2 |
| **C++ 后端（权威）** | `Ant-Game/game/src/coin.cpp:11-19` |
| **Python SDK** | `Ant-Game/SDK/backend/constants.py:325` |

## 证据一：C++ 后端（规则的最终权威）

`Ant-Game/game/src/coin.cpp`：

```cpp
constexpr int INITIAL_COIN = 50, INITIAL_TOWER_BUILD_PRICE = 15,
              INITIAL_BARRACK_BUILD_PRICE = 0, INITIAL_BASIC_INCOME = 3,
              INITIAL_PENALTY = 0;
constexpr int COMBAT_KILL_REWARD = 18;

int tower_build_cost_for_count(int tower_count) {
    tower_count = std::max(tower_count, 0);
    int cost = INITIAL_TOWER_BUILD_PRICE;          // 15
    for (int index = 0; index < tower_count / 2; ++index)
        cost *= 3;                                  // × 3^(n/2)
    if (tower_count % 2 == 1)
        cost *= 2;                                  // 奇数再 ×2
    return cost;
}
```

**这个文件就在你本地**，而且我逐文件核对过：AAArena 的 antwar2 C++ 后端（`games/antwar2/backend/logic/gamecode_logic/game/`）与你本地 `Ant-Game/game/` 的 23 个 C++ 源文件里 **21 个逐字节相同**（忽略换行符差异），差异只在 `src/comm_judger.cpp` 和 `src/main.cpp`（AAArena 的评测接线）。

`coin.cpp` **就在那 21 个相同文件里** → 上面这段代码同时是 AAArena 的规则和你能在本地验证的规则。

## 证据二：Python SDK（你实际调用的接口）

`Ant-Game/SDK/backend/constants.py:325`：

```python
def tower_build_cost_for_count(tower_count: int) -> int:
    tower_count = max(int(tower_count), 0)
    cost = TOWER_BUILD_BASE_COST * (3 ** (tower_count // 2))
    if tower_count % 2 == 1:
        cost *= 2
    return cost
```

与 C++ **完全一致**。`其他版本ai/rule_v4/SDK/backend/constants.py` 里那份也逐字节相同。

## 正确 vs README 的数值表

| 已有塔数 n | **真实造价** | README 的说法 | 差额 |
|---:|---:|---:|---:|
| 0 | 15 | 15 | — |
| 1 | 30 | 30 | — |
| 2 | **45** | 60 | −15 |
| 3 | **90** | 120 | −30 |
| 4 | **135** | 240 | −105 |
| 5 | **270** | 480 | −210 |
| 6 | **405** | 960 | −555 |
| 7 | **810** | 1,920 | −1,110 |
| 8 | **1,215** | 3,840 | −2,625 |

（真实序列：15, 30, 45, 90, 135, 270, 405, 810, 1215 … 即每两座塔 ×3，中间那座再 ×2）

## 影响面

**运行时不受影响** —— 只要你调的是 SDK 的 `build_tower_cost()` / `tower_build_cost_for_count()`（`SDK/backend/engine.py` 里的 `build_tower_cost()` 就是转发到它），算出来的就是正确值。

**会踩坑的地方**：

1. 写策略启发式时**按 README 的公式手算**成本 / 预算
2. **硬编码**"塔超过 N 座就不建了"之类的阈值
3. 自己实现了一个价格估算函数（比如从 README 抄的）

**偏差方向**：README 会**越来越严重地高估**造塔成本 → 策略会**偏保守、建塔不足**。第 4 座塔你以为要 240（实际 135），第 6 座你以为 960（实际 405）。

## 自查方法

在自己的 AI 代码里搜这些模式，确认没有手写的错误公式：

```powershell
# 找手写的价格计算 / 硬编码价格表
Select-String -Path <你的代码> -Pattern '2 \*\* |2\*\*|<< *tower|tower.*\* *2|15 *\* *\(2'
# 确认用的是 SDK 函数（这是对的）
Select-String -Path <你的代码> -Pattern 'build_tower_cost|tower_build_cost_for_count'
```

如果你确实需要一份价格表，**以本文的「正确造价」那一列为准**，别用 README 的。

## 顺带做的其他常数核对（这些**都对得上**）

用同样的方式把 README 的其余关键数字对了一遍 C++ 常数，**没有发现第二处冲突**：

| 项 | C++ 常数 | README |
|---|---|---|
| 初始金币 | `INITIAL_COIN = 50` | ✅ 一致 |
| 基础收入 | `INITIAL_BASIC_INCOME = 3`（每两轮） | ✅ 一致 |
| 击杀奖励 | `COMBAT_KILL_REWARD = 18` | ✅ 一致 |
| 初始塔价 | `INITIAL_TOWER_BUILD_PRICE = 15` | ✅ 一致 |
| 武器花费 | 闪电 90 / EMP 135 / 偏转 60 / 闪避 60 | ✅ 一致 |
| 基地升级 | 出兵速度 200 / 蚂蚁等级 250 | ✅ 一致 |
| 胜负判定链 | 基地 HP → 击杀敌蚁数 → **较少**超级武器使用量 → **较低** AI 总耗时 → 先手(P0)胜 | ✅ 一致（README §8「胜负判定」） |
| 同时归零 | 同一结算双方基地均归零 → **P0 获胜** | ✅ 一致 |

⚠️ 这是**关键词级抽查**，不是逐条穷举。如果以后怀疑别的数字，建议照本文的办法直接去 `game/src/*.cpp` 查权威实现。

## 权威规则在哪

- 本地副本（已解包，方便直接翻）：[`其他版本ai/AAArena规则/`](../其他版本ai/AAArena规则/README.md)
- antwar2 的三个权威来源，优先级从高到低：
  1. **C++ 后端** `Ant-Game/game/src/*.cpp`（= AAArena `games/antwar2/backend/logic/gamecode_logic/game/src/`）
  2. `其他版本ai/AAArena规则/antwar2/rules.md`（49 行精炼版，已注明"按本仓官方后端核对"）
  3. `其他版本ai/AAArena规则/antwar2/decision_space.yaml`（候选接口契约）
- 你本地 `Ant-Game/README.md`（454 行，细节最全，**但价格那一处有错**）

## 待办

- [ ] 到上游 `Ant-Game` 仓库提个 issue / PR 修 `README.md` 的造塔价格公式（可选，但能帮到别人）
