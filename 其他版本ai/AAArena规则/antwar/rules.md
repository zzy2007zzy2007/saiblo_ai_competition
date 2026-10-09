# AntWar / 蚁国争霸 (27th) —— 规则（rules.md）

> 第 27 届 AntWar：双人回合制**塔防对抗**。双方在镜像地图上放置防御塔、使用超级武器，
> 拦截敌方蚂蚁、护送己方蚂蚁抵达对方基地扣血，最终摧毁对方基地或血量领先者胜。
> 本文从本 GamePack 的后端源码 `backend/`
> （`game.cpp` 流程、`comm_judger.cpp` 协议、`operation.h`）与 `public_sdk/`
> 提炼，以后端逻辑为准。参赛者不需要、也不得读取正式选手池源码。

## 1. 玩家与角色

- 恰好 **2 名玩家**，角色标识 `P0` / `P1`。
- 地图为六边形网格，`MAP_SIZE = 2*EDGE-1 = 19`；两名玩家的基地与高地互为**镜像**。
- 完整信息博弈（除随机信息素初始化外，双方可见全局塔/蚂蚁/金币/基地血量）。

## 2. 回合流程（`main.cpp` 主循环）

每回合依次：
1. 后端向选手 0 推送当前局面（OJ 文本），选手 0 提交操作集合。
2. 选手 0 的操作转发给选手 1；后端向选手 1 推送局面，选手 1 提交操作。
3. 后端结算：防御塔攻击蚂蚁、蚂蚁按信息素移动、生成新蚂蚁、更新金币与信息素、
   判定基地血量。
- 最多 `MAX_ROUND = 512` 回合；超过则按血量/击杀数裁定（`judge_winner`）。

## 3. 资源与经济（`common.hpp`）

- 初始金币 `COIN_INIT = 50`，每回合基础收入 `BASIC_INCOME = 1`（另有击杀蚂蚁收益）。
- 建塔价 `TOWER_BUILD_PRICE_BASE = 15`，按已有塔数指数增长（`RATIO = 2`）。
- 升级：二级塔 60、三级塔 200；拆塔返还 80%。
- 基地升级：二级 200、三级 250（提升出兵速度/蚂蚁血量等级）。

## 4. 操作类型（选手提交，`OperationType` / `Operation::Type`）

| type | 操作 | 参数 |
| --- | --- | --- |
| 11 | BuildTower 建塔 | x y（位置） |
| 12 | UpgradeTower 升级塔 | tower_id（+目标类型） |
| 13 | DowngradeTower 降级/拆塔 | tower_id |
| 21 | UseLightningStorm 闪电风暴 | x y |
| 22 | UseEmpBlaster EMP 干扰 | x y |
| 23 | UseDeflector 力场护盾 | x y |
| 24 | UseEmergencyEvasion 紧急疏散 | x y |
| 31 | UpgradeGenerationSpeed 升级出兵速度 | （无参） |
| 32 | UpgradeGeneratedAnt 升级蚂蚁 | （无参） |

一个回合可提交**多条操作**（受金币与合法性约束）。

## 5. 塔与超级武器

- 塔类型（`TowerType`）：Basic(0)、Heavy 系(11/12/13：重炮/冰冻/加农)、Quick 系(2/21/22)
  等，各有攻击力、攻速(cd)、射程三元组。
- 超级武器（`SuperWeaponType`）：闪电风暴(1)、EMP(2)、力场(3)、紧急疏散(4)，各有
  持续时间与作用范围，使用有 CD。

## 6. 蚂蚁

- 蚂蚁沿信息素路径向对方基地移动；到达对方基地（`Success`）对其扣血；血量耗尽为 `Fail`；
  存活过久为 `TooOld`；被 EMP 冻结为 `Frozen`。
- 生成周期由基地出兵速度等级决定（`GENERATION_CYCLE_INFO = {4,2,1}`）。

## 7. 胜负判定

- **基地摧毁**：`judge_base_camp`——对方基地血量 ≤ 0 则己方胜（双方同时为 0 判 P0 胜）。
- **回合上限**：达 512 回合调 `judge_winner`——依次比较基地血量（高者胜）、击败
  对方蚂蚁数（高者胜）、超级武器使用次数（少者胜）和 AI 累计用时（少者胜）；仍相同
  时由 P0 获胜。这是后端的完整字典序，不存在最终平局。
- **选手非法/超时/崩溃**：后端 `set_AI_state_IO`/错误消息将该玩家判负（对手 winner），
  end_state 标记为 RE/TLE/OLE/IA。本仓库对战器记 `game_error`，判对手胜。

## 8. 随机性与种子

- 后端仅在**信息素初始化**（`map.init_pheromon(random_seed)`）使用随机种子。init 消息
  的 `config.random_seed` 若提供则采用，否则后端自取随机。本仓库对战器**固定种子**
  （见 `evaluator/arena.py` 的 `FIXED_SEED`）以保证可复现。

## 9. 对战器实现位置

- 后端：`backend/`
  （C++ Makefile，输出 `output/main`）。
- 公开选手接口：`public_sdk/`；模板是 Python，正式选手池可以包含多种实现语言。
- 协议：4 字节大端长度前缀 + 4 字节 object_id 的 stdio 帧；选手侧为 **OJ 文本**协议
  （`std::cin` 读状态、4 字节长度前缀写操作）。详见 `evaluator/` 与 `replay_format.md`。
