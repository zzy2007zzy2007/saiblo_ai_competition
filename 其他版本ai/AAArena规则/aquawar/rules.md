# AquaWar / 鱼洋陷役 —— 规则（rules.md）

> 第 25 届贺岁赛 AquaWar。双人、隐匿信息、三局两胜的回合制 SRPG。
> 本文从后端源码 `games/aquawar/backend/logic/gamecode_logic/`
> （`aqua_sdk.cpp` 交互与流程、`game.h/game.cpp` 状态机、`fish.h/fish.cpp` 鱼与技能、
> `fishset.*` 阵型）以及经过认证的 `public_sdk_cpp/` 提炼，以后端逻辑为准。

## 1. 玩家与角色

- 恰好 **2 名玩家**，角色标识 `P0` / `P1`。
- 每名玩家从 **12 种鱼**中选 **4 条**上场组成己方阵型（位置 0..3）。
- 隐匿信息：己方所选鱼的种类对对手隐藏，直到被「断言」命中或因技能/死亡而暴露。

## 2. 鱼种与属性（`fish.h`）

12 种鱼，编号 1..12。初始 **HP=400，ATK=100，DEF=0**（各鱼在此基础上有主动技能与被动）：

| id | 鱼 | 主动技能（active）方向 | 被动（passive）方向 |
| --- | --- | --- | --- |
| 1 | 射水鱼 spray | 群体攻击 (AOE) | 反伤 |
| 2 | 喷火鱼 flame | 友方增益/内战 | 反伤 |
| 3 | 电鳗 eel | 群体攻击 (AOE) | 友方伤害转移 |
| 4 | 翻车鱼 sunfish | 友方增益 | 友方伤害转移 |
| 5 | 梭子鱼 barracuda(blind) | 单体暴击 | 免伤 |
| 6 | 蝠鲼 mobula(flyfish) | 友方减伤 | 免伤 |
| 7 | 海龟 turtle | 友方增益（限使用次数，带 3 层盾 shell） | 护盾 shield |
| 8 | 章鱼 octopus | 友方回血/减伤 | 限量回血 |
| 9 | 大白鲨 whiteshark | 攻击最低血量敌人（弱暴击） | 限量回血 |
| 10 | 锤头鲨 hammerhead(goblinshark) | 攻击最低血量敌人（弱暴击） | 死亡爆炸 dead_boom |
| 11 | 小丑鱼 clownfish(joker) | 友方增益 + 群体 | 伤害转移 injury_transfer |
| 12 | 拟态鱼 imitator | 模仿一条 1..11 号鱼的技能/被动 | 随所模仿对象 |

- **拟态鱼 (id=12)**：选择上场时必须额外指定 `ImitateFish`（1..11），本局按被模仿鱼行动。
- Buff（`FishBuff`）：SWELL 膨胀、HEAL 低血自动恢复、SHIELD 下次减伤 70%、
  LIFEONHIT 受击回血、DEFLECT 下次被击后队友分摊。

## 3. 对局结构（三局两胜）

一场比赛由至多 **3 个小局（round）** 组成，先赢 **2 局**者胜（后端 `game.score`
达 ±2 或打满 3 局判胜；`round_end` 中 `rounds==3` → `over=true`）。每个小局包含：

1. **准备阶段 READY**：确定本小局先后手。首局先手固定为 P0；此后由上一小局负者先手。
2. **选择阶段 PICK**：双方各选 4 条鱼（含可选拟态目标）。分两回合各自提交。
3. **断言阶段 ASSERT**：轮到的玩家可对敌方某个位置断言其鱼种（`Pos`+`ID`），
   命中则暴露该鱼并造成额外效果；也可选择 `Null` 不断言。
4. **行动阶段 ACTION**：轮到的玩家用一条存活鱼执行「普通攻击」或「技能」。

断言与行动在两名玩家间交替，直到一方全部鱼阵亡 → 该小局分出胜负。

## 4. 动作空间（选手提交的 JSON，见 `aqua_sdk.cpp` 的 checker）

- **Pick**：`{"Action":"Pick","ChooseFishs":[id,id,id,id],"ImitateFish":k?}`
  - `ChooseFishs` 恰 4 个、取值 1..12、不重复；含 12 时须给 `ImitateFish`(1..11)。
- **Assert**：`{"Action":"Assert","Pos":p,"ID":id}` 或 `{"Action":"Null"}`
  - `Pos` ∈ 0..3，`ID` ∈ 1..12。
- **Action**：
  - 普通攻击 `{"Action":"Action","Type":0,"MyPos":m,"EnemyPos":e}`（`m,e` ∈ 0..3）。
  - 技能 `{"Action":"Action","Type":1,"MyPos":m,"MyList":[...],"EnemyList":[...]}`
    （目标位置 ∈ 0..3、不重复；具体合法性由鱼种决定）。
- **Finish**：小局结束时选手回 `{"Action":"Finish"}` 确认。

## 5. 胜负与判负

- **正常终局**：某小局一方全鱼阵亡则该小局判负；先胜 2 局或打满 3 局定整场胜负。
- **回合上限**：单小局阶段推进超过 `STATE_LIMIT=128` 触发 `state_limit_exceeded`，
  按「活鱼多 → 总血高 → 单鱼最高血高 → 后手」顺序裁定该小局。
- **选手非法/超时/崩溃**：后端 `invalid_operation()` 将 `score` 置为 ±3 并正常结束，
  由对手获胜（在本仓库对战器中记为 `game_error`，判对手胜）。

## 6. 隐藏信息与观测

- 选手每回合从 judger 收到本方视角的 `GameInfo`（己方鱼的 id/hp/atk、敌方鱼的
  hp 与「已暴露」的 id）、对手上一步的 `EnemyAction`/`EnemyAssert` 摘要、
  以及小局胜负回执（`Result`: Win/Lose）。敌方未暴露鱼的种类不可见。

## 7. 确定性与种子

- AquaWar 后端不消费随机种子作为对局输入（首局先手固定、流程确定）。相同的两名
  选手对战产出一致结果。对战器的 `seed` 仅用于产物分目录，不改变对局逻辑。

## 8. 对战器实现位置

- 后端：`games/aquawar/backend/logic/gamecode_logic`（C++ Makefile）。
- 公开选手接口：资源包内经过真实对局认证的 C++17 SDK；完整目录用所带 Makefile 构建。
- 协议：4 字节大端长度前缀 + 4 字节 object_id 的 stdio 帧；终局以后端广播
  `{"state":-1,"end_info":...}` 标志。详见 `evaluator/` 与 `replay_format.md`。
