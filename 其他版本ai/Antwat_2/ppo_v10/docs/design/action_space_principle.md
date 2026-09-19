# PPO_V10 动作空间改造原则

> 基于 ppov10 开发过程中关于动作空间改造的设计讨论、代码实现审查与实际决策整理。

---

## 一、合法性机制

### 1. GA 不依赖 action mask 注入

GA 不使用 action mask 做 `-inf` 注入来屏蔽非法动作。网络始终在所有维度上自由采样，非法动作通过运行时校验退化为 NOOP。mask 仅作为验证参考，不参与前向计算。

### 2. NOOP 惩罚 = 加速淘汰

非法动作不会被修正为随机合法动作，而是直接返回 NOOP。GA 视角下，非法率高的个体 → NOOP 多 → 操作节奏落后 → 败北 → 被种群淘汰。该机制等价于自然选择压力，无需引入额外的惩罚系数或正则项。

---

## 二、网络输出架构

### 3. 语义分段向量

网络输出从「分层 one-hot」改为「语义分段向量」。总输出维度 27，划分为 4 个独立语义段：

- `[0:3)` strategy（3 维）
- `[3:9)` type（6 维）
- `[9:19)` sub_type（10 维）
- `[19:27)` position（8 维）

### 4. 每段独立解码

每段各自做 `argmax`，得到该段的离散索引。不存在段间联合采样或依赖关系。解码结果通过映射表转换为具体游戏操作。

> 说明：GA 为确定性决策，softmax 作为单调函数不改变 argmax 结果，故直接对 logits 取 argmax，省去冗余的 softmax 计算。

### 5. 无 explore 分支

GA 不需要探索噪声。所有动作输出均为确定性选择（`deterministic=True`），不设 `logit_noise_std`，不做随机采样。

### 6. 删除 PPO 遗留结构

以下 PPO-only 结构全部删除，GA 不使用：

- Value 编码器（`value_cnn`、`value_mlp`、`value_proj`、`value_layer_norm`）
- ValueHead（价值头）
- Auxiliary Heads（TowerDamageHead、GoldIncomeHead、BaseDamageHead × 2 套）
- `evaluate_actions` 方法（完整 PPO 训练通路）
- `get_value_only` 方法（PPO 估值入口）
- `forward` 方法（PPO 训练入口）
- `log_prob` 计算（GA 不调用）
- `logit_noise_std`（GA 无噪声需求）

仅保留 Policy 编码器（`policy_cnn`、`policy_mlp`、`policy_proj`、`policy_layer_norm`）作为策略特征提取核心。

---

## 三、Strategy 段

### 7. strategy 三值设计

strategy 段共 3 个取值，分别代表：

| 索引 | 语义 |
|------|------|
| 0 | 防御（defensive） |
| 1 | 无操作（NOOP） |
| 2 | 进攻（offensive） |

### 8. strategy=无操作时强制 NOOP

当 `strategy=1` 时，无论 `type_idx` 取何值，解码结果均为 NOOP。不依赖 type 段输出。

---

## 四、Type 段

### 9. type_idx 无固有语义

`type_idx` 本身不携带任何操作语义。同一 `type_idx` 在不同 `strategy` 下代表完全不同的操作。只有 `(strategy, type_idx)` 联合起来才唯一确定一个操作种类。

### 10. 共享表示空间

type 空间（0-5 共 6 个索引）对所有 strategy 共享。不按 strategy 拆分或重叠，所有 strategy 共用同一套 6 维逻辑。

### 11. 攻防成对反向

在每个 `type_idx` 下，防御方动作和进攻方动作互为"反向操作"——加强己方 vs 削弱敌方：

| type_idx | 防御 | 进攻 | 说明 |
|:--------:|:----:|:----:|:----:|
| 0 | BUILD_TOWER（建己方塔） | DOWNGRADE_TOWER（削敌方塔） | 非超武·塔↔塔 |
| 1 | UPGRADE_TOWER（升己方塔） | DOWNGRADE_TOWER（削敌方塔·冗余） | 非超武·塔↔塔 |
| 2 | LIGHTNING_STORM（杀伤敌方蚂蚁） | DEFLECTOR（保护己方蚂蚁） | **超武↔超武**·蚂蚁 |
| 3 | LIGHTNING_STORM（杀伤敌方蚂蚁·冗余） | EVASION（己方蚂蚁闪避） | **超武↔超武**·蚂蚁 |
| 4 | LIGHTNING_STORM（杀伤敌方蚂蚁·冗余） | EMP_BLASTER（压制敌方塔） | **超武↔超武**·混合目标 |
| 5 | →NOOP | TECH_UPGRADE（加强己方蚂蚁） | 非超武·进攻方独有 |

### 12. 超武只与超武成对

超武类型的动作只与其他超武动作配对：

- LIGHTNING_STORM（杀伤敌方蚂蚁）↔ DEFLECTOR（保护己方蚂蚁）
- LIGHTNING_STORM（杀伤敌方蚂蚁）↔ EVASION（己方蚂蚁闪避）
- LIGHTNING_STORM（杀伤敌方蚂蚁）↔ EMP_BLASTER（压制敌方塔）

非超武动作（BUILD_TOWER / UPGRADE_TOWER / DOWNGRADE_TOWER / TECH_UPGRADE）只与非超武成对。超武与非超武之间不存在配对映射。

### 13. 非对称映射

防御方和进攻方的映射表不必对称。允许一方有 slot 而另一方为空（退化为 NOOP）。例如 `type_idx=5` 在防御方为空，在进攻方为 TECH_UPGRADE。这给不同 strategy 不同的表达能力。

### 14. 冗余映射

同一动作可以映射到多个 `type_idx`：

- 防御方：`type_idx=2,3,4` 都映射为 LIGHTNING_STORM（3 个命中点）
- 进攻方：`type_idx=0,1` 都映射为 DOWNGRADE_TOWER（2 个命中点）

冗余映射帮助 GA 更容易命中有效动作——多条通路通向同一结果，降低了搜索难度。

### 15. 无纯空置 slot

type 空间不保留纯 NOOP 条目。每个 slot 至少被一方使用（或防御方映射有效动作，或进攻方映射有效动作），不存在攻防双方都为空置的 type_idx。

---

## 五、进攻/防守归类原则

### 16. 归类逻辑

操作按效果分为"加强己方"和"削弱敌方"两大类，统一编入 strategy：

| 动作 | 效果方向 | Strategy | 理由 |
|------|:--------:|:--------:|------|
| BUILD_TOWER | 加强己方 | 防御 | 强化己方塔 |
| UPGRADE_TOWER | 加强己方 | 防御 | 强化己方塔 |
| DOWNGRADE_TOWER | 削弱己方 | 进攻 | 降级己方塔换金币，用于实施其他攻击操作 |
| EMP_BLASTER | 削弱敌方 | 进攻 | 压制/削弱敌方塔 |
| LIGHTNING_STORM | 削弱敌方 | 防御 | 杀伤敌方蚂蚁 = 削弱敌方蚂蚁 |
| DEFLECTOR | 加强己方 | 进攻 | 保护己方蚂蚁 = 加强己方蚂蚁 |
| EVASION | 加强己方 | 进攻 | 己方蚂蚁闪避 = 加强己方蚂蚁 |
| TECH_UPGRADE | 加强己方 | 进攻 | 加强己方蚂蚁 |

简化规则：
- 与塔相关的操作：加强己方塔 → 防御，削弱敌方塔 → 进攻，降级己方塔换币进攻 → 进攻
- 与蚂蚁相关的操作：加强己方蚂蚁 → 进攻，削弱敌方蚂蚁 → 防御

### 17. TECH_UPGRADE 不因 strategy 分裂

TECH_UPGRADE 固定在进攻方（strategy=2），不因 strategy 变化而分裂到多个槽位。其具体升级内容（generation_speed / generated_ant）由 position 段决定。

---

## 六、Sub_type 段

### 18. sub_type 10 维结构

sub_type 段共 10 维，仅在动作类型为 UPGRADE_TOWER 时使用。进一步分解为三个子段：

| 子段 | 维度 | 索引范围 | 取值 |
|:----:|:----:|:--------:|:----:|
| main_tree | 3 | [9:12) | HEAVY / QUICK / MORTAR |
| sub_variant | 3 | [12:15) | variant_A / variant_B / variant_C |
| producer | 4 | [15:19) | PRODUCER / PRODUCER_FAST / PRODUCER_SIEGE / PRODUCER_MEDIC |

### 19. BASIC→combat/producer 升级分立

BASIC 塔升级时，producer 线（生产系）使用独立的 4 个 slot，与 combat 升级树（HEAVY/QUICK/MORTAR）完全区分。二者通过 main_tree 和 producer 的取值互斥选择。

### 20. sub_type 语义随塔类型变化

sub_type 的实际含义取决于目标塔的当前类型：

- 目标塔为 BASIC → main_tree / producer 决定一级升级方向
- 目标塔为 combat（HEAVY/QUICK/MORTAR）→ main_tree 决定上哪个枝，sub_variant 决定该枝的二級子类
- 目标塔为 PRODUCER → producer 决定二級子类

---

## 七、Position 段

### 21. position 8 维编码

position 段共 8 维（`[19:27)`），输出值 ∈ {0..7}。8 维是覆盖全部 8 个玩家塔位所需的最小维度，同时为超武和科技升级提供充足的冗余映射空间。

### 22. 多表映射：position 语义随动作类型变化

position 值本身不固定对应某一坐标。同一 position 值在不同动作类型下查不同的映射表解码：

| 动作类型 | 映射表 | 含义 |
|---------|--------|------|
| BUILD / UPGRADE / DOWNGRADE_TOWER | 玩家塔位表（`_PLAYER_TOWER_POS`） | 8 个固定塔位的索引 |
| LIGHTNING_STORM / EMP_BLASTER / DEFLECTOR / EVASION | 超武冗余映射表（`_WEAPON_POS_REDUNDANCY`） | 5 个预设位 + 3 个冗余入口 |
| TECH_UPGRADE | 科技冗余映射表（`_TECH_POS_MAP`） | 2 种科技类型，各 4 个冗余入口 |
| NOOP | 忽略 | — |

### 23. Player-specific 塔位表

塔操作（BUILD/UPGRADE/DOWNGRADE_TOWER）的 position 解码引入玩家身份作为上下文。同一 position 值为不同玩家解码出不同坐标，双方镜像对称：

| position | Player 0 | Player 1 | 战略排序依据 |
|:--------:|:--------:|:--------:|-------------|
| 0 | (5, 9) | (13, 9) | 中线核心位 >> 中线辅助 >> 中央战场 >> 侧翼 >> 角落 |
| 1 | (4, 9) | (14, 9) | ↓↓↓ |
| 2 | (7, 8) | (10, 7) | ↓↓↓ |
| 3 | (6, 7) | (12, 6) | ↓↓↓ |
| 4 | (8, 7) | (11, 5) | ↓↓↓ |
| 5 | (5, 6) | (12, 9) | ↓↓↓ |
| 6 | (6, 14) | (11, 14) | ↓↓↓ |
| 7 | (4, 2) | (13, 15) | 角落防守 |

**排序原则**：按战略价值从高到低排列。中线核心位（pos 0-1）优先级最高，中央战场（pos 2-4）次之，侧翼和角落（pos 5-7）最后。这种排序让网络的注意力自然集中在高价值区域——即使 early stopping 未充分训练，靠前的 position 值也倾向于对应战略意义更大的位置。

**效果**：8 个 position 值全部对应合法的高地格，编码空间零浪费。仅在该格已被建塔或被 EMP 覆盖时才会导致 NOOP。

### 24. 超武位置：冗余映射最大化合法率

超级武器只有 5 个预设释放位置，但 position 提供 8 个入口值。后 3 个值（5-7）冗余映射到中线/核心位（位置 2-4），而非留空：

```
position:    0   1   2   3   4   5   6   7
             ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
映射到:     [0] [1] [2] [3] [4] [2] [3] [4]
                    ↑ 核心区        ↑ 冗余
```

设计意图：
- **前 5 个值**（0-4）：精确选择 5 个预设位，给网络细粒度控制
- **后 3 个值**（5-7）：冗余到中线/核心位，超武漂移场景下中线区价值最高

**收益**：8/8 = 100% 编码利用率（对比 4 维方案下 4/5 = 80%）。

### 25. 科技升级：平均冗余覆盖

TECH_UPGRADE 只有 2 种科技类型（generation_speed / generated_ant）。将 8 个 position 值均分为两组：

- position 0-3 → generation_speed（4 个冗余入口）
- position 4-7 → generated_ant（4 个冗余入口）

给 GA 4 倍的命中窗口，降低搜索难度。

### 26. position 不追求坐标级精度

position 映射的是"战略位置"而非精确 x/y 坐标。这是有意的设计取舍：

- **建塔场景**：塔位表固定为 8 个预选高地格，覆盖了 16 个 TOWER_POSITIONS 的核心位置。即便网络选到已被占的塔位退化 NOOP，8 选 1 的命中率也远高于原 16 选 1 且 50% 无效的架构。
- **超武场景**：超级武器会每回合随机漂移 1 格，本身就缺乏精确性。冗余映射导致的位置偏差（如 pos 5 → 位置[2] 代替位置[4]）在实际效果上差异可忽略。
- **科技升级**：不涉及坐标，position 仅作为选择器使用。

### 27. 编码空间利用最大化

| 动作类 | 可用编码值 | 实际有效映射 | 利用率 | 原 4 维方案 |
|-------|:---------:|:----------:|:-----:|:----------:|
| 塔操作 | 8 | 8（全部高地格） | **100%** | ~70%（region 分组可能全满） |
| 超级武器 | 8 | 5（含 3 冗余） | **100%** | 80%（值 4 不可达） |
| 科技升级 | 8 | 2（各 4 冗余） | **100%** | 50%（仅值 0,1 有效） |
