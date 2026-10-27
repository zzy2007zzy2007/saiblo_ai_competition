# 引擎随机数（RNG）机制调查 + "在 rollout 里主动消费引擎 RNG" 的可行性与改造方案

- 时间：2026-10-27
- 范围：**只读**源码 + 独立探针（不动 `native_game.pyd`、不动 `code/my_ai/az_intent/*`、不 kill 任何进程）；全程 1 个 python 进程 + 2 次 g++（每次约 40 s），实际占用 ≈ 1 核
- 依据版本：**官方 C++ 源码** `Ant-Game/game/src`（`game.cpp` 与我们 `code/cpp_engine/src/game.cpp` MD5 完全相同 = `2E3066766C1B5010EF714F7791151AD2`）；官方 python 引擎的结论一律不采信
- 交付物：本文档 + 两个探针（`code/test_match/probe_engine_rng.py`、`code/cpp_engine/probe_rng/build_probe_pyd.py`）

---

## 0. 三句话结论

1. **RNG 用在哪**：引擎没有 `std::mt19937`，用的是**自实现的 48-bit LCG**（`s ← 25214903917·s + 11 mod 2^48`），单一状态 `Game::rng_state`。消费点共 11 处，**85–89% 的抽取来自"每只蚂蚁每回合的走位 softmax 抽签"**（`game.cpp:1217`），其余是刷怪抽签、传送、武器漂移、魅惑目标。一条 440 回合的真实对局共抽 **5344** 次（12.2 次/回合）。
2. **是否确定性**：**是，逐位确定**。给定 seed + 双方操作序列，连跑两次（同进程 / 跨进程 / 交错再开一局 / clone 后重跑）**440 回合全等**；更强的证据是官方判题器 `judge_O0.exe` 与我们的 pyd 在 **440 回合 × 722 格信息素**上 `max|diff| = 0`（信息素由"每回合每只蚂蚁位置"驱动 ⇒ 等价于**每次 RNG 抽取的序列都相同**）。
3. **改造值不值得**：**接口改造 1–2 小时（3 个 header inline 方法 + 3 行 pybind），风险近零**（不消费时逐位不变，已用 parity 验证）。但"主动多消费 RNG 来制造多样性"这个用法**不是更真实的采样，而是把估计量换成"反事实种子的平均"（引入平滑/偏差）**；真正干净、能确定带来收益的用法是**配对（CRN）**：所有候选共享同一组 RNG 偏移，实测让候选差的 sd 降 **13%–56%**、排序更稳，成本 ≈ 0。

---

## 1. RNG 在引擎里用在哪

### 1.1 RNG 本体：自实现 48-bit LCG（不是 mt19937）

```cpp
// Ant-Game/game/src/game.cpp:25-28
constexpr unsigned long long RNG_MASK = (1ULL << 48) - 1;
constexpr unsigned long long RNG_MULTIPLIER = 25214903917ULL;
constexpr unsigned long long RNG_INCREMENT = 11ULL;
constexpr int RANDOM_FLOAT_BITS = 24;
```

```cpp
// Ant-Game/game/src/game.cpp:174-189
unsigned long long Game::next_random() {
    rng_state = (RNG_MULTIPLIER * rng_state + RNG_INCREMENT) & RNG_MASK;
    return rng_state;
}
double Game::random_float() {
    return static_cast<double>((next_random() >> (48 - RANDOM_FLOAT_BITS)) &
                               ((1ULL << RANDOM_FLOAT_BITS) - 1ULL)) /
           static_cast<double>(1ULL << RANDOM_FLOAT_BITS);      // 取高 24 bit / 2^24
}
int Game::random_index(int bound) {
    if (bound <= 1) return 0;                                   // ← bound<=1 **不消耗**
    return static_cast<int>(next_random() % static_cast<unsigned long long>(bound));
}
```

要点（都影响"怎么改造"）：

| 属性 | 值 | 影响 |
|---|---|---|
| 类型 | 自实现 LCG，乘子 25214903917（= `java.util.Random` 的乘子）、增量 11、模 2^48 | 不是 `std::mt19937`，没有 `seed_seq`/引擎内置分布 |
| 状态 | `unsigned long long rng_state`（`game.hpp:69`，**官方源码里就有**） | 单个 48-bit 整数 ⇒ `get/set` 就是一次 64-bit 拷贝 |
| 周期 | 2^48（乘子 ≡ 1 mod 4、增量奇数 ⇒ 满周期） | 偏移 k 步在周期内唯一 ⇒ 可以从"前后状态"反解抽了几次（见 §2.3） |
| 取值 | `random_float()` = 高 24 bit；`random_index(n)` = `state % n`（**有模偏差**，但确定性无影响） | 改造时不要"顺手改成 `std::uniform_int_distribution`"——那会改变现有对局 |
| 是否可克隆 | 是，`m1_support.cpp:414` `rng_state = other.rng_state;`（注释：*RNG is state: clones evolve independently*） | **同一状态 clone 出的两个 rollout 共享同一条 RNG 流** ⇒ 这是"idle rollout 完全一样"的根因 |
| 每回合是否重置/固定推进 | **都不**。只在当回合"真有事件"时按需推进；没有"每回合固定抽 n 次" | 抽签次数是**状态相关**的（见 §1.3 的分布） |

### 1.2 种子如何进入

| 入口 | 位置 | 行为 |
|---|---|---|
| 官方判题器链路 | `game.cpp:1361-1375`（`Game::init()`） | 读 init 包 `config.random_seed`：unsigned / integer（负数→0）；**字段缺失时用 `std::random_device`**（那才是唯一不可复现的分支） |
| 我们的 pybind 链路 | `m1_support.cpp:57-59`（`Game::init_game`） | 逐行等价复刻：`random_seed = seed; rng_state = (random_seed ^ RNG_MULTIPLIER) & RNG_MASK;` |
| 同一个 seed 的第二个用途 | `game.cpp:1377` / `m1_support.cpp:59` → `map.cpp:135-147` | `map.init_pheromon(random_seed)` 用的是**原始 seed**（不是异或后的），并且走**另一条 LCG**：`map.cpp:3` 的**文件级全局** `unsigned long long lcg_seed`，递推 `lcg_seed = (25214903917 * lcg_seed) & (2^48-1)`（**没有 +11**），共抽 `2×19×19 = 722` 次初始化信息素 |

信息素初值 = `80000 + ((v*10000)>>46)` ⇒ 整数 `80000..119999` ⇒ 真实值 `8.0..12.0`（均值 10 = `TAU_BASE`）。

> ⚠️ **`seed = 0` 是退化地图**：`map.cpp:136` 的乘性 LCG **没有增量**，`lcg_seed = 0` 是不动点 ⇒ 722 格信息素全部 = 8.0（实测 `uniq = 1`）。这**不是**我们和官方不一致（官方同样退化），但别用 seed 0 做实验，会得到一个"完全没有信息素梯度"的地图。

### 1.3 RNG 消费点全表（官方 `game.cpp` 行号）

统计口径：`match_results/ladder_logs/VS_RUNNERUP_cells4/10.log`（seed 10，440 回合，真实对局重放）与 `RU_ENT128/1014.log`（512 回合）。数字来自探针 `probe_engine_rng.py sites`（在副本里给每个消费点打 site tag，见 §4）。

| # | 位置 | 函数 / 语义 | 影响什么 | 10.log 计数 | 1014.log 计数 |
|---|---|---|---|---|---|
| 1 | `game.cpp:1217` | `choose_ant_move_enhanced`：按 `exp(score/T)`（`T=1.75`）做**逆 CDF 抽签**，抽一个 `random_float()` 当阈值 | **每只 Default 行为蚂蚁每回合的走位**（`Conservative`/`ControlFree` 走 argmax 分支 `:1195-1204`，**不抽签**；`Bewitched` 转 #2） | **4567 (85.5%)** | **7846 (89.2%)** |
| 2 | `game.cpp:1025` | `choose_ant_move_legacy`：同上（legacy 策略，或 enhanced 里的 `Bewitched` 分支，`T=1.5/1.75`） | 走位 | 0 | 0 |
| 3 | `game.cpp:1041` | `choose_ant_move_enhanced`：`Randomized` 行为 → 均匀抽一个合法方向 | 走位（被冰塔/脉冲塔打中的蚂蚁、以及 10% 天生 Randomized 的蚂蚁） | 107 (2.0%) | 48 (0.5%) |
| 4 | `game.cpp:934` | `choose_ant_move_legacy` 的 `Randomized` 分支 | 走位（legacy） | 0 | 0 |
| 5 | `game.cpp:236` | `choose_random_legal_move`：均匀抽一个合法方向（**唯一调用点 = #7**，即传送后那 3 步乱走） | 走位 | 90 (1.7%) | 147 (1.7%) |
| 6 | `game.cpp:815` | `random_bewitch_target`：从"自己半场靠后"的格子集合里均匀抽一个当**魅惑目标** | 被魅惑蚂蚁的目标格（#1 的输入） | 0 | 0 |
| 7 | `game.cpp:1290` | `resolve_random_move_steps`：每步先抽 `random_index(3) < 2`（2/3 均匀乱走、1/3 走正常选择） | 传送后连走 3 步 | 153 (2.9%) | 255 (2.9%) |
| 8 | `game.cpp:1312` | `teleport_ants`：每 10 回合（`round+1)%10==0`）抽 `10%` 的蚂蚁传送 | 蚂蚁位置 | 51 (1.0%) | 85 (1.0%) |
| 9 | `game.cpp:1333` | `drift_items`：闪电风暴/EMP 的坐标每回合漂到"自身 + 6 邻格"共 7 格里的一格 | 武器落点 | 180 (3.4%) | 185 (2.1%) |
| 10 | `game.cpp:1549` | `generate_ants` 的 `draw_spawn_profile`：按 `{0.4,0.35,0.10,0.15}`（`game.cpp:72`）抽新蚂蚁的 `kind/behavior` | 新蚂蚁类型（工作/战斗）与行为（Default/Conservative/Randomized） | 196 (3.7%) | 228 (2.6%) |
| 11 | `game.cpp:1684` | `generate_ants` 的攻城塔分支：`random_float() <= siege_spawn_chance` | 攻城塔额外刷一只战斗蚁 | 0 | 0 |
| — | `game.cpp:1373` | `std::random_device`（**仅当 config 里没有 `random_seed`**） | 整局种子 | 0（我们永远传 seed） | 0 |
| — | `map.cpp:144`（`init_pheromon`） | **另一条**全局 LCG × 722 次 | 开局信息素场 | 722（一次性，不计入上表） | 722 |

**总量（`rng_state` 的推进次数，已用两种独立方法交叉验证：LCG 反解 + site 计数）**：

```
[census] 10.log   440 回合 总消费 5344，平均 12.15/回合，消耗>0 的回合 436/440 (99.1%)，单回合 0/12/32（min/中位/max）
[census] 1014.log 512 回合 总消费 8794，平均 17.18/回合，消耗>0 的回合 512/512 (100.0%)，单回合 1/17/37
```

⇒ **引擎每回合都在抽签（几乎所有回合）**，但抽的是"走位细节"，不是"整局走向"。

> 补充：`increase_ant_age`（`game.cpp:1750`）里"Randomized 行为 5 回合后自然衰减"（`RANDOM_ANT_DECAY_TURNS`）是**确定性**的，不抽签；`SPAWN_BEHAVIOR_PROBS` 里 10% 的 Randomized 蚂蚁才是随机性的来源。

### 1.4 每回合的抽签顺序（决定了"在哪注入最有效"）

`game.cpp:1882-1919` 的 `next_round()` 顺序：`attack_ants`（塔攻击 → #6 魅惑抽签）→ `move_ants`（#1/#3，**主要消费**）→ `teleport_ants`（#7/#8）→ `update_pheromone`（不抽）→ `manage_ants`（不抽）→ `generate_ants`（#10/#11）→ `increase_ant_age` → `update_coin` → `update_items`（#9 漂移）→ `round++`。

⇒ 一次 `consume_rng(k)` 会把**这一回合及以后的所有抽签**整体错位 k 步。但"错位"不等于"立刻分叉"：

- 在真实对局的切片上（有很多 Default 蚂蚁在抽签），偏移 1 就能让 rollout 的第 1 回合分叉（§5.1 表里"与分支 0 首次分歧回合 = 0"）；
- 反过来，若当回合**根本不抽签**（没有 Default 蚂蚁、没有刷怪、没有武器在漂），偏移 1 可以连续几回合都看不出来 —— 本机实测：无操作重放第 120 回合 `consume_rng(1)` 后推进 1 回合，16 个重复试验的状态**完全相同**（因为那一回合抽了 0 次）。

⇒ **"扰动幅度"与"是否分叉"不是线性关系**，别指望靠调小 k 精确控制失真度。

---

## 2. 确定性结论 + 证据

### 2.1 结论

**给定 seed + 给定双方操作序列，官方引擎（以及我们逐字节相同的 pybind 引擎）逐位可复现。** 唯一非确定的分支是"config 里不给 `random_seed`"（`game.cpp:1372-1375` 走 `std::random_device`）——我们从 `GameStateFacade.initial(seed=...)` 起就总是给 seed。

判定"没有别的随机来源"的完整检查（`Ant-Game/game/src` 全目录）：

```
$ grep -rniE 'mt19937|random_device|std::rand|srand|default_random_engine|uniform_int_distribution|uniform_real_distribution|shuffle' Ant-Game/game
Ant-Game/game/include/json.hpp:3286: std::random_access_iterator_tag      ← 与随机无关
Ant-Game/game/include/json.hpp:6548: std::random_access_iterator_tag      ← 与随机无关
Ant-Game/game/src/game.cpp:1373:        std::random_device rd;            ← 仅"没给 seed"时
```

即：没有 `rand()/srand()`、没有 `mt19937`、没有时间/线程 id 参与判定；`unordered_map` 只用于 `enhanced_tower_claims` / `enhanced_move_cells`（按 key 查询，不迭代求和），不引入顺序不确定。

### 2.2 证据（探针 `code/test_match/probe_engine_rng.py`）

摘要函数 `digest()` 把**所有可观测量**灌进 sha256：`round / base_hp / coins / winner / is_ended / rng_state / tower_details / ant_details / weapon_cds / base_levels / active_effects / 722 维信息素（double 二进制）`。

```
$ python code/test_match/probe_engine_rng.py determinism      # 默认 log = VS_RUNNERUP_cells4/10.log
[log] ...\10.log seed=10 回合数=440 记录终局={'rounds': 440, 'hp': (0, 28)}
[run1] 440 回合 终局 hp=0,28 rng=155510713659591
[run2] 440 回合 终局 hp=0,28 rng=155510713659591
[同进程 二次运行] 逐回合粗摘要完全一致: True
[同进程 二次运行] rng_state 序列完全一致: True
[交错另一局] 与单独运行逐回合摘要一致: True
```

- **跨进程**：连跑两次把摘要落盘，`SHA256 = 3B03F034C546A5159D879A3792CE1B7F5DDE31822558C19ADB07F09B79BF0982` 两次完全相同。
- **交错第二局**（进程内再开一局 seed=12345，每回合插在中间推进）本局摘要不变 ⇒ `map.cpp:3` 那个**文件级全局** `lcg_seed` 不会造成跨局污染（因为只在 `init_pheromon` 里用，而 `init_pheromon` 每次建局都重置它）。
- **clone 后重跑**（同一状态切两份，跑同一串真实操作 40 回合）：

```
$ python code/test_match/probe_engine_rng.py clone --cut 120 --horizon 40
[clone] 在第 120 回合切片：rng_state=219551896323594 hp=42,33 ants=14
[clone] 两个 clone 的初始摘要相同: True
[clone] 同一串操作推 40 回合，逐回合摘要一致: True
[clone] 终局 hp: A=37,30  B=37,30  rng A=121942865722801 B=121942865722801
```

⇒ **这是"flat rollout 里引擎 RNG 提供 0 多样性"的直接证明**：`m1_support.cpp:414` 连 `rng_state` 一起 clone，所以同一起点 + 同一动作序列 ⇒ 同一条 RNG 流 ⇒ 同一个叶子。这也正是 `flat_rollout_greedy.py:66-69` 注释里"idle 模式下 samples=1/16/64 分数逐位相同"的原因。

- **与官方判题器的最强对拍**（信息素 722 格逐元素比较）：

```
$ python code/test_match/probe_engine_rng.py official --module probe
[official] judge rc=0 440 条 replay 终局 camps=[0, 28]
[official] 我们的引擎 终局 hp=0,28 （440 回合）
[official] 信息素逐回合比对：440 回合 × 722 元素，不相等元素总数=0  max|diff|=0.000e+00
[official] 终局 ants 记录是**增量**（不是全量快照）⇒ 不作逐只比对；信息素每回合由全部蚂蚁位置决定，它逐元素相同已蕴含'每回合每只蚂蚁位置都相同'
[official] 每回合 base_hp 一致: True
[official] 结论: ✅ 引擎 RNG 消费序列与官方判题器逐位一致
```

`317,680` 次比较全部相等。因为 `update_pheromone` 每回合对**每只活蚂蚁**写一次信息素，只要任何一回合任何一只蚂蚁位置不同，722 格里就会出现差异 ⇒ **两边的抽签次数与顺序逐次相同**。这比只比 `base_hp/coins/towers` 强得多（补充了 `docs/engine_consistency_probe.md` 的 1792 回合对拍）。

### 2.3 附带工具：不用改引擎也能"数抽了几次"

LCG 是双射且周期 2^48 ⇒ 由"回合前后 `rng_state_now()`"可以唯一反解这一回合抽了 k 次。`probe_engine_rng.py census` 就是这么做的（纯只读，不需要任何新接口）：

```
def draws_between(s0, s1):  # s_{n+1} = (M*s_n + 11) & (2^48-1)
    for k in range(1, LIMIT):
        s = step(s); 
        if s == s1: return k
```

两种独立方法（反解 vs 引擎内计数器）在 440 / 512 回合上给出**完全相同的总数**（5344 / 8794）⇒ 互证。

---

## 3. 当前 pybind 暴露面

`code/cpp_engine/binding.cpp`（103 行）暴露的东西：

| 类别 | 方法 | 位置 |
|---|---|---|
| 建局 | `init_game(seed, movement_policy="enhanced", cold=True)` | `binding.cpp:43-45` → `m1_support.cpp:18-69` |
| 出招 | `apply_operation_list(player, [(t,a0,a1)])`（cold 语义：非法静默跳过） | `binding.cpp:46-59` → `m1_support.cpp:171-207` |
| 推进 | `advance_round()` = `&Game::next_round` | `binding.cpp:60` |
| 查询 | `round / is_ended / winner / base_hp / coin / base_levels / weapon_cds / super_weapon_usage / valid_cell / tower_build_cost` | `binding.cpp:61-77` |
| 快照 | `tower_snapshot / ant_snapshot / tower_details / ant_details / pheromone_flat / active_effects` | `binding.cpp:65-76` → `m1_support.cpp:71-151` |
| 合法性 | `can_apply_cold`（真 clone+apply）/ `can_apply_dry`（复刻检查，便宜） | `binding.cpp:78-101` → `m1_support.cpp:253-400` |
| 分支 | `clone()` = `deep_clone` | `binding.cpp:102` → `m1_support.cpp:402-461` |
| **RNG** | **`rng_state_now()` —— 只读** | `binding.cpp:69` → `game.hpp` 新增 inline（官方 `game.hpp` 无此方法，`rng_state` 成员本身是官方的） |

**回答用户怀疑的那点**：*RNG 可以从 Python 观测（`rng_state_now()` 早就有，实测可用），但**不能控制**——没有 `set/consume`，也不能"多抽几次"。*

```
$ python -c "import sys; sys.path.insert(0,'code/cpp_engine'); import native_game as ng; g=ng.NativeGame(); g.init_game(seed=7,...)"
pyd: D:\2026智能体大赛_新2_distill\code/cpp_engine\native_game.pyd
['active_effects','advance_round','ant_details','ant_snapshot','apply_operation_list','base_hp',
 'base_levels','can_apply_cold','can_apply_dry','clone','coin','init_game','is_ended','pheromone_flat',
 'rng_state_now','round','super_weapon_usage','tower_build_cost','tower_details','tower_snapshot','valid_cell',
 'validate_self','weapon_cds','winner']
rng_state_now -> 25214903914       # == (7 ^ 25214903917) & (2^48-1)  ✅ 初始化公式确认
after 1 round -> 179732370459700   # 第 1 回合抽了 2 次（两次刷怪抽签）
after 2 rounds -> 179732370459700  # 第 2 回合抽了 0 次（蚂蚁都是 Conservative ⇒ argmax 不抽签）
```

> ⚠️ 注意最后两行：**抽签次数是状态相关的**，可以是 0。所以"按回合固定 inject"与"按抽取次数 inject"不是一回事。

**顺带发现 1（重要，写进 §6 风险）**：现役 `native_game.pyd` 暴露了 `validate_self`，而**当前 `binding.cpp` 里没有这个符号**（全仓 `grep -r validate_self` 的 py/md 命中数 = 0）。⇒ 现役 pyd 不是用当前这份 `binding.cpp` 编出来的（与 `docs/engine_consistency_probe.md` §7.4 的"未做逐位校验"呼应）。

**顺带发现 2（把这个隐患基本排掉了）**：我顺手用**当前源码 + `build.sh` 的 flags** 重新编了一份 pyd（`C:\Temp\rng_probe\out\live_rebuild\native_game.pyd`），与现役 pyd 在 440 回合上对拍：

```
[digest] module=live 回合=440 终局hp=0,28 rng=155510713659591   ← 重建产物
[digest] module=live 回合=440 终局hp=0,28 rng=155510713659591   ← 现役 pyd
重建产物 == 现役 pyd 逐回合摘要: True   rng 序列: True
```

⇒ 现役 pyd 多出来的只有 `validate_self`（不影响判定、全仓无调用），**其余行为与"当前源码重建"逐位一致**。这对"要不要重建 pyd"是个明显的减风险信号。

---

## 4. 改造方案（只加接口，不改任何游戏逻辑）

### 4.1 建议新增的最小接口（3 个，全部 header inline）

```cpp
// code/cpp_engine/include/game.hpp  —— 紧跟在已有的 rng_state_now() 后面
unsigned long long rng_state_now() const { return rng_state; }          // 已有
// ── 新增 ──
unsigned long long get_rng_state() const { return rng_state; }
void set_rng_state(unsigned long long state) {
    rng_state = state & ((1ULL << 48) - 1);   // 折叠到 48 bit，保持 LCG 不变量
}
void consume_rng(int count) { for (int i = 0; i < count; ++i) next_random(); }
```

- **不改 `game.cpp` 一个字节**（`next_random()` 是 private，但 header inline 方法在类内 ⇒ 可以调）；
- 三个方法都**不改变任何判定**：不调用它们时引擎行为逐位不变（已证，§4.4）；
- `set_rng_state` 做 48-bit 折叠而不是断言，是为了"任意 64-bit 输入都合法"（探针实测 `set_rng_state(1<<50)` → `0`）；
- 不建议顺手加 `random_float/uniform` 之类的高层接口（会被诱导去"改分布"，反而破坏与官方的一致性）。

**可选的第 4 个（探针专用，不进现役）**：`rng_site_counts()` 之类的 per-site 计数器，用于统计"哪个消费点抽了多少次"（本文档 §1.3 的数字就来自它）。

### 4.2 pybind 绑定（3 行）

```cpp
// code/cpp_engine/binding.cpp:69 之后
        .def("rng_state_now", &Game::rng_state_now)
        .def("get_rng_state", &Game::get_rng_state)
        .def("set_rng_state", &Game::set_rng_state, py::arg("state"))
        .def("consume_rng", &Game::consume_rng, py::arg("count"))
```

facade 侧（可选，1 行 passthrough；不改它也能用 `sim._g.consume_rng(k)`）：

```python
# code/my_ai/az_intent/game_state_facade.py（**本次不动，仅示意**）
    def consume_rng(self, count: int) -> None:
        self._g.consume_rng(int(count))
```

### 4.3 重建步骤（含 MinGW / 中文路径 / build.sh 的坑）

> **本次没有重建现役 `native_game.pyd`**（约束：有训练进程在跑；重建会改变现役数值行为）。我是在**副本**上完成全部改造 + 编译 + 验证的，脚本见 `code/cpp_engine/probe_rng/build_probe_pyd.py`。下面的步骤是"真要落地时"照抄的版本。

```powershell
# 0) 前置：MinGW + 两个 python（编译用 traing env 的 python 3.11 头文件，pybind11 头文件来自 base 环境）
#    C:\mingw64\bin\g++.exe  (g++ 16.1.0)
#    D:\anaconda3\envs\pytorch-gpu\python.exe   (3.11.15)  ← 目标 ABI
#    D:\anaconda3\python.exe                     (3.13.9)   ← 只用来取 pybind11 头文件（header-only）

# 1) 中文路径 + TMP 的坑：cc1plus 用 TMP 写 .o，默认 C:\Users\<中文名>\AppData\Local\Temp
#    会 Permission denied。必须把 TMP/TEMP/TMPDIR 指到纯 ASCII 目录。
New-Item -ItemType Directory -Force C:\Temp\rng_probe\tmp | Out-Null

# 2) build.sh 有两个已知坑，重建前必须先处理：
#    (a) build.sh:8  REPO="D:/2026智能体大赛_新2"   ← 指向**另一个目录**，本工作区是 _distill
#    (b) build.sh:29 用的是 -O2，而官方 Makefile 没有 -O（=-O0，见 docs/engine_consistency_probe.md §1）
#    最小改法（二选一）：
#      - 把 REPO 改成本工作区绝对路径；并把 -O2 改成 -g（想对齐官方）
#      - 或者干脆照 build.sh 的 flags 手敲一遍（本文档用的就是这条，flags 与现役 pyd 一致才可比特对照）

# 3) 编译（产物**不要**覆盖现役 pyd；下面是**实测跑通**的版本，rc=0）
#    注意 pyd 的文件名必须等于 PYBIND11_MODULE 的模块名，否则 import 会报
#    "dynamic module does not define module export function (PyInit_xxx)"。
#    所以"另存一份"要放**另一个目录**里、仍叫 native_game.pyd，不能叫 native_game_new.pyd。
mkdir -p /c/Temp/rng_probe/out
bash -lc 'set -e
export TMP="C:\Temp\rng_probe\tmp" TEMP="C:\Temp\rng_probe\tmp" TMPDIR="C:\Temp\rng_probe\tmp"
cd /d/2026智能体大赛_新2_distill
PBI_INC=$(/d/anaconda3/python.exe -c "import pybind11;print(pybind11.get_include())")
PY_INC=$(/d/anaconda3/envs/pytorch-gpu/python.exe -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])")
/c/mingw64/bin/g++.exe -std=c++17 -O2 -fPIC -shared \
  -I"$PBI_INC" -I"$PY_INC" -Icode/cpp_engine/include \
  code/cpp_engine/src/game.cpp code/cpp_engine/src/ant.cpp code/cpp_engine/src/map.cpp \
  code/cpp_engine/src/building.cpp code/cpp_engine/src/coin.cpp code/cpp_engine/src/item.cpp \
  code/cpp_engine/src/operation.cpp code/cpp_engine/src/aco.cpp \
  code/cpp_engine/src/comm_judger.cpp code/cpp_engine/src/output.cpp \
  code/cpp_engine/m1_support.cpp code/cpp_engine/binding.cpp \
  -L"D:/anaconda3/envs/pytorch-gpu/libs" -lpython311 \
  -o C:/Temp/rng_probe/out/live_rebuild/native_game.pyd
echo rc=$?'
cp /c/mingw64/bin/{libstdc++-6,libgcc_s_seh-1,libwinpthread-1}.dll /c/Temp/rng_probe/out/live_rebuild/
# 实测：rc=0，1,232,056 bytes（2026-10-27，g++ 16.1.0）

# 4) 冒烟 + 与现役 pyd 对拍（-O2 与现役一致才可比）
$env:PYTHONIOENCODING="utf-8"
$py = "D:\anaconda3\envs\pytorch-gpu\python.exe"
& $py code/test_match/probe_engine_rng.py digest --engine-dir C:\Temp\rng_probe\out\live_rebuild --out C:\Temp\rng_probe\out\a.json
& $py code/test_match/probe_engine_rng.py digest --out C:\Temp\rng_probe\out\b.json
# 两次 [digest] 行应该完全一样（回合数 / 终局hp / rng）
```

一键版（本次实际用的）：

```powershell
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe code/cpp_engine/probe_rng/build_probe_pyd.py --smoke
#   → C:\Temp\rng_probe\out\native_game_rng.pyd  + 冒烟 5 项
```

`build_probe_pyd.py` 做的事（= 本文档改造方案的**可执行原型**）：把 `code/cpp_engine/{src,include,m1_support.cpp,binding.cpp}` 拷到 `C:\Temp\rng_probe\eng\`，在**副本**上打 4 类补丁（header 加 3 个方法 / `next_random()` 加计数器 / 11 个消费点打 site tag / `binding.cpp` 改模块名 + 加 4 个 `.def`），用**与 `build.sh` 相同的 flags** 编译到 `C:\Temp\rng_probe\out\`，并把 MinGW 运行库拷到旁边。每个锚点都断言"恰好出现 1 次"，源码一变就会中止而不是静默打错。

### 4.4 如何验证"不消费时行为与现在逐位一致"

这是整个改造的**验收条件**，一条命令：

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:CUDA_VISIBLE_DEVICES=""
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_engine_rng.py parity
```

实测输出（现役 `native_game.pyd` vs 打了改造补丁的 `native_game_rng.pyd`，真实对局 440 回合）：

```
[digest] module=live  回合=440 终局hp=0,28 rng=155510713659591
[digest] module=probe 回合=440 终局hp=0,28 rng=155510713659591
[parity] 逐回合强摘要完全一致: True
[parity] rng_state 序列完全一致: True
```

`parity` 之所以要分两个子进程，是因为 pybind11 会拒绝"同一进程注册两个同名 C++ 类型"（`TypeError: generic type: type "NativeGame" is already registered`）——这也是**同步验证**时唯一的环境坑。

配套的三条回归（都建议在换 pyd 前后各跑一次）：

```powershell
python code/test_match/probe_engine_rng.py determinism      # 同进程/交错/跨进程逐位一致
python code/test_match/probe_engine_rng.py official --module probe   # vs 官方 judge_O0.exe：722 格信息素全等
python code/test_match/probe_engine_rng.py clone --cut 120 --horizon 40  # clone 后重跑一致（改造没引入隐藏状态）
```

---

## 5. 实验设计建议（含本次实测）

### 5.0 先把"这个方向到底在做什么"说清楚

因为**引擎在给定 seed + 操作序列时是完全确定的**，所以"主动多消费 RNG"产生的是：

> **在这个 seed 下永远不会发生的局面**（同一个 seed 的同一个状态，只有一条未来）。

它**不是**"更真实的采样"。它等价于"换一个反事实的种子之后会怎样的世界"。所以它只能当**平滑/降噪工具**用，不能当"更准的世界模型"用。这一点在 §6 的第 1 条风险里再重复一次。

在这个前提下，"有没有用"取决于两件事：
- (a) 它能不能把**叶子抽样噪声**降下来（能，且几乎免费）；
- (b) 它会不会把**候选之间的真实差异**抹平（会，抹平多少要实测）。

### 5.1 五个候选方案 + 实测结果

实验设置（探针 `probe_engine_rng.py perturb`，全部用真实对局 `10.log` 的操作流，切片起点 = 第 120 回合，`horizon=40`，`N=64` 分支；"action" 组复刻现役 `decay C=0.3,k=1` 的对手随机出招）：

| 方案 | 做法 | 不同叶子轨迹 | 不同 (hp0,hp1) | sd(Δhp) | 相对"不扰动分支(k=0)"的位移 | 额外成本 |
|---|---|---|---|---|---|---|
| **engine**（推荐先试） | `clone()` 后 `consume_rng(idx)`，之后双方都按日志原操作走 | **64/64** | 23/64 | 1.94 | \|mean−k0\|/se = **2.78** | `consume_rng(64)` ≈ **0.4 µs**/分支 |
| engine_pr | 每回合以 `p(i)=C·e^(−i/k)` 多消费 1 次 | **8/64** ❌ | 7/64 | 0.76 | 1.32 | 同左（更少） |
| reroll | 每回合以 `p(i)` 把 `rng_state` 重掷成随机值 | 25/64 | 20/64 | 1.59 | 1.50 | 同左 |
| **action**（现役配置） | 只随机对手出招（`decay .3,1`） | **20/64** | 8/64 | 1.13 | 3.98 | 每个随机回合都要真的 `apply+advance`（≈ 现役 4.8 s 的来源） |
| both | engine + action | 64/64 | 26/64 | 2.12 | 10.90 | ≈ action |

同一套实验在 `cut=300` 上复现（结论不变，`engine` 64/64、`action` 20/64、`engine_pr` 8/64）。

**三个可以直接用的结论**：

1. **"起点扰动一次"性价比压倒性最高**：0.4 µs 换 64/64 条不同轨迹；而现役"对手随机出招"在同样 N=64 下只有 **20/64**（其中一半分支从头到尾一次都没触发，`首次分歧回合 = 40` 即"永不分歧"）——这复现了你观察到的 **17/64**。也就是说：**引擎 RNG 扰动比现役的动作随机更"能出多样性"，而且几乎不花时间。**
2. **"每回合以概率 p 多消费 1 次"是五个方案里最差的一个（8/64）**。原因不是"幅度小"，而是**相邻偏移高度相似 + 分支之间只差"第一次触发发生在第几回合"**：`p(i)=0.3e^{−i}` 的 40 回合里只有 24/64 个分支会触发≥1 次，触发回合又只有少数几种 ⇒ 大量分支逐位相同。（这条对"想法本身"是个负面证据，值得先知道。）
3. **叠加是有效的**：`both` 给出 64/64 轨迹 + 26/64 个不同终局 hp，sd 从 1.13（纯动作随机）升到 2.12。

> 读表注意：`|mean−k0|/se` 这一列，只有 `engine` / `engine_pr` / `reroll` 可以当"平滑位移"读（它们的 k=0 分支 = 完全不扰动 = 真实流）。`action` / `both` 的 k=0 分支**本身也带动作随机**，所以那一列主要反映"换了对手策略"，不是"RNG 平滑"。

### 5.2 更值得做的用法：**配对（common random numbers）**

搜索真正要的是**候选之间的差**；RNG 抖动对同一个候选是"噪声"，对两个候选是**共模噪声**（只要它们的抽签次数还对得上）。所以：**给所有候选喂同一组 RNG 偏移**，差值上的噪声会抵消。实测（`probe_engine_rng.py crn`，N=64 偏移，H=40）：

| 切片 | 候选 A / B | 配对 sd | 不配对 sd | sd 降幅 | \|t\| 配对 / 不配对 |
|---|---|---|---|---|---|
| cut=60 | 建塔 (4,2) vs (4,3) | **1.097** | 2.491 | **−56%** | 0.11 / 0.05 |
| cut=80 | 建塔 (4,2) vs (4,3) | 1.811 | 2.372 | −24% | 1.10 / 0.84 |
| cut=120 | 降级塔#21 vs 降级塔#22 | 1.861 | 2.128 | −13% | 0.81 / 0.70 |
| cut=200 | HOLD vs 降级塔#40 | 3.137 | 3.610 | −13% | 2.47 / 2.15 |
| cut=300 | HOLD vs 降级塔#75 | 1.832 | 2.880 | −36% | 0.20 / 0.13 |

**配对稳定地把"候选差"的 sd 压掉 13%–56%（等价于免费多拿 1.2–2.3 倍有效样本），且一次都不变差。** 机制上的上限来自"两个候选一旦局面分叉，抽签次数就不同、流就错位了"——所以降幅随 horizon/分叉早晚变化（cut=60 好处最大）。

> 这也解释了为什么"配对"不能做到像统计里 CRN 那样彻底消噪：引擎 RNG 的**消耗次数是状态相关**的（§1.3 里同一局不同回合 0→37 次），一旦分叉，对齐就断了。

### 5.3 建议的落地顺序

1. **先只加接口**（§4.1/4.2），跑 `parity` + `official` 两个回归，确认"不消费 = 逐位不变"。这一步不需要改任何 AI 逻辑。
2. **A/B 实验 1（变量：叶子噪声）**：`rollout_mode=random`（现役）vs `rollout_mode=random + 起点 consume_rng(idx)`，同预算（总回合数相同）比 `(σ_within/√N_eff)/|β|` 与对局胜率。预期：`σ_within` 降（叶子更分散但仍独立）、`N_eff` 升；风险：`β` 也被抹平（见下）。
3. **A/B 实验 2（变量：配对）**：候选 × 共享偏移矩阵 vs 独立偏移，比**同一个候选对的排序稳定性**（同一局面重复决策，看 argmax 抖动率）。这是最可能稳定赢的一项，且完全不改"要不要引入额外随机"的取舍。
4. **不要**先做"逐回合概率注入"（实测最差，§5.1）。

### 5.4 哪些方案"加偏差"、哪些"降方差"（明确区分）

| 方案 | 性质 | 说明 |
|---|---|---|
| engine（起点 consume k） | **加平滑/改估计量**（可疑偏差） | 目标从"这个 seed 下会发生的未来"变成"反事实种子集合上的平均"。它降低**叶子抽样噪声**，但把"与实际 RNG 流的耦合"抹平；且实测 `\|mean−k0\|/se ≈ 2.8` ⇒ **不守恒**真实流的值 |
| engine_pr / reroll | 同上，但**多样性收益更小**（8/64、25/64） | 不建议 |
| action（现役） | **更接近真实不确定性**（对手未知），但代价是墙钟时间 | 是"合法"的随机源：对手的策略确实是未知的 |
| both | 两者叠加 | 多样性最高，平滑最强；只建议在"确认 β 不被抹平"之后用 |
| **CRN 配对（§5.2）** | **纯降方差**（同一偏移下比较） | 不改变"给定偏移下的值"，只让候选**差**的噪声抵消 ⇒ 若只用一个偏移（如 k=0），配对是完全无偏的方差缩减 |
| 用多偏移平均 + 配对 | 降方差 + 轻度平滑 | 折中：偏移数 M 控制平滑强度，M=1 退化成"完全无偏+CRN" |

> 一句话：**"多加随机"不是目的；"让候选之间可比"才是。** 如果只能做一件事，做 CRN 配对（§5.2），而不是"多消费 RNG"。

---

## 6. 风险清单

1. **与官方引擎的等价性**：*只要不调用新接口*，逐位等价（§4.4 已证：440 回合、rng_state 序列、722 格信息素）。**一旦在 rollout 里调用 `consume_rng/set_rng_state`，该 rollout 的局面在这个 seed 下永远不会出现** ⇒
   - ❌ 绝对不能用于**判题、回放、数据采集/自对弈采样**（会污染训练标签：标签应该是"真实对局的结局"）；
   - ✅ 只能用于**搜索内部的打分**（rollout 叶子）。
2. **扰动幅度不可控（既不是"轻微"也不是"永远立刻炸"）**：在真实对局切片上偏移 1 步就能让 rollout 第 1 回合分叉；而在"当回合根本不抽签"的局面里，偏移 1 步可以连续几回合看不出来（§1.4 实测）。但**一旦分叉就永久错位**（之后所有回合的 `rng_state` 都不同），所以"失真度"这个旋钮**不存在**，只能选"改/不改"，或者用"多偏移平均"控制平滑的期望强度。
3. **值网分布外风险（最大的实际风险）**：值网训练数据来自**真实对局**；RNG 扰动后的叶子可能出现"双方蚂蚁被随机乱走"这类真实对局罕见的局面 ⇒ 值网在这些点上的评分**没有校准保证**，可能系统性偏移。表现可能是"分数更平"而不是"更准"。**必须用对局胜率 A/B 判定，不能只看 σ 变小。**
4. **现役 pyd 与源码不完全同步（已量化，风险可控）**：`native_game.pyd` 多一个 `validate_self`（源码里没有、全仓无调用）；用当前源码 + `build.sh` flags 重建的 pyd 在 440 回合上与现役**逐位一致**（§3、§4.3 实测）⇒ 行为漂移基本排除。真正的风险只剩"**换 pyd 的时刻**"：换上去会改变正在跑的那批实验的数值行为（哪怕理论上等价）。建议：等当前批跑完 → 换 → 先跑 `determinism`/`parity`/`official` 三条回归，再开新批。
5. **`build.sh` 本身有两个坑**：`build.sh:8` 的 `REPO` 指向 `D:/2026智能体大赛_新2`（不是 `_distill`）；`build.sh:29` 是 `-O2` 而官方 Makefile 是 `-O0`（见 `docs/engine_consistency_probe.md` §1）。重建前必须处理，否则要么编不过，要么与官方二进制不同源。
6. **中文路径坑**（本机已验证）：MinGW `std::ofstream` 在中文路径下**静默失败**（官方判题器的 replay）、`cc1plus` 在中文 `TMP` 下**Permission denied**。所有编译/落盘一律用 `C:\Temp\...`。
7. **`seed = 0` 退化**：`map.cpp:136` 的乘性 LCG 无增量 ⇒ 信息素全场 8.0（§1.2）。若实验里用 seed 0，会得到一张没有信息素梯度的地图，结论不可外推。
8. **`set_rng_state` 的边界**：必须 mask 到 48 bit（已实现）。`set_rng_state(0)` 是合法状态（下一步 `next_random()` 给出 11），不要拿 0 当"未初始化"标记。
9. **并发**：pybind 绑定没有 `call_guard<gil_scoped_release>` ⇒ 从 Python 侧调用会**持有 GIL**、天然串行；但引擎里有一个**文件级全局** `lcg_seed`（`map.cpp:3`，只在 `init_pheromon` 里写）。多进程各自独立没问题（每个进程有自己的数据段），但**将来若有人给 pybind 加上 GIL 释放**，"一个进程里并发建局"会因为共享 `lcg_seed` 而互相污染信息素初始化。新接口不改变这点，只是提醒别顺手加 GIL 释放。
10. **`rng_state` 与 `random_seed` 是两个东西**：`set_rng_state` 不会改 `random_seed`（`output.save_seed` 落盘的仍是原 seed）。如果做实验要"换种子重放"，要么用 `init_game(seed)`，要么清楚 `random_seed` 只影响 replay 记录与信息素初始化。

---

## 7. 复现清单（全部命令都跑过）

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:CUDA_VISIBLE_DEVICES=""
$py = "D:\anaconda3\envs\pytorch-gpu\python.exe"

# ── 只读部分（不需要编译，用现役 pyd）──
& $py code/test_match/probe_engine_rng.py determinism            # 确定性：同进程/交错/跨进程
& $py code/test_match/probe_engine_rng.py census                 # 每回合抽了几次（LCG 反解）
& $py code/test_match/probe_engine_rng.py clone --cut 120 --horizon 40

# ── 探针版引擎（副本编译，不碰现役 pyd）──
& $py code/cpp_engine/probe_rng/build_probe_pyd.py --smoke       # → C:\Temp\rng_probe\out\native_game_rng.pyd
& $py code/test_match/probe_engine_rng.py parity                 # ✅ 不消费 RNG 时与现役逐位一致
& $py code/test_match/probe_engine_rng.py official --module probe  # ✅ vs 官方 judge_O0.exe：722 格信息素全等
& $py code/test_match/probe_engine_rng.py sites  --module probe  # 消费点分布（§1.3 的表）
& $py code/test_match/probe_engine_rng.py perturb --module probe --cut 120 --horizon 40 --branches 64 --prob-mode decay
& $py code/test_match/probe_engine_rng.py crn    --module probe --cut 60  --horizon 40 --branches 64
```

| 文件 | 作用 |
|---|---|
| `code/test_match/probe_engine_rng.py` | 10 个模式：`determinism / census / clone / interleave / parity / sites / perturb / crn / official / digest`（只读 + 可选探针 pyd） |
| `code/cpp_engine/probe_rng/build_probe_pyd.py` | 把改造方案**自动化**成"复制→打补丁→编译→冒烟"，产物在 `C:\Temp\rng_probe\out\` |
| `code/cpp_engine/build_probe/rng_probe/*.json` | 摘要/统计落盘（已 gitignore） |

---

## 8. 未验证 / 未做（如实列出）

1. **没有重建现役 `native_game.pyd`**（约束：有训练进程在跑；且重建会改数值行为）。改造方案在**副本**上编译 + 全量验证通过（`parity` / `official`），但"把新 pyd 换上去并跑一轮真实对局/训练"没做。
2. **没有把 CRN 或 engine 扰动接进 `FlatRolloutGreedy` 做真实 A/B 对局**（约束：不改 `code/my_ai/az_intent/*`）。注入点已经确定：`flat_rollout_greedy.py:427` `sim = child.clone()` 之后（起点偏移）或 `:435` 的 `for _ in range(self.horizon)` 循环内（逐回合重掷）——注意该文件正在被别人改（当前工作区已加 `rollout_prob_mode="reverse"`），行号会漂移，按代码位置找而不是按行号。收益必须靠对局胜率判，不能靠本文档的 `sd` 数字。
3. **`σ_within/√N_eff / |β|` 里的 β 没有量**：本次用的是 `Δhp` 作为叶子分数的代理（不开值网，避免干扰在跑的实验）。真正的 β 需要用值网跑候选打分才能得到。
4. **没有做 MSVC 构建**（沿用 `docs/engine_consistency_probe.md` §7.1 的遗留问题：`double` 打分 + 排序在 MSVC 下是否有 FP 差异仍未排除）。
5. **`native_game.pyd` 的 `validate_self` 来源仍未查清**（疑似历史实验构建残留；全仓无调用，故重建不会打断脚本）。但"行为是否等于当前源码"已经用 440 回合逐位对拍确认（§3 顺带发现 2）。
6. **官方 `main.exe` 未跑**：`official` 模式用的是 `code/cpp_engine/build_probe/judge_O0.exe`（2026-10-26 用官方 `game/src` + 官方 flags 编的），不是 `Ant-Game/game/output/main.exe`（仓库里本来就没有预编译产物）。
