# 引擎 RNG 接口 + rollout 主动消费：「实现报告」

- 时间：2026-10-27
- 前置调研：`docs/engine_rng_survey.md`（RNG 消费点全表、确定性证明、改造方案）
- 结论摘要：**引擎 RNG 接口已落地（独立 pyd，现役一个字节不改）**；**rollout 开关已落地（默认关、零影响已逐位验证）**；`k=48/N=64/H=16/share_prefix=1` 实测：**只消费引擎 RNG（const p=0.5）拿到 64.0/64 条不同叶子、耗时 7.63 s，而现状"动作随机 decay(1.0,3)"是 62.4/64、11.15 s ⇒ 用 68% 的时间拿到略高的多样性**（A/E 两组在**同一进程、同一批局面/重复**里复测过，见 §3.1）。

---

## 0. 交付物 & 快速上手

| 文件 | 作用 |
|---|---|
| `code/cpp_engine/build_rng_variant.py` | 打 2 处补丁（`game.hpp` 3 个 header-inline + `binding.cpp` 3 行 `.def`）+ 编译，产出 **`code/cpp_engine/rng_build/native_game.pyd`**（现役 pyd 不动） |
| `code/my_ai/az_intent/rng_engine.py` | 引擎选择器：`enable()` 把 `rng_build/` 插到 `sys.path[0]`；`available()/describe()` |
| `code/my_ai/az_intent/flat_rollout_greedy.py` | **新增**开关 `rollout_rng_perturb` / `rollout_rng_consume` / `rollout_rng_key` + 前缀树缓存键改造 |
| `code/test_match/probe_engine_rng.py` | 调研期探针（`parity` 等 10 个模式），本次给它加了 `--logs` 批量 parity |
| `code/test_match/probe_rng_perturb_zero_impact.py` | **零影响证明**：改动前/后同一进程、同模型、同快照，逐字段哈希比对 |
| `code/test_match/expl_rng_perturb.py` | ③ 的四组（+两组补充）实测 |
| `code/test_match/smoke_rng_perturb_deploy.py` | 部署链路冒烟：只靠环境变量打开开关，走 `az_bridge_ai.build_engine()` |

```powershell
$py = "D:\anaconda3\envs\pytorch-gpu\python.exe"
# 1) 建独立 pyd（不进 git：.gitignore 里有 *.pyd / *.dll）+ 冒烟 + 与现役 pyd 对拍
& $py code/cpp_engine/build_rng_variant.py --check
# 2) 零影响（默认关闭时逐位不变）
& $py code/test_match/probe_rng_perturb_zero_impact.py --before-rev HEAD --out code/cpp_engine/build_probe/rng_impl/zero_impact.log
# 3) 四组实测
& $py code/test_match/expl_rng_perturb.py --ablate-key
# 4) 部署链路冒烟（只加环境变量）
$env:AZAI_FLAT_RNG_PERTURB="1"; & $py code/test_match/smoke_rng_perturb_deploy.py
```

---

## 1. ① 引擎 RNG 接口（独立模块，现役 pyd 一字未动）

### 1.1 改了什么

唯一的源码改动是 **2 处、共 3+3 行**，`game.cpp` 与官方源码 **MD5 相同**（构建脚本每次都会断言这一点）：

```cpp
// code/cpp_engine/include/game.hpp（紧跟已有的 rng_state_now() 之后）
unsigned long long get_rng_state() const { return rng_state; }
void set_rng_state(unsigned long long state) {
    rng_state = state & ((1ULL << 48) - 1);   // 折叠到 48 bit，保持 LCG 不变量
}
void consume_rng(int count) { for (int i = 0; i < count; ++i) next_random(); }
```
```cpp
// code/cpp_engine/binding.cpp（.def("rng_state_now", ...) 之后）
.def("get_rng_state", &Game::get_rng_state)
.def("set_rng_state", &Game::set_rng_state, py::arg("state"))
.def("consume_rng", &Game::consume_rng, py::arg("count"))
```

三个方法都是 header-inline ⇒ `game.cpp` 不需要重编改动（虽然整体仍要重链），语义上"只加接口、不改逻辑"。

### 1.2 为什么产物叫 `rng_build/native_game.pyd`（而不是 `native_game_rng.pyd`）

pybind11 的模块名在 `PYBIND11_MODULE(native_game, m)` 里写死，**Python 要求 `.pyd` 文件名 == 模块名**（否则 `ImportError: dynamic module does not define module export function (PyInit_native_game_rng)`，我实测踩过）。因此"独立模块"的正确做法是**目录隔离**：

- 现役：`code/cpp_engine/native_game.pyd`（mtime 2026-08-09，本次**未触碰**）
- 变体：`code/cpp_engine/rng_build/native_game.pyd`（+ 3 个 MinGW DLL）

好处：`import native_game` 的所有既有代码（`game_state_facade.initial`）**一行都不用改**，只靠 `sys.path` 顺序切换；坏处：两套引擎不能同进程共存（pybind11 会报 `type "NativeGame" is already registered`），所以 `rng_engine.enable()` 必须在**第一次建局之前**调用 —— 官方链路 `az_bridge_ai.main()` 正好是 `build_engine()`（构造搜索器）→ `GameState.initial()`，顺序天然满足（§3 实测）。

```python
from my_ai.az_intent import rng_engine
rng_engine.enable()          # 之后 GameStateFacade 拿到的就是 RNG 变体
rng_engine.describe()        # 'native_game=...\rng_build\native_game.pyd rng_build=✓'
```

### 1.3 parity 证据：不消费 RNG 时与现役 pyd **逐位一致**

`parity` 比较"逐回合强摘要（`round/base_hp/coins/winner/rng_state/tower_details/ant_details/weapon_cds/base_levels/active_effects/722 维信息素 double`）"与 `rng_state` 序列，**分两个子进程**跑（同进程会被 pybind11 拒绝）：

```
$ python code/test_match/probe_engine_rng.py parity --logs <6 条真实对局日志>
[parity] 候选引擎 = code/cpp_engine/rng_build/native_game.pyd
[parity] 10.log     回合= 440 hp=(0, 28) vs (0, 28)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 7.log      回合= 463 hp=(0, 19) vs (0, 19)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 865.log    回合= 512 hp=(38, 24) vs (38, 24)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 1014.log   回合= 512 hp=(16, 31) vs (16, 31)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 10.log     回合= 426 hp=(9, 0) vs (9, 0)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 10.log     回合= 377 hp=(15, 0) vs (15, 0)  逐回合强摘要一致=True  rng_state 序列一致=True
[parity] 汇总: ✅ 全部逐位一致 （6 条日志）
```

共 **2730 回合**，含 3 个不同 ladder 目录/对手。另有调研期的更强证据（未变）：官方 `judge_O0.exe` vs 我们 → **440 回合 × 722 格信息素 `max|diff| = 0`**（`probe_engine_rng.py official`），说明 RNG 抽取序列与官方判题器逐次相同。

### 1.4 构建的两个坑（脚本已处理）

1. **MinGW 的 ld 不能把 `-o` 指到含中文的仓库路径**：`ld.exe: cannot open output file D:\2026...\rng_build\native_game.pyd: No such file or directory`（目录存在也没用，是窄字符 `fopen` 的问题）⇒ 先链到 `C:\Temp\rng_variant_build\out\`，再用 Python 的 `shutil.copy2` 拷回（Python 走宽字符 API 没问题）。
2. `TMP/TEMP/TMPDIR` 必须是纯 ASCII（否则 `cc1plus` 写 `.o` 报 Permission denied）。

---

## 2. ② rollout 侧：开关式消费机制

### 2.1 参数 / 环境变量

| 参数 | 默认 | 说明 |
|---|---|---|
| `rollout_rng_perturb` | `None` ⇒ 读 `AZAI_FLAT_RNG_PERTURB`（默认 `0`=关） | 总开关。开时 `__init__` 会 `rng_engine.enable()` |
| `rollout_rng_consume` | `1` | 每回合 Bernoulli 命中时**消费几次** |
| `rollout_rng_key` | `"state"` | 前缀树缓存键是否带 `rng_state`。`"none"` **仅诊断用**（会错误合并，见 §4.3） |

**概率调度复用现有 `rollout_prob_mode` / `rollout_decay_c` / `rollout_decay_k` / `rollout_act_prob`**：把原来内联在 `_decide_actions` 里的 `p(r)` 抽成 `_prob_for_round(r)`（`const | decay | first | reverse` 四种语义逐位不变），动作随机与 RNG 消费**共用同一个 p(r)**。

### 2.2 语义（精确到"哪个时刻"）

```
for r in 0..H-1:
    ops = 对手随机出招（若 rollout_mode=="random"）
    apply(ops)
    if Bernoulli(p(r)):  sim.consume_rng(k)     # ← 消费位置：apply 之后、advance 之前
    sim.advance_round()
```

- 消费对象是 **`child.clone()` 出来的 rollout 实例**，裁判/本局状态完全不受影响 —— 实测 `search()` 前后本局 `rng_state` **完全不变**（§3 冒烟输出 `before == after`）。
- 消费决策用**独立**的 numpy 流（种子 = `((rollout_seed*1000003 + decision_idx)*100003 + cand*101 + k) + 500003`），**不和动作抽签抢 draw** ⇒ 开/关 perturb 不会改变动作随机的抽样序列，两种随机源正交、可单独 A/B。
- `rollout_mode="idle"` 时 `_decide_actions` 现在显式返回 `[]`（新加的 `rollout_mode != "random"` 守卫）；这条守卫对现有配置是 no-op（前缀树路径原本只在 `rollout_mode=="random"` 时才进）。

### 2.3 前缀树缓存键：`(parent_id, ops_token, rng_state)`

```python
key = (cur, token, self._rng_after(parent_rng_state, k))     # 开 perturb 时
key = (cur, token)                                           # 关闭时（与改动前**完全一样**）
```

**键里的 `rng_state` 对应哪个时刻**：*本回合 `ops` 已施加、`advance_round()` 之前*的引擎 RNG 状态（即"父状态的 rng_state 前推 k 步"）。理由：`(父状态, ops, 这个 rng_state)` 三者一起唯一决定子状态（`advance_round()` 是确定的），所以它是**精确判据**；又因为 LCG 是双射，"前推 k 步后的状态"与"消费了 k 次"一一对应，两者信息等价，但**用状态表达不依赖"k 只在 1 这个取值"这个假设**，将来 `rollout_rng_consume>1` 或换成 `set_rng_state(随机值)` 都不用改键。

- `_rng_after()` 只是一个 2 行的 LCG 前推（`s←(25214903917*s+11)&(2^48-1)`），一次乘法级别；它与引擎 `consume_rng` 的一致性由探针断言（`consume_rng(5) == 手推 LCG` → True）。
- **前缀共享会"摊薄"消费**：消费只发生在**新建节点**时，共享同一前缀的轨迹共享同一个消费决策（语义正确：同一状态 + 同一 rng ⇒ 同一世界）。所以实际消费次数远小于 `N×H`（例：C 组 64 条分支 × 16 回合，实际只消费 2299 次 ≈ 48 次/候选）。
- `leaf_tokens`（去重加权的"轨迹签名"）也带上了消费次数，否则两条消费模式不同的轨迹会被当成同一条。

### 2.4 默认关闭 ⇒ **逐位不变**（证据）

`probe_rng_perturb_zero_impact.py`：同一进程里装"改动前版本"（从 `git show HEAD` 取）与"改动后版本"，共用同一份 `net_fn / value_model / feature_extractor` 和同一串真实对局快照，逐字段打哈希（`last_scores` 的 float64 原始字节、`visit_policy`、`chosen_index`、`chosen_bundle`、`last_rollouts`、`last_rounds`、`last_effective_samples`、**`last_tree_stats`(节点/叶子/去重叶子)**、`last_rng_consumed`）：

```
[zero] idle+prefix    r40   p0 | 一致 ✅ | scores=fe0ac8bf4fe14f91 tree=None chosen=39 rng_used=0 rollouts=64 rounds=49152
[zero] idle+prefix    r200  p1 | 一致 ✅ | scores=db3868d5e7bf614c tree=None chosen=16 rng_used=0 rollouts=64 rounds=49152
[zero] random+prefix  r40   p0 | 一致 ✅ | scores=9ccf817e441d93f5 tree=(389, 64, 25) chosen=39 rng_used=0 rollouts=64 rounds=49152
[zero] random+prefix  r40   p1 | 一致 ✅ | scores=b0cd8ab48d086b92 tree=(334, 64, 22) chosen=31 rng_used=0 rollouts=64 rounds=49152
[zero] random+prefix  r200  p0 | 一致 ✅ | scores=5b184c6ea195dcf1 tree=(367, 64, 24) chosen=0  rng_used=0 rollouts=64 rounds=2048
[zero] random+prefix  r200  p1 | 一致 ✅ | scores=d9e5c314f9a0ca82 tree=(90, 64, 6)   chosen=31 rng_used=0 rollouts=64 rounds=49152
[zero] random+naive   ...     | 全部一致 ✅（4 个快照）
[zero] 结论：✅ 全部逐位一致（默认关闭 ⇒ 零影响）
```

**关键点**：`tree=(节点, 叶子, 去重叶子)` 也逐位相同 ⇒ **共享率没有变化**。这是因为默认路径的缓存键**仍是老的 `(cur, token)`**（代码里 `if self._rng_on and rollout_rng_key=="state"` 才加 rng_state），而不是"加了键以后恰好一样"。覆盖了前缀树路径、朴素路径、idle 路径三条。

### 2.5 部署链路冒烟（只靠环境变量）

```
$ $env:AZAI_FLAT_RNG_PERTURB="1"; python code/test_match/smoke_rng_perturb_deploy.py
[smoke] AZAI_FLAT_RNG_PERTURB=1  rng_build 已构建=True
[rng_engine] ✓ 已把 ...\code\cpp_engine\rng_build 插到 sys.path[0]（之后 import native_game 命中 RNG 变体）
[smoke] searcher=FlatRolloutGreedy rollout_rng_perturb=True _rng_on=True
[smoke] 建局引擎支持 consume_rng=True rng_state=23591231540191  native_game=...\rng_build\native_game.pyd rng_build=✓
[smoke] 候选=45 chosen=32 tree(node,leaf,uniq)=(719, 64, 64) rng_consumed=16067 rng_missing=0
[smoke] 本局状态 rng_state: before=23591231540191 after=23591231540191 -> 未被 rollout 影响 ✅
[smoke] 结论: ✅ 通过
```

对照（`AZAI_FLAT_RNG_PERTURB=0`）：`_rng_on=False`、建局引擎是现役 pyd（`consume_rng=False`）、`rng_consumed=0`、`tree=None`（idle 模式走朴素路径）—— 与改动前一致。

---

## 3. ③ 实测：`k=48 / N=64 / horizon=16 / share_prefix=1`

口径：3 个局面（`GameStateFacade.initial(seed=10/11/12)` 空推 60 回合）× 3 个重复（`rollout_seed=53/61/71`），每格是**每候选**的均值（48 个候选再平均）。硬上界：**每候选 ≤ 1 + N×H = 1025 节点**。日志：`training_history/runs/20261027_170830_expl_rng_perturb/`、`..._171928_expl_rng_perturb_const/`、`..._173021_expl_rng_AE_sameprocess/`（A/E 的复测见 §3.1）。

| 组 | 随机化来源 | 调度 | 节点/候选 | 占上界 | **去重叶子/64** | **总耗时 s** | rollout s | ms/千节点 | RNG 消费 |
|---|---|---|---|---|---|---|---|---|---|
| **A**（现状） | 动作随机 | decay C=1.0,k=3.0 | 938.7 | 91.6% | **62.4** | **11.169** | 11.082 | 0.246 | 0 |
| B | 仅 RNG | decay C=0.3,k=1.0 | 99.1 | 9.7% | 6.7 | 3.149 | 3.052 | 0.641 | 277 |
| **C** | 仅 RNG | decay C=1.0,k=3.0 | 410.2 | 40.0% | 39.8 | 5.435 | 5.352 | 0.272 | 2299 |
| D | 叠加 | 都 decay C=1.0,k=3.0 | 968.6 | 94.5% | 63.9 | 11.324 | 11.237 | 0.242 | 8391 |
| **E**（补充） | 仅 RNG | **const p=0.5** | 710.7 | 69.3% | **64.0（满）** | **7.588** | 7.501 | 0.220 | 17041 |
| F（补充） | 仅 RNG | const p=1.0 | 17.0 | 1.7% | **1.0** ⚠️ | 2.519 | 2.435 | — | 768 |

A 组的 **11.17 s 与父任务记录的 11.0 s 一致** ⇒ 口径可比（同一 k/N/H、同一 ckpt、同一批局面类型；value 网在 **CPU** 上跑，`load_three_models` 不自动 `.cuda()`，实测 device=cpu、553k 参数）。

### 3.1 直接回答"核心问题"

**A 与 E 在同一进程、同一批局面/重复里复测**（排除机器负载漂移，`training_history/runs/20261027_173021_expl_rng_AE_sameprocess/`，27 次搜索，171 s）：

| 组 | 节点/候选 | 占上界 | 去重叶子/64 | 总耗时 s | rollout s |
|---|---|---|---|---|---|
| A 动作随机 decay(1.0,3) | 938.7 | 91.6% | 62.4 | **11.153** | 11.064 |
| **E 仅RNG const p=0.5** | 710.7 | 69.3% | **64.0** | **7.627** | 7.543 |

⇒ **E 用 A 的 68.4% 墙钟、75.7% 节点，拿到比 A 更高（且封顶）的叶子多样性**；9 次重复里每次都是 64.0/64（方差为 0），A 在 62.1~62.7 之间波动。

- **C 组（decay 1.0,3）达不到 64/64，只有 39.8/64**，原因不是"消费太少"，而是 **decay 把消费挤到前几回合**：`p(0)=1.0, p(1)=0.72, p(2)=0.51, …` 的熵集中在前 ~4 回合 ⇒ 64 条分支的"消费模式"只有 ~40 种互不相同的。耗时 5.44 s（A 的 49%）。
- **换成均匀调度 `const p=0.5`（E 组）就满了**：**64.0/64、7.63 s = A 的 68%**、节点只占上界的 69%（A 是 92%）。
  ⇒ **"用远低于 A 的耗时拿到不亚于 A 的多样性"是成立的，但前提是把 p(r) 从 decay 换成 const**；decay 这种"只在开头随机"的调度天生给不出大 N 的多样性（对动作随机同理：A 组也是 62.4/64 而不是 64）。
- **D（叠加）没有额外收益**：63.9/64 @ 11.32 s，与 A 基本同价同效 ⇒ 如果要用引擎 RNG，应该**替换**动作随机，而不是叠加。

### 3.2 一个必须记住的反例：F 组（p=1.0）多样性 = 1

`p=1` 时每条分支每回合都消费 1 次 ⇒ **所有分支的消费模式完全相同** ⇒ RNG 流始终对齐 ⇒ 前缀树把 64 条分支全部合并成 1 条（17 节点 = H+1，去重叶子 1.0/64），耗时还只有 2.5 s。
⇒ **多样性来自"消费决策的随机性"，不是"消费的总次数"**。这一点和"动作随机"是对称的（双方都不出招时 64 条分支也只有 1 条叶子）。**任何"确定性多消费 k 次"的实现都是无效的。**

### 3.3 把 `rng_state` 加进缓存键的代价（父任务要求的第 2 点）

`rollout_rng_key="none"`（= 忘记把 rng_state 加进键，**会错误合并**不同世界）vs `"state"`（正确），每候选节点数：

| 组 | key=none | key=state | 涨幅 | 说明 |
|---|---|---|---|---|
| B（弱 decay .3,1） | 17.0 | 98.1 | **+477%** | 弱随机化时**比例涨幅最大**（原本共享极多） |
| C（强 decay 1.0,3） | 17.0 | 408.4 | **+2302%** | 绝对代价仍远低于上界 1025 |

`key=none` 会退化成 17 节点（= H+1 的单链）——这不是"省了节点"，而是**把 39.8 种不同的世界合并成 1 个**（去重叶子签名仍记录到 39.7 种，但树只保留 1 条）⇒ 数值上是**错的**。所以这个 ablation 的意义是："必须加，且加了以后的绝对代价仍然 ≤ 上界（强随机化 40%、E 组 69%）"。

**最坏比例情况下（弱随机化 B）节点数涨 4.8 倍，但绝对值只有 98/1025（9.6%）**，耗时 2.49 s → 3.07 s。

### 3.4 "每新增一个节点的成本"：RNG 消费 vs 随机动作

用 6 个组做两点估计（固定成本 = F 组的 2.43 s，即 48 候选 × 64 分支 × 16 回合 = 49152 回合的纯循环/推进开销 ≈ **49 µs/回合**）：

| 组 | rollout s | 节点总数 | 扣除固定成本后的边际成本 |
|---|---|---|---|
| A 动作随机 | 11.082 | 45058 | (11.08−2.43)/(45058−816) = **0.196 ms/千节点** |
| E 仅RNG const.5 | 7.501 | 34114 | (7.50−2.43)/(34114−816) = **0.152 ms/千节点** |
| C 仅RNG decay(1,3) | 5.352 | 19690 | 0.155 ms/千节点 |
| B 仅RNG decay(.3,1) | 3.052 | 4759 | 0.158 ms/千节点 |

⇒ **引擎 RNG 消费每新增一个节点比"随机动作"便宜约 22%（0.152 vs 0.196 ms/千节点）**，方向和父任务的猜测一致（少一次 numpy 抽样 + 少一次 `Operation` 构造 + 少一次 `apply` 的 cold 路径校验），但**幅度不大**；真正的大头是每个节点必须付的 `clone()` + `advance_round()`（以及每回合 49 µs 的固定循环成本）。
⇒ **这条路的价值主要来自"用更少的节点/更少的回合拿到同样的多样性"（E vs A: 71% 节点、68% 耗时），而不是"节点更便宜"。**

---

## 4. 风险 / 边界

1. **默认关闭已验证零影响**，但**开启后**的局面在这个 seed 下永远不会出现（等价于"换一个反事实种子"）⇒ 只能用于搜索打分，**绝不能**用于判题/回放/数据采集/自对弈采样（会污染标签）。裁判侧（`match_via_sdk.py`）本次**一行未改**，而且它跑在**另一个进程**里（AI 是子进程），不会加载 `rng_build`。
2. **必须用随机化的消费决策**（§3.2 的 F 反例）：`const p=0.5` 或 `decay` 都行，`p=1` 或"每回合固定消费"会得到 0 多样性。
3. **`enable()` 的时序**：RNG 变体必须赶在**第一次建局之前**生效；若 `native_game` 已被 import 成现役 pyd，`enable()` 会返回 False 并打一行警告，rollout 里 `_consume_rng` 会计 `last_rng_missing`（**不会**静默出错）。部署链路天然满足（`build_engine` 在建局前）。
4. **两套引擎不能同进程共存**（pybind11 类型重复注册）⇒ 同一进程里不要既 `import native_game`（现役）又要 `import rng_build` 的。
5. **值网分布外**：RNG 扰动产生的叶子（例如"蚂蚁走位被换了一条流"）在真实对局里同样可能出现（它就是引擎自己会生成的局面，只是不属于这个 seed），但仍可能与训练数据分布有偏差 ⇒ **多样性 ≠ 有效**，必须用对局 A/B 判定（下一步）。
6. **前缀树里"消费只发生在新节点"**：`share_prefix=1` 时节点共享会把消费次数摊薄（2299 次 vs 理论 64×16×48），这是正确语义，但意味着"想要更多独立样本"必须靠**更多互不相同的消费模式**（p 或 N），不是靠"多消费几次"。
7. `rollout_rng_key="none"` 是**诊断口径**，别在正式实验里用（会把不同 rng_state 的分支合并）。
8. 本次顺带删掉了父任务在 `_decide_actions` 里留下的死变量 `_p_reverse = True`（未被读取）。

---

## 5. 下一步（建议的对局 A/B）

代码就绪后，只需给 AI 子进程加环境变量即可 A/B（`match_via_sdk` 的 AI 子进程继承环境）：

```powershell
# 对照组（现状）
$env:AZAI_FLAT_ROLLOUT_MODE="random"; $env:AZAI_FLAT_PROB_MODE="decay"
$env:AZAI_FLAT_DECAY_C="1.0"; $env:AZAI_FLAT_DECAY_K="3.0"; $env:AZAI_FLAT_RNG_PERTURB="0"
# 实验组（只消费引擎 RNG，同预算）
$env:AZAI_FLAT_ROLLOUT_MODE="idle"; $env:AZAI_FLAT_PROB_MODE="const"
$env:AZAI_FLAT_ACT_PROB="0.5"; $env:AZAI_FLAT_RNG_PERTURB="1"
# 两组都建议：AZAI_FLAT_SHARE_PREFIX=1
```

要注意"等预算"：E 组的 rollout 回合数相同（49152），但耗时只有 A 的 68% ⇒ 若要等**时间**对照，E 组应把 `AZAI_FLAT_ROUNDS`（或 N）乘 ~1.47，让两边墙钟对齐，才是"等时间、不同随机源"的干净对比。

## 6. 未做

1. **没有跑对局 A/B**（本轮只坐到"引擎接口 + rollout 开关 + 离线多样性/耗时"）；建议按 §5 的口径跑 32~64 局。
2. **没有在 GPU 上跑值网**：`load_three_models` 不自动 `.cuda()`，实测 value 网 device=**cpu**（553k 参数），所以本报告所有计时是 **CPU 口径**（`AZAI_TORCH_THREADS=1`，避免干扰在跑的训练）。**组间相对比较有效**；绝对值与 GPU 口径不可直接比（A 组 11.15 s 与父任务 11.0 s 接近，多半因为他们的 harness 也是同一份 CPU 路径）。
   另外 A/D 与 B/C 是两次独立进程跑的（相隔 ~40 分钟，机器负载可能不同），所以**只用同一进程复测过的 A vs E 下结论**（§3.1）；B/C/D 仅作趋势参考。
3. `set_rng_state` 目前只用于探针/`reroll` 类实验，**没有**接进 rollout 的正式路径（按需求只实现"每回合 Bernoulli 消费 1 次"）。
4. 没有验证 MSVC 构建（沿用 `docs/engine_consistency_probe.md` §7.1 的遗留项）。
