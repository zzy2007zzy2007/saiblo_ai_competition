# 亚军候选注入（runner-up candidate injection）：设计与验证

> 2026-10-26 起。**开关式实验功能**：决策时把"亚军（cpp_lure_v4）在我们这个局面的出招"
> 作为**额外候选**并入搜索，默认**完全关闭**（不建子进程、不加任何耗时、数值行为逐位不变）。
>
> 相关既有文档：
> - `docs/runnerup_oracle_protocol.md`：裁判↔AI 协议的确切报文 + `RunnerUpOracle` 查询器（30/30、90/90 已验）
> - `code/test_match/runnerup_oracle.py`：常驻会话实现（本轮直接复用，**未改一行**）

## 0. 一句话

`AZAI_ORACLE_INJECT=<亚军 exe>` ⇒ 每局按 (seed, player) 起一个常驻亚军会话、**每回合**把
「当前局面 + 对手操作块」喂进去，把它这一手的 ops 变成一个候选 bundle，附加到搜索候选列表末尾，
由**同一个 rollout 评估器**给它打分 ⇒ 它能赢就选它，赢不了就丢弃。
开关为空 ⇒ 一行都不执行（默认关闭）。

---

## 1. 开关（环境变量）

| 变量 | 默认 | 含义 |
|---|---|---|
| `AZAI_ORACLE_INJECT` | 空（关） | 亚军 exe 的路径。**空 = 不启用**：不 import、不建子进程、不加耗时 |
| `AZAI_ORACLE_TIMEOUT` | `2.5` | 单次查询的读包超时（秒）。超时 ⇒ 放弃注入、继续正常决策 |
| `AZAI_ORACLE_VERBOSE` | `0` | `1` ⇒ 每回合把 oracle 的 ops / 合法性 / 耗时打到 stderr |
| `AZAI_ORACLE_DEBUG` | `0` | `1` ⇒ `RunnerUpOracle(verbose=True)`（它内部逐条收发摘要） |

⚠️ 全部输出的落点是 **stderr**（`_log`）。`az_bridge_ai` 的 stdout 是**二进制协议通道**
（4 字节大端长度前缀），一个 `print` 就会毁掉整局 —— 这点在文件头已有历史注释。

启动示例（单局，我们对亚军）：

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P code/test_match/match_via_sdk.py --trace-ops \
  --ai0=code/test_match/az_bridge_ai.py \
  --ai1-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_SEARCH=flat;AZAI_MODE=pos-only" \
  --ai1=code/test_match/az_bridge_ai.py \
  3000
```
（`--ai0-env=...` 里再串上 `AZAI_ORACLE_INJECT=其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe`
就是"开注入"那一臂。）

---

## 2. 注入通道（搜索侧改动，默认零影响）

### 2.1 `FlatRolloutGreedy`（**冠军配置走的就是这条**：`AZAI_SEARCH=flat`）

`code/my_ai/az_intent/flat_rollout_greedy.py`：

```python
search(state, player, *, temperature=0.0, extra_bundles=None) -> BundleSearchResult
searcher.set_extra_bundles([( (11,10,11), ), ( (13,0), (11,8,2) ), ()])   # 也可用属性
```

- `extra_bundles=None`（默认）⇒ **一个字节都不多看**，候选列表/顺序/预算/分数与改动前完全一致
  （§4.1 用哈希逐位验证）。
- 非空 ⇒ 在 `root.bundles` **末尾追加**（去重：与已有 bundle 完全相同的跳过；
  同一个 extra 内部重复也去重），`prior` 补 `AZAI_ORACLE_PRIOR`（默认 0.0）。
  - **追加在末尾**是关键：既有候选的下标、展开顺序、`_leaf_values` 的批组成都不变
    ⇒ 不开注入时数值路径本身就等价。
  - 追上后走**同一条** `children` 构造 + rollout + 值网路径 ⇒ 它和普通候选**同权竞争**，
    "亚军这一手好不好"由我们自己的评估器判，不由它的分数判（协议文档 §8 的推荐用法 3）。
- `HOLD` = **空 bundle `()`**（与 `bundles` 里既有 HOLD 表示一致）。
- 落选时它只是多花一次 rollout 的预算；**命中**（`chosen_index >= 网络候选数`）会在统计里单独计数。

### 2.2 `UnifiedActionSearch`（`AZAI_SEARCH=unified`）

同一个参数/属性；额外候选的内部分类通道号是 `100`（`CHANNEL_TAG[100] = "oracle"`），
在 `last_actions` 里显示为 `oracle`。它的 `rollout_state()` 已改为"按 bundle 逐条 apply"
（单 op 候选的 apply 序列与改动前**完全相同**，仍是一条 `Operation`）。

---

## 3. 桥接侧（`code/test_match/az_bridge_ai.py`）

### 3.1 会话生命周期

- 在 `main()` 里 `build_engine()` 之后、进入回合循环之前 `_oracle_init(seed, player)`：
  读 `AZAI_ORACLE_INJECT`；空 ⇒ `_ORACLE["off"]=True`，**什么都不做**（连 import 都不做）。
- 非空 ⇒ `RunnerUpOracle(seed=seed, player=player, timeout=AZAI_ORACLE_TIMEOUT)`。
  构造失败（文件不存在 / 进程起不来 / DLL 缺）⇒ `_log` 一条、**永久关闭**，对局照常。
- 对局结束（`main()` 末尾）+ `_report()`（atexit / SIGTERM）都会打统计。⚠️ bridge 收尾会硬杀
  子进程，所以**对局中每 25 个决策也打一行累计**（和既有的 `⏱️ 耗时累计` 同一个节奏）。

### 3.2 每回合喂（这是硬约束）

`RunnerUpOracle` 是**有状态长会话**：`state.round_index` 必须严格 +1（跳回合 = 读端永久错位）。
所以注入必须**从 round 0 起、每回合都问**，不管这一回合要不要用它的候选：

```
decide() 的第一件事：oracle_step()
    ├─ player==1: ask(facade, opponent_ops=本回合对手 ops)      # 主循环刚收到、已 apply
    └─ player==0: ask(facade)                                  # 对手 ops 在下一回合开头登记
```

对手操作块的登记严格照协议（`docs/runnerup_oracle_protocol.md` §2.2）：

| 我们执 | 协议里它读的顺序 | 我们何时登记对手 ops |
|---|---|---|
| P1 | `state(N)` → `opp_ops(N)` → 写自己的 | **`ask(N)` 之前**（`ask(opponent_ops=)`） |
| P0 | `opp_ops(N-1)` → `state(N)` → 写自己的 | **`ask(N-1)` 之后**（`set_opponent_ops(N-1, ops)`） |

- **喂进去的局面**：直接用 `facade`（`ask` 内部 `state_to_text(facade)`）。这不是"近似"：
  桥接每回合都在做 `state_to_text(facade)` 与裁判发来的字节**逐字节对拍**（`mismatch`
  计数），所以 `facade` 就是收到的局面文本，且省掉一次文本往返。
- 空过（HOLD）**也注入**（用户要求）：亚军"这一手什么都不做"本身就是个决策。

### 3.3 合法性

`facade.apply_operation_list` 会**静默跳过非法操作**（`binding.cpp:46-59` 的返回值被 facade 丢掉），
所以注入前逐条查：`can_apply_operation(player, op, pending=已接受的前序 ops)`（`pending`
让它把前序 op 的金币/塔位影响算进去，与亚军自己 `cpp_lure_v4.cpp:122-128` 的做法一致）。

- 有一条非法 ⇒ **整条候选丢弃**（不是"删掉那一条"——删一条就变成另一个动作了），
  `oracle_illegal` +1，日志里留一行原文。

### 3.4 兜底（绝不让决策失败）

`ask` 的失败**全部**被包进 `try/except`：

| 失败 | 处理 |
|---|---|
| `OracleTimeout`（> `AZAI_ORACLE_TIMEOUT`） | `oracle_timeouts` +1；**永久停用注入**（该会话读端已错位），本回合及以后照常决策 |
| `OracleDesync` / `OracleDead` / 其它异常 | 同上，日志带异常类型与消息 |

- 超时的那一回合**不注入**（候选列表 = 纯网络候选，与关闭时一致）。
- 不做"重建会话"：重建必须**从 r0 重放**（协议硬约束），代价与实现复杂度都不值得；
  超时后该局退化为"无注入"，统计里能看见（见 §3.5）。这是**已知限制**，写在这里而不是藏起来。

### 3.5 统计字段（全部进 stderr）

```
[azai][oracle] 注入统计: asks=… ok=… inject=… hit=… illegal=… timeout=… dead=…
                          avg_ms=… p50_ms=… max_ms=… hit_rate=…%
```

| 字段 | 含义 |
|---|---|
| `asks` | 实际调用 `ask` 的回合数（成功+失败） |
| `ok` | 成功拿到回包的次数 |
| `inject` | 真正把候选交给搜索的次数（含 HOLD；`ok` 里扣掉丢弃的非法候选） |
| `hit` | 注入的候选被搜索**选中**的次数（`chosen_index` 落在额外候选上） |
| `hit_rate` | `hit / inject` |
| `timeout` / `dead` / `illegal` | 三类失败各自计数 |
| `avg_ms` / `p50_ms` / `max_ms` | 每次 `ask` 的墙钟（含它自己的搜索） |

> 为什么 `hit` 是这轮最关键的读数：它是"亚军这一手在我们的评估器眼里到底值不值得下"的
> 直接证据。`hit=0` 也有信息量（= 我们现在的候选没输给它），但 `hit>0` 说明它带来的是
> **我们网络给不出、而 rollout 认账**的动作。

---

## 4. 验证

（见 §4.1/§4.2 的真实输出，随本轮实验回填）

### 4.1 零影响（`AZAI_ORACLE_INJECT` 为空）

两层证据：

1. **逐位哈希**（最强）：`code/test_match/probe_oracle_inject_zero_impact.py` 在**真实对局局面**
   （从 `match_results/ladder_logs/<tag>/*.log` 的 `--trace-ops` 行重放，含塔/蚁/金币）
   上跑同一批 `search`，把 `last_scores` 的 float64 字节、`visit_policy`、`chosen_bundle`
   全部打哈希。改动前 / 改动后两份日志**逐行相同**。
2. **真对局**：`AZAI_ORACLE_INJECT` 为空跑 2 局，`verdict=engine`、无 timeouts、正常终局。

### 4.2 注入生效（`AZAI_ORACLE_INJECT=<exe>`）

4 局（2 seed × 2 座位），要求：每局 `verdict=engine`、`inject>0`、`hit>0`、决策耗时 p50/p95 不爆。

---

## 5. 已知限制 / 不做的事

1. **成本**：亚军单次查询 0.1s（早期）→0.7s（r30）→2.0s（r90），而我们的决策 ~2.5–4s
   ⇒ 注入大约让每决策**+50% 墙钟**。`AZAI_ORACLE_TIMEOUT=2.5` 在长局偏紧（r90 的亚军查询
   接近 2s），慢局会以 `timeout` 停用注入——统计里可见，需要时调大。
2. **它的前瞻里对手是"空过"**（协议文档 §8 硬约束 3）：它以为对手这一回合不动，
   所以它的出招对"会反抗的对手"有系统性偏差。我们只把它当**候选**、不当先验。
3. **不做跨局复用**：每局一个新会话（seed 变了局面就变了，会话状态无法迁移）。
4. **不做重建**：超时/崩溃后该局停用（见 §3.4）。
