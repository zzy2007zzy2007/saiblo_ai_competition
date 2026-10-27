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

### 4.1 零影响（`AZAI_ORACLE_INJECT` 为空）—— 两层逐位证据

**① 搜索层：同一批真实局面，改动前后 `search` 的输出逐位相同。**

`code/test_match/probe_oracle_inject_zero_impact.py` 把**真实对局**（`VS_RU_dump/8r.log`，
501 回合、我方 vs 亚军）的双方操作重放成局面，在回合 3/10/26/47/83/200 × 2 个座位
= 12 个决策上跑 `search`，把 `last_scores`（float64 原始字节）、`visit_policy`、
`res.bundles`、`last_actions`（含每个候选的浮点均值）、`chosen_bundle` 全部打哈希：

```
$ python code/test_match/probe_oracle_inject_zero_impact.py --tag VS_RU_dump --seed 8 \
      --rounds 3 10 26 47 83 200 --search flat --out .../zero_impact_flat_AFTER.log
[probe] r47   p0 | 候选=1    额外候选=0  rollouts=16  回合=1024    | chosen_idx=0 chosen=() root_value=0.045388877391815186
[probe]            | scores_sha=553280d8b43ad89d actions_sha=dc937b59892604f5 policy_sha=... bundles_sha=... sum_scores=0.045388877391815186
...
[probe] 完成：12 个决策全部打印哈希 ⇒ 与另一份日志逐行 diff 应无差异
```

| 对比 | 结果 |
|---|---|
| flat：BEFORE vs BEFORE2（**确定性**自检） | **逐行相同**（不先证明这个，BEFORE/AFTER 的差异就分不清是"改了行为"还是"本来就抖"） |
| flat：BEFORE vs AFTER（改动前后） | **逐行相同** |
| unified：BEFORE vs BEFORE2 / BEFORE vs AFTER | **逐行相同** |

日志：`match_results/oracle_inject/zero_impact_{flat,unified}_{BEFORE,BEFORE2,AFTER}.log`。

> 覆盖到的形态：92 个候选的大局面（含闪电）、只有 1 个候选（HOLD）的回合、
> 多 op 的 bundle（`((13,6,-1),(13,10,-1))`）、p0/p1 两条决策路径。

**② 对局层：同一批 seed，改动前后**整局**逐位相同。**

用 `ab_search_modes.py`（两侧都是同一份冠军配置、都不注入）跑 seed 3000/3001，
一次用**改动前**的代码（`git checkout fced917 -- <3 个文件>`），一次用**改动后**的代码，
两边都开 `AZAI_AZ_DUMP`：

```
RESULT seed=3000 ... winner_side=p0 verdict=engine rounds=469 terminal=True base_hp=16,0 coins=538,264   ← 改动前
RESULT seed=3000 ... winner_side=p0 verdict=engine rounds=469 terminal=True base_hp=16,0 coins=538,264   ← 改动后
RESULT seed=3001 ... winner_side=p1 verdict=engine rounds=434 terminal=True base_hp=0,20 coins=206,514    ← 改动前
RESULT seed=3001 ... winner_side=p1 verdict=engine rounds=434 terminal=True base_hp=0,20 coins=206,514    ← 改动后
```

4 个 `AZAI_AZ_DUMP` jsonl（**每回合的候选束 / 访问分布 / chosen / root_value 全在里面**）：

```
az_seed03000_p0  new=65C8BC44E681D735(258606 B)  old=65C8BC44E681D735(258606 B)  same=True
az_seed03000_p1  new=631CB41F03756FC3(257105 B)  old=631CB41F03756FC3(257105 B)  same=True
az_seed03001_p0  new=1374C2618935E064(240325 B)  old=1374C2618935E064(240325 B)  same=True
az_seed03001_p1  new=97A729D26641AFD6(239795 B)  old=97A729D26641AFD6(239795 B)  same=True
```

⇒ **对局正常结束（`verdict=engine`）、且 903 条决策的候选集与访问分布逐字节相同**。

### 4.2 注入生效（`AZAI_ORACLE_INJECT=<exe>`）

（本节输出随冒烟回填，见下）

### 4.3 怎么跑

单局（我们 vs 亚军，注入开在我方）：

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
CHAMP="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_SEARCH=flat;AZAI_MODE=pos-only"
EXE="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"

# 注入臂 = ai0（我方先手），ai1 = 亚军本体
$P -u code/test_match/match_via_sdk.py --trace-ops \
   --ai0=code/test_match/az_bridge_ai.py --ai1=其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe \
   --ai0-env="$CHAMP;AZAI_ORACLE_INJECT=$EXE;AZAI_ORACLE_VERBOSE=1" \
   3000
```

4 局冒烟（2 seed × 2 座位，注入臂 vs 不注入臂，同一份冠军配置）：

```bash
bash code/test_match/run_oracle_inject_ab.sh 3000 3001      # 建议套 run_logged
```

32 局 A/B（**16 个 seed × 2 座位**；注入臂与不注入臂只差 `AZAI_ORACLE_INJECT`）：

```bash
bash code/run_logged.sh oracle_ab32 \
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py \
  --paired-sides --jobs 4 --out-dir match_results/oracle_inject/ab32 \
  --seeds 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3015 \
  --flat-env="$CHAMP;AZAI_ORACLE_INJECT=$EXE" --mcts-env="$CHAMP"
# 读法：`--flat-env` 那一臂就是**注入臂**（ab_search_modes 里 arm A 叫 "flat"，与本实验无关），
#       `flat 胜率` = 注入臂得分（配对座位 ⇒ 每 seed 两局各坐一次先后手）。
```

⚠️ 用 `ab_search_modes --paired-sides` 时**注入统计只能读最后存活的那一局**：它的 stderr 文件名
只含 `(side, label, seed)`（`match_via_sdk.py:373-374`），配对的两局同一个 seed **并发** ⇒
两个进程用 `"wb"` 打开同一个 `ai?_az_bridge_ai_seedN.stderr.log`。要看**每局**的
`inject/hit/耗时`，用 `run_oracle_inject_ab.sh`（它把两局错开 seed 并发，并在每局结束时另存 stderr）。


---

## 5. 已知限制 / 不做的事

1. **成本**：亚军单次查询 0.1s（早期）→0.7s（r30）→2.0s（r90），而我们的决策 ~2.5–4s
   ⇒ 注入大约让每决策**+50% 墙钟**。`AZAI_ORACLE_TIMEOUT=2.5` 在长局偏紧（r90 的亚军查询
   接近 2s），慢局会以 `timeout` 停用注入——统计里可见，需要时调大。
2. **它的前瞻里对手是"空过"**（协议文档 §8 硬约束 3）：它以为对手这一回合不动，
   所以它的出招对"会反抗的对手"有系统性偏差。我们只把它当**候选**、不当先验。
3. **不做跨局复用**：每局一个新会话（seed 变了局面就变了，会话状态无法迁移）。
4. **不做重建**：超时/崩溃后该局停用（见 §3.4）。
