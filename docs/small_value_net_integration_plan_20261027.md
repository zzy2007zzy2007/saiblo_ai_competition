# 小值网 32/3 接入 rollout 叶子评估：实施计划（2026-10-27）

> 前置：`docs/small_value_net_20261027.md`（离线评估：R²=0.4716、ρ=0.9603、前向 5.13×）
> 与 `docs/value_net_pipeline_survey.md` §5（12 条坑）。
> 本文只写**计划与判据**，数字在 `docs/small_value_net_integration_20261027.md`。

## 0. 目标（A–D）

| # | 目标 | 完成判据 |
|---|---|---|
| A | 把 `y_std/y_mean` 折进 `value_head.2`，产出**可直接当现役值网用**的 ckpt | `tanh(folded)` vs `tanh(老师)` 的 Spearman ≈ 0.96；且 folded 与 `raw*y_std+y_mean` **逐位等价**（max abs diff < 1e-5） |
| B | `AZAI_VALUE_CKPT` 开关（默认关） | 空值：`probe_oracle_inject_zero_impact.py` 的哈希日志与改动前**逐行相同**；非空：同一探针的输出**必须变化**（否则开关是死的） |
| C | 同一局面上计时（现役 vs 小网） | 3 局面 × 3 重复取中位；给出总耗时、`_leaf_values` 累计、值网占比、节点数、去重叶子数，并换算"等 10 s 预算能多放多少 N / k" |
| D | 8 局冒烟（开关各 4 局） | 每局 `verdict=engine`、无异常退出、有每局耗时 |

## 1. 关键风险与对策（都不是"读一遍代码觉得没问题"）

| 风险 | 对策 |
|---|---|
| **量纲错**（z-score 标签未折回 ⇒ 放大 2.6×） | 折进 `value_head.2`（照 `graft_value_ckpt.py:68-76`）；自检**两条**：①显式比 `raw*y_std+y_mean`；②抽 1000 帧与老师比 Spearman |
| **默认行为被改**（消费点从写死 `["value"]` 改成兼容写法） | 改为"dict 走原路、裸张量才走新路"的取值函数（`value_of`）；再用**哈希探针**（`probe_oracle_inject_zero_impact.py`）改动前/后各跑一次 diff |
| **开关写死在搜索逻辑里** ⇒ 以后想整臂回退很麻烦 | 开关只在 `az_bridge_ai.build_engine` 里"换叶子值网对象"这一层；`_leaf_values`/`value_batch` 只加**取值兼容**，不加分支逻辑 |
| **计时被机器负载污染**（当前有 8 局对局在跑） | 两侧**同一进程背靠背**跑、交替先后顺序、3 次取中位；只引用比值与同批次绝对值 |
| **`_leaf_values` 里含特征抽取**（每叶固定成本），前向 5.1× 不等于叶子 5.1× | 额外用"包装值网对象"的方式单独计**纯前向**耗时，把 `encode_observation` 的固定成本分离出来；换算用**双点斜率**（N=64/128、k=48/96）外推，不拿平均成本硬乘 |

## 2. 改动清单

| 文件 | 改动 | 默认（`AZAI_VALUE_CKPT` 空）行为 |
|---|---|---|
| `code/my_ai/az_intent/value_override.py` | **新增**：`ValueModelAdapter`（把独立 `ValueNetwork` 包成返回 `{"value": ...}`）、`value_of(out)`（dict/裸张量通用）、`load_override()`、`maybe_override()`；折归一化（若 ckpt 未标记 `y_folded`） | 不加载任何模型、不被调用 |
| `code/my_ai/az_intent/value_batch.py:55` | `model(...)["value"]` → `value_of(model(...))` | **逐位不变**（dict 分支返回同一个张量对象） |
| `code/my_ai/az_intent/flat_rollout_greedy.py:284` | 同上 | **逐位不变** |
| `code/test_match/az_bridge_ai.py` | `flat`/`unified`/`mc_score=net` 三处**叶子值网**换成 `maybe_override(...)`；文件头补 `AZAI_VALUE_CKPT` 说明 | `maybe_override` 原样返回同一个对象 ⇒ **逐位不变** |
| `code/test_match/fold_small_value_ckpt.py` | **新增**：折归一化 + 自检（A） | 新增脚本 |
| `code/test_match/bench_leaf_value_swap.py` | **新增**：同局面计时（C） | 新增脚本 |
| `code/test_match/_smoke_small_value.sh` | **新增**：8 局冒烟（D） | 新增脚本 |

> 不动的：`bundle_mcts.py`（`net_fn` 的三网、M6c 仲裁的 value）、`az_selfplay.load_three_models`
>（**故意不在这里挂钩**——否则连 `net_fn` 的 value 都被换掉，M6c 仲裁会跟着变，超出"只换叶子值网"的范围）。

## 3. 执行顺序

1. **A** 折 ckpt → `checkpoints/small32_3/v_32_3_folded.pt`，跑自检（日志落 `training_history/logs/`）。
2. **B** 写开关 + 消费点兼容；`probe_oracle_inject_zero_impact.py` 改动前 → 改动后 diff（应无输出）→ 开开关再跑一次（应有差异）。
3. **C** 3 个局面（不同回合快照）× 2 臂 × 3 重复；另跑 k=96 / N=128 双点斜率外推。
4. **D** 8 局（4 seed × 2 座位）× 开关开/关，`run_logged.sh` 落盘。
5. 报告 + `git commit`。

## 4. 约束

- 不 kill 任何正在跑的实验；本任务最多 4 核（`OMP_NUM_THREADS=4` 上限，计时用**部署口径 1 线程**）。
- 不跑 128 局 A/B（明早的事）。
- 日志全部落盘（`run_logged.sh` 或 `--log-file`），不加管道过滤。
