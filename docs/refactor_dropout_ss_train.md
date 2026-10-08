# 重构计划：Dropout 移到 Agent + 改用 ss_train

## 背景

当前 es_train.py 用 ES + BC 混合方式训练，BC 部分数据收集残缺（只存 argmax）、损失函数简陋（无 label_smoothing/KL/soft target），且 ES 梯度与 BC 更新冲突。

ss_train.py 是 ES+BC 思路的完整进化版，但缺 `--small` 和 `--action-dropout`。

当前 action dropout 实现在 es_train `_eval_worker` 中（第 150-157 行），直接操作 ActionCatalog，架构不干净。

## 改动详请

### 改动 1：agent.py — NeuralAgent 加 action_dropout 支持

**位置**：`code/my_ai/agent.py`

**1a. __init__ 加参数**（第 35-49 行）

```python
def __init__(
    self,
    model: AntWarNetwork | None = None,
    seed: int | None = None,
    max_actions: int = 96,
    *,
    allowed_classes: list[int] | None = None,
    action_dropout: float = 0.0,   # ← 新增
):
```

加一行：`self.action_dropout = action_dropout`

文件头加：`import random`

**1b. _choose_operations 末尾加 dropout 逻辑**（第 97-98 行之后）

```python
# 解码得到模型选择的 ops
operations = decode_network_output(output, state, player,
                                   allowed_classes=self.allowed_classes)

# ── Action Dropout：按概率用随机 legal action 覆盖 ──
self.dropout_this_turn = False  # 新增字段，供 BC 数据收集判断
if self.action_dropout > 0 and random.random() < self.action_dropout:
    from SDK.utils.actions import ActionCatalog
    catalog = ActionCatalog()
    bundles = catalog.build(state, player)
    if bundles:
        chosen = random.choice(bundles)
        operations = list(chosen.operations)
        self.dropout_this_turn = True
```

注意：
- `last_output` 保持存模型原始输出，不受 dropout 影响
- `dropout_this_turn` 标记当前回合是否被 dropout 覆盖，供 ss_train 数据收集时跳过

### 改动 2：ss_train.py — _eval_worker 改用实际动作标签

**位置**：`code/my_ai/ss_train.py`

**2a. _eval_worker 加 action_dropout 参数**（第 44-68 行）

```python
def _eval_worker(
    params_flat: np.ndarray,
    opp_params_flat: np.ndarray,
    seed: int,
    num_heads: int = 3,
    bc_dir=None,
    gen=0,
    ind=0,
    action_dropout: float = 0.0,   # ← 新增
) -> dict:
```

创建 agent 时传入：`agent = NeuralAgent(model=model, action_dropout=action_dropout)`

**2b. 数据收集：dropout 回合直接跳过**（第 78-107 行）

当前代码每回合先调 `agent._choose_operations()`，然后收集数据。

改为：仅在**没有触发 dropout** 的回合收集数据。触发 dropout 的回合跳过（不
存 board/stats/labels），因为该回合的 (state → action) 映射被随机动作污染了。

在 agent 侧标记当前回合是否触发了 dropout，例如：
```python
# NeuralAgent._choose_operations 内
self.dropout_this_turn = False  # 新增字段，每回合重置
if self.action_dropout > 0 and random.random() < self.action_dropout:
    # ... 随机覆盖操作
    self.dropout_this_turn = True
```

在 _eval_worker 的数据收集处：
```python
ops_us = agent._choose_operations(state, our_player)
if bc_dir and not agent.dropout_this_turn:  # ← 跳过 dropout 回合
    feat = agent.feature_extractor.encode_observation(...)
    # ... 正常收集 board/stats/class/action_map/head_logits
```

这样：
- dropout 回合的数据不进入训练集，不污染标签
- dropout 改变了游戏状态（建塔消耗金币），后续正常回合的数据仍是干净的
- 模型在后续回合自己做出拆塔决策 → 这些数据进入 top-K → BC 学到拆塔

### 改动 3：ss_train.py — build_eval_args 传 action_dropout

**位置**：第 677-699 行

在参数元组中加一项：

```python
all_args.append((
    params_list[idx], opp_params_list[k_idx], base_seed,
    num_heads,
    bc_dir, gen, idx,
    action_dropout,  # ← 新增
))
```

`build_eval_args` 函数签名加 `action_dropout: float = 0.0` 参数。

### 改动 4：ss_train.py — 加 --small 命令行参数

**位置**：第 753-790 行（argparse 区域）

加：
```python
parser.add_argument("--small", action="store_true",
                    help="use small model (87K params, 1 head)")
```

**模型创建处**（第 817 行）：
```python
model = create_model(num_heads=args.num_heads, small=args.small)
```

**参数计数**（第 818 行）：
```python
param_count = model.count_parameters()
```
不动，自动适配。

**训练循环中的参数创建**（第 930-941 行，`bc_mutate_population` 内）：
`bc_mutate_population` 内第 442 行：`model_template = create_model(num_heads=N)` → 需改为 `model_template = create_model(num_heads=N, small=args.small)`

但这函数不直接访问 args，需在循环外保存 small 标志，或修改 `bc_mutate_population` 加参数。

**bc_mutate_population 函数签名**（第 424 行）：
```python
def bc_mutate_population(mean_model, dataset, device, n_individuals,
                          p_mutate=0.1, temperature=3.0,
                          mutation_step=0.01, sigma=0.0002,
                          n_dirs=20, log=None, gen=0,
                          small=False):  # ← 新增
```

调用处（第 931 行）传 `small=args.small`。

**检查点保存/加载**（第 640-655 行）：
保存时加 `"small": small`，加载时恢复。

### 改动 5：ss_train.py — 加 --action-dropout 命令行参数

**位置**：第 753-790 行（argparse 区域）

加：
```python
parser.add_argument("--action-dropout", type=float, default=0.0,
                    help="probability of replacing model action with random legal action")
```

**训练循环中创建 eval 任务时**（第 1013-1063 行）：
`build_eval_args` 调用处传入 `action_dropout=args.action_dropout`。

### 改动 6：ss_train.py — 命令行参数默认值调整

当前 ss_train 默认值：
- `--pop-size 64`（偏大，小模型可减小）
- `--sigma 0.0002`（ES 噪声，ss 不需要大噪声）
- `--games 18`（偏多，小模型可减小）

默认值不动，用户按需传参。

### 改动 7：ss_train.py — bc_mutate_population 内 model_template 加 small

**位置**：第 442 行

```python
model_template = create_model(num_heads=N, small=small)
```

否则加载 state_dict 时大/小模型参数不匹配会报错。

## 启动命令

停止当前 es_train 后启动：

```bash
python code/my_ai/ss_train.py \
    --pop-size 24 --games 8 --workers 16 \
    --generations 500 --num-heads 1 \
    --small \
    --action-dropout 0.15 \
    --k 3 --epochs 3 --lr 1e-3 \
    --p-hold 0.1 \
    --save-every 10 \
    --out-dir training_history/ss_dropout
```

## 涉及文件和改动汇总

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agent.py` | L35-49 __init__ | 加 `action_dropout` 参数和 `self.action_dropout` |
| `agent.py` | L79-99 _choose_operations | 末尾加 dropout 覆盖逻辑，新增 `self.last_action_class` |
| `agent.py` | 文件头 | 加 `import random` |
| `ss_train.py` | L44-68 _eval_worker 签名 | 加 `action_dropout` 参数 |
| `ss_train.py` | L62-64 agent 创建 | 传 `action_dropout=action_dropout` |
| `ss_train.py` | L88-96 数据收集 | 改用 `agent.last_action_class` 做标签 |
| `ss_train.py` | L677-699 build_eval_args | 加 `action_dropout` 参数和传递 |
| `ss_train.py` | L753-790 argparse | 加 `--small` 和 `--action-dropout` |
| `ss_train.py` | L817-818 模型创建 | `small=args.small` |
| `ss_train.py` | L424-427 bc_mutate_population | 加 `small=False` 参数 |
| `ss_train.py` | L442 model_template | `small=small` |
| `ss_train.py` | L931 调用 bc_mutate | 传 `small=args.small` |
| `ss_train.py` | L1013-1063 build_eval_args 调用 | 传 `action_dropout=args.action_dropout` |
