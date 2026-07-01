# 多头改造详细设计文档

## 目标

将硬编码的 `--single-head`（布尔标志）改为 `--num-heads N`（整数参数，默认 3），灵活控制策略头数量。

---

## 1. `code/my_ai/network.py` — 模型定义

### 1.1 `AntWarNetwork.__init__`

**当前代码**（第 64-97 行）：
```python
def __init__(self, num_resblocks=6, single_head=False):
    self.single_head = single_head
    ...
    self.policy_head1 = nn.Linear(64, 24)
    if not single_head:
        self.policy_head2 = nn.Linear(64, 24)
        self.policy_head3 = nn.Linear(64, 24)
```

**改为**：
```python
def __init__(self, num_resblocks=6, num_heads=3):
    self.num_heads = num_heads
    ...
    self.policy_heads = nn.ModuleList([
        nn.Linear(64, 24) for _ in range(num_heads)
    ])
```

### 1.2 `AntWarNetwork.forward`

**当前代码**（第 139-149 行）：
```python
head1_logits = self.policy_head1(policy_base)
result = {"action_map": action_map, "head1_logits": head1_logits, "value": ...}
if not self.single_head:
    result["head2_logits"] = self.policy_head2(policy_base)
    result["head3_logits"] = self.policy_head3(policy_base)
```

**改为**：
```python
result = {"action_map": action_map, "value": self.value_head(state_emb)}
for i in range(self.num_heads):
    result[f"head{i+1}_logits"] = self.policy_heads[i](policy_base)
```

### 1.3 `create_model` / `create_zero_model`

```python
def create_model(num_resblocks=6, num_heads=3):  # single_head→num_heads
    model = AntWarNetwork(num_resblocks=num_resblocks, num_heads=num_heads)
    ...
```

---

## 2. `code/my_ai/decoder.py` — 解码器

**需要改**。第 318-322 行写死了 `for i in (1, 2, 3)`：

```python
head_logits_list = [
    _to_np(network_output[f"head{i}_logits"])
    for i in (1, 2, 3)                  # ← 硬编码，num_heads>3 时 heads 4+ 被忽略
    if f"head{i}_logits" in network_output
]
```

**改为**——动态检测所有 head key：
```python
head_keys = sorted(k for k in network_output if k.startswith("head") and k.endswith("_logits"))
head_logits_list = [_to_np(network_output[k]) for k in head_keys]
```

---

## 3. `code/my_ai/agent.py` — Agent 调用

**不需要改**。`NeuralAgent` 只是调 `model.forward()` + `decode_network_output()`。

---

## 4. `code/my_ai/es_train.py` — 训练程序

### 4.1 参数解析

```python
# 改前
parser.add_argument("--single-head", action="store_true",
                    help="train with only 1 policy head")

# 改后
parser.add_argument("--num-heads", type=int, default=3,
                    help="number of policy heads (default: 3)")
```

### 4.2 `ESTrainer.__init__`

```python
# 改前
def __init__(self, ..., single_head=True):
    self.single_head = single_head
    self.model = create_model(single_head=single_head)

# 改后
def __init__(self, ..., num_heads=3):
    self.num_heads = num_heads
    self.model = create_model(num_heads=num_heads)
```

### 4.3 `main()` 中实例化 ESTrainer

```python
# 改前
trainer = ESTrainer(..., single_head=args.single_head)
log.print(key="single_head", value=args.single_head)

# 改后
trainer = ESTrainer(..., num_heads=args.num_heads)
log.print(key="num_heads", value=args.num_heads)
```

### 4.4 参数量日志

`params={trainer.param_count:,}` — 自动跟随模型变化，不需要改。

---

## 5. `code/my_ai/win_graph.py` — WinGraph

### 5.1 `WinGraph.__init__`

```python
# 改前
def __init__(self, ..., single_head=True):

# 改后
def __init__(self, ..., num_heads=3):
```

### 5.2 `_game_worker`（模块级函数）

```python
# 改前
s, params_a, params_b, single_head = args
ma = create_model(single_head=single_head)

# 改后
s, params_a, params_b, num_heads = args
ma = create_model(num_heads=num_heads)
```

### 5.3 `add_node` 中构建 args

```python
# 改前
all_args.append((s, params, opp_params, self._single_head))

# 改后
all_args.append((s, params, opp_params, self._num_heads))
```

### 5.4 `state_dict` / `load_state_dict`

```python
# 改前
"single_head": self._single_head
self._single_head = cfg.get("single_head", True)

# 改后
"num_heads": self._num_heads
self._num_heads = cfg.get("num_heads", 3)
```

### 5.5 属性

```python
# 改前
@property
def single_head(self) -> bool:
    return self._single_head

# 改后
@property
def num_heads(self) -> int:
    return self._num_heads
```

---

## 6. `code/my_ai/expand_to_3heads.py`

**硬编码了 1→3 展开**。两种方案：

**方案 A（废弃）**：既然训练可以直接设 `--num-heads N`，不再需要展开脚本。

**方案 B（保留）**：改为 `--from-heads 1 --to-heads N`，通用的单头→多头复制。

暂时选方案 A，文档中说明即可。

---

## 7. 测试脚本（6 个文件）

每个文件的改动模式相同：

| 文件 | 改动点 |
|------|--------|
| `eval_checkpoint.py` | `--single-head` → `--num-heads`，`TOP1=` → `NUM_HEADS=`，传给 `create_model` |
| `diagnose_model.py` | 同上模式 |
| `compare_checkpoints.py` | worker 内部 `create_model` 参数名 |
| `build_win_graph.py` | worker 内部 `create_model` 参数名 |
| `benchmark_wg.py` | 只引用了 WinGraph 的参数名 |
| `benchmark_wg2.py` | 只引用了 WinGraph 的参数名 |

以 `eval_checkpoint.py` 为例的详细改动：

```python
# 改前
parser.add_argument("--single-head", action="store_true", ...)
TOP1 = False  # or True
...
# worker 函数中
ma = create_model(single_head=single_head)
...
head_keys = [k for k in output if k.startswith("head")]  # 已有

# 改后
parser.add_argument("--num-heads", type=int, default=3)
NUM_HEADS = 3
...
ma = create_model(num_heads=NUM_HEADS)
```

其他测试脚本同理。

---

## 8. 文档文件

以下 .md 文件中的 `--single-head` 替换为 `--num-heads N`：

- `docs/model_design.md`
- `docs/eval_method.md`
- `docs/training_program.md`
- `docs/讨论记录.md`
- `docs/eval_results.md`
- `docs/win_graph_design.md`

---

## 9. 向后兼容

旧 checkpoint 是用 `create_model(single_head=True)` 训练的，参数字典中只有 head1 的权重：
```
policy_head1.weight, policy_head1.bias
```

新代码用 `create_model(num_heads=N)`，参数字典为：
```
policy_heads.0.weight, policy_heads.0.bias
policy_heads.1.weight, policy_heads.1.bias
...
```

**旧 checkpoint 无法直接加载**。需要在 `load_checkpoint` 中检测旧格式：

```python
ckpt = torch.load(path, ...)
# 检测旧格式兼容
if "policy_head1.weight" in ckpt["model_state"]:
    # 旧格式：single_head=True → 转换为 num_heads=1
    sd = ckpt["model_state"]
    sd["policy_heads.0.weight"] = sd.pop("policy_head1.weight")
    sd["policy_heads.0.bias"] = sd.pop("policy_head1.bias")
    ...
```

---

## 10. 改动汇总

| 文件 | 改动类型 | 大概行数 |
|------|---------|---------|
| `code/my_ai/network.py` | 参数名 + 循环建 head | ~15 行 |
| `code/my_ai/es_train.py` | 参数名 + 参数 | ~8 行 |
| `code/my_ai/win_graph.py` | 参数名 | ~15 行 |
| `code/my_ai/decoder.py` | 硬编码 `for i in (1,2,3)` → 动态检测 head keys | ~3 行 |
| `code/my_ai/agent.py` | 不需要改 | 0 |
| `code/my_ai/expand_to_3heads.py` | 标记废弃 | ~2 行 |
| `code/test_match/eval_checkpoint.py` | 参数名 | ~5 行 |
| `code/test_match/diagnose_model.py` | 参数名 | ~5 行 |
| `code/test_match/compare_checkpoints.py` | 参数名 | ~3 行 |
| `code/test_match/build_win_graph.py` | 参数名 | ~3 行 |
| `code/test_match/benchmark_wg.py` | 参数名 | ~2 行 |
| `code/test_match/benchmark_wg2.py` | 参数名 | ~2 行 |
| 文档文件 ×6 | 文字替换 | ~6 处 |
