# 评估阶段温度采样方案

## 问题

当前解码器在 `head_logits` 上取 argmax 确定动作类：

```python
class_id = int(np.argmax(head_logits))
```

用 argmax 的问题：logits 微小波动会导致行为彻底跳变（BUILD ↔ LIGHTNING），使适应度评估充满 threshold 噪声。

## 方案

用温度采样替代 argmax：

```python
# 1. z-score 归一化（解耦温度与 logits 量级）
logits = (logits - logits.mean()) / (logits.std() + 1e-8)
# 2. 温度采样
if temperature > 0:
    probs = softmax(logits / temperature)
    class_id = np.random.choice(24, p=probs)
else:
    class_id = int(np.argmax(head_logits))
```

- `temperature=0` = argmax（当前行为）
- `temperature>0` = 概率性选择，logits 接近的类被选中的概率也接近

归一化的必要性：模型输出 logits 范围在不同训练阶段可能相差百倍（早期 ±2 vs 后期 ±200），不归一化的话相同 temperature 在不同阶段的含义完全不同。

## 改动点

### 1. 解码器 `decoder.py`

`decode_head()` 中：

```python
class_id = int(np.argmax(head_logits))  # 当前
→
if temperature > 0:
    probs = softmax(head_logits / temperature)
    class_id = int(rng.choice(24, p=probs))
else:
    class_id = int(np.argmax(head_logits))
```

解码器目前是纯 numpy 函数，加上 `rng` 参数和 `temperature` 参数即可。

### 2. 调用链

- `_eval_worker` → `decode_network_output` → `decode_head`
- `train_value_net.py` → 也用 decode_head

每个调用点都需要传递 `temperature` 参数。

### 3. 命令行参数

```bash
# 在 ga_ss_train 和 eval_checkpoint 等工具中加：
--eval-temperature 0.3    # 评估时的采样温度，0 = argmax
```

### 4. 变异关系

`mutate_class_labels_soft` 已经在对 label 做温度采样变异（`temperature=0` 时均匀随机）。评估的温度采样是**独立**的——只影响评估阶段动作的选择，不影响训练数据。

## 对现有系统的影响

| 场景 | 改变 |
|---|---|
| GA 训练评估 | `--eval-temperature 0.3` 平滑选择压力 |
| 诊断工具 diagnose.py | `--eval-temperature 0` 保持 argmax 诊断准确性 |
| 最终性能测试 | `--eval-temperature 0` 取最高分动作 |

## 参数建议

```bash
# GA 训练（平滑评估）
--eval-temperature 0.3

# 最终评测（最佳性能）
--eval-temperature 0.0
```
