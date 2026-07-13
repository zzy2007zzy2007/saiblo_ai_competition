# 软标签变异与蒸馏式行为克隆

## 背景

ga_ss_train 的交叉和变异依赖 BC 训练。BC 训练可以使用硬标签（cross-entropy on argmax）或软标签（KL divergence on raw logits），两者在变异时存在一致性问题。

## 标签类型

| 标签 | 形状 | 含义 |
|---|---|---|
| `class_label` | (T, N_heads) | argmax 硬标签，每帧每个 head 选哪个动作 |
| `head_logits` | (T, N_heads, 24) | 模型原始输出 logits，保留完整的置信度分布 |

## 问题

变异只改了 `class_label`（硬标签），但 `head_logits` 未变。

- 用硬标签训练（`--lambda-class > 0`）：学到变异后的动作
- 用软目标训练（`--lambda-soft > 0`）：学到原始策略，变异无效
- 两者混用：两个损失互相矛盾

## 解决方案

`mutate_class_labels_soft()` 同时变异 `class_label` 和 `head_logits`：

1. 对 `class_label` 做随机替换（`temperature=0` 时均匀采样）
2. 对对应的 `head_logits`：把旧最高类（变异前的 label）和新类（变异后的 label）的 logit 值互换

这样：
- 软目标分布指向新动作
- 保留了原始分布的相对结构（同类的相似关系不变）
- 硬标签和软目标的信号一致

## 推荐训练配置

```bash
# 纯蒸馏模式（不用硬标签）
--lambda-class 0 --lambda-soft 1.0

# 混合模式（推荐起步）
--lambda-class 0.3 --lambda-soft 0.7

# 经典 BC（不用软标签）
--lambda-class 1.0 --lambda-soft 0.0
```

## 相关函数

- `ga_ss_boilerplate.mutate_class_labels_soft()` — 变异标签 + logits
- `ga_ss_boilerplate.mutate_class_labels()` — 只变异标签
- `mutate_dataset()` — 在 mutation 流程中调用上述函数
