# 行为克隆（Behavior Cloning）设计

## 动机

当前的 ES 训练在 550K 参数空间下无法学到有效策略。11 代后网络仍输出 BUILD→DOWNGRADE 循环（`_eval_results.md` 验证），根本原因是维度诅咒：

- pop=54 产生的梯度信噪比 ≈ 噪声水平
- 参数随机游走，学不出任何稳定策略

BC 通过**利用 ExampleAI 的知识**，为网络提供一个合理的初始化点，再配合 ES 微调。

## 整体流程

```
collect_data.py          train_bc.py           es_train.py (微调)
ExampleAI 自对弈 ───→  监督训练 class head ──→ 加载 BC 权重继续 ES
    ↓                      ↓                      ↓
  离线 .npz 文件          bc_checkpoint.pt        bc+es 最终模型
```

## 数据采集（collect_data.py）

### 输入

- ExampleAI 自对弈 N 局（N=500 起步，约 25 万条样本）
- 多进程并行加速

### 每回合记录

| 字段 | 形状 | 说明 |
|------|------|------|
| `board` | (28, 19, 19) float32 | 官方 FeatureExtractor 的 board 特征 |
| `stats` | (42,) float32 | 同上，stats 特征 |
| `class_label` | int (0-22) | ExampleAI 执行的动作类 |

注意：只记录先手玩家（player=0）的回合，后手玩家动作受对手影响大，学习价值低。

### 输出

`{timestamp}_ex_data.npz`，内存：

```python
{
    "board": (N_samples, 28, 19, 19),
    "stats": (N_samples, 42),
    "class_label": (N_samples,),  # int64
}
```

## 模型训练（train_bc.py）

### 加载方式

用 `create_model(single_head=True)` 创建网络，加载 .npz 数据训练。

### 训练目标

冻结 encoder（初始卷积 + 6×ResBlock + stats_MLP + action_map_conv），只训练 class head 部分：

```
冻结  encoder + stats_mlp            ← board_emb 特征已够用
冻结  action_map_conv                 ← 位置预测不动
训练  policy_base (Linear 64→64)     ← 学习从 board_emb 映射到动作类
训练  policy_head1 (Linear 64→23)    ← 学习的动作类权重
```

### 损失函数

`CrossEntropyLoss(class_logits, class_label)`，Adam 优化器，lr=1e-3。

### 数据分割

80% 训练 / 20% 验证，batch_size=1024。

### 评估

验证集 top-1 准确率。预期 80%+（ExampleAI 的策略相对确定性高）。

### 输出

`bc_checkpoint.pt`，保存 state_dict + mean 向量，与 ES checkpoint 格式兼容。

## 与 ES 的衔接

1. `train_bc.py` 输出 `bc_checkpoint.pt`（参数格式与 ES 兼容）
2. ES trainer 新增 `--load-bc <path>` 参数：
   - 加载 BC checkpoint 的 mean 作为初始 `self.mean`
   - `self.velocity` 初始化为零（动量从零开始）
   - 继续正常 ES 训练
3. ES 只微调**所有参数**（包括 action_map 和 encoder），因为 BC 产出的位置头还是随机的

## 预期效果

| 阶段 | 动作类准确率 | vs ExampleAI 胜率 |
|------|:-----------:|:----------------:|
| 纯 ES (gen 11) | ~4% | 2-4% |
| BC 冷启动后 | 80%+ | ~30-50%（位置头随机，但类正确） |
| BC + ES 微调 100 代 | 85%+ | 50%+ |
