# AlphaZero 第三轮：策略网 / 价值网拆分为两个独立网络

> 目标：解决共享骨干反复出现的耦合问题——价值训练拉偏策略、冻结一头骨干变、价值头校准难。拆成两个独立网络解耦，各自干净地优化。
> 日期：2026-08-07。状态：计划阶段。

## 1. 背景：共享骨干的耦合已经咬了我们三次

| # | 现象 | 根因 |
|---|------|------|
| 1 | 第一轮价值训练把策略拉成"纯闪电"（建塔 0 次） | 价值梯度经共享骨干反向影响策略，anchor 只是创可贴 |
| 2 | 想冻结价值头只训策略，但共享骨干会变、冻结头失准 | 骨架服务于两个头，动一个影响另一个 |
| 3 | 价值头重训劣化（比"预测0"还差）、量级饿死（std 0.04 vs 0.21） | 小数据（80 局）撑不起兼顾两头的骨干，价值信号被稀释 |

**AlphaZero 共享骨干的理论优势**（表征复用、样本效率）在我们这种数据规模（~80 局）下不成立，反而引入难以控制的耦合。actor-critic 用独立策略/价值网络是 RL 的标准做法。

## 2. 设计

```
策略网 policy_net：骨干 + 3 个策略头（输出 action_map, head1-3_logits）
价值网 value_net ：骨干 + 价值头（输出 value）
```

- **两个网都从 gen0120_warm 权重 warm start**（拆 state_dict）。价值网不能从零学表征——80 局学不动，必须继承预训练骨干。
- **互不干扰**：训练策略网时价值网权重完全不动，反之亦然。两阶段/双优化器都安全，anchor 对抗价值漂移的创可贴可以简化。
- **搜索接口不变**：`net_fn(state, player)` 跑两个网，返回 `{action_map, head_logits, value}`——bundle_mcts 无感知。

## 3. 数据与损失

```
策略网：策略目标 = 当前 batch（CE + anchor，同第二轮方案 B）
价值网：价值标签 = 加权未来血量差（--tau 20, --label-scale ~6，累计数据）
```

- 数据范围拆分照旧：策略只用当前 batch（目标会过时），价值用累计（事实可积累）。
- `--label-scale 6` 保留：第二轮已实证缩放修复量级饿死（std 0.04→0.22），拆分后仍需要。
- 每 epoch：策略 batch → 策略网一步；价值 batch → 价值网一步。两个 optimizer。

## 4. 实现改动

### 4.1 检查点格式（向后兼容）

单个 checkpoint 存两份权重：
```python
torch.save({
    "model_state": policy_state,      # 向后兼容：load_model_from_ckpt 直接得到策略网
    "value_state": value_state,       # 新增：价值网
    "num_heads": 3, "no_bn": True, "latent_dim": 64, "num_resblocks": 6,
}, ckpt)
```

### 4.2 加载

- `load_model_from_ckpt`：返回策略网（原逻辑不变，向后兼容 raw 解码等只用到策略的场景）。
- 新增 `load_split_models(ckpt_path) -> (policy_model, value_model)`：构建两个网，分别 load。
- 搜索类脚本（az_selfplay、az_search_vs_search、eval.py --bundle-mcts）改用 `load_split_models` + 拆分版 `make_net_fn`。

### 4.3 拆分版 net_fn

```python
def make_split_net_fn(policy_model, value_model, feat):
    def net_fn(state, player):
        obs = feat.encode_observation(state, player, np.zeros(96))
        board, stats = to_tensor(obs)
        with torch.no_grad():
            p = policy_model(board, stats)
            v = value_model(board, stats)
        return {"action_map": p["action_map"],
                "head_logits": [p[f"head{i+1}_logits"] for i in range(3)],
                "value": v["value"]}
    return net_fn
```

### 4.4 训练

- 新脚本 `az_train_split.py`（或 az_train.py 加 `--split` 模式）：
  - 加载 gen0120_warm → 拆成 policy_model / value_model
  - 策略网：policy-pool 数据，分解 CE + anchor，自己的 optimizer
  - 价值网：value-pool 数据，加权缩放标签 MSE，自己的 optimizer
  - 每 epoch 两个循环（或一个循环两步），保存合并 checkpoint

### 4.5 改动清单

| 文件 | 改动 |
|------|------|
| `az_train.py` 或新 `az_train_split.py` | 拆分训练逻辑 |
| `az_selfplay.py` | `load_split_models` + 拆分 net_fn |
| `az_search_vs_search.py` | 加载拆分模型 |
| `eval.py` | `--bundle-mcts` 模式加载拆分模型 |
| `run_az_batches.sh` | 调新训练脚本 |

## 5. 验证计划

1. **训练一轮**（policy=batch8、value=累计 80 局、tau 20、label-scale 6），看两个网各自的损失是否下降
2. **价值网诊断**：输出量级（std 应 ≈0.2）+ 与终局结果的相关性
3. **search vs search**（128/深度4，6 局）：拆分后训练 vs gen0120_warm，>50% 为通过
4. **raw vs raw**（12 局）：策略网没退化
5. 与 az_r2c（单网+缩放）结果对照：若拆分显著更好 → 耦合确实是主因

## 6. 风险

| 风险 | 缓解 |
|------|------|
| 价值网从 gen0120_warm 微调但数据仍少（80 局） | 缩放 + 加权标签已缓解；必要时价值池扩局 |
| 两个网前向 2 次（搜索成本略增） | 前向仅占扩展 ~4%，可忽略 |
| 拆分后 policy 网价值头闲置（浪费） | 计算量小，不影响；后续可加 mode 跳过 |
| 检查点/加载改动引入兼容问题 | 向后兼容设计（model_state 仍是策略网） |

## 7. 一句话总结

**第三轮 = 把策略网和价值网拆成两个独立网络（都从 gen0120_warm warm start），策略只吃当前 batch 的 CE+anchor、价值只吃累计数据的加权缩放标签，搜索接口不变——彻底解耦"共享骨干打架"这一类问题，让每个头能干净地学习。**
