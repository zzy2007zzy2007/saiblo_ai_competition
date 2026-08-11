# AlphaZero 训练改进：解耦价值网络训练步数

> 日期：2026-08-11。状态：计划阶段。目标：修复"价值网络训练曝光不足"导致的价值头未训透问题。

## 1. 背景：价值训练量绑定在策略池大小上

当前 `az_train.py` 的 `train_split` 循环：

```python
for start in range(0, n, batch_size):   # n = 策略池大小（当前 batch ~13K 样本）
    # 策略步：p_batch 训练策略
    # 价值步：v_batch 训练价值（同一循环顺带）
```

**每步同时做策略和价值更新 → 价值训练步数 = 策略训练步数。**

但池子大小悬殊：

| 池 | 大小 | 每 epoch 曝光 |
|----|------|--------------|
| 策略池（当前 batch 16 局）| ~13K | 100%（完整一遍）|
| 价值池（最近 16 个 batch）| ~213K | **~6%（只有 13K）** |

价值每 epoch 只抽到自己池子的 ~6%，5 epochs 累计 ~30%（有放回）。

**后果（已实测验证）**：
- az_r62_v50（价值单独训透 2.2M 样本次）：vs rule_v4 = 31.2%
- az_r69（价值随训练仅 456K 样本次）：vs rule_v4 = 25.0%
- **价值曝光差 ~5 倍 → 胜率差 ~6 个百分点**，且 az_r69 的价值是从 tau=20 起点朝 tau=50 漂移，远没训透。

## 2. 方案：价值循环独立遍历自己的池子

每个 epoch 改成两个独立循环：

```python
for epoch in range(epochs):
    # 策略：1 遍策略池（当前行为，不变）
    order = shuffle(policy pool)
    for start in range(0, n, batch_size):
        策略步（CE + anchor）

    # 价值：value_passes 遍价值池（新增，解耦）
    for _ in range(value_passes):
        v_order = shuffle(value pool)
        for start in range(0, nv, batch_size):
            价值步（加权标签 MSE）
```

- 策略训练完全不变（仍只过当前 batch 一遍）
- 价值每 epoch 完整过价值池 `value_passes` 遍（默认 3），彻底解决曝光不足
- `value_only` 模式保持不变（本来就在价值池上迭代）

## 3. 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--value-passes` | 3 | 每 epoch 价值池完整过几遍（策略仍 1 遍）|

- 默认 3 遍 → 价值曝光 ~213K×3 = 640K/epoch，5 epochs = 3.2M（超过 az_r62_v50 的 2.2M）
- 可调 1（退化为接近现状）/5/10

## 4. 计算开销

- 价值步比策略步便宜（无 CE/anchor，只有 MSE）
- 完整一遍价值池 = 213K/32 ≈ 6663 步；3 遍 = 20K 步/epoch
- GPU 上实测价值步 ~2-3ms → 3 遍 × 5 epochs ≈ 5-10 分钟/batch（采集 ~30 分钟，占比小）

## 5. 实现改动

- `az_train.py`：`train_split` 加 `value_passes` 参数；循环重构为"策略 1 遍 + 价值 N 遍"双循环
- `main()`：加 `--value-passes` CLI 参数并传递
- `run_az_batches.sh`：训练调用加 `--value-passes 3`
- 其余文件不动

## 6. 验证计划

1. 用现有数据（batch 30-69）重训一轮（init = gen_0120 策略 + az_r62_v50 价值 或 az_r69），看价值损失是否明显低于 az_r69（~0.033 → 应降到接近 az_r62_v50 的水平）
2. search-vs-search：新模型 vs gen0120_warm_cpp（应 ≥ az_r69 的 75%）
3. vs rule_v4：32 局，对比 az_r69（25%）、az_r62_v50（31.2%）、gen_0120+tau50 组合（46.9%）
4. 若价值训透后 vs rule_v4 仍 <31.2%，则确认问题在策略侧（需 t_class 1.0 防坍缩）

## 7. 一句话总结

**把价值训练从"跟着策略步数顺带"改为"独立完整遍历价值池（每 epoch 3 遍）"，解决价值曝光不足（~30%→300%），让价值头真正训透 tau=50 标签——若价值训透后仍不强，则问题锁定在策略侧。**
