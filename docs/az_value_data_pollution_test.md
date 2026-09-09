# 实验 A：验证"自对弈数据污染"假设

## 起因（2026-09-09）

补测裸策略基线后得到新事实：

| 模型 | 裸策略 vs rule_v4 | 带搜索（+原版价值头） |
|------|------------------|---------------------|
| gen0120 | 1W/31L = 3.1% | 18.8% |
| az_r10 | 0W/32L = 0% | 34.4% |

**两个裸策略都≈0**。所以 §23 的"az 迭代训练让策略大幅进步"要改写成：
**az 策略在裸打上并不比 gen0120 强，它强在"作为搜索先验"**（配原版价值头
18.8% → 34.4%）。真正干活的是搜索 + 价值头，价值头是瓶颈。

## 假设：az_fixed 的自对弈数据被退化价值头污染

`run_az_batches.sh` 是链式采集，每个 batch 用上一批 checkpoint 自对弈：

```
batch1  ← gen0120_bn_init.pt（= BN 转换后的 gen0120_warm_cpp，好价值头）
batch2  ← az_r1（价值头已训过 1 轮）
...
batch10 ← az_r9（价值头已退化 9 轮）
```

az_fixed 实际启动命令（从会话记录恢复）：
```
run_az_batches.sh 10 1 training_history/az_fixed/gen0120_bn_init.pt 256 1 \
    training_history/az_fixed/data training_history/az_fixed 16 16 0.5 0.3 24 3 abs
```
即 256 iter / depth4 / k=24 / t_class=0.5 / t_pos=0.3 / native / 16 局每 batch。

**推论**：160 局数据里只有 batch1 的 16 局是好价值头引导的搜索，其余 144 局
由逐步退化的 az 价值头搜出来 —— 动作目标和价值目标都带坏价值头的偏差。
这正好解释 §26：这批数据训价值头只能到 9.4%。

（顺带排除：az_r1/5/10 的 value 权重 std 37→367→1250 看着像发散，拆开看全是
BatchNorm `running_var`，可学习参数没爆炸。）

## 实验设计

1. **采集**：用 `mix_r10p_bn0v.pt`（az_r10 策略 + gen0120_warm_cpp 价值头
   = 34.4% 那个模型）自对弈 160 局，配置与 az_fixed 完全一致：
   ```
   az_selfplay.py --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt \
       --games 160 --workers 32 --iterations 256 --max-depth-rounds 4 \
       --max-rounds 512 --out-dir training_history/az_fixed/data_goodvalue \
       --seed 20000 --native-engine --t-class 0.5 --t-pos 0.3 --k 24
   ```
2. **转换**：`convert_pkl_to_warm_npz.py`（用 gen_0120 重新前向生成 anchor 目标）
3. **训价值头**：`value_warmup.py`（gen_0120 init + 策略 anchor + 终局 HP-diff
   标签，10 epochs / lr 1e-3），与 §26 的复现配方一致。
   需给 value_warmup.py 加 `--skip-collect`（直接读 `--data-dir` 里已有的 npz）。
4. **合成**：`make_mix_checkpoint.py --policy az_r10 --value <新价值头>`
   → `mix_r10p_goodvalue.pt`
5. **测试**：vs rule_v4 32 局，配置同 34.4% 那次的命令：
   ```
   eval.py --checkpoint <新> --opponent rule_v4 --bundle-mcts --iterations 256 \
       --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 \
       --games 32 --workers 16
   ```

## 判读

| 新价值头成绩 | 结论 |
|-------------|------|
| ≈34% 或更高 | 污染假设成立 —— 只要搜索用好的价值头，搜索数据就能训出好价值头，AlphaZero 闭环可修 |
| ≈9% | 搜索产生的数据本身不适合训价值头（或 az 策略访问的状态分布有问题），需换数据来源 |

## 对照参考

| init | 数据来源 | 训练方法 | vs rule_v4 |
|------|---------|---------|-----------|
| gen_0120 | gen_0120 raw 重放 200 局 | value_warmup | **34.4%** |
| gen_0120 | az_fixed 搜索自对弈 160 局（坏价值头） | value_warmup | **9.4%** |
| gen_0120 | **新：好价值头搜索自对弈 160 局** | value_warmup | 本次待测 |

## 时间预估

参考 az_fixed：16 局 / 16 workers ≈ 16 分钟 → 160 局 / 32 workers ≈ 80 分钟。
