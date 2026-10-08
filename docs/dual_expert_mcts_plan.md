# 双专家模型 + MCTS 方案

## 核心思路

用 ss_train 自然训练出的两个已有多样策略模型（gen_0030 闪电流、gen_0101 塔防流）作为 MCTS 的策略先验来源。每回合两个模型各解码出一个具体操作，MCTS 以这两个操作作为候选分支进行搜索，用价值网络评估局面优劣，选出胜率更高的选择。

不再依赖单一模型在所有局面下输出正确动作，也不训练门控网络来挑选专家——让 MCTS 的搜索来做这个决策。

## 现有材料

- **gen_0030**（ss_20260709_191050）：Lightning + Downgrade 闪电流，90.6% vs ExampleAI
- **gen_0101**（ss_20260710_155327）：纯塔防（Mortar+/Medic），67.2% vs ExampleAI
- **ss_train.py 已有 value 标签收集**：_eval_worker 保存 `[hp_us, hp_opp]` 到每个 npz，SSDataset 已支持加载

## 需要新建的文件

### 1. `code/my_ai/value_network.py` — 独立价值网络

**设计原则**：与策略网络共享相同的 board encoder 结构，但去掉所有策略相关的头（3个policy head + action_map_conv）。价值回归只需要学到"这个局面最终领先/落后多少血量"，不需要关心动作。

**从随机初始化训练**，不加载策略网络的 encoder 权重。原因：当前策略网络的 encoder 质量不确定，加载可能有偏的权重不如从头学。

**详细结构**：

```
输入:
  board:  (B, 28, 19, 19) — 棋盘特征
  stats:  (B, 42)          — 统计信息

Encoder (同 AntWarNetwork):
  initial_conv = Conv2d(28->64, k=7, pad=3) + BatchNorm + ReLU     → (B, 64, 19, 19)
  resblocks   = 6× ResBlock(64, Conv3x3)                            → (B, 64, 19, 19)
  gap         = GlobalAvgPool                                        → (B, 64)          board_emb
  
stats_mlp:
  Linear(42->64) + ReLU                                             → (B, 64)          stats_emb

value_head:
  state_emb = concat(board_emb(64), stats_emb(64))                  → (B, 128)
  Linear(128->64) + ReLU                                            → (B, 64)
  Linear(64->1)                                                     → (B, 1)  裸输出，无Tanh
```

**参数总量**：约 97K（encoder 87K + value_head ~9K）

**与策略网络的区别**：
1. 去掉 action_map_conv(64->23) → 节省 ~1.5K 参数
2. 去掉 policy base(64->64) + 3×policy head(64->24) → 节省 ~15K 参数
3. value_head 简化为 2 层 MLP（128→64→1），无 LayerNorm、无 Dropout
4. 输出层无 Tanh，裸 Linear 输出，不限制范围

```python
class ValueNetwork(nn.Module):
    def __init__(self):
        self.initial_conv = nn.Sequential(
            nn.Conv2d(28, 64, 7, padding=3), nn.BatchNorm2d(64), nn.ReLU())
        self.resblocks = nn.ModuleList([ResBlock(64) for _ in range(6)])
        self.stats_mlp = nn.Sequential(
            nn.Linear(42, 64), nn.ReLU())
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    
    def forward(self, board, stats):
        x = self.initial_conv(board)
        for block in self.resblocks:
            x = block(x)
        board_emb = x.mean(dim=[2, 3])        # GAP → (B, 64)
        stats_emb = self.stats_mlp(stats)       # (B, 64)
        state_emb = torch.cat([board_emb, stats_emb], dim=1)  # (B, 128)
        return self.value_head(state_emb).squeeze(-1)  # (B,) 裸标量
```

### 2. `code/my_ai/train_value_net.py` — 价值网络训练脚本

**标签设计**：使用 **双方血量差** 作为监督信号，归一化到合理范围但不限制输出

```
value_target = (hp_us - hp_opp) / 20.0  # 归一化到 [-1, 1] 左右
```

- 最终血量差比简单胜负更精细：20:0 大胜 vs 13:11 险胜，价值不同
- 单局中即使输了，血量差也能反映"差点赢"的梯度信号
- 除以 20（最大血量）归一化到 [-1,1] 量级，使 MSE loss 尺度合理
- 输出层不用 Tanh，裸 Linear 输出——避免 Tanh 在 ±1 附近的梯度饱和

**数据来源**：复用 ss_train 的 .npz 数据，使用已有的 `[hp_us, hp_opp]` 标签
- 也可以专门用 gen_0030 / gen_0101 打新局收集更多数据

**训练流程**：
1. 扫描 npz 数据目录
2. 加载 board(28,19,19) + stats(42) 作为输入
3. 标签 = (hp_us - hp_opp) / 20.0，值域约 [-1, 1]
4. 数据量估计：几万局 × ~200 帧/局 = 百万级样本
5. 损失：MSE
6. 优化器：AdamW，weight_decay=1e-4（对回归任务起正则作用）
7. 训练 50-100 个 epoch

**输出**：`value_net.pt` 文件

### 3. `code/test_match/mcts_agent.py` — MCTS 智能体

**输入**：
- 两个策略模型（gen_0030 和 gen_0101）
- 价值网络（value_net.pt）
- MCTS 参数：搜索深度、宽度、模拟次数

**MCTS 流程**（每回合）：

```
1. 当前局面为 root node
2. 用两个模型各解码一个操作 → 2 个候选动作分支
3. 对每个候选动作：
   a. 应用到游戏状态 → 新局面
   b. 用 value_net 评估新局面的胜率
   c. 或者继续展开搜索（更深）
4. 选择价值最高的分支执行
```

**关键特性**：
- 搜索分支不超过 2 个（两个模型各一），branching factor 极小
- 可以不用 roll-out——直接用 value_net 评估叶子节点，节省时间
- 支持多 depth：走一步后让两个模型再解码下一回合动作，继续评估

**MCTS 参数设计**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 候选动作数 | 2 | 两个模型各出一个，不再扩展 |
| 搜索深度 | 3-5 | 向前看几步（≈3-5回合） |
| 价值估计 | value_net | 调用的 value_net 预测，不 rollout 打完整局 |
| 选择策略 | max avg value | 选择平均价值估计最高的分支 |

由于分支极少，MCTS 核心其实退化为：**展开两个候选路径 → value_net 评估 → 选更好的**。没有复杂的 UCB/uct 计算需求。

## 需要修改的文件

### 4. `code/test_match/eval_checkpoint.py` — 评估脚本扩展

在原评估逻辑上新增 `--mcts` 模式：
- 加载两个策略模型 + 价值网络
- 用 MCTS 智能体替代单一模型打局
- 报告胜率对比（纯 gen_0030 vs 纯 gen_0101 vs MCTS 组合）

### 5. （可选）`code/my_ai/leaderboard.py` — 用于 MCTS 的对手池

如果 MCTS 智能体需要加入训练循环的 leaderboard 作为对手，可能需要修改添加适配器。

## 不需要动的文件

- `code/my_ai/ss_train.py` — 训练流程不变，继续产模型
- `code/my_ai/network.py` — 原策略网络不变
- `code/my_ai/decoder.py` — 解码逻辑不变
- `code/my_ai/agent.py` — 原 NeuralAgent 不变

## 数据收集方式

每局对战双方都从 {gen_0030, gen_0101, ExampleAI} 中随机选择，记录其中一方的 board/stats + 最终血量差。这样可以覆盖多种对战风格：

| 对战组合 | 特点 |
|---------|------|
| 0030 vs Example | 强攻 vs 弱守，局面倾斜 |
| 0101 vs Example | 龟缩 vs 弱攻，低血量差 |
| 0030 vs 0101 | 强攻 vs 龟缩，风格冲突，最多样 |
| 0030 vs 0030 | 自对弈，镜像开局 |
| 0101 vs 0101 | 自对弈，塔防内战 |

## 实施步骤

| 步骤 | 内容 | 预计时间 | 状态 |
|------|------|--------|------|
| 1 | 写 `value_network.py` — 独立价值网络定义 | 30min | ✅ |
| 2 | 写 `collect_value_data.py` — 混战数据收集 | 30min | ✅ |
| 3 | 正式收集数据（~1000局，16 workers） | ~30min | 待跑 |
| 4 | 写 `train_value_net.py` — 价值网络训练 | 30min | ✅ |
| 5 | 训练价值网络 | ~10min | 待跑 |
| 6 | 写 `mcts_agent.py` — MCTS 核心逻辑 | 1h | 待写 |
| 7 | 修改 `eval_checkpoint.py` — MCTS 模式评估 | 30min | 待写 |
| 8 | 测试 MCTS vs gen_0030/gen_0101 对比 | 30min | 待跑 |

## 已知风险和问题

1. **价值网络的训练数据分布**：如果只用 gen_0030 打出的数据训价值网络，它对塔防局面的评估可能不准。最好混用两个模型的数据训，或者用强规则AI（rule_v4）打的局。

2. **value_net 的泛化能力**：价值网络看到的局面和 MCTS 搜索中展开的局面可能分布不同（MCTS 会生成两个模型都不会自然到达的"杂交局面"）。如果价值网络在这种 OOD 局面下乱猜，MCTS 反而可能选错。

3. **同步 vs 异步 MCTS**：走一步后让两个模型再解码动作需要重新推理策略网络。如果推理慢（17ms/次），深度3步需要 2×3=6次推理/回合，还在可接受范围（~100ms/回合）。

4. **gen_0030 和 gen_0101 可能在某些局面下输出相同操作**：比如都在 HOLD，导致实际上只有1个候选分支，MCTS 白跑了。

5. **先有鸡还是先有蛋**：价值网络需要足够多的高质量游戏数据。目前 ss_train 的 npz 数据可能就存了千局量级，如果用两个模型打新局收集数据本身也需要时间。

6. **与 ss_train 训练循环的整合**：初期可以先做 standalone 的 MCTS 智能体（只用于评估，不参与训练），后期再考虑是否把 MCTS 的选择结果作为更好的标签反馈进 ss_train。
