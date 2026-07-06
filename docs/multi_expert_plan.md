# 多专家（Multi-Expert）方案设计文档

## 1. 动机

当前训练的核心问题：BC-only 训练导致模型收敛到单一策略（Thunder），无法学到局面感知的多策略行为。根因是 Thunder 是一个极强的局部最优，任何其他动作组合在 fitness 上都无法与之竞争。

核心思路：将 24 类动作分组为若干"专家"，每个专家只拥有 2-6 类动作的权限，独立训练。因为专家没有 Thunder 可用（或其动作集被严格限制），它必须学会该子集内的有效策略。最后用一个门控网络根据局面选择专家。

## 2. 总体架构

```
Game State
    │
    ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Gating   │──→│ Expert A │   │ Expert B │   │ Expert C │
│ Network  │   │ (Thunder │   │ (Towers) │   │ (Support)│
│ (8K)     │   │  + Econ) │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
    │               │
    ▼               ▼
Selected Expert's Operations
```

### 2.1 专家分组方案

建议分 4 个专家，覆盖不同的玩法路线：

| 专家 | 动作类 | 包含动作 | 策略定位 |
|------|--------|---------|---------|
| A | 17, 21, 22, 23 | Thunder + 基地升级 + HOLD | 纯Thunder经济流 |
| B | 0-16, 23 | 所有塔动作(建/升级/降级) + HOLD | 纯塔攻防 |
| C | 18, 19, 20, 21, 22, 23 | EMP/Deflectors/Evasion + 基地升级 + HOLD | 辅助武器 |
| D | 4, 13, 14, 15, 16, 23 | Producer系 + 降级 + HOLD | 产兵运营 |

**选择依据：**
- 专家 A：保留现有的 Thunder 强势策略，叠加基地升级
- 专家 B：最大动作集（16 类），覆盖所有塔类组合
- 专家 C：三大辅助武器的不同组合
- 专家 D：经济产兵路线，与塔攻防路线互补

### 2.2 门控网络

极小网络，只有 ~8K 参数：

```python
class GatingNetwork(nn.Module):
    """Select which expert to use given game state."""
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)   # input = state_emb
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_experts)
    
    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state_emb))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits, no softmax
```

输入复用专家模型的 `state_emb`（128 维，board_emb + stats_emb），不需要额外特征提取。

## 3. 代码改动

### 3.1 decoder.py — 加 allowed_classes 参数

改动最小方案：在 `decode_head` 和 `decode_network_output` 加一个 `allowed_classes` 参数。

```python
def decode_head(
    head_logits, action_map, class_mask, position_mask, state, player,
    allowed_classes: list[int] | None = None,  # ← 新增
) -> Operation | None:
    
    if allowed_classes is not None:
        # Filter class_mask: only allow specified classes
        allowed = np.zeros(len(class_mask), dtype=bool)
        allowed[allowed_classes] = True
        class_mask = class_mask & allowed  # ← 核心改动
        # Also filter position_mask for disallowed classes
        for ch in range(len(position_mask)):
            if ch not in allowed_classes:
                position_mask[ch] = False
    
    # 其余代码不变...
    class_id = int(np.argmax(head_logits))
    # 如果 argmax 在不允许的类上，class_mask 已设为 False，返回 None
```

同理 `decode_network_output` 透传此参数。

### 3.2 agent.py — 支持多专家切换

```python
class NeuralAgent(BaseAgent):
    def __init__(self, ..., allowed_classes: list[int] | None = None):
        ...
        self.allowed_classes = allowed_classes  # ← 新增
    
    def _choose_operations(self, state, player):
        ...
        operations = decode_network_output(
            output, state, player,
            allowed_classes=self.allowed_classes,  # ← 透传
        )
        return operations
```

训练每个专家时，用不同的 `allowed_classes`：
- 专家 A：`allowed_classes=[17, 21, 22, 23]`
- 专家 B：`allowed_classes=list(range(17)) + [23]`  # 0-16 + 23
- 等

### 3.3 门控网络的推理集成

在 agent.py 中新增门控逻辑：

```python
class MultiExpertAgent(BaseAgent):
    """Agent that uses a gating network to select among expert models."""
    def __init__(self, experts: list[NeuralAgent], gate: GatingNetwork):
        self.experts = experts
        self.gate = gate
        self.feature_extractor = FeatureExtractor(...)
    
    def _choose_operations(self, state, player):
        # 1. 提取特征
        obs = self.feature_extractor.encode_observation(...)
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        
        # 2. 所有专家前向（共用专家的 encoder 输出 state_emb）
        #    为效率，只跑一次 encoder，再分别过 policy head
        state_embs = []
        for expert in self.experts:
            with torch.no_grad():
                out = expert.model(board, stats)
            state_embs.append(out)  # 存下来后续选中的专家直接取结果
        
        # 3. 门控网络选专家
        #    用第一个专家的 state_emb（所有专家共享同架构，相近但权重不同）
        #    更精确：用一个单独的 encoder 或直接用 stats 子集
        gate_input = state_embs[0]["state_emb"]  # 或专门的门控特征
        gate_logits = self.gate(gate_input)
        expert_idx = gate_logits.argmax().item()
        
        # 4. 用选中的专家的输出解码
        operations = decode_network_output(
            state_embs[expert_idx], state, player,
            allowed_classes=self.experts[expert_idx].allowed_classes,
        )
        return operations
```

### 3.4 训练过程中的支持

专家训练复用现有的 `es_train.py` 管线：

```bash
# 专家 A: Thunder + base upgrades
python code/my_ai/es_train.py --pop-size 64 --games 16 --workers 12 \
    --small --bc --bc-k 3 --bc-epochs 3 --bc-lr 1e-3 \
    --load-bc distill_data_v2/model_all_bal.pt \
    --save-every 3 --generations 100 \
    --allowed-classes "17,21,22,23"  # ← 新增命令行参数
```

需要在 `es_train.py` 的 `_eval_worker` 中把 `allowed_classes` 传给 `NeuralAgent`。

## 4. 训练流程

### 4.1 各专家的对手：混合策略对手池

不使用固定对手（ExampleAI 太弱、rule_v4 太强），也不使用自对弈（同专家内战雪上加霜）。

**核心思路：混合策略对手（Mixed Strategy Opponent）**

```
每回合对手的动作来源：
  a × 随机动作         ← 简单：产生多样化局面
  b × ExampleAI       ← 中等：有一定强度
  c × rule_v4         ← 困难：强规则AI
```

**实现方式：** 训练时每局的实际对手是一个"混合agent"——每回合以 (a,b,c) 概率随机选择一个来源，用该来源的策略选取动作。

```python
class MixedStrategyOpponent:
    """每回合从三种策略中按概率采样一个来源，用其动作。"""
    def __init__(self, probs: tuple[float, float, float]):
        # probs = (a, b, c) 对应随机/ExampleAI/rule_v4
        self.strategies = [
            lambda s: random_legal_action(s),
            lambda s: example_ai_choose(s),
            lambda s: rule_v4_choose(s),
        ]
        self.probs = probs
    
    def choose_operations(self, state, player):
        source = np.random.choice(3, p=self.probs)
        return self.strategies[source](state, player)
```

**关键优势：**

| 维度 | 固定对手 | 自对弈 | 混合策略(本方案) |
|-----|---------|-------|----------------|
| 难度阶梯 | 不连续，易跳跃 | 渐进，但同质化 | **连续可调** |
| 局面多样性 | 低 | 低（越打越像自己） | **高**（混合三种风格） |
| 跨策略鲁棒性 | 低 | 低 | **高** |
| 可调节性 | 手动换对手 | 自动但缓慢 | **自动快速调** |

#### 动态难度调节（关键创新）

每代结束后根据胜率自动调整 (a,b,c)：

```python
def adjust_probs(win_rate, target_rate=0.55):
    """让胜率保持在 target_rate 附近，fitness 信号最强。"""
    a, b, c = current_probs
    if win_rate > target_rate + 0.05:  # 太简单了 → 增c减a
        shift = min(a * 0.2, 0.1)
        a -= shift
        c = min(1, c + shift)
    elif win_rate < target_rate - 0.05:  # 太难了 → 增a减c
        shift = min(c * 0.2, 0.1)
        c -= shift
        a = min(1, a + shift)
    return normalize(a, b, c)  # 归一化到和为1
```

这样：
- 模型弱的时候：(a=0.8, b=0.2, c=0.0) — 大部分随机的对手，先学会基础
- 模型中等：(a=0.2, b=0.5, c=0.3) — 混合对手
- 模型强：(a=0.0, b=0.2, c=0.8) — 主要打 rule_v4

**如何适配不同专家：**

每个专家可以从不同的起点开始：
- 专家 A（Thunder）可能直接从 (a=0.1, b=0.4, c=0.5) 开始——Thunder 够强
- 专家 D（产兵）从 (a=0.6, b=0.3, c=0.1) 开始——产兵弱，需要更多简单对手

每个专家独立维护自己的 (a,b,c) 曲线。

**对 ES 梯度的影响：** 由于混合对手池产生不同难度的对手，个体间 fitness 的可比性需要关注。解决方案：同代内所有个体使用相同的对手序列（固定种子），保证公平比较。

### 4.2 阶段一：训练各专家（串行）

```
专家 A 训练：50-100代，只允许 Thunder + 基地升级
  ├── 初始化：distill_data_v2/model_all_bal.pt
  ├── 对手：混合策略（a=0.1, b=0.4, c=0.5 起始，动态调）
  ├── 输出：expert_a_final.pt
  └── 训练目录：training_history/expert_a/

专家 B 训练：50-100代，只允许塔动作
  ├── 初始化：distill_data_v2/model_all_bal.pt
  ├── 对手：混合策略（a=0.3, b=0.4, c=0.3 起始，动态调）
  ├── 输出：expert_b_final.pt
  └── 训练目录：training_history/expert_b/

专家 C 训练：50-100代，只允许辅助武器
  ├── 初始化：distill_data_v2/model_all_bal.pt（或随机）
  ├── 对手：混合策略（a=0.5, b=0.4, c=0.1 起始，动态调）
  ├── 输出：expert_c_final.pt
  └── 训练目录：training_history/expert_c/

专家 D 训练：50-100代，只允许产兵路线
  ├── 初始化：distill_data_v2/model_all_bal.pt（或随机）
  ├── 对手：混合策略（a=0.6, b=0.3, c=0.1 起始，动态调）
  ├── 输出：expert_d_final.pt
  └── 训练目录：training_history/expert_d/
```

每个专家训练量：50-100 gen × ~6min/gen = 5-10 小时，4 个专家串行约 20-40 小时。

### 4.3 阶段二：训练门控网络

门控网络约 8K 参数，使用 ES 训练（复用 `es_train.py`）：

```
每代流程：
1. 扰动门控权重 → 64 个变种门控
2. 每个变种 + 4 个冻结的专家模型 → 打 16 局
3. fitness = 胜率
4. ES 梯度更新门控权重

对手选择（重要）：
  门控网络使用相同的混合策略对手池（同各专家训练时一致），
  以保证门控学会在各种难度/风格的对手下正确选择专家。
```

门控参数量极小（8K），用 64 种群绰绰有余，预计 50-100 代即可收敛。

### 阶段二：训练门控网络

训练方式一（推荐）：**ES 训练门控**，复用现有 `es_train.py`。

门控网络 ~8K 参数，64 种群绰绰有余：

```
每代流程：
1. 扰动门控权重 → 64 个变种门控
2. 每个变种门控 + 4 个冻住的专家 → 打 16 局
3. fitness = 胜率
4. ES 梯度更新门控权重
```

门控本身极小，一个全连接层就 128×64=8K 参数，实际上可以用更简单的方案。

训练方式二（更简单）：**随机搜索**。

既然门控只有 ~8K 参数，甚至不需要 ES。用 `es_train.py` 的 --synthetic-test 模式（已实现）可以直接测试随机搜索能否优化。

建议阶段二也用 50-100 代，因为门控参数量极小，收敛会很快。

## 5. 与原方案的工作量对比

| 方案 | 新增代码量 | 修改文件 | 训练时间 | 风险 |
|-----|-----------|---------|---------|------|
| 多专家 + 门控 | ~200 行 | decoder(20行), agent(80行), es_train(50行), 新文件(50行) | 20-40h + 5-10h | 中等 |
| AlphaZero | ~2000+ 行 | 大量新文件 | 极长 | 高 |
| 熵正则化 | ~20 行 | elite_bc(10行) | 0（复用已有训练） | 低（但效果存疑） |

## 6. 风险与缓解

1. **门控训练数据分布偏移**：ES 训练门控时，专家是冻住的。但实际游戏中专家不会遇到所有局面。缓解：门控训练时加入各种早期/中期/后期局面的采样。

2. **专家能力不平衡**：某个专家（如纯产兵）可能 fitness 极低，门控永远不选它。缓解：如果出现这种问题，可以为每个专家预留最低选择概率（ε-greedy），或者直接去掉无效专家。

3. **门控过小表达力不足**：如果 8K 门控无法学会正确的专家选择，可以扩大门控或改用基于 stats 子集的简单规则。缓解：门控设计本身允许轻松扩展。

4. **动作类分组不合理**：某个专家的动作组合在游戏里没有协同效应。缓解：可以在第一阶段训练后，通过行为诊断调整分组。

## 7. 后续扩展

1. **专家数量可伸缩**：发现不足可以加新专家（如"纯Ice/Heavy控制流"），不影响已有专家。
2. **专家联级**：门控可以输出专家概率分布，不只是 argmax。多专家可以按序执行（类似现在的 3 个 head）。
3. **在线蒸馏**：如果最终希望得到一个单模型，可以用多专家系统收集数据，蒸馏一个单模型（用户已确认现阶段不需要）。
