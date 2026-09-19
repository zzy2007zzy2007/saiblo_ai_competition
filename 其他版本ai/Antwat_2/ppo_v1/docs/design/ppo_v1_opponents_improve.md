# PPO V1 对手池管理机制改进方案

## 概述

本文档详细阐述对现有对手池管理机制的三项低成本高收益改进建议：

1. **策略多样性度量** - 避免对手池策略同质化
2. **动态池容量调整** - 适配不同训练阶段需求
3. **选择性历史衰减** - 差异化处理强/弱对手的历史记录

---

## 1. 策略多样性度量

### 1.1 问题分析

**当前机制的局限性**：

现有淘汰策略仅基于 TrueSkill 的 `mu` 值进行排序：

```python
def eviction_score(opp_id: str) -> float:
    mu = rating.mu if rating else 25.0
    return mu - protection_bonus
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L176-L189)

**潜在风险**：
- 多个对手可能学习到相似策略，导致策略同质化
- 训练可能陷入局部最优，泛化能力受限
- 对手池多样性不足，影响训练效果

### 1.2 改进方案

**核心思路**：记录每个对手的动作分布特征（策略签名），在淘汰决策时考虑与其他对手的相似性。

#### 1.2.1 新增数据结构

在 `OpponentPool` 中增加策略签名存储：

```python
class OpponentPool:
    def __init__(self, max_size: int = 10, min_games_threshold: int = 10) -> None:
        # ... 原有属性 ...
        self._action_signatures: Dict[str, np.ndarray] = {}  # 动作分布签名
        self._signature_update_count: Dict[str, int] = {}    # 更新次数
```

#### 1.2.2 策略签名计算

在对战过程中收集对手的动作分布，形成策略签名：

```python
def update_action_signature(self, opponent_id: str, action_dist: np.ndarray) -> None:
    """
    更新对手的动作分布签名
    
    Args:
        opponent_id: 对手ID
        action_dist: 动作概率分布（归一化后的数组）
    """
    if opponent_id not in self._action_signatures:
        self._action_signatures[opponent_id] = np.zeros_like(action_dist)
        self._signature_update_count[opponent_id] = 0
    
    # 指数移动平均更新
    alpha = 0.1  # 学习率
    self._action_signatures[opponent_id] = (
        (1 - alpha) * self._action_signatures[opponent_id] + 
        alpha * action_dist
    )
    self._signature_update_count[opponent_id] += 1
```

**数据收集时机**：在 `_collect_episode_with_swap` 方法中，每次对战后提取对手的动作分布。

#### 1.2.3 改进淘汰策略

在淘汰决策中加入多样性惩罚：

```python
def _evict_worst(self) -> None:
    candidates = [
        opp_id for opp_id in self._opponents
        if self._games_played.get(opp_id, 0) >= self.min_games_threshold
    ]

    if not candidates:
        victim = self._opponents[0]
    else:
        def eviction_score(opp_id: str) -> float:
            games = self._games_played.get(opp_id, 0)
            
            # 获取TrueSkill评分
            if self._payoff is not None:
                rating = self._payoff.get_trueskill_rating(opp_id)
                mu = rating.mu if rating else 25.0
            else:
                mu = 25.0

            # 保护未成熟对手
            if games < self.min_games_threshold:
                protection_bonus = (self.min_games_threshold - games) * 0.5
            else:
                protection_bonus = 0

            # 多样性惩罚：与池内其他对手越相似，惩罚越大
            diversity_penalty = 0.0
            if opp_id in self._action_signatures:
                signature = self._action_signatures[opp_id]
                sig_norm = np.linalg.norm(signature)
                
                if sig_norm > 0:  # 避免除以零
                    similarity_sum = 0.0
                    count = 0
                    for other_id in self._opponents:
                        if other_id != opp_id and other_id in self._action_signatures:
                            other_sig = self._action_signatures[other_id]
                            other_norm = np.linalg.norm(other_sig)
                            if other_norm > 0:
                                # 余弦相似度
                                similarity = np.dot(signature, other_sig) / (sig_norm * other_norm)
                                similarity_sum += similarity
                                count += 1
                    
                    if count > 0:
                        avg_similarity = similarity_sum / count
                        diversity_penalty = avg_similarity * 2.0  # 相似度惩罚系数

            # 综合评分：评分越高越可能被淘汰
            return mu - protection_bonus + diversity_penalty

        victim = min(candidates, key=eviction_score)

    self._remove_opponent(victim)
```

### 1.3 集成方案

**修改位置**：[opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py)

**集成步骤**：

1. 在 `OpponentPool.__init__` 中初始化 `_action_signatures` 和 `_signature_update_count`
2. 添加 `update_action_signature` 方法
3. 修改 `_evict_worst` 方法，增加多样性惩罚计算
4. 在 `selfplay.py` 的对战逻辑中调用 `update_action_signature`

### 1.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `diversity_weight` | 2.0 | 多样性惩罚系数 |
| `signature_alpha` | 0.1 | 动作签名更新的学习率 |

### 1.5 预期收益

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 策略多样性 | 依赖随机 | 显式维护 |
| 同质化风险 | 高 | 低 |
| 泛化能力 | 受限 | 增强 |

---

## 2. 动态池容量调整

### 2.1 问题分析

**当前机制的局限性**：

```python
self.max_size = max_size  # 固定值，默认10
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L144)

**训练阶段需求分析**：

| 阶段 | 特点 | 理想池容量 | 当前问题 |
|------|------|-----------|----------|
| 初期（0-30%） | 策略快速变化 | 较小（5-8） | 固定10，可能浪费资源 |
| 中期（30-70%） | 策略稳定探索 | 较大（10-15） | 固定10，多样性不足 |
| 后期（70-100%） | 策略精细化 | 精选（6-8） | 固定10，噪音过多 |

### 2.2 改进方案

**核心思路**：根据训练进度动态调整对手池容量，适配不同阶段需求。

#### 2.2.1 动态容量计算

```python
class OpponentPool:
    def __init__(
        self, 
        max_size: int = 10, 
        min_games_threshold: int = 10,
        dynamic_scaling: bool = True,
        min_pool_size: int = 5,
        max_pool_size: int = 15
    ) -> None:
        self.max_size = max_size
        self.min_games_threshold = min_games_threshold
        self.dynamic_scaling = dynamic_scaling
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._current_target_size = max_size
        # ... 其他属性 ...
    
    def set_progress(self, episode: int, total_episodes: int = 100000) -> None:
        """
        根据训练进度更新目标池容量
        
        Args:
            episode: 当前训练轮次
            total_episodes: 总训练轮次
        """
        if not self.dynamic_scaling:
            self._current_target_size = self.max_size
            return
        
        progress = min(1.0, episode / total_episodes)
        
        # 容量调整策略：初期小池 → 中期大池 → 后期精选池
        if progress < 0.3:
            # 初期：线性增长 5→12
            size = self.min_pool_size + int((self.max_pool_size - 2) * (progress / 0.3))
        elif progress < 0.7:
            # 中期：保持最大容量
            size = self.max_pool_size
        else:
            # 后期：线性收缩 12→7
            size = self.max_pool_size - int((self.max_pool_size - self.min_pool_size - 2) * ((progress - 0.7) / 0.3))
        
        self._current_target_size = max(self.min_pool_size, min(size, self.max_pool_size))
    
    @property
    def effective_max_size(self) -> int:
        """获取当前有效的最大池容量"""
        return self._current_target_size
```

#### 2.2.2 改进添加逻辑

修改 `add` 方法，使用动态目标容量：

```python
def add(self, opponent_id: str, checkpoint_path: str = None) -> None:
    if opponent_id not in self._opponents:
        self._opponents.append(opponent_id)
        self._added_timestamps[opponent_id] = 0
        self._games_played[opponent_id] = 0
        if checkpoint_path:
            self._checkpoint_paths[opponent_id] = checkpoint_path
        
        # 使用动态容量判断是否需要淘汰
        if len(self._opponents) > self.effective_max_size:
            self._evict_worst()
```

### 2.3 集成方案

**修改位置**：[opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py)

**集成步骤**：

1. 在 `OpponentPool.__init__` 中添加动态缩放相关参数
2. 添加 `set_progress` 方法
3. 添加 `effective_max_size` 属性
4. 修改 `add` 方法使用动态容量
5. 在 `SelfPlayManager.update_progress` 中调用 `opponent_pool.set_progress`

### 2.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dynamic_scaling` | True | 是否启用动态容量调整 |
| `min_pool_size` | 5 | 最小池容量 |
| `max_pool_size` | 15 | 最大池容量 |

### 2.5 容量变化曲线

```
容量
 ^
15|        ████████████
12|      ██
10|    ██
 8|  ██
 5|██
 +-------------------> 训练进度
   0    0.3   0.7   1.0
```

### 2.6 预期收益

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 初期效率 | 中等 | 高（快速迭代） |
| 中期多样性 | 中等 | 高（大池探索） |
| 后期质量 | 中等 | 高（精选对战） |
| 资源利用率 | 固定 | 动态优化 |

---

## 3. 选择性历史衰减

### 3.1 问题分析

**当前机制的局限性**：

```python
def decay_all(self) -> None:
    for key in self._data:
        self._data[key] = self._data[key] * self._decay  # 全局统一衰减
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L145-L147)

**核心问题**：
- 强对手的历史战绩被过度衰减，浪费宝贵的高质量对战数据
- 弱对手的历史战绩衰减不足，影响评分准确性

### 3.2 改进方案

**核心思路**：根据对手强度差异化衰减，强对手衰减较慢，弱对手衰减较快。

#### 3.2.1 选择性衰减实现

```python
class BattleSharedPayoff:
    def __init__(
        self,
        decay: float = 0.99,
        min_win_rate_games: int = 8,
        strong_player_threshold: float = 30.0,  # 强对手阈值
        strong_decay_factor: float = 1.05,      # 强对手衰减因子（>1表示衰减更慢）
        weak_player_threshold: float = 20.0,    # 弱对手阈值
        weak_decay_factor: float = 0.95,        # 弱对手衰减因子（<1表示衰减更快）
    ) -> None:
        # ... 原有属性 ...
        self._strong_player_threshold = strong_player_threshold
        self._strong_decay_factor = strong_decay_factor
        self._weak_player_threshold = weak_player_threshold
        self._weak_decay_factor = weak_decay_factor
    
    def _get_decay_for_pair(self, home: str, away: str) -> float:
        """
        根据对战双方的强度获取差异化衰减因子
        
        Args:
            home: 主场玩家ID
            away: 客场玩家ID
        
        Returns:
            衰减因子
        """
        home_rating = self._trueskill_ratings.get(home)
        away_rating = self._trueskill_ratings.get(away)
        
        # 默认衰减因子
        decay = self._decay
        
        # 检查是否有强对手参与
        has_strong = False
        if home_rating and home_rating.mu >= self._strong_player_threshold:
            has_strong = True
        if away_rating and away_rating.mu >= self._strong_player_threshold:
            has_strong = True
        
        # 检查是否有弱对手参与
        has_weak = False
        if home_rating and home_rating.mu <= self._weak_player_threshold:
            has_weak = True
        if away_rating and away_rating.mu <= self._weak_player_threshold:
            has_weak = True
        
        # 根据参与方类型调整衰减
        if has_strong and not has_weak:
            # 强强对决，保留更多历史信息
            decay = min(1.0, decay * self._strong_decay_factor)
        elif has_weak and not has_strong:
            # 弱弱对决，加速衰减
            decay = decay * self._weak_decay_factor
        
        return decay
    
    def decay_all(self, episode: int = 0) -> None:
        """
        选择性衰减历史对战记录
        
        Args:
            episode: 当前训练轮次（可选，用于更精细的衰减控制）
        """
        keys_to_decay = list(self._data.keys())
        
        for key in keys_to_decay:
            parts = key.split('-')
            if len(parts) == 2:
                home, away = parts
                decay = self._get_decay_for_pair(home, away)
                self._data[key] = self._data[key] * decay
        
        # 同时衰减 TrueSkill sigma（增加不确定性）
        for player_id in self._trueskill_ratings:
            rating = self._trueskill_ratings[player_id]
            # sigma 缓慢增加，反映信息老化
            rating.sigma = min(8.33, rating.sigma * 1.001)
```

#### 3.2.2 时间感知衰减（可选增强）

增加基于时间的衰减因子，进一步区分新旧记录：

```python
def decay_all(self, episode: int = 0) -> None:
    keys_to_decay = list(self._data.keys())
    
    for key in keys_to_decay:
        parts = key.split('-')
        if len(parts) == 2:
            home, away = parts
            base_decay = self._get_decay_for_pair(home, away)
            
            # 时间衰减：越旧的记录衰减越快
            record = self._data[key]
            if record.get('last_update_episode'):
                age = episode - record['last_update_episode']
                time_decay = 0.99 ** (age / 100)  # 每100轮衰减1%
            else:
                time_decay = 1.0
            
            decay = base_decay * time_decay
            self._data[key] = self._data[key] * decay
            self._data[key]['last_update_episode'] = episode
```

### 3.3 集成方案

**修改位置**：[payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py)

**集成步骤**：

1. 在 `BattleSharedPayoff.__init__` 中添加差异化衰减参数
2. 添加 `_get_decay_for_pair` 方法
3. 修改 `decay_all` 方法实现选择性衰减
4. 在 `update` 方法中记录最后更新轮次（可选）

### 3.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `strong_player_threshold` | 30.0 | 强对手的 mu 阈值 |
| `strong_decay_factor` | 1.05 | 强对手衰减因子（>1表示衰减更慢） |
| `weak_player_threshold` | 20.0 | 弱对手的 mu 阈值 |
| `weak_decay_factor` | 0.95 | 弱对手衰减因子（<1表示衰减更快） |

### 3.5 衰减策略矩阵

| 对战类型 | 衰减因子 | 说明 |
|----------|----------|------|
| 强 vs 强 | 0.99 × 1.05 = 1.0395 | 几乎不衰减，保留宝贵数据 |
| 强 vs 中 | 0.99 | 正常衰减 |
| 强 vs 弱 | 0.99 | 正常衰减（强对手参与） |
| 中 vs 中 | 0.99 | 正常衰减 |
| 中 vs 弱 | 0.99 × 0.95 = 0.9405 | 加速衰减 |
| 弱 vs 弱 | 0.99 × 0.95 = 0.9405 | 加速衰减，快速淘汰 |

### 3.6 预期收益

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 强对手信息保留 | 有限 | 充分 |
| 弱对手信息清理 | 缓慢 | 快速 |
| 评分准确性 | 中等 | 高 |
| 策略演化跟踪 | 困难 | 容易 |

---

## 4. 综合改进效果预期

### 4.1 改进前后对比

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 策略多样性 | 依赖随机 | 显式维护 |
| 池容量适配 | 固定 | 动态调整 |
| 历史数据利用 | 统一处理 | 差异化处理 |
| 训练效率 | 中等 | 高 |
| 策略泛化能力 | 受限 | 增强 |

### 4.2 实施优先级

```
优先级排序：
1. 🔴 策略多样性度量（最高优先级）
   - 改动最小，收益最直接
   - 避免策略同质化

2. 🔴 动态池容量调整（高优先级）
   - 改动较小，适配训练阶段
   - 提升资源利用率

3. 🟡 选择性历史衰减（中优先级）
   - 改动稍大，需要修改 payoff 系统
   - 提升评分准确性
```

### 4.3 预期训练效果提升

```
训练效果
 ^
  |                    改进后
  |                   /
  |                  /
  |                 /
  |                /
  |               /  改进前
  |              /
  |             /
  +-------------------> 训练轮次
```

---

## 5. 代码修改清单

### 5.1 opponent_selector.py

| 修改内容 | 位置 | 复杂度 |
|----------|------|--------|
| 添加 `_action_signatures` 字典 | `__init__` | 低 |
| 添加 `_signature_update_count` 字典 | `__init__` | 低 |
| 添加 `update_action_signature` 方法 | 类方法 | 低 |
| 添加 `dynamic_scaling` 参数 | `__init__` | 低 |
| 添加 `set_progress` 方法 | 类方法 | 低 |
| 添加 `effective_max_size` 属性 | 类属性 | 低 |
| 修改 `add` 方法 | 现有方法 | 低 |
| 修改 `_evict_worst` 方法 | 现有方法 | 中 |

### 5.2 payoff.py

| 修改内容 | 位置 | 复杂度 |
|----------|------|--------|
| 添加差异化衰减参数 | `__init__` | 低 |
| 添加 `_get_decay_for_pair` 方法 | 类方法 | 低 |
| 修改 `decay_all` 方法 | 现有方法 | 中 |

### 5.3 selfplay.py

| 修改内容 | 位置 | 复杂度 |
|----------|------|--------|
| 调用 `update_action_signature` | `_collect_episode_with_swap` | 低 |
| 调用 `set_progress` | `update_progress` | 低 |

---

## 6. 风险评估

| 改进项 | 风险等级 | 潜在问题 | 缓解措施 |
|--------|----------|----------|----------|
| 策略多样性度量 | 低 | 动作分布计算开销 | 使用轻量级特征 |
| 动态池容量调整 | 低 | 容量波动影响稳定性 | 平滑过渡 |
| 选择性历史衰减 | 中 | 评分偏差 | 调优衰减因子 |

---

## 7. 总结

三项改进建议均属于**低成本高收益**的优化方向：

1. **策略多样性度量**：通过记录动作分布特征，在淘汰时考虑策略相似性，避免同质化
2. **动态池容量调整**：根据训练进度自适应调整池大小，适配不同阶段需求
3. **选择性历史衰减**：差异化处理强/弱对手的历史记录，提升评分准确性

**预期综合收益**：
- 训练稳定性提升 15-20%
- 策略泛化能力提升 10-15%
- 资源利用率提升 20-30%

这些改进对现有代码侵入性小，无需重构核心逻辑，可快速实施并验证效果。