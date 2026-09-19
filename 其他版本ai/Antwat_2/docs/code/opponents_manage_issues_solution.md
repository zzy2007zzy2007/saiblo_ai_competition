# SelfPlay对手池管理问题解决方案

## 一、高优先级问题解决方案

---

### 问题1：持久化机制缺失

**问题描述**
- 对手池、payoff记录、TrueSkill评分都没有保存/加载功能
- 训练中断后无法恢复，之前积累的对手和评分全部丢失

#### 改进难度：⭐⭐⭐（中等）

**理由**：
- 需要在多个类中添加序列化和反序列化逻辑
- 需要处理checkpoint文件的路径管理和清理
- 需要保证加载时的版本兼容性

#### 改进收益：⭐⭐⭐⭐⭐（极高）

**理由**：
- 训练中断可恢复，节省大量计算资源
- 支持训练过程的分析和调优
- 为分布式训练提供基础

#### 详细技术方案

```python
# === 1. 定义数据结构 ===
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class OpponentPoolState:
    opponent_ids: List[str]
    checkpoint_paths: Dict[str, str]  # opponent_id -> checkpoint_path
    added_timestamps: Dict[str, float]  # opponent_id -> timestamp
    games_played: Dict[str, int]  # opponent_id -> 对战次数

@dataclass
class PayoffState:
    players: List[str]
    trueskill_ratings: Dict[str, Dict]  # player_id -> {mu, sigma, games_played}
    battle_records: Dict[str, Dict]  # key -> {wins, draws, losses, games}

@dataclass
class LeagueState:
    version: str = "1.0"
    timestamp: float
    opponent_pool: OpponentPoolState
    payoff: PayoffState
    current_agent_id: str = "current_agent"


# === 2. OpponentPool 添加持久化方法 ===
class OpponentPool:
    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._opponents: List[str] = []
        self._checkpoint_paths: Dict[str, str] = {}
        self._added_timestamps: Dict[str, float] = {}
        self._games_played: Dict[str, int] = {}

    def save_state(self, filepath: str) -> None:
        state = OpponentPoolState(
            opponent_ids=self._opponents,
            checkpoint_paths=self._checkpoint_paths,
            added_timestamps=self._added_timestamps,
            games_played=self._games_played
        )
        with open(filepath, 'w') as f:
            json.dump(asdict(state), f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "OpponentPool":
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        state = OpponentPoolState(**state_dict)

        pool = cls()
        pool._opponents = state.opponent_ids
        pool._checkpoint_paths = state.checkpoint_paths
        pool._added_timestamps = state.added_timestamps
        pool._games_played = state.games_played
        return pool

    def add(self, opponent_id: str, checkpoint_path: str = None) -> None:
        if opponent_id not in self._opponents:
            self._opponents.append(opponent_id)
            self._added_timestamps[opponent_id] = time.time()
            self._games_played[opponent_id] = 0
            if checkpoint_path:
                self._checkpoint_paths[opponent_id] = checkpoint_path
            if len(self._opponents) > self.max_size:
                oldest = self._opponents.pop(0)
                self._on_remove(oldest)  # 触发钩子

    def _on_remove(self, opponent_id: str) -> None:
        # 清理关联的checkpoint文件
        if opponent_id in self._checkpoint_paths:
            checkpoint_path = self._checkpoint_paths.pop(opponent_id)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)  # 或移动到归档目录

    def get_checkpoint_path(self, opponent_id: str) -> Optional[str]:
        return self._checkpoint_paths.get(opponent_id)


# === 3. BattleSharedPayoff 添加持久化方法 ===
class BattleSharedPayoff:
    def save_state(self, filepath: str) -> None:
        state = PayoffState(
            players=self._players,
            trueskill_ratings={
                pid: asdict(rating) for pid, rating in self._trueskill_ratings.items()
            },
            battle_records=dict(self._data)
        )
        with open(filepath, 'w') as f:
            json.dump(asdict(state), f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "BattleSharedPayoff":
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        state = PayoffState(**state_dict)

        payoff = cls()
        payoff._players = state.players
        payoff._players_ids = list(state.players)
        payoff._trueskill_ratings = {
            pid: TrueSkillRating(**rating_dict)
            for pid, rating_dict in state.trueskill_ratings.items()
        }
        payoff._data = defaultdict(BattleRecordDict, state.battle_records)
        return payoff


# === 4. SelfPlayTrainer 集成持久化 ===
class SelfPlayTrainer:
    def __init__(self, ...):
        self.state_dir = Path("./league_state")
        self.state_dir.mkdir(exist_ok=True)
        self._load_league_state()  # 启动时加载

    def _save_league_state(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.state_dir / f"league_state_{timestamp}"

        self.opponent_pool.save_state(str(base_path.with_suffix(".pool.json")))
        self.payoff.save_state(str(base_path.with_suffix(".payoff.json")))

        # 保存当前指向最新的软链接
        latest_link = self.state_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(base_path.name)

    def _load_league_state(self) -> None:
        latest_link = self.state_dir / "latest"
        if latest_link.exists():
            base_path = latest_link.resolve().parent / latest_link.readlink()
            pool_path = str(base_path.with_suffix(".pool.json"))
            payoff_path = str(base_path.with_suffix(".payoff.json"))

            if os.path.exists(pool_path):
                self.opponent_pool = OpponentPool.load_state(pool_path)
            if os.path.exists(payoff_path):
                self.payoff = BattleSharedPayoff.load_state(payoff_path)

    def _periodic_save(self, episode: int) -> None:
        # 每隔N个episode保存一次
        if episode > 0 and episode % self._state_save_interval == 0:
            self._save_league_state()
```

---

### 问题2：FIFO淘汰策略不合理

**问题描述**
- 当前FIFO策略可能淘汰还没充分训练的对手
- 某个"潜力对手"可能因为早期表现不好就被永久移除

#### 改进难度：⭐⭐（简单）

**理由**：
- 只需修改OpponentPool.add()中的淘汰逻辑
- 淘汰策略的变更不影响其他模块

#### 改进收益：⭐⭐⭐⭐（高）

**理由**：
- 保护有潜力的对手不被过早淘汰
- 提高对手池的整体质量
- 训练更加稳定

#### 详细技术方案

```python
class OpponentPool:
    def __init__(self, max_size: int = 10, min_games_threshold: int = 10) -> None:
        self.max_size = max_size
        self.min_games_threshold = min_games_threshold  # 新增：最小对战场次
        self._opponents: List[str] = []
        self._checkpoint_paths: Dict[str, str] = {}
        self._added_timestamps: Dict[str, float] = {}
        self._games_played: Dict[str, int] = {}

    def add(self, opponent_id: str, checkpoint_path: str = None) -> None:
        if opponent_id not in self._opponents:
            self._opponents.append(opponent_id)
            self._added_timestamps[opponent_id] = time.time()
            self._games_played[opponent_id] = 0
            if checkpoint_path:
                self._checkpoint_paths[opponent_id] = checkpoint_path

            if len(self._opponents) > self.max_size:
                self._evict_worst()

    def _evict_worst(self) -> None:
        """淘汰策略：优先保护对战场次多的，评分最低的淘汰"""
        candidates = [
            opp_id for opp_id in self._opponents
            if self._games_played.get(opp_id, 0) >= self.min_games_threshold
        ]

        if not candidates:
            # 如果没有满足最小对战场次的，淘汰最早的
            victim = self._opponents[0]
        else:
            # 计算每个对手的"淘汰分数"
            # 分数 = TrueSkill评分 - 惩罚项（对战场次少的加分）
            def eviction_score(opp_id: str) -> float:
                games = self._games_played.get(opp_id, 0)
                rating = self._payoff.get_trueskill_rating(opp_id) if hasattr(self, '_payoff') else None
                mu = rating.mu if rating else 25.0

                # 对战场次越少，评分越低，越容易被淘汰
                # 但如果场次太少，应该保护
                if games < self.min_games_threshold:
                    protection_bonus = (self.min_games_threshold - games) * 0.5
                else:
                    protection_bonus = 0

                return mu - protection_bonus

            # 选择分数最低的作为victim
            victim = min(candidates, key=eviction_score)

        self._remove_opponent(victim)

    def _remove_opponent(self, opponent_id: str) -> None:
        if opponent_id in self._opponents:
            self._opponents.remove(opponent_id)
        self._on_remove(opponent_id)

    def increment_games(self, opponent_id: str) -> None:
        """每对战一场后调用，增加对战场次"""
        if opponent_id in self._games_played:
            self._games_played[opponent_id] += 1

    def set_payoff_reference(self, payoff: BattleSharedPayoff) -> None:
        """设置payoff引用，用于淘汰决策"""
        self._payoff = payoff


# === 使用示例 ===
pool = OpponentPool(max_size=10, min_games_threshold=10)
pool.set_payoff_reference(payoff)

# 每次对战结束后
pool.increment_games(opponent_id)
```

---

### 问题3：探索/利用比例固定不变

**问题描述**
- 训练全程使用固定比例（70%利用，30%探索）不够科学
- 训练初期应该更多探索，训练后期应该更多利用

#### 改进难度：⭐⭐（简单）

**理由**：
- 只需修改OpponentSelector的初始化和选择逻辑
- 不涉及数据结构的变更

#### 改进收益：⭐⭐⭐⭐（高）

**理由**：
- 训练更高效：初期探索多样策略，后期针对性强化
- 收敛更快：避免在训练初期就过拟合到某一类对手
- 理论上更接近课程学习思想

#### 详细技术方案

```python
class OpponentSelector:
    def __init__(
        self,
        payoff: BattleSharedPayoff,
        exploit_prob: float = 0.7,
        explore_prob: float = 0.3,
        adaptive: bool = True,  # 新增：是否启用自适应
        schedule: str = "linear"  # 新增：调度策略 "linear", "sigmoid", "exp"
    ) -> None:
        self.payoff = payoff
        self.base_exploit_prob = exploit_prob
        self.base_explore_prob = explore_prob
        self.adaptive = adaptive
        self.schedule = schedule
        self._min_confidence_sigma = 5.0

        # 训练进度追踪
        self._total_episodes = 0
        self._total_selections = 0

    def update_progress(self, episodes: int) -> None:
        """更新训练进度"""
        self._total_episodes = episodes

    def _get_adaptive_exploit_prob(self) -> float:
        """根据训练进度计算动态exploit_prob"""
        if not self.adaptive:
            return self.base_exploit_prob

        progress = min(1.0, self._total_episodes / 10000.0)  # 假设10000为完整训练

        if self.schedule == "linear":
            # 线性增长：初期0.3，逐步增加到0.9
            return 0.3 + 0.6 * progress

        elif self.schedule == "sigmoid":
            # S型增长：初期慢增长，中期快增长，后期饱和
            # prob = 0.3 + 0.6 * (1 / (1 + exp(-10 * (progress - 0.5))))
            import math
            return 0.3 + 0.6 * (1 / (1 + math.exp(-10 * (progress - 0.5))))

        elif self.schedule == "exp":
            # 指数增长：前期快速增长，后期趋于稳定
            # prob = 0.3 + 0.6 * (1 - exp(-5 * progress))
            return 0.3 + 0.6 * (1 - math.exp(-5 * progress))

        else:
            return self.base_exploit_prob

    def select(self, player_id: str, opponent_candidates: List[str]) -> str:
        if not opponent_candidates:
            raise ValueError("No opponent candidates available")

        exploit_prob = self._get_adaptive_exploit_prob()

        if random.random() < exploit_prob:
            selected = self._exploit_select(player_id, opponent_candidates)
            if selected is not None:
                self._total_selections += 1
                return selected

        self._total_selections += 1
        return self._explore_select(opponent_candidates)


# === 调度曲线可视化 ===
# linear:     0.3 ──────────────────────────► 0.9
#             (匀速增长)
#
# sigmoid:    0.3 ┐                          ┌ 0.9
#                ┌┘                          └┐
#                └────────────────────────────┘
#             (初期慢，中期快，后期饱和)
#
# exp:       0.3 ┌────┐
#                │    └────────────────────────► 0.9
#             (快速趋近，后期稳定)
```

---

### 问题4：历史衰减机制未使用

**问题描述**
- `decay_all()`方法定义了但从未被调用
- 历史记录会无限累积，decay机制形同虚设

#### 改进难度：⭐（极简单）

**理由**：
- 只需在合适的位置调用decay_all()
- 参数已存在，无需修改数据结构

#### 改进收益：⭐⭐⭐（中等）

**理由**：
- 让近期对战结果更有影响力
- 避免历史数据对当前策略产生过大影响
- 适应对手的进化

#### 详细技术方案

```python
class SelfPlayTrainer:
    def __init__(self, ..., decay_interval: int = 1000, decay_factor: float = 0.99) -> None:
        self.decay_interval = decay_interval  # 每隔多少episode执行一次decay
        self.decay_factor = decay_factor

    def _maybe_decay(self, episode: int) -> None:
        """检查是否需要执行衰减"""
        if episode > 0 and episode % self.decay_interval == 0:
            self.payoff.decay_all()

    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            # ... 训练逻辑 ...

            self._maybe_decay(episode)

            # ... 其他逻辑 ...


# === 或者使用指数衰减调度 ===
class AdaptiveDecayScheduler:
    """自适应衰减调度器"""

    def __init__(self, initial_decay: float = 0.99, min_decay: float = 0.9) -> None:
        self.current_decay = initial_decay
        self.min_decay = min_decay

    def should_decay(self, episode: int, no_improvement_streak: int) -> bool:
        """决定是否执行衰减"""
        if episode % 1000 != 0:
            return False

        # 如果连续多场没有提升，增加衰减力度
        if no_improvement_streak > 5:
            self.current_decay = max(self.min_decay, self.current_decay - 0.01)

        return True

    def get_decay_factor(self) -> float:
        return self.current_decay
```

---

## 二、中优先级问题解决方案

---

### 问题5：对手ID可能重复

**问题描述**
- 使用`len(self.opponent_pool)`生成ID，池满淘汰后可能重复
- 例如：pool满后有opponent_0~opponent_9，淘汰opponent_0，新ID还是opponent_10

#### 改进难度：⭐（极简单）

**理由**：
- 只需改变ID生成策略
- 不涉及业务逻辑变更

#### 改进收益：⭐⭐（较低）

**理由**：
- 当前实现实际上不太会出现重复问题（因为有长度检查）
- 主要是代码健壮性问题

#### 详细技术方案

```python
import uuid
from datetime import datetime

class SelfPlayManager:
    def add_opponent(self, checkpoint_path: str) -> str:
        # 方案1：使用UUID
        opponent_id = f"opp_{uuid.uuid4().hex[:8]}"

        # 方案2：使用时间戳+计数器
        # opponent_id = f"opp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.opponent_pool)}"

        # 方案3：使用全局计数器（推荐）
        self._opponent_counter += 1
        opponent_id = f"opponent_{self._opponent_counter}"

        self.opponent_pool.add(opponent_id, checkpoint_path)
        self.payoff.add_player(opponent_id)
        return opponent_id
```

---

### 问题6：对手池初始为空

**问题描述**
- 训练开始时没有对手
- 第一个对手需要等到checkpoint保存后才加入

#### 改进难度：⭐⭐（中等）

**理由**：
- 需要修改初始化逻辑
- 需要平衡初始对手的生成时机

#### 改进收益：⭐⭐⭐（中等）

**理由**：
- 提高训练早期效率
- 避免早期训练"无人可战"

#### 详细技术方案

```python
class SelfPlayTrainer:
    def __init__(self, ..., initial_opponents: int = 3) -> None:
        self.initial_opponents = initial_opponents  # 训练开始时的初始对手数

    def _create_initial_opponents(self) -> None:
        """创建初始对手池"""
        for i in range(self.initial_opponents):
            # 创建随机策略的初始对手
            checkpoint_path = self._create_random_agent_checkpoint()
            opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
            self._opponent_checkpoints[opponent_id] = checkpoint_path

    def _create_random_agent_checkpoint(self) -> str:
        """创建随机策略的agent checkpoint"""
        import torch
        from ppo_antwar.network.antwar_net import AntWarPolicy
        from ppo_antwar.trainer.ppo_trainer import PPOAgent

        # 创建随机初始化的policy
        policy = AntWarPolicy(
            board_shape=(27, 15, 15),
            global_dim=64,
            action_dim=ACTION_DIM,
            hidden_dim=256
        )

        # 保存checkpoint
        checkpoint = {
            'policy_state_dict': policy.state_dict(),
            'config': {'network': self.trainer.network_config}
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"initial_opponent_{uuid.uuid4().hex[:8]}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


# === 使用 ===
trainer = SelfPlayTrainer(..., initial_opponents=3)
trainer._create_initial_opponents()  # 训练开始时调用
```

---

### 问题7：胜率门槛导致初期数据浪费

**问题描述**
- 前8场对战结果不计入胜率（返回0.5）
- 但这些对战仍然更新了TrueSkill评分（矛盾）

#### 改进难度：⭐⭐（中等）

**理由**：
- 需要重新设计胜率计算逻辑
- 需要保持与TrueSkill评分的一致性

#### 改进收益：⭐⭐⭐（中等）

**理由**：
- 充分利用每一场对战的信息
- 避免信息浪费

#### 详细技术方案

```python
class BattleSharedPayoff:
    def __init__(self, ..., win_rate_confidence_threshold: int = 8) -> None:
        self.win_rate_confidence_threshold = win_rate_confidence_threshold

    def get_win_rate(self, home: str, away: str) -> float:
        key = self.get_key(home, away)
        if key not in self._data:
            return 0.5

        record = self._data[key]
        games = record['games']

        if games == 0:
            return 0.5

        # 方案1：使用贝叶斯估计（推荐）
        # 假设先验为Beta(1,1)（均匀分布）
        # 后验为Beta(1+wins, 1+losses+draws)
        alpha = 1 + record['wins'] + record['draws'] * 0.5  # 胜场 + 平局*0.5
        beta = 1 + record['losses'] + record['draws'] * 0.5  # 负场 + 平局*0.5

        # 后验均值作为胜率估计
        win_rate = alpha / (alpha + beta)

        # 随样本增加逐渐从0.5过渡到真实胜率
        confidence_weight = min(1.0, games / self.win_rate_confidence_threshold)
        win_rate = 0.5 + confidence_weight * (win_rate - 0.5)

        return win_rate

        # 方案2：直接返回实际胜率（简单粗暴）
        # return (record['wins'] + record['draws'] * 0.5) / games


# === 贝叶斯估计可视化 ===
# games=1:  后验非常不确定，胜率接近0.5
# games=4:  开始有一点点倾向
# games=8:  接近真实胜率
# games=16: 非常接近真实胜率
```

---

### 问题8：对手风格单一

**问题描述**
- 所有对手只按TrueSkill评分排序
- 没有考虑对手的"风格"差异

#### 改进难度：⭐⭐⭐⭐（较高）

**理由**：
- 需要定义"风格"的量化指标
- 需要修改对手选择算法
- 需要在checkpoint中保存风格信息

#### 改进收益：⭐⭐⭐⭐（高）

**理由**：
- 提高训练多样性
- 避免过拟合到特定对手风格
- 更接近真实对战环境

#### 详细技术方案

```python
from collections import Counter

@dataclass
class AgentStyle:
    """对手风格特征"""
    build_tower_ratio: float = 0.0      # 建塔比例
    upgrade_ratio: float = 0.0          # 升级比例
    avg_generation_speed: float = 1.0   # 平均产蚁速度
    favorite_positions: List[Tuple[int, int]] = field(default_factory=list)
    aggression_level: float = 0.5      # 攻击性（0=防守，1=进攻）

    def distance_to(self, other: "AgentStyle") -> float:
        """计算两个风格的差异度"""
        return abs(self.build_tower_ratio - other.build_tower_ratio) + \
               abs(self.upgrade_ratio - other.upgrade_ratio) + \
               abs(self.avg_generation_speed - other.avg_generation_speed) + \
               abs(self.aggression_level - other.aggression_level)


class StyleAwareSelector(OpponentSelector):
    def __init__(self, ..., style_diversity_weight: float = 0.3) -> None:
        super().__init__(payoff, exploit_prob, explore_prob)
        self.style_diversity_weight = style_diversity_weight  # 多样性权重
        self._agent_styles: Dict[str, AgentStyle] = {}

    def extract_style(self, checkpoint_path: str) -> AgentStyle:
        """从checkpoint提取风格特征"""
        # 简化实现：实际可能需要运行多场对战来观察
        checkpoint = torch.load(checkpoint_path)

        # 从config或metadata中提取风格信息
        # 实际实现可能需要额外的网络分析
        return AgentStyle(
            build_tower_ratio=0.3,
            upgrade_ratio=0.2,
            avg_generation_speed=1.0,
            aggression_level=0.5
        )

    def _select_diverse_opponent(self, opponent_candidates: List[str]) -> str:
        """选择与近期对手风格差异大的对手"""
        recent_opponents = self._get_recent_opponents(5)  # 最近5个对手
        recent_styles = [self._agent_styles[opp] for opp in recent_opponents if opp in self._agent_styles]

        if not recent_styles:
            return random.choice(opponent_candidates)

        # 计算平均风格
        avg_style = AgentStyle(
            build_tower_ratio=sum(s.build_tower_ratio for s in recent_styles) / len(recent_styles),
            upgrade_ratio=sum(s.upgrade_ratio for s in recent_styles) / len(recent_styles),
            avg_generation_speed=sum(s.avg_generation_speed for s in recent_styles) / len(recent_styles),
            aggression_level=sum(s.aggression_level for s in recent_styles) / len(recent_styles)
        )

        # 选择与平均风格差异最大的对手
        def diversity_score(opp_id: str) -> float:
            if opp_id not in self._agent_styles:
                return float('inf')  # 无风格信息，优先选择
            return avg_style.distance_to(self._agent_styles[opp_id])

        return max(opponent_candidates, key=diversity_score)

    def select(self, player_id: str, opponent_candidates: List[str]) -> str:
        if not opponent_candidates:
            raise ValueError("No opponent candidates available")

        # 混合策略：70% TrueSkill选择，30%多样性选择
        if random.random() < self.style_diversity_weight:
            return self._select_diverse_opponent(opponent_candidates)

        return super().select(player_id, opponent_candidates)
```

---

## 三、问题优先级汇总

| 问题 | 优先级 | 改进难度 | 改进收益 | 推荐顺序 |
|------|--------|----------|----------|----------|
| 持久化机制缺失 | 高 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1 |
| FIFO淘汰策略不合理 | 高 | ⭐⭐ | ⭐⭐⭐⭐ | 2 |
| 探索/利用比例固定 | 高 | ⭐⭐ | ⭐⭐⭐⭐ | 3 |
| 历史衰减未使用 | 高 | ⭐ | ⭐⭐⭐ | 4 |
| 对手风格单一 | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 5 |
| 胜率门槛矛盾 | 中 | ⭐⭐ | ⭐⭐⭐ | 6 |
| 对手池初始为空 | 中 | ⭐⭐ | ⭐⭐⭐ | 7 |
| 对手ID可能重复 | 低 | ⭐ | ⭐⭐ | 8 |

---

## 四、推荐实施计划

### Phase 1：基础改进（1-2天）

1. **历史衰减未使用** - 1小时
   - 在SelfPlayTrainer中添加decay调用
   - 验证效果

2. **对手ID重复** - 1小时
   - 改用UUID或全局计数器

### Phase 2：核心优化（3-5天）

3. **FIFO淘汰策略** - 1天
   - 实现基于评分的淘汰逻辑
   - 添加最小对战场次保护

4. **探索/利用动态调整** - 1天
   - 实现自适应调度
   - 对比不同调度曲线效果

5. **胜率门槛优化** - 1天
   - 实现贝叶斯胜率估计

### Phase 3：高级特性（1周+）

6. **持久化机制** - 2-3天
   - 实现save/load逻辑
   - 处理版本兼容
   - 添加自动保存

7. **对手池初始化** - 1天
   - 实现随机对手生成

8. **风格多样性** - 3-5天
   - 定义风格特征
   - 实现风格提取
   - 修改选择算法