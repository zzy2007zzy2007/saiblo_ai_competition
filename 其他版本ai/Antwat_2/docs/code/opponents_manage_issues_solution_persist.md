# 问题7：持久化机制缺失 - 解决方案

## 一、问题描述

当前ppo_v1的SelfPlay系统存在以下问题：
- 对手池、payoff记录、TrueSkill评分都没有保存/加载功能
- 训练中断后无法恢复，之前积累的对手和评分全部丢失

## 二、解决方案设计

### 设计原则

1. **简单直接**：不做过度设计，满足基本需求即可
2. **与checkpoint同步**：保存时机与模型checkpoint一起执行
3. **向后兼容**：不影响现有代码逻辑
4. **易于维护**：使用JSON格式存储，方便查看和调试

### 核心思路

在每次保存模型checkpoint时，同时保存：
1. 对手池状态（对手ID列表、checkpoint路径）
2. Payoff记录（对战胜负记录）
3. TrueSkill评分（每个玩家的mu和sigma）

## 三、详细技术方案

### 3.1 数据结构设计

```python
# === 持久化数据结构 ===
from dataclasses import dataclass, asdict
import json
from typing import Dict, List, Optional

@dataclass
class PersistState:
    """持久化状态数据结构"""
    version: str = "1.0"
    timestamp: float = 0.0
    
    # 对手池状态
    opponent_ids: List[str] = field(default_factory=list)
    opponent_checkpoint_paths: Dict[str, str] = field(default_factory=dict)
    
    # Payoff记录
    players: List[str] = field(default_factory=list)
    battle_records: Dict[str, Dict] = field(default_factory=dict)  # key -> {wins, draws, losses, games}
    
    # TrueSkill评分
    trueskill_ratings: Dict[str, Dict] = field(default_factory=dict)  # player_id -> {mu, sigma, games_played}
```

### 3.2 OpponentPool 添加持久化方法

```python
class OpponentPool:
    def save_state(self, filepath: str) -> None:
        """保存对手池状态"""
        state = {
            'opponent_ids': self._opponents,
            'checkpoint_paths': self._checkpoint_paths,
            'added_timestamps': self._added_timestamps,
            'games_played': self._games_played,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "OpponentPool":
        """加载对手池状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        pool = cls()
        pool._opponents = state.get('opponent_ids', [])
        pool._checkpoint_paths = state.get('checkpoint_paths', {})
        pool._added_timestamps = state.get('added_timestamps', {})
        pool._games_played = state.get('games_played', {})
        return pool
```

### 3.3 BattleSharedPayoff 添加持久化方法

```python
class BattleSharedPayoff:
    def save_state(self, filepath: str) -> None:
        """保存payoff状态"""
        state = {
            'players': self._players,
            'players_ids': self._players_ids,
            'trueskill_ratings': {
                pid: {
                    'mu': rating.mu,
                    'sigma': rating.sigma,
                    'games_played': rating.games_played
                }
                for pid, rating in self._trueskill_ratings.items()
            },
            'battle_records': dict(self._data),
            'decay': self._decay,
            'min_win_rate_games': self._min_win_rate_games,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "BattleSharedPayoff":
        """加载payoff状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        payoff = cls(
            decay=state.get('decay', 0.99),
            min_win_rate_games=state.get('min_win_rate_games', 8)
        )
        payoff._players = state.get('players', [])
        payoff._players_ids = state.get('players_ids', [])
        payoff._trueskill_ratings = {
            pid: TrueSkillRating(**rating_dict)
            for pid, rating_dict in state.get('trueskill_ratings', {}).items()
        }
        payoff._data = defaultdict(BattleRecordDict, state.get('battle_records', {}))
        return payoff
```

### 3.4 SelfPlayTrainer 集成持久化

```python
class SelfPlayTrainer:
    def __init__(self, ...):
        # 新增：持久化目录
        self.persist_dir = self.checkpoint_dir / "league_state"
        self.persist_dir.mkdir(exist_ok=True)
        
        # 尝试加载之前的状态
        self._load_league_state()

    def _save_league_state(self) -> None:
        """保存整个league状态（与checkpoint同步）"""
        # 保存对手池
        pool_path = self.persist_dir / "opponent_pool.json"
        self.opponent_pool.save_state(str(pool_path))
        
        # 保存payoff
        payoff_path = self.persist_dir / "payoff.json"
        self.payoff.save_state(str(payoff_path))
        
        # 保存checkpoint映射
        checkpoint_map_path = self.persist_dir / "checkpoint_map.json"
        with open(checkpoint_map_path, 'w') as f:
            json.dump(self._opponent_checkpoints, f, indent=2)

    def _load_league_state(self) -> None:
        """加载之前保存的league状态"""
        pool_path = self.persist_dir / "opponent_pool.json"
        payoff_path = self.persist_dir / "payoff.json"
        checkpoint_map_path = self.persist_dir / "checkpoint_map.json"
        
        if pool_path.exists():
            try:
                self.opponent_pool = OpponentPool.load_state(str(pool_path))
                self.opponent_pool.set_payoff_reference(self.payoff)
            except Exception as e:
                logger.warning(f"加载对手池状态失败: {e}")
        
        if payoff_path.exists():
            try:
                self.payoff = BattleSharedPayoff.load_state(str(payoff_path))
            except Exception as e:
                logger.warning(f"加载payoff状态失败: {e}")
        
        if checkpoint_map_path.exists():
            try:
                with open(checkpoint_map_path, 'r') as f:
                    self._opponent_checkpoints = json.load(f)
            except Exception as e:
                logger.warning(f"加载checkpoint映射失败: {e}")

    def train(self, num_episodes: int):
        # ... 训练循环 ...
        
        if episode % opponent_update_interval == 0 and episode > 0:
            checkpoint_path = self.trainer.save_checkpoint(
                str(self.checkpoint_dir / f"model_{episode}.pt")
            )
            new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
            self._opponent_checkpoints[new_opponent_id] = checkpoint_path
            
            # 同步保存league状态
            self._save_league_state()  # <--- 新增：与checkpoint同步保存
```

## 四、保存时机

### 触发条件

持久化操作在以下时机触发：

1. **定期保存**：每次保存模型checkpoint时（`episode % opponent_update_interval == 0`）
2. **训练结束**：训练完成时保存最终状态
3. **异常处理**：可选，捕获异常时保存状态（可选增强）

### 文件结构

```
checkpoints/
├── model_1000.pt          # 模型checkpoint
├── model_2000.pt          # 模型checkpoint
├── ...
└── league_state/          # league状态目录
    ├── opponent_pool.json  # 对手池状态
    ├── payoff.json         # payoff记录和TrueSkill评分
    └── checkpoint_map.json # 对手ID到checkpoint路径的映射
```

## 五、兼容性考虑

### 版本管理

- 在状态文件中包含`version`字段
- 加载时检查版本，支持向后兼容

### 容错处理

- 文件不存在时使用默认初始化
- JSON解析失败时记录警告并继续
- 部分加载失败不影响整体训练

## 六、改进难度与收益

### 改进难度：⭐⭐⭐（中等）

**理由**：
- 需要在多个类中添加序列化和反序列化逻辑
- 需要处理文件路径管理
- 需要保证加载时的兼容性

### 改进收益：⭐⭐⭐⭐⭐（极高）

**理由**：
- 训练中断可恢复，节省大量计算资源
- 支持训练过程的分析和调优
- 为分布式训练提供基础

## 七、实施步骤

1. **第一步**：在`OpponentPool`中添加`save_state()`和`load_state()`方法
2. **第二步**：在`BattleSharedPayoff`中添加`save_state()`和`load_state()`方法
3. **第三步**：在`SelfPlayTrainer`中添加`_save_league_state()`和`_load_league_state()`方法
4. **第四步**：在训练循环中调用`_save_league_state()`（与checkpoint同步）
5. **第五步**：在初始化时调用`_load_league_state()`

## 八、注意事项

1. **性能影响**：JSON序列化开销较小，与checkpoint同步执行可接受
2. **磁盘空间**：状态文件较小，不会显著增加存储需求
3. **并发安全**：训练过程为单线程，无需考虑并发写入问题

## 九、总结

本方案提供了一个简单直接的持久化解决方案：

- 使用JSON格式存储状态，易于查看和调试
- 与模型checkpoint同步保存，保证数据一致性
- 支持加载之前的状态，实现训练中断恢复
- 保持向后兼容，不影响现有代码逻辑