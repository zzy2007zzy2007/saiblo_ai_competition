# Leaderboard Pool — 轻量级对手池设计

## 目标

替代 WinGraph 的对手选择功能，但更轻量——不维护有向图、不重新计算 SCC、不需要额外对战验证。核心思路：**维护一个从强到弱排序的对手列表，只允许"打败最强者"的个体加入。**

```
WinGraph:       A → B → C (DAG + SCC 缩点 + 边权重)
Leaderboard:   [top1, top2, top3, ...] (线性排序，假设不存在循环克制)
```

## 和 WinGraph 的关系

| 维度 | WinGraph | Leaderboard Pool |
|:----|:---------|:----------------|
| 数据结构 | 有向图 + SCC + 边权重 | 有序列表 |
| 对手选择 | DAG 顶层节点 | 按排名加权采样 |
| 新节点验证 | 与所有现存节点对战 | 只与当前最强节点对战 |
| 复杂度 | O(N²) 场对战验证 | O(1) 场对战验证 |
| 克制关系 | 完全建模（包含循环） | 线性假设（无循环） |
| 剪枝 | Kosaraju + 深度 | 直接砍尾部 |
| 适用场景 | 策略高度多样、克制关系复杂 | 策略空间相对单调、追求收敛速度 |

Leaderboard 适用于 ss_train 的当前阶段——策略空间还不够丰富，线性假设足够成立。后续如果发现循环克制严重，可以再加 WinGraph。

## 数据结构

```python
@dataclass
class LeaderboardEntry:
    gen: int               # 第几代加入的
    params: np.ndarray     # 参数向量（mean 或 top1）
    score: float           # 加入时的 fitness（与最强者对战的胜率）

class Leaderboard:
    """从强到弱排序的对手池。"""
    def __init__(self, max_size: int = 20):
        self.entries: list[LeaderboardEntry] = []  # 从强到弱排序
        self.max_size = max_size
```

**排序假设**：entries[i] 打得过 entries[i+1]（胜率 > 50%）。基于当前训练经验，策略空间线性假设是合理的——后期如果发现循环克制再升级。

## API 设计

接口和 WinGraph 保持一致，方便未来切换：

```python
class Leaderboard:
    def __init__(self, max_size: int = 20):
        """初始化对手池。
        Args:
            max_size: 池子最大容量（默认 20）
        """
        ...

    def add_candidate(self, gen: int, params: np.ndarray, score: float) -> bool:
        """尝试加入一个新候选个体。

        流程：
        1. 如果池子为空 → 直接加入（第一代）
        2. 如果池子非空 → 用 score 作为"与当前最强者对战"的估计，
           插入到对应位置
        3. 如果池子满了 → 删除尾部最弱的

        Args:
            gen: 代数
            params: 参数向量
            score: 与当前最强者对战的胜率（由调用方提供）

        Returns:
            True 表示成功加入，False 表示被拒绝（弱于现有最弱且池子满）
        """
        ...

    def get_opponents(self, k: int = 5) -> list[dict]:
        """从池中按排名加权采样 k 个对手。

        采样权重：排名越靠前（越强）概率越高。
        - top-1: ~30%
        - top-2: ~20%
        - top-3: ~15%
        - top-5~10: 均匀低权重
        - 如果 k > 池子大小，补齐 ExampleAI 参数

        Returns:
            [{"gen": gen, "params": params}, ...]
            按权重降序排列（权重高的在前面）
        """
        ...

    def get_strongest(self) -> dict | None:
        """返回当前最强的对手。"""
        ...

    def prune(self, max_size: int) -> list[int]:
        """裁剪到 max_size，返回被删除的代数列表。"""
        ...

    def state_dict(self) -> dict:
        """序列化保存。"""
        ...

    def load_state_dict(self, d: dict):
        """从序列化恢复。"""
        ...

    def get_info(self) -> dict:
        """统计信息。"""
        ...

    def get_ranked_entries(self) -> list[dict]:
        """返回所有个体的排名列表，从强到弱。

        和 WinGraph 的 get_node_depths 类似，方便外部灵活使用。

        Returns:
            [{"gen": int, "params": np.ndarray, "rank": int, "score": float}, ...]
            rank=0 为最强，依次递增。
        """
        ...

    def get_opponents(self, k: int = 5) -> list[dict]:
        """按排名加权采样 k 个对手。

        基于 get_ranked_entries 实现，保证一致性。
        采样权重：排名越靠前（越强）概率越高。
        如果 k > 池子大小，补齐 ExampleAI 参数。

        Returns:
            [{"gen": gen, "params": params}, ...]
        """
        ...
```

## 使用方式

### 初始化

```python
leaderboard = Leaderboard(max_size=20)
opponents = leaderboard.get_opponents(k=k_per_ind)
```

如果第一次调用时池子为空，`get_opponents`（内部调用 `get_ranked_entries`）返回 ExampleAI 的参数（冷启动）。

### 添加到池

```python
# 每代评估结束后：
# 让当前 mean 与 leaderboard 最强打一局 => 获取 score
score = compute_win_rate(trainer.mean, leaderboard.get_strongest().params)
leaderboard.add_candidate(gen, trainer.mean.copy(), score)
```

**不需要额外对战验证**——score 从评估阶段的数据中已经可得，或者只需要和最强打一局。

### 对手采样

```python
# 每代评估前：
opp_list = leaderboard.get_opponents(k=k_per_ind)
# opp_list = [{"gen": 5, "params": ...}, {"gen": 12, "params": ...}, ...]
# 替换 select_opponents 的逻辑
```

## 冷启动

| 代数 | 池内容 | 对手采样来源 |
|:---|:------|:-----------|
| gen_00 | 空 | 全部用 ExampleAI |
| gen_01 | [gen_00 top1] | ExampleAI + gen_00 top1 |
| gen_02 | [gen_01 top1, gen_00 top1] | 前 2 代 top1 + 补 ExampleAI |
| gen_N | [top1, top2, ..., topN] | 按排名加权从池中采样 |

## 与 ss_train 的集成点

ss_train 的主循环中，`select_opponents` 调用替换为 `leaderboard.get_opponents`：

```python
# 当前：
opp_indices = select_opponents(args.pop_size, args.games, rng)
opp_params_list = [params_list[i] for i in opp_indices]

# 改后：
opp_list = leaderboard.get_opponents(k=args.games // 2)
opp_params_list = [entry["params"] for entry in opp_list]
```

每代结束后更新 leaderboard：

```python
# 在 SS 监督训练之后、保存 checkpoint 之前：
if leaderboard is not None:
    # 让当前 mean 与最强者对战
    strongest_params = leaderboard.get_strongest()["params"]
    score = compute_win_rate(trainer.mean, strongest_params, pool)
    leaderboard.add_candidate(gen, trainer.mean.copy(), score)
```

## 复杂度

| 操作 | WinGraph | Leaderboard |
|:---|:--------|:-----------|
| 新个体验证 | O(N × edge_games) 场对局 | O(1) 场对局 |
| 对手采样 | O(N log N) 拓扑排序 | O(k) 加权采样 |
| 序列化 | 节点 + 边权重 | 有序列表 |
| 内存 | 图结构 + 边记录 | 有序列表 |

## 与 Checkpoint 的集成

Leaderboard 保存在 checkpoint 中，加载后恢复：

```python
# save_checkpoint:
if leaderboard is not None:
    data["leaderboard"] = leaderboard.state_dict()

# load_checkpoint:
if "leaderboard" in ckpt:
    leaderboard = Leaderboard(...)
    leaderboard.load_state_dict(ckpt["leaderboard"])
```

## 后续升级路径

如果线性排序假设不成立（出现循环克制），可以：

1. **给每个个体加一条额外信息：输给过谁**，当发现 A > B > C > A 时降级为 WinGraph
2. 接口一致，不需要改调用方代码

最终形态是 WinGraph。Leaderboard 是通往 WinGraph 的中间步骤。
