# WinGraph 模块设计

## 目标

维护一个有向图，记录历史精英个体之间的克制关系（谁打得过谁），用于：

1. **挑对手**：只从 DAG 顶层挑最强的对手来评估当前种群，而不是把所有历史精英混在一起
2. **去噪声**：自动剪枝底层弱个体，减小 elite pool 的噪声
3. **建数据集**：积累所有代际之间有可比性的对战数据

---

## 数据结构

### 有向图

- **节点**：每个节点对应一个精英个体（checkpoint 的 top1 或 mean 参数向量），存储 `(gen, param_vector, example_win_rate?)`，以及该个体接入图时与其他节点的对战记录
- **有向边**：`A → B` 表示 A 打 B 的胜率超过阈值（如 55%）。边存储 `(win_a, win_b, draws)` 原始计数，支持后续更新
- **SCC 缩点**：使用 Kosaraju 算法缩点后，形成一个 DAG

### 图的状态维度

```
n_max: 20~30 个节点（上限）
edge_games: 每对对比 10~20 局
win_threshold: 建边阈值 55%（与之前测试一致）
```

---

## API 设计

### `WinGraph` 类

```python
class WinGraph:
    def __init__(self, n_max=30, edge_games=10, win_threshold=0.55):
        ...

    def add_node(self, gen: int, params: np.ndarray) -> dict:
        ...
    
    @property
    def n_max(self) -> int:
        return self._n_max
    
    @n_max.setter
    def n_max(self, value: int):
        """动态调整节点上限。如果设小了，自动 prune 掉超出的节点。"""
        self._n_max = value
        if len(self.nodes) > self._n_max:
            self.prune(len(self.nodes) - self._n_max)
    
    @property
    def edge_games(self) -> int:
        return self._edge_games
    
    @edge_games.setter
    def edge_games(self, value: int):
        """动态调整每对比试局数。之后新增的节点按新值执行。"""
        self._edge_games = value

    def remove_node(self, gen: int) -> bool:
        """
        加入一个新的精英个体。
        步骤：
        1. 与图中所有现有节点各对战 edge_games 局
        2. 统计胜/负/平，添加反向边
        3. 如果有向边涵盖了所有节点对，则自动按 win_threshold 简化
        4. 返回新增的对战结果 dict

        Returns:
            {"gen": gen, "wins": int, "losses": int, "draws": int, "win_rate": float}
        """

    def remove_node(self, gen: int) -> bool:
        """按 gen 编号移除指定节点及其所有边。"""

    def prune(self, k: int = 1) -> list[int]:
        """
        从 DAG 的 sink SCCs（最底层）中移除 k 个最弱的节点。
        如果底层不够 k 个，向上层递补。
        Returns: 被移除的 gen 编号列表
        """

    def get_top_opponents(self, k: int = 5) -> list[dict]:
        """
        从 DAG 的 source SCCs（最顶层）中返回 top-k 对手。
        如果 source SCC 中不够 k 个，向下层递补。
        Returns: [{"gen": int, "win_rate_vs_example": float}, ...]
        """

    def recompute_scc(self) -> tuple[list, list]:
        """
        重新运行 SCC 缩点和 DAG 拓扑排序。
        Returns: (scc_list, topo_order)
        """

    def get_graph_info(self) -> dict:
        """
        获取图的当前状态统计（用于日志和监控）。
        Returns:
            {"n_nodes": int, "n_edges": int, "n_scc": int,
             "topo_ranks": [[gen, ...], ...],  # 按拓扑序排列的 SCCs
             "edge_games": int, "n_max": int}
        """

    def state_dict(self) -> dict:
        """序列化为可保存的 dict（包含所有节点和边的原始数据）。"""

    def load_state_dict(self, d: dict):
        """从 dict 恢复。"""
```

### 核心工具函数

```python
def compute_win_rate(params_a: np.ndarray, params_b: np.ndarray,
                     games: int, workers: int) -> tuple[int, int, int]:
    """
    让两个个体对战 games 局（平分先后手），
    Returns: (wins_a, wins_b, draws)
    """
```

---

## 建边规则

1. **方向判定**：`win_rate > win_threshold` 时建边 `A → B`
2. **对称情况**：如果 `|win_rate - 0.5| < 0.05`（即 45-55% 之间），认为势均力敌，不建边
3. **反向情况**：如果 `win_rate < 1 - win_threshold`，建反向边 `B → A`
4. **已有边更新**：如果两个节点已有对战记录，新数据可以累加到旧记录上（增加统计置信度）

---

## SCC → DAG 缩点流程

```
1. 从当前有向边构建邻接表
2. Kosaraju 算法：
   a. 第一次 DFS：按完成时间排序
   b. 第二次 DFS（反向图）：收集 SCC group
3. 在 SCC groups 之间建 DAG 边
4. 拓扑排序
5. Source SCCs = in_degree == 0 的节点组
6. Sink SCCs = out_degree == 0 的节点组
```

---

## 淘汰策略

每次加入新节点后，如果节点数超过 `n_max`：

1. 找出 DAG 的所有 sink SCCs（最底层组）
2. 从 sink SCC 中选择一个节点移除
3. 如果 sink SCC 中有多个节点，优先移除：
   - 对战次数最少的
   - 或 gen 最小的
4. 如果 sink SCC 为空或只剩一个，向上层递补

---

## 与 ES 训练对接

在 `es_train.py` 中，每代结束后：

```python
# 初始化
wg = WinGraph(n_max=30, edge_games=10)

# 每代结束后
wg.add_node(generation, mean_params)      # 加入种群均值
wg.add_node(generation, top1_params)       # 加入 top1

# 从 DAG 顶层挑对手
top_opponents = wg.get_top_opponents(k=5)

# 在评估适应度时使用 top_opponents 作为对手
# 而不是从 elite_pool 中随机选
```

---

## 文件结构

```
code/my_ai/win_graph.py    ← WinGraph 类实现
docs/win_graph_design.md   ← 本文档
```

---

## 风险和开放问题

1. **计算成本**：每次加新节点要和所有现有节点各打 `edge_games` 局。10 个节点时 = 10×10=100 局/代，20 个 = 200 局/代。可以先用 2 workers 跑，约 2-3 分钟/代
2. **阈值选择**：55% 是经验值，可能需要调优。太松（如 50%）导致全连通图没信息，太严（如 70%）导致边太少
3. **冷启动**：前几代节点很少时，无法做 SCC 缩点。需要至少 5 个节点才有意义
4. **ExampleAI 评分可选**：图中不一定需要存 vs ExampleAI 的胜率，但可以作为辅助维度帮助决策
