# Antwar Python 选手模板

这个目录是从经过审计的历史选手包中恢复的 Antwar Python 兼容模板。文件来源、
复用数量和原始恢复哈希见 `SDK_PROVENANCE.json`。

## 复制与编写策略

```bash
cp -R games/antwar/public_sdk ./my-antwar-player
```

选手包根目录必须保留 `main.py`。编辑其中的
`simple_ai(my_seat, state)`，返回本回合的 `list[Operation]`；
`run_antwar_ai(simple_ai)` 会完成初始化、双方操作交换和局面更新。

`antwar/` 包中的主要入口为：

- `controller.py`：`GameController` 和 `run_antwar_ai()`；
- `gamestate.py`：可在本地模拟操作的 `GameState`；
- `protocol.py`：`Operation` 及建塔、升级、拆除、超级武器和基地升级助手；
- `coord.py`、`gamedata.py` 和 `pheromone.py`：坐标与公开局面数据结构。

AA-Arena 可以直接将复制后的目录作为 `PlayerRef.code_path`：

```python
from agentbench.core import PlayerRef

player = PlayerRef("my-antwar-ai", code_path="/absolute/path/to/my-antwar-player")
```

选手模板本身是 Python。Antwar 裁判后端则位于同一 GamePack 的 `backend/`，评测准备时
会使用系统 `make` 和 `g++` 以 C++17 编译。这两个工具需由运行评测的机器预先安装，
不会在对局进程中下载。
