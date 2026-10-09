# LostSpace 兼容 SDK

这个目录提供可直接复制的 LostSpace Python 选手包。`legacy_main.py` 是从经过
审计的历史选手中恢复的原始文件，内含特定选手的完整策略，仅作来源证据和
兼容参考。`lostspace_sdk/`、新 `main.py` 和本文档是 AA-Arena 根据裁判协议
重建的兼容层，不是上游单独发布的官方 SDK。详细来源见
`SDK_PROVENANCE.json`。

## 复制与运行

```bash
cp -R games/lostspace/public_sdk ./my-lostspace-player
python ./my-lostspace-player/main.py
```

评测器会在选手包根目录执行 `main.py`。只需编辑其中的 `AI.play(state)`，并让
它返回按执行顺序排列的 `tuple[Action, ...]`。每个回合的最后一项必须是
`Action.finish()`：

```python
from lostspace_sdk import Action, Client, TurnState


class AI(Client):
    def play(self, state: TurnState) -> tuple[Action, ...]:
        return (Action.detect((1, 1, 1)), Action.finish())


if __name__ == "__main__":
    AI().run()
```

AA-Arena 可以把复制后的目录直接作为 `PlayerRef.code_path`：

```python
from agentbench.core import PlayerRef

player = PlayerRef("my-lostspace-ai", code_path="/absolute/path/to/my-lostspace-player")
```

`lostspace_sdk` 不导入 AA-Arena 内部对象，因此复制后仍是独立选手进程。

## 状态与行动

`Client` 只在 `roundbegin.inturn` 与本玩家相同且玩家存活时调用
`play()`。`TurnState` 及其 `Inventory`、`Opponent` 和 `VisibleNode` 均为不可变
快照；只包含裁判已下发的自身状态、物品栏、可见对手和视野节点，不提供
隐藏地图节点、不可见陷阱或裁判内部状态。

可用行动助手为：

- `Action.move((x, y, z))`
- `Action.attack((x, y, z), player_id)`
- `Action.interact("Box" | "KeyMachine")`
- `Action.collect("Materials", tool_name)`
- `Action.escape(to_escape)`
- `Action.trap("LandMine" | "Sticky")`
- `Action.tool("Kit")` 或 `Action.tool("Transport", (x, y, z))`
- `Action.detect((x, y, z))`
- `Action.finish()`

只能在自己存活的行动回合发出操作。同一回合可以返回多个行动，`Client` 会按顺序
发送，并在每个非 `finish` 行动后等待裁判响应。可重载
`on_action_response(action, response)` 观察响应，或重载 `on_event(message)` 观察
回合外消息。

坐标统一为 `(x, y, z)`：`x` 是行，`y` 是列，`z=0` 为逃生舱层，`z=1`
为出生点层，`z=2` 为上层。

## 帧格式

- 裁判到选手：4 个 ASCII 十进制长度字符 + UTF-8 JSON。
- 选手到裁判：4 字节有符号大端长度 + UTF-8 JSON。
- 长度均按 UTF-8 编码后的字节数计算，单帧上限为 64 MiB。

如需单独测试协议，可直接使用 `encode_message()` 和 `decode_message()`。
