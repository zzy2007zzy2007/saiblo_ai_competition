# Miracle 兼容 SDK

这个目录是从经过审计的历史选手包中恢复出来的 Python 兼容 SDK，并由
AA-Arena 补充了协议助手、文档和中性示例。它不是一个上游单独发布、独立版本化的
官方 SDK；具体文件来源和复用计数见 `SDK_PROVENANCE.json`。

## 快速开始

将整个目录复制为一个选手包：

```bash
cp -R games/miracle/public_sdk ./my-miracle-player
```

选手包根目录必须保留可执行入口 `main.py`。在 `main.py` 的 `AI.choose_cards()` 中选择
卡组，在 `AI.play()` 中编写每回合策略。`example_ai.py` 给出了一个固定卡组、
每回合只调用 `end_round()` 的中性示例；它用于展示钩子，不会自动替换
`main.py`。

AA-Arena 会直接从传给 `PlayerRef.code_path` 的目录启动 `main.py`，例如：

```python
from agentbench.core import PlayerRef

player = PlayerRef("my-miracle-ai", code_path="/absolute/path/to/my-miracle-player")
```

SDK 本身不导入 AA-Arena 模块，因此复制后仍可作为普通选手进程运行。

## 协议与坐标

- 裁判发给选手的状态帧使用 **6 位 ASCII 十进制字节数**作为头部。
- 选手发给裁判的操作帧使用 **4 字节有符号大端整数**作为头部。
- 两种长度都按 UTF-8 编码后的字节计数，而不是 Python 字符数。
- 地图位置使用六边形立方坐标 `(x, y, z)`，并满足 `x + y + z == 0`。

`ai_client.py` 中的 `encode_message()`、`read_state()` 和 `build_operation()` 可用于
单独检查帧与操作数据；`AiClient` 中原有的召唤、移动、攻击、使用神器和结束
回合方法保持原有裁判协议。
