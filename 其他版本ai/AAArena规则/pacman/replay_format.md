# Pacman replay format

Agent 只会看到 bounded JSON，不会看到原始 AI9 文本/zip。顶层字段：

```json
{
  "schema_version": 1,
  "format": "ai9-text-rounds 或 ai9-json-rounds",
  "rounds": [
    {
      "round": 1,
      "observations": ["..."],
      "actions": [{"action": "move", "detail": "..."}]
    }
  ],
  "winner": 0
}
```

- `round` 从 1 开始；`observations` 保留 SDK 状态摘要。
- `actions` 是该回合执行/提交的动作。
- `winner: 0/1` 是 P0/P1；`-2` 表示平局。
- JSON 结果型游戏（MoneCraft/Pacman）还会保留终局 `result`。
- 超出 256 KiB 时框架会自动截断和稀疏化，优先保留首尾回合与动作。

## 如何读
1. 先看终局分数与胜者，再回到中期转折回合。
2. 对比己方与对手的资源、位置、血量和动作顺序。
3. 把规律写成与对手 ID、seed、回放 ID 无关的规则。
4. 如果 JSON 与 narrate 冲突，停止迭代并记录证据。
