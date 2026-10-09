# LOTA 规则摘要

**目标**：MOBA 式推塔对抗。

## 玩法
- 每位玩家选择英雄，兵线会沿固定路线前进。
- 英雄可移动、攻击、施放技能、加点和复活。
- 摧毁主基地，或在回合上限时按剩余防御/击杀/经验判定。

## Agent 接口
- `public_sdk/ai.cpp` 是唯一需要编辑的策略文件。
- 具体 callback 与编译方法见 `public_sdk/INTERFACE.md`。
- 公开 SDK 只包含协议和状态访问；不要访问后端内部文件或对手代码。

## 回放
- `replay/format.md` 说明 bounded JSON 的字段。
- `replay/guide.md` 说明如何把回合、状态增量和动作转成可证伪策略假设。
