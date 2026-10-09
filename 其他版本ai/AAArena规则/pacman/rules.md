# Pacman 规则摘要

**目标**：吃豆、躲避与埋雷。

## 玩法
- 双方 Pacman 在对称迷宫中吃普通豆和金币豆。
- 吃到怪物会受伤并可能被送回出生点；可释放地雷改变路径。
- 连续吃豆可进入加速状态。
- 600 回合后按分数判定。

## Agent 接口
- `public_sdk/ai.cpp` 是唯一需要编辑的策略文件。
- 具体 callback 与编译方法见 `public_sdk/INTERFACE.md`。
- 公开 SDK 只包含协议和状态访问；不要访问后端内部文件或对手代码。

## 回放
- `replay/format.md` 说明 bounded JSON 的字段。
- `replay/guide.md` 说明如何把回合、状态增量和动作转成可证伪策略假设。
