# MoneCraft AI9 GamePack

本目录包含自包含 AI9 后端、评测器、公开 SDK 与真实/样例选手池。评测使用
仓库内 AI9 runtime；agent iteration 使用 `public_sdk/` 与回放资料。

- 规则：`rules.md`
- 公开 SDK：`public_sdk/`
- 回放格式：`replay_format.md`
- 回放阅读方法：`replay_skill.md`
- 评测器：`evaluator/`

注意：AI9 不使用 Saiblo stdio 协议，而是后端 + `ailoader` +选手 `.so`；
这些细节由 evaluator 处理，agent 只编辑 public SDK 中的策略文件。
