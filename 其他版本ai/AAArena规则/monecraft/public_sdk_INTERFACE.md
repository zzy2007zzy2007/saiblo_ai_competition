# MoneCraft AI9 interface

- Entry point: ``player_ai(Mine&, Map&, GameInfo&, RoundInfo&, Action&)``.
- Edit `ai.cpp`; keep SDK headers and runtime sources unchanged.
- Local build: `make`, producing `player.so`.
- Evaluator compilation is independent and will compile your `ai.cpp` with the private game SDK.
- stdout is not a diagnostic channel.
