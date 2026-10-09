# Rollman replay facts

Rollman replays are JSONL state streams. Every decodable object is a state frame. A level reset can make the per-level `round` value return to zero, so file order is the global frame order.

## Observable state

Frames can contain `level`, `round`, `score`, `pacman_coord`, `ghosts_coord`, `pacman_skill_status`, and a board on reset frames. The narrator reports coordinate snapshots, level changes, and adjacent score changes for pacman and the ghost side.

## Roles

Pacman supplies one direction per step. The ghost side supplies three directions per step. Role labels reflect the evaluator perspective when one is provided.

## Terminal facts

The last available score determines the recorded winner when the official result is absent. The last frame can contain `StopReason`; maximum recorded round and total state-frame count are reported separately.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Position lines retain global frame references. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
