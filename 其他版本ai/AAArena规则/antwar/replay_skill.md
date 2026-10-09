# AntWar replay facts

AntWar replay files are non-empty JSON arrays. Each element is one round with `op0`, `op1`, and `round_state`.

## Observable actions

`op0` and `op1` contain the submitted operations. Public operation fields include `type`, `pos`, `id`, and `args`. An empty list records that no operation was submitted in that round.

## Observable state

`round_state` may contain `round`, `camps`, `coins`, `speedLv`, `anthpLv`, `towers`, `ants`, `pheromone`, `winner`, and `message`. Tower entries are incremental: an entry whose `type=-1` removes that tower identifier from the maintained public tower set. The narrator reports adjacent-frame changes for base HP, coins, and both level vectors. Ant terminal records are deduplicated by `(player, id)`.

## Terminal facts

The final frame supplies `winner`, base HP, coins, levels, and optional per-player termination codes in `message`. `message` can be a JSON-encoded string or a two-element array.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Every action line includes a round reference. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
