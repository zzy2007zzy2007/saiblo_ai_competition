# SnakeGo replay facts

SnakeGo replay files are JSON objects with `game_config`, `item_list`, `round_info`, `operations`, and `end_info`.

## Observable actions

`operations` is an ordered list of per-round operation batches. Each operation's `basic` object identifies the snake and normalized action. Result fields can include `got_item`, `extra_length`, `new_snake_id`, `dead_snake`, `should_solid`, and `extra_solid_area`.

## Observable state

`game_config` supplies board dimensions and round limit. `item_list` supplies item identifiers, types, appearance times, and coordinates. Operation results supply numeric length growth, spawned identifiers, collected item identifiers, and death lists.

## Terminal facts

`end_info` supplies winner, termination type, error text, and final score. Reported rounds come from the evaluator when present or from `round_info` length.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Each operation line includes round and snake references. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
