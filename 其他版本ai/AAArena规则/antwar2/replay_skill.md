# AntWar2 replay facts

AntWar2 replay files are non-empty JSON arrays. Each element contains both players' operation lists and a public state snapshot.

## Observable actions

`op0` and `op1` contain operation objects. Codes 11–13 describe tower changes, 21–24 describe activated devices, and 31–32 describe level changes. Empty operation lists are represented as `HOLD` facts.

## Observable state

`round_state` contains `round`, `camps`, `coins`, `speedLv`, `anthpLv`, `weaponCooldowns`, `towers`, `ants`, `activeEffects`, and `winner`. Tower entries are deltas; the decoder maintains the public tower set across frames. The narrator reports adjacent changes in base HP, coins, levels, and tower count.

## Terminal facts

The final canonical frame supplies the winner, base HP, coins, levels, tower set, ant set, and active effects. Official winner and round values take precedence when supplied by the evaluator.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Action lines carry `state_id` references. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
