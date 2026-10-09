# AquaWar replay facts

AquaWar replay files are JSON arrays of state snapshots. Non-object elements, including a trailing `null` sentinel, are ignored.

## Observable actions

`operation[].Action` is a string. Recorded values include `Pick`, `Assert`, `Null`, `Action`, and `Finish`. Action objects retain public positions, fish identifiers, action type, and target lists.

## Observable state

Each snapshot may contain `gamestate`, `cur_turn`, `rounds`, `players`, and `winner`. Player records expose `my_fish` and `fight_fish`; fish records expose identifiers, HP, attack, and exposure status. The narrator compares fish HP by player and fish identifier in adjacent snapshots.

## Terminal facts

The last object supplies the winner, round count, and final visible fish records. An externally supplied official result takes precedence.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Each action line carries a frame reference. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
