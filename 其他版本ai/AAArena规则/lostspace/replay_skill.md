# LostSpace replay facts

LostSpace replay files are JSON arrays with birth positions first, nested round records in the middle, and a score dictionary last.

## Observable actions

Nested player records include `move`, `attack`, `detect`, `inspect`, `interact`, `getkey`, `keymachine`, `escape_capsule`, `died`, `escaped`, and public update events. Coordinates, targets, key identifiers, held tools, and HP values are retained when present.

## Observable state

Birth positions are recorded as coordinate triples. `hp_update` values are compared per player across the ordered event stream. Score dictionary values are ordered numerically and displayed with their player identifiers.

## Terminal facts

The final dictionary maps player identifiers to placement scores. The greatest score identifies the first-place player. Escape, death, and key events retain their source round.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Each action line includes round and player references. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
