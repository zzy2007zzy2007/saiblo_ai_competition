# Miracle replay facts

Miracle replays are big-endian signed int32 streams. The first seven integers form a header; every following group of seven integers is one media record.

## Observable events

A media record contains round, event code, and five raw payload integers. Known codes name turn, spawn, movement, attack, damage, death, heal, artifact, summon, and terminal events. Raw non-zero payload values remain visible in the narration.

## Observable state

The header supplies map type and day/night value. The event stream supplies round transitions and numeric payloads. The file does not contain a complete public board snapshot, so no absent state is reconstructed.

## Terminal facts

Event code 10 is `GameEnd`; its first payload integer is the recorded winner. The greatest non-negative event round is the decoded round count. Files with fewer than half of their event codes in the public enum are reported as unreadable.

## Narration contract

For readable files, generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Each event line includes a round reference. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
