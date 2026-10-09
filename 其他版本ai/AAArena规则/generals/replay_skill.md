# Generals replay facts

Generals replays are JSON objects written one per line. Non-JSON lines are skipped by the decoder.

## Observable actions

Each record contains `Round`, `Player`, and a variable-length integer `Action`. Codes 1–8 represent submitted operations; code 9 is the terminal record. Operation parameters preserve coordinates, directions, counts, identifiers, and selected types.

## Observable state

This replay format is an operation log and does not contain a complete state snapshot for every round. The narrator reports per-player operation counts and retains numeric operation parameters. It does not reconstruct unrecorded board values.

## Terminal facts

An `Action == [9]` record can supply the winning player and `Content` terminal text. The largest recorded `Round` is the decoded round count.

## Narration contract

The generated Markdown has `终局事实`, `状态变化与数值增量`, `可观测动作`, and `截断元数据` sections. Each action line includes round and player references. Full output reports disabled folding and an unbounded line limit; digest output reports source-event and rendered-line counts.
