# Multi-head Refactoring Spec

## Why
Replace hardcoded `--single-head` (bool: 1 or 3 heads) with `--num-heads N` (int: arbitrary head count). This allows flexible exploration of different head counts during ES training — more heads increase action diversity (as seen in gen_5's breakthrough at 90% vs ExampleAI), fewer heads reduce parameter count and training noise.

## What Changes
- **network.py**: `single_head: bool` → `num_heads: int`, replace `policy_head1/2/3` with `ModuleList`
- **decoder.py**: hardcoded `for i in (1, 2, 3)` → dynamic key detection
- **es_train.py**: `--single-head` → `--num-heads N`, internal rename
- **win_graph.py**: `single_head` → `num_heads` throughout
- **Test scripts** (6 files): rename parameter, pass `num_heads` to `create_model`
- **expand_to_3heads.py**: mark deprecated
- **Docs** (6 files): `--single-head` → `--num-heads N`
- **Backward compat**: state_dict key mapping for old checkpoints

## Impact
- Affected specs: train, eval, test, compare
- Affected code: 10 source files + 6 doc files
- **BREAKING**: Old checkpoints saved with `single_head` have different state_dict keys (`policy_head1.weight` → `policy_heads.0.weight`). Loader must auto-convert.
- **Not breaking**: `--num-heads 1` equivalent to old `--single-head`; `--num-heads 3` equivalent to old default (no flag).

## ADDED Requirements
### Requirement: Configurable head count
The system SHALL accept `--num-heads N` where N is any positive integer.

#### Scenario: Train with --num-heads 5
- **WHEN** user runs `python es_train.py --num-heads 5`
- **THEN** model creates 5 policy heads, decoder processes all 5, output log shows `num_heads=5`

### Requirement: Dynamic decoder
The decoder SHALL process any number of head logits from the network output.

#### Scenario: Variable head count
- **WHEN** network outputs 1, 3, or 5 head logits
- **THEN** decoder detects and decodes all of them without hardcoded ranges

## MODIFIED Requirements
### Requirement: Checkpoint backward compatibility
The system SHALL load old `single_head` checkpoints (both True and False variants) by mapping state_dict keys.

**Key mapping**:
- `policy_head1.weight` → `policy_heads.0.weight` (single_head=True or False)
- `policy_head2.weight` → `policy_heads.1.weight` (single_head=False only)
- `policy_head3.weight` → `policy_heads.2.weight` (single_head=False only)

### Requirement: Parameter serialization order preserved
The flattened parameter vector order for `num_heads=1` and `num_heads=3` SHALL match the old `single_head=True` and `single_head=False` order respectively.

## REMOVED Requirements
### Requirement: --single-head flag
**Reason**: Replaced by `--num-heads N`. `--num-heads 1` achieves the same effect.
**Migration**: Change `--single-head` to `--num-heads 1`.

### Requirement: expand_to_3heads.py
**Reason**: Training directly supports `--num-heads N`. No need for a post-training expansion script.
**Migration**: Train with `--num-heads 3` directly instead of training with 1 head and expanding.
