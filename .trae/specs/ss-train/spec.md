# Strategy-Space Training Program (ss_train.py) Spec

## Why
ES gradient updates in 553K parameter space cause good strategies to collapse within 1-2 generations (gen_25→gen_26). Elite BC removed ES gradient but had no exploration mechanism. Strategy-space mutation adds controlled label noise during supervised training, enabling exploration entirely in behavior space.

## What Changes
- `code/my_ai/ss_train.py` — **NEW**: Complete training program, independent of es_train.py and elite_bc.py
  - Mirrored sampling (same as ES)
  - `_eval_worker` with full data collection (board, stats, class_, action_map, head_logits)
  - `SSDataset` with HOLD downsampling + class mutation (temperature sampling) + position mutation (Gaussian noise)
  - `ss_supervised_update` with CE(class) + KL(map) loss
  - Training loop with top-K selection + checkpoint + cleanup

## Impact
- Affected specs: Training pipeline (new strategy-space training option)
- Affected code: `code/my_ai/ss_train.py` (new), depends on `network.py`, `agent.py`, `decoder.py`
- `es_train.py` and `elite_bc.py` are NOT modified

## ADDED Requirements
### Requirement: SSTrainer
The system SHALL provide a training program `ss_train.py` that operates entirely in strategy space (parameter space only for sampling).

#### Scenario: Training loop
- **WHEN** `ss_train.py` runs with `--pop-size N --games G --workers W`
- **THEN** it samples N individuals via mirrored sampling, evaluates them, collects .npz data, selects top-K, and trains the mean model via supervised learning

### Requirement: Data Collection with head_logits
The system SHALL collect head_logits and full action_map in .npz files for strategy-space mutation.

#### Scenario: _eval_worker writes .npz
- **WHEN** `_eval_worker` completes a game
- **THEN** it writes a .npz file with board(float16), stats(float16), class_(int64), action_map(float16, 23×19×19), head_logits(float16, N_heads×24)

### Requirement: Strategy-space mutation in SSDataset
The system SHALL mutate class labels via temperature sampling and position maps via Gaussian noise during training.

#### Scenario: Class mutation
- **WHEN** `SSDataset.__getitem__` is called with `p_mutate > 0`
- **THEN** it randomly replaces class labels using softmax(saved_logits / temperature) with probability p_mutate

#### Scenario: Position mutation
- **WHEN** `SSDataset.__getitem__` is called with `pos_noise_std > 0`
- **THEN** it adds Gaussian noise N(0, pos_noise_std) to action_map with probability p_mutate

### Requirement: KL divergence for map loss
The system SHALL use KL divergence between predicted and target action_map distributions (not CE on argmax).

#### Scenario: Map loss computation
- **WHEN** `ss_supervised_update` computes map loss
- **THEN** it computes KL(log_softmax(pred_map), softmax(target_map).detach())

### Requirement: Checkpoint
The system SHALL save checkpoints every `--save-every` generations with mean, model_state, generation, and config.

## MODIFIED Requirements
None.

## REMOVED Requirements
None.
