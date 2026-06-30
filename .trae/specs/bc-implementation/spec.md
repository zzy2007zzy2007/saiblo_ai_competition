# Behavior Cloning Implementation Spec

## Why
ES training on 550K parameters fails to produce a viable policy (BUILD→DOWNGRADE cycle, 2-4% win rate). BC provides supervised initialization from ExampleAI demonstrations before ES fine-tuning.

## What Changes
- `code/bc/collect_data.py` — Collect training data from ExampleAI self-play
- `code/bc/train_bc.py` — Supervised training of class prediction head
- `code/my_ai/es_train.py` — Add `--load-bc` flag for loading BC checkpoint

## Impact
- Affected specs: Training pipeline (ES → BC + ES)
- Affected code: `code/bc/`, `code/my_ai/es_train.py`

## ADDED Requirements
### Requirement: Data Collection
The system SHALL collect board+stats features and action class labels from ExampleAI self-play games.

#### Scenario: Collect N games
- **WHEN** `collect_data.py` runs with `--games N --workers W`
- **THEN** it saves an `.npz` file with board (N_samples, 28, 19, 19), stats (N_samples, 42), and class_label (N_samples,) arrays

### Requirement: Supervised Training
The system SHALL train the class prediction head using cross-entropy loss while freezing the encoder and action_map.

#### Scenario: Training loop
- **WHEN** `train_bc.py` runs with `--data <npz> --epochs E`
- **THEN** it outputs a BC checkpoint with trained policy_head weights, compatible with ES trainer format

### Requirement: ES Integration
The ES trainer SHALL support loading a BC checkpoint as initialization.

#### Scenario: Load BC weights
- **WHEN** `es_train.py` runs with `--load-bc <path>`
- **THEN** it initializes model parameters from the BC checkpoint and resets velocity to zero

## MODIFIED Requirements
None.

## REMOVED Requirements
None.
