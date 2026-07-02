# Elite BC (Behavior Cloning from Top-K Individuals) Spec

## Why
ES gradient updates in 553K parameter space are fragile — good strategies (gen_5's BUILD/DOWNGRADE/lightning balance) collapse in 1-2 generations. Elite BC replaces ES gradient with supervised learning on top-K individuals' game data, learning from behavior space rather than parameter space.

## What Changes
- `code/my_ai/elite_bc.py` — **NEW**: EliteSelector, BCDataset, supervised_update, BC data I/O, BCConfig
- `code/my_ai/es_train.py` — Add `--bc` flag, modify `_eval_worker` for data collection, skip ES gradient in BC mode, add BC training loop
- `code/my_ai/agent.py` — Add `self.last_output` field for collecting network output during eval

## Impact
- Affected specs: Training pipeline (ES → CEM-style with BC)
- Affected code: `code/my_ai/elite_bc.py` (new), `code/my_ai/es_train.py`, `code/my_ai/agent.py`

## ADDED Requirements
### Requirement: Data Collection in eval worker
The system SHALL collect board, stats, class argmax, and map argmax from all individuals during evaluation when BC mode is enabled.

#### Scenario: Write game data to .npz
- **WHEN** `_eval_worker` runs with `bc_dir` set
- **THEN** it collects per-turn (board, stats, class labels, map labels) and saves as `gen_NNNN_ind{idx}_seed{seed}.npz` in bc_dir

### Requirement: Skip ES gradient in BC mode
The system SHALL skip ES gradient+momentum+mean update when BC mode is enabled.

#### Scenario: step() returns without updating mean
- **WHEN** `step()` is called and `self.bc_config.enabled` is True
- **THEN** it skips all gradient computation and mean update, returns fitness/ind_details as normal

### Requirement: Supervised training from top-K data
The system SHALL load top-K individuals' .npz files and train the mean model using supervised learning.

#### Scenario: BC training loop
- **WHEN** main loop detects `trainer.bc_config.enabled`
- **THEN** it selects top-K by fitness, reads their .npz files, builds BCDataset (with HOLD downsampling at p_hold=0.1), runs `supervised_update()` on GPU, updates `trainer.mean`

### Requirement: HOLD downsampling
The system SHALL downsample all-HOLD turns in BCDataset to prevent HOLD from dominating the loss.

#### Scenario: HOLD filter activates
- **WHEN** BCDataset loads data
- **THEN** it computes `(class_label == 23).all(dim=1)` and keeps each HOLD turn with `p_hold=0.1` probability, removing the rest

### Requirement: Cleanup after training
The system SHALL delete all .npz files for the current generation after BC training completes.

#### Scenario: cleanup_gen_npz
- **WHEN** BC training finishes for generation N
- **THEN** `cleanup_gen_npz(bc_dir, N)` deletes all `gen_NNNN_*.npz` files

## MODIFIED Requirements
### Requirement: NeuralAgent._choose_operations
The system SHALL save the raw network output to `self.last_output` after each forward pass.

#### Scenario: Store last_output
- **WHEN** `_choose_operations` returns
- **THEN** `self.last_output` contains the full network output dict

## REMOVED Requirements
None.
