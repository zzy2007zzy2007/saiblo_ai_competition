# Tasks

## Phase 1: Network definition changes

- [ ] Task 1: Refactor `network.py` — Replace `single_head: bool` with `num_heads: int` and `ModuleList`
  - Steps:
    1. Change `AntWarNetwork.__init__(self, ..., single_head=False)` to `__init__(self, ..., num_heads=3)`
    2. Replace `self.policy_head1/2/3` with `self.policy_heads = nn.ModuleList([nn.Linear(...) for _ in range(num_heads)])`
    3. Update `forward()` to loop over `self.policy_heads` and output `head{i+1}_logits`
    4. Update `create_model()` and `create_zero_model()` signatures
    5. Update docstrings
  - Files: `code/my_ai/network.py`
  - Dependencies: none

## Phase 2: Decoder changes

- [ ] Task 2: Fix `decoder.py` — Dynamic head key detection
  - Steps:
    1. Replace hardcoded `for i in (1, 2, 3)` with `sorted(k for k in network_output if k.startswith("head") and k.endswith("_logits"))`
    2. Update docstring
  - Files: `code/my_ai/decoder.py`
  - Dependencies: none

## Phase 3: Training program changes

- [ ] Task 3: Update `es_train.py` — Parameter and init changes
  - Steps:
    1. Replace `--single-head` arg with `--num-heads type=int default=3`
    2. Change `ESTrainer.__init__` parameter from `single_head=True` to `num_heads=3`
    3. Update `create_model(...)` calls to use `num_heads`
    4. Update log output
    5. Add backward compat in `load_checkpoint` for old state_dict keys
  - Files: `code/my_ai/es_train.py`
  - Dependencies: Task 1

## Phase 4: WinGraph changes

- [ ] Task 4: Update `win_graph.py` — parameter rename
  - Steps:
    1. Rename `single_head` to `num_heads` in `__init__`, properties, `_game_worker`, `add_node`, `state_dict`/`load_state_dict`
    2. Change default from `True` to `3`
  - Files: `code/my_ai/win_graph.py`
  - Dependencies: Task 1, Task 3

## Phase 5: Test script changes (batch)

- [ ] Task 5: Update `eval_checkpoint.py` — parameter rename
  - Steps:
    1. Replace `--single-head` arg with `--num-heads type=int default=3`
    2. Update `create_model(...)` calls in main and worker
  - Files: `code/test_match/eval_checkpoint.py`
  - Dependencies: Task 1

- [ ] Task 6: Update `diagnose_model.py` — parameter rename
  - Steps:
    1. Same pattern as eval_checkpoint.py
  - Files: `code/test_match/diagnose_model.py`
  - Dependencies: Task 1

- [ ] Task 7: Update `compare_checkpoints.py` — worker create_model call
  - Steps:
    1. Rename `single_head` to `num_heads` in worker function
  - Files: `code/test_match/compare_checkpoints.py`
  - Dependencies: Task 1

- [ ] Task 8: Update `build_win_graph.py` — worker create_model call
  - Steps:
    1. Rename `single_head` to `num_heads` in worker function
  - Files: `code/test_match/build_win_graph.py`
  - Dependencies: Task 1

- [ ] Task 9: Update benchmark scripts (2 files) — parameter rename
  - Steps:
    1. Rename `single_head` references in `benchmark_wg.py` and `benchmark_wg2.py`
  - Files: `code/test_match/benchmark_wg.py`, `code/test_match/benchmark_wg2.py`
  - Dependencies: Task 4

## Phase 6: Deprecation

- [ ] Task 10: Mark `expand_to_3heads.py` as deprecated
  - Steps:
    1. Add deprecation warning at top of script: "This script is deprecated. Use --num-heads 3 directly during training."
  - Files: `code/my_ai/expand_to_3heads.py`
  - Dependencies: none

## Phase 7: Documentation updates

- [ ] Task 11: Update doc files (6 files) — `--single-head` → `--num-heads N`
  - Steps:
    1. Search-and-replace in `model_design.md`, `eval_method.md`, `training_program.md`, `讨论记录.md`, `eval_results.md`, `win_graph_design.md`
  - Files: docs/*.md
  - Dependencies: none

## Phase 8: Verification

- [ ] Task 12: Verify backward compatibility
  - Steps:
    1. Load old checkpoint (single_head=True) with new code
    2. Load old checkpoint (single_head=False) with new code
    3. Verify parameter counts match
    4. Verify evaluation results match
  - Dependencies: Task 1-5

- [ ] Task 13: Run benchmark to confirm num_heads=3 behavior unchanged
  - Steps:
    1. Run `benchmark_wg2.py` with `num_heads=3` (equivalent to old default)
    2. Confirm timing matches previous results
  - Dependencies: Task 1-9

# Task Dependencies

- Task 1, 2: independent (can run in parallel)
- Task 4 depends on Task 1, 3
- Task 5-8 depend on Task 1
- Task 12 depends on Task 1-5
