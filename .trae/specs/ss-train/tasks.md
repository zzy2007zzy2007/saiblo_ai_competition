# Tasks

- [x] Task 1: Create `code/my_ai/ss_train.py` — full implementation
  - [x] Argument parser with all --bc-* and --ss-* params
  - [x] `_eval_worker` function with full .npz data collection
  - [x] `SSDataset` with HOLD downsampling + class mutation + position mutation
  - [x] `ss_supervised_update` with CE(class) + KL(action_map) loss
  - [x] Training loop: mirrored sampling → evaluate → top-K → train → update → checkpoint → cleanup
  - [x] Data I/O: write_npz, collect_npz, cleanup_gen_npz
  - [x] save_checkpoint / load_checkpoint
  - [x] Signal handler for Ctrl+C

# Task Dependencies
- No external dependencies
