# Tasks

- [ ] Task 1: Write `code/bc/collect_data.py`
  - Multi-process ExampleAI self-play, record board/stats/class_label
  - Save as `.npz` file with timestamp
  - Only record player-0 turns

- [ ] Task 2: Write `code/bc/train_bc.py`
  - Load `.npz` data, freeze encoder + action_map_conv
  - Train policy_base + policy_head1 with CrossEntropyLoss
  - Save BC checkpoint compatible with ES format
  - Accept `--freeze-encoder` flag (default on)

- [ ] Task 3: Add `--load-bc` to `es_train.py`
  - Load BC checkpoint, initialize model.mean from it
  - Reset velocity to zero

# Task Dependencies
- [Task 1] no dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
