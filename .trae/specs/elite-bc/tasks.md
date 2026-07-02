# Tasks

- [ ] Task 1: Create `code/my_ai/elite_bc.py` with all components
  - EliteSelector ABC + TopKSelector
  - BCDataset with HOLD downsampling (p_hold=0.1)
  - supervised_update() function
  - write_bc_npz, collect_bc_data, cleanup_gen_npz
  - BCConfig class

- [ ] Task 2: Modify `code/my_ai/agent.py` — add `self.last_output`
  - NeuralAgent.__init__: add `self.last_output = None`
  - NeuralAgent._choose_operations: add `self.last_output = output` at end

- [ ] Task 3: Modify `code/my_ai/es_train.py` — all changes
  - Add import for elite_bc components
  - Add `--bc*` argparse arguments
  - ESTrainer.__init__: add `self.bc_config`, `self.bc_dir`
  - main(): initialize BC config/bc_dir when `--bc` is set
  - _eval_worker: add bc_dir/gen/ind params + data collection code
  - step(): add all_args construction with bc_dir/gen/ind
  - step(): skip ES gradient when BC enabled
  - Main loop: add BC training step after step()
  - save_checkpoint / load_checkpoint: persist BC config

# Task Dependencies
- [Task 1] no dependencies
- [Task 2] no dependencies
- [Task 3] depends on [Task 1], [Task 2]
