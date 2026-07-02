# Checklist

## Task 1: elite_bc.py
- [x] EliteSelector ABC defines `select(fitness) -> list[int]`
- [x] TopKSelector selects top-K by fitness
- [x] BCDataset loads .npz files and concatenates along time dimension
- [x] BCDataset filters all-HOLD turns with p_hold=0.1 downsampling
- [x] BCDataset reports HOLD filter statistics
- [x] supervised_update() runs Adam training on GPU with class_loss + map_loss
- [x] supervised_update() moves model back to CPU after training
- [x] supervised_update() returns dict with class_loss, map_loss, total_loss, samples
- [x] write_bc_npz() saves board(float16), stats(float16), class_, map_ as compressed .npz
- [x] collect_bc_data() glob-matches files by gen and selected indices
- [x] cleanup_gen_npz() deletes all files for a given gen
- [x] BCConfig stores all BC hyperparameters with defaults
- [x] BCConfig.from_args() constructs from argparse namespace
- [x] BCConfig.device falls back to CPU if CUDA unavailable

## Task 2: agent.py
- [x] NeuralAgent.__init__ initializes `self.last_output = None`
- [x] NeuralAgent._choose_operations stores `self.last_output = output` at end

## Task 3: es_train.py
- [x] Imports elite_bc components at file top
- [x] argparse adds all --bc* arguments
- [x] ESTrainer.__init__ accepts bc_config, creates self.bc_config, self.bc_dir
- [x] main() creates bc_data dir and initializes BC config when --bc is set
- [x] _eval_worker accepts bc_dir/gen/ind parameters
- [x] _eval_worker collects board/stats/class/map per turn when bc_dir is set
- [x] _eval_worker writes .npz file after game completes
- [x] step() constructs all_args with bc_dir/gen/ind for all individuals
- [x] step() skips ES gradient+momentum+mean update when bc_config.enabled
- [x] step() still computes and returns ind_details in BC mode
- [x] Main loop selects top-K, loads .npz files, runs BCDataset + supervised_update
- [x] Main loop calls cleanup_gen_npz after BC training
- [x] save_checkpoint persists bc_config dict when BC enabled
- [x] load_checkpoint restores bc_config from checkpoint
- [x] Without --bc flag, training behavior is completely unchanged
