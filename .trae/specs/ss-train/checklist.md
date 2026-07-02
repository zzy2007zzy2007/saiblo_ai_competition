# Checklist

## SSDataset & Data
- [x] SSDataset loads .npz files with board, stats, class_, action_map, head_logits
- [x] SSDataset performs HOLD downsampling with p_hold=0.1 (all heads selecting class 23 kept with 10% probability)
- [x] SSDataset prints HOLD filter statistics
- [x] SSDataset mutates class labels using temperature sampling with p_mutate probability
- [x] SSDataset adds Gaussian noise to action_map with p_mutate probability
- [x] write_npz saves board(float16), stats(float16), class_(int64), action_map(float16), head_logits(float16) as compressed .npz
- [x] collect_npz glob-matches files by gen and selected indices
- [x] cleanup_gen_npz deletes all .npz for a given gen

## Training
- [x] ss_supervised_update runs Adam on GPU with CE(class) + KL(action_map) loss
- [x] KL loss uses log_softmax(pred) vs softmax(target).detach()
- [x] Model moved to GPU for training and back to CPU after training
- [x] Returns dict with class_loss, map_loss, total_loss, samples

## _eval_worker
- [x] _eval_worker accepts params, opp_params, seed, num_heads, bc_dir, gen, ind
- [x] _eval_worker collects board, stats, class labels, action_map, head_logits per turn
- [x] _eval_worker writes .npz after game completes
- [x] head_logits saved as (T, N_heads, 24) float16
- [x] action_map saved as (T, 24, 19, 19) float16

## Training Loop
- [x] Mirrored sampling produces (mean + sigma*noise) and (mean - sigma*noise) pairs
- [x] Opponents selected randomly from current population (each pair plays twice, first/second player)
- [x] Top-K selection by fitness
- [x] SSDataset + ss_supervised_update execute after evaluation
- [x] mean updated from model.get_parameters_as_vector()
- [x] Checkpoint saved every --save-every generations
- [x] Checkpoint contains mean, model_state, generation, config
- [x] Interrupt checkpoint saved on Ctrl+C
- [x] cleanup_gen_npz runs after training each generation
- [x] All CLI arguments accepted and used correctly
