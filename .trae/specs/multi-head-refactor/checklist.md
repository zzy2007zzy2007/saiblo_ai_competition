* [ ] `AntWarNetwork.__init__` accepts `num_heads: int = 3` instead of `single_head: bool`

* [ ] `create_model()` / `create_zero_model()` use `num_heads` parameter

* [ ] Model uses `ModuleList` for policy heads, all heads named `policy_heads.N`

* [ ] `forward()` outputs `head1_logits`...`headN_logits` dynamically

* [ ] `decoder.py` uses dynamic head key detection (no hardcoded range)

* [ ] `es_train.py` uses `--num-heads` (int, default 3) instead of `--single-head`

* [ ] `ESTrainer.__init__` accepts `num_heads=3`, passes to `create_model`

* [ ] Old checkpoint (`policy_head1.weight` keys) loads correctly via key remapping

* [ ] `win_graph.py` uses `num_heads` parameter consistently

* [ ] `eval_checkpoint.py` accepts `--num-heads` and passes to `create_model`

* [ ] `diagnose_model.py` accepts `--num-heads` and passes to `create_model`

* [ ] `compare_checkpoints.py` passes `num_heads` in worker

* [ ] `build_win_graph.py` passes `num_heads` in worker

* [ ] `benchmark_wg.py` and `benchmark_wg2.py` use updated WinGraph API

* [ ] `expand_to_3heads.py` has deprecation notice

* [ ] 6 doc files updated: `--single-head` → `--num-heads N`

* [ ] Benchmark confirms num\_heads=3 behavior matches old default

* [ ] `git commit` all changes

