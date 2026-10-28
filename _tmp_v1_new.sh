#!/usr/bin/env bash
# V1（最终代码版）：默认行为不变 —— 不加任何新参数，跑同一个 2 局小批次。
# 与 v1_old（HEAD 版驱动）的命令逐字相同，只有驱动文件/输出目录不同。
export PYTHONIOENCODING=utf-8
TINY='AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=16;AZAI_DEPTH=2;AZAI_K=8;AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off'
FLAT="$TINY;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000"
MCTS="$TINY;AZAI_SEARCH=mcts"
OUT=match_results/retry_resume_test/v1_new
echo "[v1] 目标目录（会先清空）: $(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
rm -rf "$OUT"
bash code/run_logged.sh v1_new_2games D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py --paired-sides --seeds 900 --jobs 1 --out-dir "$OUT" --flat-env="$FLAT" --mcts-env="$MCTS"
