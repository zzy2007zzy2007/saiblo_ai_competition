#!/usr/bin/env bash
# V4（补充，几秒钟）：--resume 与 --retry-invalid 同时开、且全部已完成 ⇒
#   期望「已完成 2 个、待跑 0 个」+「没有无效局了」+ 有效率 2/2，**一局都不重跑**。
export PYTHONIOENCODING=utf-8
TINY='AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=16;AZAI_DEPTH=2;AZAI_K=8;AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off'
FLAT="$TINY;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000"
MCTS="$TINY;AZAI_SEARCH=mcts"
bash code/run_logged.sh v4_resume_noop D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py --seeds 901 902 --jobs 1 --out-dir match_results/retry_resume_test/v3_resume --resume --retry-invalid 1 --game-timeout-s 600 --flat-env="$FLAT" --mcts-env="$MCTS"
