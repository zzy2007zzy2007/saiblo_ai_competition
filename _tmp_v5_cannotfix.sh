#!/usr/bin/env bash
# V5（补充，~25s）：**重试救不回来的情况** —— 看门狗阈值故意设成 5s（远小于正常单局墙钟），
# 于是每局都被杀、重试 1 轮仍然被杀 ⇒ 期望：重跑清单 2 局、最终有效率 0/2、仍然无效清单。
# 同时这也是用**已提交的最终代码**再验一次"重跑过的局"那行的统计口径。
export PYTHONIOENCODING=utf-8
OUT=match_results/retry_resume_test/v5_retry_cannot_fix
echo "[v5] 目标目录（会先清空）: $(pwd)/$OUT"
rm -rf "$OUT"
TINY='AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=16;AZAI_DEPTH=2;AZAI_K=8;AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off'
FLAT="$TINY;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000"
MCTS="$TINY;AZAI_SEARCH=mcts"
bash code/run_logged.sh v5_retry_cannot_fix D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py --seeds 920 921 --jobs 1 --out-dir "$OUT" --game-timeout-s 5 --retry-invalid 1 --flat-env="$FLAT" --mcts-env="$MCTS"
