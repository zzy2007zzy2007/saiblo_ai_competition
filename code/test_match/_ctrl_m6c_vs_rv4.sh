#!/usr/bin/env bash
# 对照组：**M6c-ROOT 逐字配置 vs rule_v4**，验证这条臂是否复现已知的 0.7344。
# 若显著低于 0.7344 ⇒ 我的 M6c 臂被削弱了，前面的 86% 不成立。
set -euo pipefail
export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=5 AZAI_MC_HORIZON=256 AZAI_MC_ROOTONLY=1
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
exec D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_ladder.py \
  --tag=CTRL_m6croot_vs_rv4_64 --jobs=8 \
  --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py \
  7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r \
  18 18r 19 19r 20 20r 21 21r 22 22r 23 23r 24 24r 25 25r 26 26r 27 27r \
  28 28r 29 29r 30 30r 31 31r 32 32r 33 33r 34 34r 35 35r 36 36r 37 37r 38 38r
