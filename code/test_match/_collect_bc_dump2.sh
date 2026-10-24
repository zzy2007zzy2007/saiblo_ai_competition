#!/usr/bin/env bash
# BC 数据第二批：再采 16 局（seeds 2130-2145）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
C="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_K=160;AZAI_FLAT_ROLLOUTS=1;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_AZ_DUMP=D:/2026智能体大赛_新2_distill/training_history/az_dump_bc;AZAI_AZ_DUMP_RAW=1"
echo "############ BC 数据第二批（16 局）  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2130 2145 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_bc_collect2 \
  --flat-env="$C" --mcts-env="$C"
echo ""
echo "第二批结束 ($(date +%H:%M:%S))"
ls -1 training_history/az_dump_bc/*.jsonl 2>/dev/null | wc -l