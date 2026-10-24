#!/usr/bin/env bash
# 关键对照：同 k=48、同采样数、同成本，只差“选哪 48 个候选”
#   A = MCTS 先验加权采样（TCLASS=0.5 / TPOS=1.0 / TEMP=1e-6）
#   B = 近似均匀随机采样（TCLASS=1e5 / TPOS=1e5 / TEMP=1e5）⇒ 先验被抹平
# 判读：A 明显 > B ⇒ 先验在随机化 rollout（昂贵）配置下真的有用
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
C="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=17;AZAI_K=48"
echo "等待「k48 vs k160 对照」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 240); do
  marker=$(ls -d training_history/runs/*k48_vs_k160_control* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ A: k48+MCTS先验加权 vs B: k48+均匀随机（随机化 rollout） 32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 1990 2005 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_prior_vs_random \
  --flat-env="AZAI_CKPT=checkpoints/prior_distill_mcts16.pt;$C;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6" \
  --mcts-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$C;AZAI_TCLASS=100000;AZAI_TPOS=100000;AZAI_TEMP=100000"
echo ""
echo "先验vs随机 对照结束 ($(date +%H:%M:%S))"