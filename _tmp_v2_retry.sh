#!/usr/bin/env bash
# V2：看门狗 + 自动重试。
#   故障注入：让**第一局**的 flat AI 卡住 600s（一次性，靠 mkdir 原子抢占）。
#   看门狗 150s ⇒ 该局被杀（只杀它的 PID 树），另一局正常跑完（~60-90s）。
#   跑完自动重试那 1 局（此时注入已被用掉 ⇒ 正常跑）⇒ 最终有效率 2/2。
export PYTHONIOENCODING=utf-8
OUT=match_results/retry_resume_test/v2_retry
rm -rf _tmp_fi_claim
mkdir -p _tmp_fi_claim
echo "[v2] 目标目录（会先清空）: $(pwd)/$OUT"
rm -rf "$OUT"
TINY='AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=16;AZAI_DEPTH=2;AZAI_K=8;AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off'
FAULT="PYTHONPATH=D:/2026智能体大赛_新2_distill/_tmp_faultinject;FAULTINJECT_CLAIM_DIR=D:/2026智能体大赛_新2_distill/_tmp_fi_claim;FAULTINJECT_SLEEP_S=600"
FLAT="$TINY;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000;$FAULT"
MCTS="$TINY;AZAI_SEARCH=mcts"
bash code/run_logged.sh v2_retry_watchdog D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py --seeds 910 911 --jobs 1 --out-dir "$OUT" --game-timeout-s 150 --retry-invalid 2 --flat-env="$FLAT" --mcts-env="$MCTS"
