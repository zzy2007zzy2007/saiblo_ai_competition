#!/usr/bin/env bash
# 内部对打：**有显式升级**（A） vs **无升级**（B），其余逐字相同。
#
# 背景（2026-10-22）：
#   · `cells4 + MC_UPG=1` 打亚军仍是"基地被打到 0、对手剩 24-37 血"，和 1W-31L 形态几乎一样；
#   · 但机制检查显示升级已达标（我们 10.3% vs 亚军 8.6%）。
#   ⇒ 说明"升级"不是那 30 分差距的原因。但那只是**对亚军的观察**，分辨率低。
#     要回答"升级到底值多少分"，必须做**内部对打**（同代码、配对种子、血量连续可分）。
#
# 单变量：A 比 B 只多 `AZAI_MC_UPG=1`（其余 15 个 env 逐字相同）。
# 附带验证：A 侧 upgrade 应 ≈10%、B 侧应 =0%（否则说明开关没生效）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 500 563 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_UPG="$BASE;AZAI_MC_UPG=1"
B_NOUPG="$BASE"

echo "############ 有升级(A) vs 无升级(B) 64 局  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_upg1_vs_none \
    --flat-env="$A_UPG" --mcts-env="$B_NOUPG"
echo ""
echo "升级对照结束 ($(date +%H:%M:%S))"
