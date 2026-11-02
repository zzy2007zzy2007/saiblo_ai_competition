#!/usr/bin/env bash
# hinge90 分段版对局扫描的**队列启动器**（排在线性扫描后面，WMI detached 起）。
#
# 流程：
#   1. 等线性扫描的 48 局 engine 都出（match_results/leaf_coin_alpha/a*/seed*_p*.log）；
#   2. 跑 hinge90 扫描（ALPHAS="0.05 0.1 0.3 1.0"、MODE=hinge90、8 局/α vs pin，JOBS=8）；
#   3. 等 hinge90 的 32 局 engine 都出；
#   4. 跑 hinge90 分析 → match_results/leaf_coin_hinge90/_analysis.txt + summary.json。
#
# 用法（WMI detached，见 docs/lightning_rollout_mode_20261102.md 末尾注释口径）：
#   用 Invoke-CimMethod Win32_Process Create 起：bash -c "bash code/test_match/_hinge90_queue.sh"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
LOG="match_results/leaf_coin_alpha/_hinge90_queue.log"

echo "[hinge90q] $(date '+%F %T') 启动：等线性 48 局 engine" | tee -a "$LOG"

# 1) 等线性 48 局 engine（最多 4 小时）
N=0
for _i in $(seq 1 240); do
    N=$(grep -l "verdict=engine" match_results/leaf_coin_alpha/a*/seed*_p*.log 2>/dev/null | wc -l | tr -d ' ')
    [ "${N:-0}" -ge 48 ] && break
    sleep 60
done
echo "[hinge90q] $(date '+%F %T') 线性 engine=$N/48，起 hinge90 扫描" | tee -a "$LOG"

# 2) 跑 hinge90 扫描（8 局/α × 4 α，vs pin）
export ALPHAS="0.05 0.1 0.3 1.0"
export MODE="hinge90"
bash code/run_logged.sh leaf_coin_hinge90_scan bash code/test_match/_leaf_coin_alpha_scan.sh >> "$LOG" 2>&1

# 3) 等 hinge90 32 局 engine
H=0
for _i in $(seq 1 240); do
    H=$(grep -l "verdict=engine" match_results/leaf_coin_hinge90/a*/seed*_p*.log 2>/dev/null | wc -l | tr -d ' ')
    [ "${H:-0}" -ge 32 ] && break
    sleep 60
done
echo "[hinge90q] $(date '+%F %T') hinge90 engine=$H/32，跑分析" | tee -a "$LOG"

# 4) 分析
"$P" -u code/test_match/analyze_leaf_coin_alpha.py \
    --alphas 0.05,0.1,0.3,1.0 --seeds 100-103 --mode hinge90 \
    --json match_results/leaf_coin_hinge90/summary.json \
    > match_results/leaf_coin_hinge90/_analysis.txt 2>&1
echo "[hinge90q] $(date '+%F %T') 分析完成 → match_results/leaf_coin_hinge90/_analysis.txt" | tee -a "$LOG"
echo "[hinge90q] ALL_DONE" | tee -a "$LOG"
