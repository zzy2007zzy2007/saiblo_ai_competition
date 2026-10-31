#!/usr/bin/env bash
# 安慰剂注入 A/B（2026-10-28；报告 `docs/placebo_inject_ab_20261028.md`）
#
# 目的：把"注入"的**信息量**抽掉、只留"多一个候选"，用来分离 A/B 里 +31pp 的两种解释
#       （知识 vs 候选数/预算效应）。
#
#   P 臂（被测，默认）= BASE + AZAI_ORACLE_PLACEBO=1
#                       ⇒ 每回合本地随机抽一个**合法单 op**（hold_p=0，干预面最大 = 上界检验）
#   P2 臂            = BASE + AZAI_ORACLE_PLACEBO=1 + AZAI_ORACLE_PLACEBO_HOLD_P=0.78
#                       ⇒ 78% HOLD + 22% 随机合法 op（对齐"A 臂 78% 的决策是 HOLD"）
#   B 臂（基准）      = BASE（--mcts-env，不注入）
#
# 两臂其余配置**完全相同**（BASE 串逐字照抄 `_oracle_inject_ab.sh`，与 A/B 那次一字不差），
# `AZAI_FLAT_SEED=53` 固定；地图 seeds 3600-3607（8）× 2 座位 = 16 局，`--jobs 8`，`--paired-sides`。
# ⚠️ **绝不开** `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`（P1 雷区，且会污染主读数）。
# ⚠️ P 臂**不查 oracle** ⇒ 不受 `AZAI_ORACLE_TIMEOUT` 影响（A 臂那次是 15s）。
#
# 为什么 `MVS_STDERR_SUFFIX=1`：`--paired-sides` 让同一 seed 的两局**并发**，而 match_via_sdk 的
# stderr 文件名原来只含 (seat,label,seed) ⇒ 同一 seed 的两局会写同一个文件。这个 opt-in 把驱动
# 已按局设好的 `DSH_PROGRESS_SUFFIX`(`_p0`/`_p1`) 拼进文件名 ⇒ 每局一份 stderr，注入统计逐局可读。
#
# 用法：
#   bash code/run_logged.sh placebo_inject_ab  bash code/test_match/_placebo_inject_ab.sh
#   bash code/run_logged.sh placebo2_inject_ab bash code/test_match/_placebo_inject_ab.sh 0.78
#   （续跑：同一条命令原样再敲一遍 —— `--resume` 会跳过已 verdict=engine 的局）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export MVS_STDERR_SUFFIX=1

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
HOLD_P="${1:-0}"                       # 0 = P 臂；0.78 = P2 臂

# 配置基线（**逐字**照抄 `_oracle_inject_ab.sh:41` 的 BASE —— 与已完成的那次 A/B 同一个串）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
INJ="$BASE;AZAI_ORACLE_PLACEBO=1"
if [ "$HOLD_P" != "0" ]; then
    INJ="$INJ;AZAI_ORACLE_PLACEBO_HOLD_P=$HOLD_P"
fi
# out-dir 带 hold_p 标签（0 -> h0，0.78 -> h078）⇒ P 与 P2 的读数不会互相覆盖
if [ "$HOLD_P" = "0" ]; then
    TAG="h0"
else
    TAG="h$(printf '%s' "$HOLD_P" | tr -d '.')"
fi
OUT="match_results/placebo_inject_ab/ab16_${TAG}"

echo "[ab] 安慰剂臂（被测，flat 侧）= $INJ"
echo "[ab] 基准臂（mcts 侧）      = $BASE"
echo "[ab] hold_p=$HOLD_P  out-dir=$OUT  seeds=3600-3607 × 2 座位 = 16 局  jobs=8  $(date '+%Y-%m-%d %H:%M:%S')"

"$P" -u code/test_match/ab_search_modes.py \
    --paired-sides --jobs 8 \
    --seeds 3600 3601 3602 3603 3604 3605 3606 3607 \
    --out-dir "$OUT" \
    --flat-env="$INJ" \
    --mcts-env="$BASE" \
    --resume --retry-invalid 2 --game-timeout-s 5400
RC=$?

echo ""
echo "================ RESULT 行（每局一行）================ $(date '+%Y-%m-%d %H:%M:%S')"
grep -h "RESULT seed" "$OUT"/seed*.log 2>/dev/null | sort -t= -k2,2n || echo "  （没有 RESULT 行）"
echo ""
# stderr 落点是 `match_results/`（match_via_sdk 的 keep_dir 写死在那里）。
# 带 `MVS_STDERR_SUFFIX=1` 后名字 = `ai{座位}_<label>_seed{N}_p{flat座位}.stderr.log`，
# 且**被测臂永远满足 座位==flat_side**（文件名里两个数字相同）⇒ 被测臂日志 = `ai{s}_..._p{s}`。
mkdir -p "$OUT/stderr"
cp -f match_results/ai?_az_bridge_ai_seed36*_p?.stderr.log "$OUT/stderr/" 2>/dev/null
echo "================ 每局注入/闸门/非法读数（被测臂 stderr 摘要；已归档到 $OUT/stderr/）================"
for f in "$OUT"/stderr/ai0_*_p0.stderr.log "$OUT"/stderr/ai1_*_p1.stderr.log; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f") =="
    grep -h "注入统计\|安慰剂注入已启用\|退出报告\|illegal-summary" "$f" | tail -4
done

exit $RC
