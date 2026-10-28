#!/usr/bin/env bash
# 任务 A：**规则内时间预算探针**（2026-10-28；报告 docs/oracle_inject_deploy_ab_20261028.md）
#
# 目的：在与主实验（任务 B）**同一套 canonical 配置**下，量"注入开 / 注入关"两侧的
#       **决策墙钟**（`dec_n / p50 / p95 / max / dec_ge10 / dec_ge20 / dec_ge30`），
#       确认注入开启后每步决策仍 ≤ 10 s（游戏规则的每步上限）。
#
# 形态刻意照抄主实验的负载：8 局**并发**（4 个 seed × 2 种座位分配），
# 每局都是「A 臂 + B 臂」两个桥接进程 ⇒ 同时 8 个注入臂 + 8 个基准臂，与任务 B 的
# `--paired-sides --jobs 8` 一模一样（只是用 `--rounds` 把每局截短，10 分钟量级）。
#
# 用法：
#   bash code/run_logged.sh oracle_inject_deploy_probe bash code/test_match/_oracle_inject_deploy_probe.sh canon
#   # 若 dec_max 超 10s，用第 2 个参数覆盖单项重测（例：降 FLAT_ROUNDS）：
#   bash code/run_logged.sh oracle_inject_deploy_probe2 bash code/test_match/_oracle_inject_deploy_probe.sh canon30000 AZAI_FLAT_ROUNDS=30000
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
# `MVS_STDERR_SUFFIX=1` ⇒ stderr 文件名带上 `DSH_PROGRESS_SUFFIX`（本脚本按局设 `_ap0`/`_ap1`），
# 否则同一 seed 的两种座位分配会写同一个 stderr 文件（互相覆盖）。
export MVS_STDERR_SUFFIX=1

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
BR=code/test_match/az_bridge_ai.py
RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"     # 必须 `D:/...` 形态（POSIX 路径会让注入静默不启用）

TAG="${1:-canon}"
EXTRA="${2:-}"                            # 追加到 canonical 的覆盖项（用于"降档重测"）
ROUNDS="${PROBE_ROUNDS:-200}"             # 每局回合上限（探针只需要到中盘）
SEEDS="${PROBE_SEEDS:-3700 3701 3702 3703}"
TO="${PROBE_ORACLE_TIMEOUT:-10}"          # 与新的默认值一致；显式写出来便于逐字核对
OUT="match_results/oracle_inject_deploy/probe_${TAG}"

# ── 安全阀：这两个开关是 P1 的雷区（会污染主读数），探针一律不碰 ──────────────────
for _bad in AZAI_NO_DOWNGRADE AZAI_BAN_OPS; do
    if [ -n "${!_bad:-}" ]; then
        echo "[probe] !! 拒绝运行：环境里有 $_bad=${!_bad}（本实验约定绝对不开）" >&2
        exit 2
    fi
done
if [ ! -f "$RU" ]; then echo "[probe] !! 亚军 exe 不存在: $RU" >&2; exit 2; fi

# ── 部署口径 canonical（= `ab_search_modes.DEFAULT_FLAT_ENV` 逐字 + `AZAI_FLAT_SEED=53`）──
CANON_BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;\
AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;\
AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;\
AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;\
AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_SEED=53"
CANON="$CANON_BASE"
[ -n "$EXTRA" ] && CANON="$CANON;$EXTRA"
BASE="$CANON"                                                        # B 臂 = 注入关
INJ="$CANON;AZAI_ORACLE_INJECT=$RU;AZAI_ORACLE_TIMEOUT=$TO"           # A 臂 = 注入开

mkdir -p "$OUT"
echo "[probe] tag=$TAG rounds=$ROUNDS seeds=$SEEDS $(date '+%Y-%m-%d %H:%M:%S')"
echo "[probe] B 臂（注入关）= $BASE"
echo "[probe] A 臂（注入开）= $INJ"
echo "[probe] out=$OUT"

# 8 局并发：每个 seed 两种座位分配。A 臂**永远**是被观察的那一侧。
#   `_ap0` = A 坐 p0（ai0=A, ai1=B）；`_ap1` = A 坐 p1（ai0=B, ai1=A）。
for s in $SEEDS; do
    (
        DSH_PROGRESS_SUFFIX=_ap0 "$P" -u code/test_match/match_via_sdk.py \
            --rounds="$ROUNDS" --ai0="$BR" --ai1="$BR" \
            --ai0-env="$INJ" --ai1-env="$BASE" "$s" > "$OUT/seed${s}_ap0.out" 2>&1
        echo "$s ap0 rc=$?" >> "$OUT/rc.txt"
    ) &
    (
        DSH_PROGRESS_SUFFIX=_ap1 "$P" -u code/test_match/match_via_sdk.py \
            --rounds="$ROUNDS" --ai0="$BR" --ai1="$BR" \
            --ai0-env="$BASE" --ai1-env="$INJ" "$s" > "$OUT/seed${s}_ap1.out" 2>&1
        echo "$s ap1 rc=$?" >> "$OUT/rc.txt"
    ) &
done
wait

# ── stderr 归档（match_via_sdk 固定写到 match_results/）────────────────────────────
mkdir -p "$OUT/stderr"
for f in match_results/ai?_az_bridge_ai_seed*_ap?.stderr.log; do
    [ -f "$f" ] || continue
    case "$f" in *seed37*) cp -f "$f" "$OUT/stderr/" ;; esac
done
echo ""
echo "================ 各局胜负（截断局 = hp_tiebreak，只证明对局跑起来了）================"
grep -h "RESULT seed" "$OUT"/*.out 2>/dev/null | sed 's/^  //' || echo "  （无）"
echo ""
echo "================ 决策耗时（每个 stderr 的最后两条耗时行）================"
for f in "$OUT"/stderr/*.stderr.log; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f") =="
    grep -h "耗时累计\|决策耗时\|退出报告" "$f" | tail -2
done
echo ""
echo "================ 注入侧（A 臂）================"
for f in "$OUT"/stderr/*.stderr.log; do
    [ -f "$f" ] || continue
    grep -q "注入已启用" "$f" || continue
    echo "== $(basename "$f") =="
    grep -h "注入已启用\|注入累计" "$f" | tail -1
done
echo "[probe] 结束 $(date '+%Y-%m-%d %H:%M:%S')"
