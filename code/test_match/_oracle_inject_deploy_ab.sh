#!/usr/bin/env bash
# 任务 B：**规则内（≤10 s/决策）口径**下的亚军候选注入 A/B（2026-10-28）
# 报告：`docs/oracle_inject_deploy_ab_20261028.md`
#
# 设计（与任务书逐字一致）：
#   A = 被测臂 = **注入开**（`--flat-env`，`ab_search_modes` 里这一臂叫 "flat"，与本实验无关）
#   B = 基准臂 = **注入关**（`--mcts-env`）
#   两臂其余**完全相同**；`AZAI_FLAT_SEED=53` 固定；**绝不开** `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`。
#   规模：地图 seeds 3700-3707（8）× 2 座位 = 16 局，`--paired-sides --jobs 8`。
#
# 配置 = **部署口径 canonical**（= `ab_search_modes.DEFAULT_FLAT_ENV` 逐字 + FLAT_SEED=53）：
#   K=24 / MC_EVERY=5 / FLAT_ROLLOUTS=16 / FLAT_HORIZON=64 / FLAT_ROUNDS=39000。
#   它是"能参赛"的那一档：贵配置（K=48/ROLLOUTS=64/ROUNDS=49152）实测 dec_max 25-75 s，
#   根本不能参赛（游戏硬约束 10 s/决策），所以"增益还在不在"这个问题必须在**规则内**量。
#
# 用法：
#   bash code/run_logged.sh oracle_inject_deploy_ab bash code/test_match/_oracle_inject_deploy_ab.sh
#   （续跑：同一条命令原样再敲 —— `--resume` 跳过已 verdict=engine 的局）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export MVS_STDERR_SUFFIX=1     # 每局一份 stderr（配对两局并发，否则互相覆盖）

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"     # `D:/...` 形态；POSIX 路径会让注入**静默不启用**
OUT="match_results/oracle_inject_deploy/ab16"
TO="${ORACLE_TIMEOUT:-10}"                # = 新默认值（旧默认 2.5s 会中途丢注入）

for _bad in AZAI_NO_DOWNGRADE AZAI_BAN_OPS; do
    if [ -n "${!_bad:-}" ]; then
        echo "[ab] !! 拒绝运行：环境里有 $_bad=${!_bad}（本实验约定绝对不开）" >&2
        exit 2
    fi
done
if [ ! -f "$RU" ]; then echo "[ab] !! 亚军 exe 不存在: $RU" >&2; exit 2; fi
echo "[ab] 亚军 exe（Windows 绝对路径）= $RU"

CANON_BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;\
AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;\
AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;\
AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;\
AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_SEED=53"
CANON="$CANON_BASE"
[ -n "${CANON_EXTRA:-}" ] && CANON="$CANON;${CANON_EXTRA}"
BASE="$CANON"
INJ="$CANON;AZAI_ORACLE_INJECT=$RU;AZAI_ORACLE_TIMEOUT=$TO"

echo "[ab] A（注入开启）= $INJ"
echo "[ab] B（注入关闭）= $BASE"
echo "[ab] out-dir=$OUT  seeds=3700-3707 × 2 座位 = 16 局  jobs=8  $(date '+%Y-%m-%d %H:%M:%S')"

"$P" -u code/test_match/ab_search_modes.py \
    --paired-sides --jobs 8 \
    --seeds 3700 3701 3702 3703 3704 3705 3706 3707 \
    --out-dir "$OUT" \
    --flat-env="$INJ" \
    --mcts-env="$BASE" \
    --resume --retry-invalid 2 --game-timeout-s 5400
RC=$?

echo ""
echo "================ RESULT 行（每局一行）================ $(date '+%Y-%m-%d %H:%M:%S')"
grep -h "RESULT seed" "$OUT"/seed*.log 2>/dev/null | sed 's/^  //' | sort -t= -k2,2n || echo "  （没有 RESULT 行）"
echo ""
# stderr 落点固定是 `match_results/`；带 MVS_STDERR_SUFFIX=1 后名字 = `ai{座位}_..._p{座位}`
# ⇒ **A 臂永远满足 文件名里两个数字相同**（`ai0_*_p0` / `ai1_*_p1`），B 臂是另一个。
mkdir -p "$OUT/stderr"
cp -f match_results/ai?_az_bridge_ai_seed37*_p?.stderr.log "$OUT/stderr/" 2>/dev/null
echo "================ 每局 A 臂读数（注入 / 闸门 / 非法）—— 已归档到 $OUT/stderr/ ================"
for f in "$OUT"/stderr/ai0_*_p0.stderr.log "$OUT"/stderr/ai1_*_p1.stderr.log; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f") =="
    grep -h "注入统计\|停用注入\|退出报告\|illegal-summary\|耗时累计" "$f" | tail -3
done

exit $RC
