#!/usr/bin/env bash
# 亚军候选注入 A/B（2026-10-28；报告 `docs/oracle_inject_ab_20261028.md`）
#
# 设计（严格遵守约定）：
#   A = 被测臂 = **注入开启**（`--flat-env`，ab_search_modes 里叫 "flat"）
#   B = 基准臂 = **注入关闭**（`--mcts-env`）
#   两臂其余配置**完全相同**（BASE 串逐字照抄任务书），`AZAI_FLAT_SEED=53` 固定同一种子。
#   ⚠️ **绝不开** `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`（那是 P1 的雷区，且会污染主读数）。
#   规模：地图 seeds 3600-3607（8）× 2 座位 = 16 局，`--jobs 8`，`--paired-sides`。
#
# 为什么 `MVS_STDERR_SUFFIX=1`：`--paired-sides` 让同一 seed 的两局**并发**，而 match_via_sdk 的
# stderr 文件名原来只含 (seat,label,seed) ⇒ 两局写同一个文件、互相覆盖（docs/oracle_inject_design.md
# §4.3）。这个 opt-in 把驱动已按局设好的 `DSH_PROGRESS_SUFFIX`(`_p0`/`_p1`) 拼进文件名
# ⇒ 每局一份 stderr，注入统计/非法自检**逐局可读**。不设它 ⇒ 文件名与改动前逐字相同。
#
# 用法：
#   bash code/run_logged.sh oracle_inject_ab bash code/test_match/_oracle_inject_ab.sh
#   （续跑：同一条命令原样再敲一遍 —— `--resume` 会跳过已 verdict=engine 的局）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export MVS_STDERR_SUFFIX=1

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
# ⚠️ 必须是**Windows 绝对路径**：`$REPO` 在 Git-bash 里是 `/d/...` 形态，而
# `az_bridge_ai._oracle_init` 只把"`is_absolute()` 为真"的路径当真（Windows 上 `/d/...`
# 不是绝对路径 ⇒ 被拼成 `<repo>/d/...`），Popen 直接 FileNotFoundError ⇒ **注入静默不启用**。
# `pwd -W` 给出 `D:/...` 形态（= 任务书要求的绝对路径）。
RU="$(pwd -W | sed 's#/$##')/$RU_REL"
OUT="match_results/oracle_inject_ab/ab16"

if [ ! -f "$RU" ]; then
    echo "[ab] !! 亚军 exe 不存在: $RU" >&2
    exit 2
fi
echo "[ab] 亚军 exe（Windows 绝对路径）= $RU"

# 配置基线（任务书逐字；两臂**只**差 `AZAI_ORACLE_INJECT`）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
# ⚠️ 唯一一处与"只有 AZAI_ORACLE_INJECT 不同"的字面偏离：`AZAI_ORACLE_TIMEOUT` 从默认 2.5s 放宽到 15s。
#    理由（**实测**，不是预防性猜测）：第一版按默认 2.5s 跑 8 并行时，8 个 A 臂里有 5 个在
#    round 55/123/150/187/207 触发 OracleTimeout（每次都是 2515ms **刚刚**超线，而 p95 只有 ~1.4s
#    ⇒ 是调度/内存抖动造成的尖峰，不是"亚军算得慢"），超时后注入**永久停用**（协议决定的，
#    读端错位不能重连）⇒ 后半局 A 臂退化成 B 臂，主读数被稀释。证据：
#    `training_history/p1_logs/ab16_attempt1_stderr/`。
#    这个阈值只影响"等亚军多久算失败"，不改搜索/网络/B 臂任何东西；放宽后仍能抓真死（15s 是 p95 的 10 倍）。
INJ="$BASE;AZAI_ORACLE_INJECT=$RU;AZAI_ORACLE_TIMEOUT=15"

echo "[ab] A（注入开启）= $INJ"
echo "[ab] B（注入关闭）= $BASE"
echo "[ab] out-dir=$OUT  seeds=3600-3607 × 2 座位 = 16 局  jobs=8  $(date '+%Y-%m-%d %H:%M:%S')"

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
# 带 `MVS_STDERR_SUFFIX=1` 后名字 = `ai{座位}_az_bridge_ai_seed{N}_p{座位}.stderr.log`，
# 且 **A 臂永远满足 座位==flat_side**（即文件名里两个数字相同）⇒ A 的日志 = `ai{s}_..._p{s}`。
mkdir -p "$OUT/stderr"
cp -f match_results/ai?_az_bridge_ai_seed36*_p?.stderr.log "$OUT/stderr/" 2>/dev/null
echo "================ 每局注入/闸门/非法读数（A 臂 stderr 摘要；已归档到 $OUT/stderr/）================"
for f in "$OUT"/stderr/ai0_*_p0.stderr.log "$OUT"/stderr/ai1_*_p1.stderr.log; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f") =="
    grep -h "注入统计\|停用注入\|退出报告\|illegal-summary" "$f" | tail -4
done

exit $RC
