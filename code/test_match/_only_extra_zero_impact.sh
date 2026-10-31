#!/usr/bin/env bash
# 「`AZAI_ONLY_EXTRA` / `AZAI_ONLY_EXTRA_RANDOM` / `AZAI_ORACLE_INJECT_B` 不设时行为逐位不变」的证据
# （2026-10-28；报告 `docs/us_vs_champion_20261028.md` §5.4）
#
# 两层证据（分工不同，缺一不可）：
#  ① **搜索器逐位**（`_only_extra_zero_impact.py`）：在**真实局面**上把"改动前的
#     `flat_rollout_greedy.py` 副本"与"现行版本"**同进程**跑同一个 `search()`，
#     对 `last_scores`(float64 原始字节)/`visit_policy`/`chosen_bundle`/候选数/rollout 数打哈希，
#     逐字段比。**不动工作区**（不换文件 ⇒ 不会污染正在跑的实验）。附正向对照：开 only_extra
#     时候选数必须从 K 条塌成"额外候选条数"。
#  ② **对局级 `RESULT` 逐字段**（本脚本第二段）：同一 seed、同一 BASE 配置，
#     `before` = `_az_bridge_ai_preonlyextra.py`（加代码那一刻的工作区原样副本，md5 见下），
#     `after` = `az_bridge_ai.py`，**不设**新开关；self-play 两局各 120 回合，
#     用 `compare_result_lines.py` 逐字段比 `RESULT` + AI 侧 `illegal-summary`。
#
# ⚠️ 只在**没有别的对局批在跑**的时候运行第二段（它自己会占 4 个并发）；
#    第一段（探针）很轻，随时可跑。
#
# 用法：
#   bash code/run_logged.sh only_extra_zero_impact bash code/test_match/_only_extra_zero_impact.sh
#   只跑探针：  PROBE_ONLY=1 bash code/test_match/_only_extra_zero_impact.sh
#   只跑对局：  GAMES_ONLY=1 bash code/test_match/_only_extra_zero_impact.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
OUT="match_results/only_extra"
BEFORE="code/test_match/_az_bridge_ai_preonlyextra.py"
AFTER="code/test_match/az_bridge_ai.py"
BEFORE_MD5="DE98B720CA2586C7E5023E00E41B60EE"    # 加代码**之前**那一份工作区文件的 md5
SEEDS="${SEEDS:-7 7r}"
ROUNDS="${ROUNDS:-120}"
mkdir -p "$OUT"
RC_ALL=0

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
[ -n "$BASE" ] || { echo "[zi] !! 读不到 BASE"; exit 2; }

# `MVS_STDERR_SUFFIX=1`：这批**不是**正式实验，stderr 名加后缀 ⇒ 不会被别的批误读、也不覆盖它们
export MVS_STDERR_SUFFIX=1
export DSH_PROGRESS_SUFFIX="_ZI"

if [ "${GAMES_ONLY:-0}" != "1" ]; then
    echo "=============================================================="
    echo "[zi] ① 搜索器逐位（真实局面上的 pre vs live；不动工作区）"
    echo "=============================================================="
    "$P" -u code/test_match/_only_extra_zero_impact.py \
        --tag "${PROBE_TAG:-VS_RU_dump}" --seed "${PROBE_SEED:-8}" \
        --rounds ${PROBE_ROUNDS:-20 80} --extra "11,5,11;13,3,-1" --only-extra \
        --out "$OUT/zero_impact_probe.log" 2> "$OUT/zero_impact_probe.stderr.log"
    RC=$?
    echo "[zi] 探针退出码=$RC（0 = 逐字段比完；日志 $OUT/zero_impact_probe.log）"
    [ "$RC" -ne 0 ] && RC_ALL=1
    # 断言：日志里不能出现"不一致"；且正向对照必须成功
    if grep -q "!! 不一致" "$OUT/zero_impact_probe.log"; then
        echo "[zi] ❌ 探针发现 pre/live 逐字段不一致（见日志）"; RC_ALL=1
    else
        echo "[zi] ✓ 没有 '!! 不一致' 行 ⇒ 开关关时搜索器逐字段相同"
    fi
    if grep -q "开关确实改变了候选集" "$OUT/zero_impact_probe.log"; then
        echo "[zi] ✓ 正向对照通过 ⇒ only_extra 不是死代码（候选集确实塌成额外候选）"
    else
        echo "[zi] ❌ 正向对照失败（开关没接上）"; RC_ALL=1
    fi
fi

if [ "${PROBE_ONLY:-0}" = "1" ]; then
    echo "[zi] PROBE_ONLY=1 ⇒ 跳过对局段"
    exit "$RC_ALL"
fi

echo ""
echo "=============================================================="
echo "[zi] ② 对局级 RESULT 逐字段（before/after self-play，$ROUNDS 回合，seeds=$SEEDS）"
echo "=============================================================="
[ -f "$BEFORE" ] || { echo "[zi] !! 缺改动前副本 $BEFORE"; exit 2; }
MD5_NOW="$(md5sum "$BEFORE" | cut -d' ' -f1 | tr 'a-f' 'A-F')"
echo "[zi] 改动前副本 md5=$MD5_NOW（期望 $BEFORE_MD5）"
if [ "$MD5_NOW" != "$BEFORE_MD5" ]; then
    echo "[zi] !! 改动前副本被改过 ⇒ 比较无效，终止"; exit 3
fi
echo "[zi] before=$(wc -c < "$BEFORE") 字节  after=$(wc -c < "$AFTER") 字节"

PIDS=()
for SEED in $SEEDS; do
    for SIDE in before after; do
        AI="$BEFORE"; [ "$SIDE" = "after" ] && AI="$AFTER"
        "$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" --trace-ops \
            --ai0="$AI" --ai1="$AI" --ai0-env="$BASE" --ai1-env="$BASE" "$SEED" \
            > "$OUT/${SIDE}_seed${SEED}.log" 2>&1 &
        PIDS+=("$!:$SIDE:$SEED")
    done
done
echo "[zi] 已起 ${#PIDS[@]} 个对局进程：${PIDS[*]}"
for e in "${PIDS[@]}"; do
    pid="${e%%:*}"; rest="${e#*:}"
    wait "$pid"; rc=$?
    echo "[zi]   pid=$pid ($rest) rc=$rc  $(date '+%H:%M:%S')"
    [ "$rc" -ne 0 ] && RC_ALL=1
done

for SEED in $SEEDS; do
    echo ""
    echo "================ seed=$SEED 逐字段比较（compare_result_lines.py）================"
    "$P" -u code/test_match/compare_result_lines.py \
        "$OUT/before_seed${SEED}.log" "$OUT/after_seed${SEED}.log"
    RC=$?
    echo "[zi] seed=$SEED 比较退出码=$RC（0 = 逐字段不变）"
    [ "$RC" -ne 0 ] && RC_ALL=1
done
echo ""
echo "[zi] 总退出码=$RC_ALL（0 = 两层证据都通过 ⇒ 新开关默认关闭时行为逐位不变）"
exit "$RC_ALL"
