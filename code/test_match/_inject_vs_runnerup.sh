#!/usr/bin/env bash
# 外部标尺：**我们（注入开 / 注入关）直接对打亚军本体**（2026-10-28；报告 docs/inject_vs_runnerup_20261028.md）
#
# 设计（跑之前登记，见报告 §2）：
#   A 臂 = BASE + AZAI_ORACLE_INJECT=<亚军 exe 的 D:/ 绝对路径> + AZAI_ORACLE_TIMEOUT=15
#   B 臂 = BASE
#   两臂其余配置**逐字相同**（BASE 与自对弈注入 A/B `_oracle_inject_ab.sh` 逐字相同 ⇒ 与 81.2% 那批可比配置）。
#   对手 = 亚军本体（ai_cpp_lure_v4.exe），两臂同一批 seed、同一座位镜像。
#   每臂 16 局 = seeds 7..14 × 2 座位（`7` = 我们执 P0；`7r` = 镜像，我们执 P1）。
#
# 为什么用 `_tmp_ladder.py`：它就是历史所有"我们 vs 亚军"批次用的**同一个 harness**
#   （`_vs_runnerup.sh` / `_vs_ru_newcfg.sh` / `_vs_ru_k48rand.sh`），日志格式、token 语义、
#   stderr 归档、单局 `TIMEOUT=7200s` 全部沿用 ⇒ 与历史读数同源。这里只补两件它没有的东西：
#     ① `--resume` 语义（跳过已有 `verdict=engine` 的 token，只补跑无效局，最多 RETRY 轮）；
#     ② 每局 stderr 挪到一边再开跑（否则上一批同 seed 的日志会被误读成本批的）。
#
# ⚠️ **禁止**按镜像名/通配符批量 kill（用户明确要求）。本脚本自己不 kill 任何进程；
#    要中止就 Ctrl-C 或用 PID 精确处理。
# ⚠️ **不开** AZAI_NO_DOWNGRADE / AZAI_BAN_OPS（与注入组合的一类非法 op 源；已有整块闸门但没必要冒险）。
#
# 用法：
#   bash code/run_logged.sh inject_vs_ru bash code/test_match/_inject_vs_runnerup.sh
#   续跑：同一条命令原样再敲（已 verdict=engine 的局会跳过）
#   只跑一个臂：ARMS="A" bash code/test_match/_inject_vs_runnerup.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-8}"
# 重试轮数：一批跑完后凡是没有 verdict=engine 的局再跑一遍，最多这么多轮
RETRY="${RETRY:-2}"
ARMS="${ARMS:-A B}"
# 2026-10-28：**C 臂（升级候选诊断臂）** = BASE + `AZAI_UPG_BUNDLES=$UPG_N`（只加升级候选，
#   不加注入）。见 docs/upgrade_bundles_20261028.md。N=0 ⇒ 等价于 B 臂（开关关 = 逐位不变）。
UPG_N="${UPG_N:-4}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"

RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
# ⚠️ 必须是 **Windows 绝对路径（D:/…）**：Git-bash 的 `/d/...` 在 Windows 上
#    `Path.is_absolute()` 判 False ⇒ `_oracle_init` 把它拼成 `<repo>/d/...` ⇒ Popen 抛
#    FileNotFoundError ⇒ **注入整局静默不启用，而 A/B 看起来照跑**（踩过一次，
#    见 docs/oracle_inject_ab_20261028.md §4.1.1 第 2 条）。`pwd -W` 给的就是 D:/ 形态。
RU="$(pwd -W | sed 's#/$##')/$RU_REL"

# 配置基线（任务书逐字；与自对弈注入 A/B 的 BASE **逐字相同**）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
INJ_EXTRA=";AZAI_ORACLE_INJECT=$RU;AZAI_ORACLE_TIMEOUT=15"

OUTROOT="match_results/inject_vs_ru"

echo "=============================================================="
echo "[ivr] 亚军 exe = $RU"
echo "[ivr] 臂 = $ARMS   seeds = $SEEDS   jobs = $JOBS   retry = $RETRY"
echo "[ivr] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ── 前置断言（三条，任一失败立刻退出，别白跑几小时）─────────────────────────
case "$RU" in
    D:/*) echo "[ivr] ✓ 亚军 exe 路径是 D:/ 形态（Windows 绝对路径）" ;;
    *)   echo "[ivr] !! 亚军 exe 不是 D:/ 形态 ⇒ 注入会静默失效，终止: $RU" >&2; exit 2 ;;
esac
[ -f "$RU" ] || { echo "[ivr] !! 亚军 exe 不存在: $RU" >&2; exit 2; }
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
    [ -f "$(dirname "$RU")/$d" ] || { echo "[ivr] !! 缺 $d（亚军会 0xC0000139 直接死）" >&2; exit 2; }
done
echo "[ivr] ✓ 亚军 exe + 3 个 DLL 都在"

# BASE 与自对弈注入 A/B（81.2% 那批）必须逐字相同 —— 否则"配置可比"这句话就不成立
BASE_REF="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_oracle_inject_ab.sh)"
if [ "$BASE" = "$BASE_REF" ]; then
    echo "[ivr] ✓ BASE 与 _oracle_inject_ab.sh 逐字相同（${#BASE} 字符）"
else
    echo "[ivr] !! BASE 与自对弈 A/B 不同 ⇒ 停（先对齐配置）" >&2
    echo "  本次: ${#BASE} 字符" >&2
    echo "  参考: ${#BASE_REF} 字符" >&2
    exit 3
fi

# 该 arm 里还没有 verdict=engine 的 token（= 需要跑的）
remaining_tokens() {
    local tag="$1" tok base seat
    for tok in $SEEDS; do
        if ! grep -q "verdict=engine" "match_results/ladder_logs/$tag/$tok.log" 2>/dev/null; then
            printf '%s ' "$tok"
        fi
    done
}

# 把上一批同 seed 的 stderr 挪到一边（只挪不删 ⇒ 历史留存；避免"读错批"）
stash_stderr() {
    local why="$1" dest f
    dest="$OUTROOT/_stale/$why"
    mkdir -p "$dest"
    for f in match_results/ai?_az_bridge_ai_seed{7,8,9,10,11,12,13,14}.stderr.log \
             match_results/ai?_runnerup_seed{7,8,9,10,11,12,13,14}.stderr.log \
             match_results/ai?_ai_cpp_lure_v4_seed{7,8,9,10,11,12,13,14}.stderr.log; do
        [ -f "$f" ] || continue
        mv -f "$f" "$dest/" 2>/dev/null && echo "[ivr]   旧 stderr 挪走: $(basename "$f") -> $dest/"
    done
}

# 跑完一个 arm：归档 stderr + 逐局读数（A 臂额外断言注入有效性）
finish_arm() {
    local arm="$1" tag="$2" tok base seat f dir ok
    dir="$OUTROOT/$arm/stderr"
    mkdir -p "$dir"
    for f in match_results/ai?_az_bridge_ai_seed{7,8,9,10,11,12,13,14}.stderr.log \
             match_results/ai?_runnerup_seed{7,8,9,10,11,12,13,14}.stderr.log; do
        [ -f "$f" ] || continue
        cp -f "$f" "$dir/" 2>/dev/null
    done
    echo ""
    echo "-------- [$arm] 逐局读数（token  us 座位  verdict  回合  asks  timeout  dead  注入已启用）--------"
    for tok in $SEEDS; do
        base="${tok%r}"
        if [ "$tok" = "$base" ]; then seat=0; else seat=1; fi
        f="$dir/ai${seat}_az_bridge_ai_seed${base}.stderr.log"
        local verdict rounds asks tmo dead en
        verdict="$(grep -ho 'verdict=[a-z_]*' "match_results/ladder_logs/$tag/$tok.log" 2>/dev/null | tail -1 | cut -d= -f2)"
        rounds="$(grep -ho 'rounds=[0-9]*' "match_results/ladder_logs/$tag/$tok.log" 2>/dev/null | tail -1 | cut -d= -f2)"
        if [ -f "$f" ]; then
            # ⚠️ 前置空格：统计行里还有 `hit_rate_asks=84.0%`，不锚定会误抓到它
            asks="$(grep -o ' asks=[0-9]*' "$f" | tail -1 | tr -d ' ' | cut -d= -f2)"
            tmo="$(grep -o 'timeout=[0-9]*' "$f" | tail -1 | cut -d= -f2)"
            dead="$(grep -o 'dead=[0-9]*' "$f" | tail -1 | cut -d= -f2)"
            en="$(grep -c '注入已启用' "$f")"
        else
            asks="-"; tmo="-"; dead="-"; en="-"
        fi
        printf '  %-4s %-4s %-10s %-6s %-6s %-9s %-6s %s\n' \
            "$tok" "P$seat" "${verdict:-无}" "${rounds:-?}" "${asks:-?}" "${tmo:-?}" "${dead:-?}" "${en:-?}"
    done
    echo "-------- [$arm] RESULT 行 --------"
    for tok in $SEEDS; do
        grep -h "RESULT seed" "match_results/ladder_logs/$tag/$tok.log" 2>/dev/null | sed "s/^/  [$tok] /"
    done
}

for arm in $ARMS; do
    tag="INJECT_VS_RU_$arm"
    out="match_results/ladder_logs/$tag"
    mkdir -p "$out"
    if [ "$arm" = "A" ]; then
        export_env="$BASE$INJ_EXTRA"
        echo ""
        echo "############ A 臂（注入开启）vs 亚军  $(date '+%H:%M:%S')"
        echo "[ivr] A env = $export_env"
    elif [ "$arm" = "C" ]; then
        # 2026-10-28：诊断臂 C —— 只多一个开关：`AZAI_UPG_BUNDLES=N`（升级进候选，由评估器裁决）
        export_env="$BASE;AZAI_UPG_BUNDLES=$UPG_N"
        echo ""
        echo "############ C 臂（升级候选 $UPG_N 条，不注入）vs 亚军  $(date '+%H:%M:%S')"
        echo "[ivr] C env = $export_env"
    else
        export_env="$BASE"
        echo ""
        echo "############ B 臂（注入关闭）vs 亚军  $(date '+%H:%M:%S')"
        echo "[ivr] B env = $export_env"
    fi
    # ⚠️ 先把上一臂的 AZAI_* 全部清掉，再 export 本臂的 —— 否则 B 臂会**继承 A 臂的注入开关**
    #    （两臂变成同一臂，整批读数作废）。`_tmp_ladder` 用 os.environ.copy() 透传；
    #    亚军 exe 忽略 AZAI_*，所以给对手带上也无害。
    for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
    for kv in $(echo "$export_env" | tr ';' '\n'); do
        [ -n "$kv" ] && export "$kv"
    done
    if [ "$arm" = "A" ]; then
        [ -n "${AZAI_ORACLE_INJECT:-}" ] || { echo "[ivr] !! A 臂没有 AZAI_ORACLE_INJECT" >&2; exit 4; }
    elif [ "$arm" = "C" ]; then
        # C 臂的三条前置断言：开关真的开了、注入真的没开、`AZAI_MC_UPG` 真的没开（别混两条路）
        [ "${AZAI_UPG_BUNDLES:-}" = "$UPG_N" ] || { echo "[ivr] !! C 臂 AZAI_UPG_BUNDLES=${AZAI_UPG_BUNDLES:-<空>} != $UPG_N" >&2; exit 4; }
        [ -z "${AZAI_ORACLE_INJECT:-}" ] || { echo "[ivr] !! C 臂竟然带着 AZAI_ORACLE_INJECT" >&2; exit 4; }
        [ -z "${AZAI_ORACLE_PLACEBO:-}" ] || { echo "[ivr] !! C 臂竟然带着 AZAI_ORACLE_PLACEBO" >&2; exit 4; }
        [ -z "${AZAI_MC_UPG:-}" ] || { echo "[ivr] !! C 臂竟然带着 AZAI_MC_UPG（那是另一条路，会污染归因）" >&2; exit 4; }
    else
        [ -z "${AZAI_ORACLE_INJECT:-}" ] || { echo "[ivr] !! B 臂竟然带着 AZAI_ORACLE_INJECT=$AZAI_ORACLE_INJECT" >&2; exit 4; }
        [ -z "${AZAI_ORACLE_TIMEOUT:-}" ] || { echo "[ivr] !! B 臂竟然带着 AZAI_ORACLE_TIMEOUT" >&2; exit 4; }
    fi
    echo "[ivr] 环境自检 ✓  AZAI_ORACLE_INJECT=${AZAI_ORACLE_INJECT:-<空>}  AZAI_FLAT_SEED=$AZAI_FLAT_SEED  AZAI_K=$AZAI_K"

    stash_stderr "${arm}_$(date '+%Y%m%d_%H%M%S')"

    attempt=0
    while [ "$attempt" -le "$RETRY" ]; do
        todo="$(remaining_tokens "$tag")"
        if [ -z "${todo// /}" ]; then
            echo "[ivr] [$arm] 没有待跑的局（全部 verdict=engine）"
            break
        fi
        echo ""
        echo "[ivr] [$arm] 第 $((attempt+1)) 轮：待跑 = $todo  $(date '+%H:%M:%S')"
        # 失败局先留档，避免被下一轮覆盖（只复制不删）
        for tok in $todo; do
            [ -f "$out/$tok.log" ] || continue
            mkdir -p "$out/_attempts"
            cp -f "$out/$tok.log" "$out/_attempts/$tok.attempt$attempt.log" 2>/dev/null
        done
        t0=$(date +%s)
        "$P" -u _tmp_ladder.py --tag="$tag" --jobs="$JOBS" \
            --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" \
            $todo 2>&1 | tee "$out/_run.log"
        rc=${PIPESTATUS[0]}
        t1=$(date +%s)
        echo "[ivr] [$arm] 第 $((attempt+1)) 轮结束 rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%H:%M:%S')"
        attempt=$((attempt+1))
        # 本轮有进展就再来一轮判断（没有待跑的会 break）
        [ "$rc" -ne 0 ] && echo "[ivr] [$arm] ⚠️ ladder rc=$rc（继续按 resume 逻辑补跑）"
    done

    finish_arm "$arm" "$tag"
    echo "[ivr] [$arm] 完成  $(date '+%H:%M:%S')"
done

echo ""
echo "=============================================================="
echo "[ivr] 全部结束 $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ivr] 读数：$P -u code/test_match/analyze_inject_vs_runnerup.py \\"
echo "         --arm inject_A=match_results/ladder_logs/INJECT_VS_RU_A \\"
echo "         --stderr inject_A=$OUTROOT/A/stderr \\"
echo "         --arm inject_B=match_results/ladder_logs/INJECT_VS_RU_B \\"
echo "         --stderr inject_B=$OUTROOT/B/stderr \\"
echo "         --pair inject_A inject_B"
echo "=============================================================="
