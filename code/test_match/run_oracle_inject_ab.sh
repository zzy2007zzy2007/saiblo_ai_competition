#!/usr/bin/env bash
# 亚军候选注入的**冒烟跑批**：2 个 seed × 2 个座位 = 4 局（注入臂 vs 不注入臂，同一套冠军配置）。
#
# 为什么不用 `ab_search_modes.py --paired-sides`：它的 stderr 文件名只含 (side, label, seed)
# （`match_via_sdk.py:373-374`），而配对的同一 seed 两局**并发**运行 ⇒ 两个进程用 "wb" 打开
# **同一个** `ai?_az_bridge_ai_seedN.stderr.log` ⇒ 日志互相覆盖，注入统计/耗时读不出来。
# 这里改成"两波、每波两个**不同 seed** 并发"：文件名不撞，每局跑完立刻把 stderr 另存一份。
#
# 用法：bash code/test_match/run_oracle_inject_ab.sh [seedA] [seedB]
#   产出：match_results/oracle_inject/inject/seed<S>_p<座位>.log           （对局账本，含 RESULT 行）
#         match_results/oracle_inject/inject/seed<S>_p<座位>_ai{0,1}.stderr.log（两个 AI 的 stderr）
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
BRIDGE=code/test_match/az_bridge_ai.py
EXE="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
# 冠军配置（用户 2026-10-26 指定；两侧完全相同，唯一差别是注入开关）
CHAMP="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_SEARCH=flat;AZAI_MODE=pos-only"
INJ="$CHAMP;AZAI_ORACLE_INJECT=$EXE;AZAI_ORACLE_VERBOSE=1"
OUT=match_results/oracle_inject/inject
mkdir -p "$OUT"
SEED_A="${1:-3000}"
SEED_B="${2:-3001}"

run_one() {   # $1=seed  $2=注入臂坐哪边(0/1)
    local seed="$1" side="$2" e0 e1
    e0="$CHAMP"; e1="$CHAMP"
    if [ "$side" = "0" ]; then e0="$INJ"; else e1="$INJ"; fi
    echo "[ab] 开局 seed=$seed 注入臂=P$side -> $OUT/seed${seed}_p${side}.log"
    "$P" -u code/test_match/match_via_sdk.py --trace-ops \
        "--ai0=$BRIDGE" "--ai1=$BRIDGE" \
        "--ai0-env=$e0" "--ai1-env=$e1" \
        "$seed" > "$OUT/seed${seed}_p${side}.log" 2>&1
    local rc=$?
    # 立刻另存 stderr（下一波会用同一个 seed 覆盖同名文件）
    for s in 0 1; do
        local src="match_results/ai${s}_az_bridge_ai_seed${seed}.stderr.log"
        [ -f "$src" ] && cp "$src" "$OUT/seed${seed}_p${side}_ai${s}.stderr.log"
    done
    echo "[ab] 结束 seed=$seed 注入臂=P$side rc=$rc"
}

echo "[ab] 波 1：seed=$SEED_A 注入臂=P0 ＋ seed=$SEED_B 注入臂=P1"
run_one "$SEED_A" 0 &
run_one "$SEED_B" 1 &
wait
echo "[ab] 波 2：seed=$SEED_B 注入臂=P0 ＋ seed=$SEED_A 注入臂=P1"
run_one "$SEED_B" 0 &
run_one "$SEED_A" 1 &
wait

echo ""
echo "================ 汇总 ================"
for f in "$OUT"/seed*_p*.log; do
    [ -f "$f" ] || continue
    grep -h "RESULT seed" "$f" || echo "  !! $f 里没有 RESULT 行"
done
echo "---- 每局注入统计（注入臂的 stderr）----"
for f in "$OUT"/seed*_p*_ai*.stderr.log; do
    [ -f "$f" ] || continue
    if grep -q "注入已启用" "$f" 2>/dev/null; then
        echo "== $(basename "$f") =="
        grep -h "注入统计\|决策耗时: n=\|停用注入\|非法" "$f" | tail -5
    fi
done
