#!/usr/bin/env bash
# 8 局冒烟：**冠军配置** × `AZAI_VALUE_CKPT` 开/关各 4 局（2 seed × 2 座位 = 4 局/批）。
#
# 目的（任务 D）：只验"接上小值网之后还能不能正常打完一局"：
#   * 每局 `verdict=engine`（官方 5 级级联判词，不是 hp_tiebreak 兜底）；
#   * 无异常退出（`_tmp_ladder` 的逐局 rc=0 + 无 INVALID/ILLEGAL）；
#   * 记录每局耗时（`[token] rc=0 X.X min`）。
# **不跑长对局**：冠军配置本身每决策 ~0.2-0.5 s，一局 5-10 min。
#
# 冠军配置（与 `docs/oracle_inject_design.md:220` 的 CHAMP 逐字一致）：
#   AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt AZAI_K=160 AZAI_FLAT_ROUNDS=39000
#   AZAI_FLAT_ROLLOUT_MODE=idle AZAI_SEARCH=flat AZAI_MODE=pos-only
# 另加 AZAI_TORCH_THREADS=1（部署口径，且遵守"最多 4 核"：2 局 × (AI 1 线程 + bridge 1 线程)）。
#
# 用法：bash code/run_logged.sh small_value_smoke bash code/test_match/_smoke_small_value.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
FOLDED=checkpoints/small32_3/v_32_3_folded.pt
SEEDS="5 5r 6 6r"        # 4 局 = 2 seed × 2 座位（r = 镜像，我们的 AI 执后手）
JOBS=2                   # 2 局并发 ≈ 4 线程（AI 1 + bridge 1）× 2

export PYTHONIOENCODING=utf-8
export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_K=160 AZAI_FLAT_ROUNDS=39000 AZAI_FLAT_ROLLOUT_MODE=idle
export AZAI_SEARCH=flat AZAI_MODE=pos-only AZAI_TORCH_THREADS=1

echo "############ ① 开关关：AZAI_VALUE_CKPT 为空（= 改动前逻辑）  $(date +%H:%M:%S)"
unset AZAI_VALUE_CKPT
"$PY" -u _tmp_ladder.py --tag=small_value_smoke_off --jobs=$JOBS \
  --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" $SEEDS
RC_OFF=$?

if [ ! -f "$FOLDED" ]; then
    echo "!! 缺 $FOLDED（先跑 code/test_match/fold_small_value_ckpt.py）" >&2
    exit 3
fi
echo "############ ② 开关开：AZAI_VALUE_CKPT=$FOLDED  $(date +%H:%M:%S)"
export AZAI_VALUE_CKPT="$FOLDED"
"$PY" -u _tmp_ladder.py --tag=small_value_smoke_on --jobs=$JOBS \
  --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" $SEEDS
RC_ON=$?

echo "############ ③ 逐局判词与耗时  $(date +%H:%M:%S)"
for TAG in small_value_smoke_off small_value_smoke_on; do
    echo "--- $TAG ---"
    for f in match_results/ladder_logs/$TAG/*.log; do
        [ -f "$f" ] || continue
        _name=$(basename "$f" .log)
        _res=$(grep -o "RESULT .*" "$f" | head -1)
        echo "  ${_name}: ${_res:-!! 没有 RESULT 行（该局异常终止）}"
    done
done

echo "############ ④ 冒烟判定"
FAIL=0
if [ $RC_OFF -ne 0 ] || [ $RC_ON -ne 0 ]; then
    echo "  ❌ 有批次非零退出（off=$RC_OFF on=$RC_ON）"
    FAIL=1
fi
_N_ENGINE=$(grep -ho "verdict=engine" match_results/ladder_logs/small_value_smoke_o*/*.log | wc -l)
_N_OTHER=$(grep -ho "verdict=[a-z_]*" match_results/ladder_logs/small_value_smoke_o*/*.log | grep -v "verdict=engine" | wc -l)
echo "  verdict=engine 的局数: ${_N_ENGINE}  （非 engine: ${_N_OTHER}）"
if [ "$_N_ENGINE" -lt 8 ] || [ "$_N_OTHER" -ne 0 ]; then
    echo "  ❌ 不是 8 局全部 verdict=engine"
    FAIL=1
fi
_ILL=$(grep -ho "ILLEGAL" match_results/ladder_logs/small_value_smoke_o*/*.log | wc -l)
echo "  ILLEGAL 出现次数: ${_ILL}"
if [ "$_ILL" -ne 0 ]; then
    echo "  ❌ 有非法操作"
    FAIL=1
fi
if [ $FAIL -eq 0 ]; then
    echo "  ✅ 8 局全部 verdict=engine、无非法操作、两批都 rc=0"
fi
exit $FAIL
