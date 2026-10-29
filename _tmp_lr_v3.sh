#!/usr/bin/env bash
# 验证 3：`--final-sweep` 生效（重试轮数已用完仍有无效局 ⇒ 再跑最后一次）
#
# 故障注入 `_tmp_lr_faultinject/sitecustomize.py` 让**前 4 个**对局进程立刻退出：
#   第一批 2 局（claim 1,2）失败 → `--retry-invalid 1` 第 1 轮 2 局（claim 3,4）又失败
#   ⇒ 重试轮数用完、仍 2 局无效 ⇒ `--final-sweep` 再跑一次（这时注入已用完 ⇒ 真对局 ⇒ engine）。
source "$(dirname "${BASH_SOURCE[0]}")/_tmp_lr_env.sh"

rm -rf _tmp_lr_claims3 && mkdir -p _tmp_lr_claims3
export PYTHONPATH="$PWD/_tmp_lr_faultinject"
export LR_FAULT_DIR="$PWD/_tmp_lr_claims3"
export LR_FAULT_K=4
export LR_FAULT_RC=3
echo "[v3] 故障注入: 前 $LR_FAULT_K 个对局进程立刻退出（rc=$LR_FAULT_RC） ⇒ 第一批 + 重试轮全废"

TAG=LR_V3_SWEEP
rm -rf "match_results/ladder_logs/$TAG"

bash code/run_logged.sh ladder_resilience_v3_final_sweep \
    "$PY" -u _tmp_ladder.py --tag=$TAG --jobs=1 --retry-invalid 1 --final-sweep \
    --ai0="$RV4" --ai1="$NOOP" 994 994r
RC=$?

echo ""
echo "===== 落盘证据 ====="
echo "--- 注入计数（应 = 4）: $(ls _tmp_lr_claims3 2>/dev/null | wc -l)"
echo "--- _attempts/（应有 4 份：attempt1 = 重试前留档，attempt2 = final-sweep 前留档）"
ls -l "match_results/ladder_logs/$TAG/_attempts"/ 2>/dev/null | sed 's/^/  /'
echo "--- 现在两局的 RESULT 行"
grep -h "RESULT seed=" "match_results/ladder_logs/$TAG"/99*.log 2>/dev/null | sed 's/^/  /'
echo ""
echo "[v3] 结束 $(date '+%Y-%m-%d %H:%M:%S')  rc=$RC"
exit $RC
