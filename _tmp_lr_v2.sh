#!/usr/bin/env bash
# 验证 2：`--retry-invalid 1` 生效（只重跑失败的那一局，并转成 engine）
#
# 故障注入 = `_tmp_lr_faultinject/sitecustomize.py`：让**第 1 个**对局进程在启动瞬间
# `os._exit(3)`（等价"一个会立刻退出的假 AI"：没有 RESULT 行、日志是空壳）。
# 期望：第 1 局无效 → 第 1 轮重试**只**重跑它 → 转 engine；第 2 局一次过、日志不被覆盖
#       （证据：`_attempts/` 里只有那一局的 attempt1；收尾表里两局 attempts 分别是 2 / 1）。
source "$(dirname "${BASH_SOURCE[0]}")/_tmp_lr_env.sh"

rm -rf _tmp_lr_claims && mkdir -p _tmp_lr_claims
export PYTHONPATH="$PWD/_tmp_lr_faultinject"
export LR_FAULT_DIR="$PWD/_tmp_lr_claims"
export LR_FAULT_K=1
export LR_FAULT_RC=3
echo "[v2] 故障注入: 前 $LR_FAULT_K 个对局进程立刻退出（rc=$LR_FAULT_RC）  dir=$LR_FAULT_DIR"

TAG=LR_V2_RETRY
rm -rf "match_results/ladder_logs/$TAG"

bash code/run_logged.sh ladder_resilience_v2_retry \
    "$PY" -u _tmp_ladder.py --tag=$TAG --jobs=1 --retry-invalid 1 \
    --ai0="$RV4" --ai1="$NOOP" 992 992r
RC=$?

echo ""
echo "===== 落盘证据 ====="
echo "--- 注入计数（应 = 1）: $(ls _tmp_lr_claims 2>/dev/null | wc -l)"
echo "--- 日志目录（full-iso 时间戳用来证明'好的那局没被重写'）"
ls -l --time-style=full-iso "match_results/ladder_logs/$TAG"/*.log | sed 's/^/  /'
echo "--- _attempts/（只应有被重跑那一局的 attempt1）"
ls -l --time-style=full-iso "match_results/ladder_logs/$TAG/_attempts"/ 2>/dev/null | sed 's/^/  /'
echo "--- 被留档的失败日志（空壳 + 注入行，没有 RESULT）"
head -5 "match_results/ladder_logs/$TAG/_attempts/"*.attempt1.log 2>/dev/null | sed 's/^/  /'
echo "--- 两局现在的 RESULT 行"
grep -h "RESULT seed=" "match_results/ladder_logs/$TAG"/99*.log 2>/dev/null | sed 's/^/  /'
echo ""
echo "[v2] 结束 $(date '+%Y-%m-%d %H:%M:%S')  rc=$RC"
exit $RC
