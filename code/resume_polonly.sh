#!/usr/bin/env bash
# 通用断点恢复（策略-only 循环）：
#   1) 找出最后一个已完成的 checkpoint az_p{k}.pt
#   2) 若 batch{k+1} 的采集数据已在但 checkpoint 缺失（被重启/打断），先补训
#   3) 从 k+2 继续跑到 TOTAL 个 batch
# 幂等：run_policy_only.sh 自身会跳过已有 checkpoint 的 batch。
#
# 用法: bash code/resume_polonly.sh [随机注入x] [目标总batch数]
set -u

REPO="D:/2026智能体大赛_新2"
PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
DATA="$REPO/training_history/az_fixed/data_polonly"
CKPT_DIR="$REPO/training_history/az_fixed"
RAND="${1:-0.0}"
TOTAL="${2:-21}"
cd "$REPO" || exit 1

last=0
for f in "$CKPT_DIR"/az_p*.pt; do
    [ -e "$f" ] || continue
    k=$(basename "$f" | sed -e 's/^az_p//' -e 's/\.pt$//')
    case "$k" in
        ''|*[!0-9]*) continue ;;
    esac
    if [ "$k" -gt "$last" ]; then last=$k; fi
done
echo "[resume] 最后一个完成的 batch = $last"
prev="$CKPT_DIR/az_p$last.pt"

next=$((last + 1))
n_pkl=$(ls "$DATA/batch$next"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_pkl" -gt 0 ] && [ ! -f "$CKPT_DIR/az_p$next.pt" ]; then
    echo "[resume] batch$next 已有 $n_pkl 个 pkl 但无 checkpoint，补训"
    "$PY" code/my_ai/az_intent/az_train.py --init "$prev" \
        --policy-dir "$DATA/batch$next" --data-dir "$DATA" \
        --checkpoint "$CKPT_DIR/az_p$next.pt" --epochs 5 --split --policy-only \
        --max-value-batches 1 --device auto
    prev="$CKPT_DIR/az_p$next.pt"
    last=$next
fi

remaining=$((TOTAL - last))
if [ "$remaining" -le 0 ]; then
    echo "[resume] 已达到 $TOTAL 个 batch（当前 $last），无需继续"
    exit 0
fi
echo "[resume] 继续 $remaining 个 batch：batch$((last + 1)) .. batch$TOTAL，注入 x=$RAND"
bash code/run_policy_only.sh "$remaining" "$((last + 1))" "$prev" "$DATA" "$CKPT_DIR" 16 16 256 "$RAND"
