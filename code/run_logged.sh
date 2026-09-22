#!/usr/bin/env bash
# 实验记录包装器：自动记录 git 版本 + 完整命令 + 输出日志，并追加到 docs/experiment_log.md
#
# 用法:
#   bash code/run_logged.sh <实验名> <命令...>
#
# 例:
#   bash code/run_logged.sh eval_bn0v_recheck \
#       D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py \
#       --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt --games 32
#
# 产出:
#   training_history/runs/<时间戳>_<名字>/{cmd.txt,meta.txt,output.log}
#   docs/experiment_log.md 末尾追加一条记录（result 需手动补）
#
# 为什么要记录 commit + dirty：同一份代码/命令，工作区有没有未提交改动
# 会直接改变结果（2026-09-09 的 34.4% 复现失败就是被这个变量卡住）。
set -u

REPO="D:/2026智能体大赛_新2"
cd "$REPO" || exit 1

# stdout 被重定向进 output.log 时，Windows 下 Python 会用 cp936 ⇒ 中文全乱码；更要命的是
# 无法编码的字符（emoji 等）会直接 UnicodeEncodeError 把脚本打断
# （2026-09-22 实测：_tmp_ladder.py 打印 "⚠️" 时崩掉，整个汇总被吃掉）。统一成 UTF-8。
export PYTHONIOENCODING=utf-8

NAME="${1:?用法: bash code/run_logged.sh <实验名> <命令...>}"
shift
if [ $# -eq 0 ]; then
    echo "[run_logged] 没有给出要运行的命令" >&2
    exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
HUMAN_TS="$(date '+%Y-%m-%d %H:%M:%S')"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
# 只数"已跟踪文件被改但没提交"（真正的代码漂移信号）；未跟踪的临时文件不算
DIRTY_N="$(git status --porcelain 2>/dev/null | grep -v '^??' | wc -l | tr -d ' ')"
UNTRACKED_N="$(git status --porcelain 2>/dev/null | grep -c '^??' | tr -d ' ')"
RUN_DIR="$REPO/training_history/runs/${TS}_${NAME}"
mkdir -p "$RUN_DIR"

printf '%s\n' "$*" > "$RUN_DIR/cmd.txt"
{
    echo "name: $NAME"
    echo "start: $HUMAN_TS"
    echo "commit: $COMMIT"
    echo "dirty_tracked_files: $DIRTY_N"
    echo "untracked_files: $UNTRACKED_N"
} > "$RUN_DIR/meta.txt"

echo "[run_logged] name=$NAME  commit=$COMMIT  dirty_tracked=$DIRTY_N  untracked=$UNTRACKED_N"
echo "[run_logged] cmd: $*"
echo "[run_logged] log: $RUN_DIR/output.log"

t0=$(date +%s)
"$@" > "$RUN_DIR/output.log" 2>&1
RC=$?
t1=$(date +%s)
{
    echo "end: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "exit_code: $RC"
    echo "duration_s: $((t1 - t0))"
} >> "$RUN_DIR/meta.txt"

{
    echo ""
    echo "## ${HUMAN_TS} — ${NAME}"
    echo ""
    echo "- **commit**: \`${COMMIT}\` (dirty: ${DIRTY_N} files)"
    echo "- **exit**: ${RC}，用时 $((t1 - t0))s"
    echo "- **cmd**:"
    echo '  ```bash'
    echo "  $*"
    echo '  ```'
    echo "- **output**: \`training_history/runs/${TS}_${NAME}/output.log\`"
    echo "- **result**: _待填_"
} >> "$REPO/docs/experiment_log.md"

echo "[run_logged] exit=$RC  ($((t1 - t0))s)  -> docs/experiment_log.md"
exit $RC
