#!/usr/bin/env bash
# 队列：等三方循环赛（_rr64.sh）跑完，自动接着跑第 4 组 f10 vs 冠军
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
echo "############ 队列启动：等待 _rr64.sh 结束  $(date +%H:%M:%S)"
while pgrep -f "code/test_match/_rr64.sh" > /dev/null 2>&1; do
  sleep 60
done
echo "############ 检测到循环赛结束，启动第 4 组  $(date +%H:%M:%S)"
bash code/test_match/_rr64_champ.sh
