#!/usr/bin/env bash
# 队列：等 training_history/.rr64_done 出现后，自动跑第 4 组（f10 vs 冠军，同种子/同 FLAT_SEED）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
echo "############ 队列启动：等待 .rr64_done  $(date +%H:%M:%S)"
for i in $(seq 1 1440); do
  if [ -f training_history/.rr64_done ]; then
    echo "############ 哨兵出现，启动第 4 组  $(date +%H:%M:%S)"
    bash code/test_match/_rr64_champ.sh
    exit 0
  fi
  sleep 60
done
echo "############ 等待超时（24 小时）  $(date +%H:%M:%S)"
