#!/usr/bin/env bash
# 验证 1：**默认行为不变**（不加任何新开关）
#   HEAD 版驱动（`git show HEAD:_tmp_ladder.py` 的副本 `_tmp_ladder_head.py`）vs 新版驱动，
#   同一 2 局小批次（token 920 = AI0 执 P0 / 920r = 镜像）。
#   判据：`RESULT` 行逐字段相同；新版 stdout 里没有新开关的区块。
# 先跑两遍 HEAD 版（`LR_V1_OLD` / `LR_V1_OLD2`）⇒ 顺便证明这个对局是可复现的
#   （否则"新旧一样"可能只是"两次运行本来就会飘"）。
source "$(dirname "${BASH_SOURCE[0]}")/_tmp_lr_env.sh"

# ⚠️ HEAD 副本只生成一次：本改动提交后 HEAD 就变了，不能每次重取。
[ -f _tmp_ladder_head.py ] || git show HEAD:_tmp_ladder.py > _tmp_ladder_head.py
echo "[v1] HEAD 副本 _tmp_ladder_head.py: $(wc -l < _tmp_ladder_head.py) 行"
echo "[v1] 新版       _tmp_ladder.py:      $(wc -l < _tmp_ladder.py) 行"

TOKENS="920 920r"
AIS="--ai0=$RV4 --ai1=$NOOP"
for t in LR_V1_OLD LR_V1_OLD2 LR_V1_NEW; do rm -rf "match_results/ladder_logs/$t"; done

echo ""
echo "===== V1-old：HEAD 版驱动（默认参数，第 1 次）====="
bash code/run_logged.sh ladder_resilience_v1_old \
    "$PY" -u _tmp_ladder_head.py --tag=LR_V1_OLD --jobs=1 $AIS $TOKENS

echo ""
echo "===== V1-old2：HEAD 版驱动（默认参数，第 2 次 ⇒ 可复现性）====="
bash code/run_logged.sh ladder_resilience_v1_old2 \
    "$PY" -u _tmp_ladder_head.py --tag=LR_V1_OLD2 --jobs=1 $AIS $TOKENS

echo ""
echo "===== V1-new：新版驱动（**默认参数**，三个新开关全关）====="
bash code/run_logged.sh ladder_resilience_v1_new \
    "$PY" -u _tmp_ladder.py --tag=LR_V1_NEW --jobs=1 $AIS $TOKENS
RC=$?

echo ""
echo "===== (a) 可复现性：HEAD 版两次对比 ====="
"$PY" _tmp_lr_cmp.py LR_V1_OLD LR_V1_OLD2 $TOKENS
REP=$?
echo ""
echo "===== (b) 默认行为不变：HEAD 版 vs 新版 ====="
"$PY" _tmp_lr_cmp.py LR_V1_OLD LR_V1_NEW $TOKENS
CMP=$?

echo ""
echo "===== 两个驱动 stdout 的逐局表 ====="
for d in $(ls -d training_history/runs/*_ladder_resilience_v1_old 2>/dev/null | tail -1) \
         $(ls -d training_history/runs/*_ladder_resilience_v1_new 2>/dev/null | tail -1); do
    echo "--- $d/output.log"
    sed -n '/=== 逐局 ===/,/=== 胜负汇总/p' "$d/output.log" | head -8
done
echo ""
echo "[v1] 结束 $(date '+%Y-%m-%d %H:%M:%S')  reproducible_rc=$REP  cmp_rc=$CMP"
exit $(( REP || CMP ))
