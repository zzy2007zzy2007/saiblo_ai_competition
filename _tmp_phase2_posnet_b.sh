#!/usr/bin/env bash
# Phase 2（重跑 3)-5) 步）：采集与过滤已完成（400 局 -> 9367 个位置样本，459 MB）
#
# 上次失败原因：az_train 的 print 里含 `⇒`(U+21D2)，run_logged 的 GBK 控制台编不出来
# ⇒ 训练直接崩、ckpt 没生成、后面评估全部 FileNotFound。已改成 ASCII，并在此显式设
# PYTHONIOENCODING=utf-8（双保险）。
#
# 判据修正：**"vs rule_v4 无搜索"没有分辨力**（raw 策略本来就是 0%，地板），
# 所以第 5 步改成用**廉价搜索**（pos-only + skip-single，实测 8 局 36s）量绝对强度。
export PYTHONIOENCODING=utf-8
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
CK3=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
W=training_history/posnet_p2
CMP="$W/compact"

echo "########## 3) 训练位置网（λ=30 / λ=150，8 epoch，pos_single）##########"
for lam in 30 150; do
  echo "---------- λ=$lam ----------"
  "$PY" code/my_ai/az_intent/az_train.py --pos-only-net --init "$CK3" \
      --policy-dir "$CMP" --checkpoint "$W/pos_l$lam.pt" \
      --epochs 8 --batch-size 32 --lr 1e-3 --lambda-anchor "$lam" --pos-single \
      || { echo "!! λ=$lam 训练失败"; exit 1; }
done

echo "########## 4) 判据2（硬闸门）：raw(新) vs raw(旧) 镜像配对 64 pair ##########"
for tag in untrained l30 l150; do
  case "$tag" in
    untrained) CK="$CK3" ;;
    l30) CK="$W/pos_l30.pt" ;;
    l150) CK="$W/pos_l150.pt" ;;
  esac
  echo "---------- $tag ----------"
  "$PY" _tmp_position_choice_ab.py --checkpoint "$CK" --opponent-checkpoint "$CK3" \
      --mode raw --pairs 64 --workers 16 --seed 0
done

echo "########## 5) 绝对强度：vs rule_v4 **带廉价搜索**（pos-only + skip-single）64 局 ##########"
for tag in untrained l30 l150; do
  case "$tag" in
    untrained) CK="$CK3" ;;
    l30) CK="$W/pos_l30.pt" ;;
    l150) CK="$W/pos_l150.pt" ;;
  esac
  echo "---------- $tag ----------"
  "$PY" code/my_ai/az_intent/eval.py --checkpoint "$CK" --opponent rule_v4 \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 --k 24 \
      --t-class 0.5 --t-pos 0.3 --search-mode pos-only --skip-single-candidate \
      --games 64 --workers 16
done
echo "########## Phase 2 重跑完成 ##########"
