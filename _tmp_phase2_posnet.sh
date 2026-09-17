#!/usr/bin/env bash
# Phase 2：位置网第一轮实跑（docs/az_three_net_split_plan.md §7 预注册判据）
#
#   1) 采集 400 局（pos-only + skip-single + skip-hold，pkl 路径——位置 CE 需要 visit/intent_counts）
#   2) 过滤成紧凑数据集（只留带位置目标的样本，2.8% 体量；否则训练时全量载入会 OOM）
#   3) 训两个 anchor 强度：λ=30（有效 anchor/CE≈1%）、λ=150（≈5%）
#   4) 判据2 硬闸门：**不搜索时** raw(新) vs raw(旧) 的镜像配对（SE=0 判据，几分钟）
#   5) 绝对强度：vs rule_v4 **无搜索** 64 局（含未训练基线作对照）
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
CK2=training_history/az_fixed/mix_r10p_vw_pol_frozen.pt
CK3=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
W=training_history/posnet_p2
RAW="$W/raw"; CMP="$W/compact"
GAMES=400

echo "########## 1) 采集 $GAMES 局（pos-only + skip-single + skip-hold）##########"
"$PY" code/my_ai/az_intent/az_selfplay.py --checkpoint "$CK2" --games "$GAMES" \
    --workers 16 --iterations 256 --max-depth-rounds 2 --max-rounds 512 --native-engine \
    --skip-hold-search --search-mode pos-only --skip-single-candidate \
    --out-dir "$RAW" --seed 910101 --t-class 0.5 --t-pos 1.0 --k 24

echo "########## 2) 过滤成紧凑数据集 ##########"
"$PY" code/my_ai/az_intent/filter_pos_samples.py --src "$RAW" --out "$CMP"

echo "########## 3) 训练位置网（λ=30 / λ=150，8 epoch，pos_single）##########"
for lam in 30 150; do
  echo "---------- λ=$lam ----------"
  "$PY" code/my_ai/az_intent/az_train.py --pos-only-net --init "$CK3" \
      --policy-dir "$CMP" --checkpoint "$W/pos_l$lam.pt" \
      --epochs 8 --batch-size 32 --lr 1e-3 --lambda-anchor "$lam" --pos-single
done

echo "########## 4) 判据2：raw(新) vs raw(旧) 镜像配对 64 pair ##########"
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

echo "########## 5) 绝对强度：vs rule_v4 无搜索 64 局 ##########"
for tag in untrained l30 l150; do
  case "$tag" in
    untrained) CK="$CK3" ;;
    l30) CK="$W/pos_l30.pt" ;;
    l150) CK="$W/pos_l150.pt" ;;
  esac
  echo "---------- $tag ----------"
  "$PY" code/my_ai/az_intent/eval.py --checkpoint "$CK" --opponent rule_v4 \
      --no-search --games 64 --workers 16
done
echo "########## Phase 2 完成 ##########"
