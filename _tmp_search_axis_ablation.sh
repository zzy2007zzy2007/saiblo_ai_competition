#!/usr/bin/env bash
# 搜索增益来自"选动作类"还是"选位置"？—— 固定一维、只搜另一维，跑 search-vs-raw。
#
# 判据：--self-raw-opponent 让对手用同一模型的 raw（temperature=0）解码，
# 我方用受限搜索。四个 arm 用同一批 seed（0..31），逐局配对。
#
#   joint             : 类与位置都采样（默认全搜索，参照）
#   class-only        : 位置固定为 action_map 的 argmax，只采样动作类
#   pos-only(argmax)  : 动作类固定为原始 logits argmax，只采样位置
#                       —— 注意：诊断显示 argmax 类常常是"买不起的闪电/不可执行"，
#                          此时候选集只剩空 bundle，该 arm 退化成 raw（见 doc §八）
#   pos-only(playable): 动作类固定为"最高 logit 且真能执行"的类，再只采样位置
#                       —— 防止上面的退化，让位置维真正有搜索空间
#
# 每个 arm 另外统计每回合诊断：候选数>=2 的比例（有没有搜索空间）、
# 搜索选择与 raw 是否相同、不同时落在类还是位置维。
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
CK=training_history/az_fixed/mix_r10p_vw_pol_frozen.pt   # 标准基线（策略冻结在 az_r10）

run () {  # $1=标签 $2=search_mode $3=pos_pin
  echo "############ $1  (search_mode=$2 pos_pin=$3, 32 局 @256/4) ############"
  "$PY" code/my_ai/az_intent/eval.py \
      --checkpoint "$CK" --self-raw-opponent --bundle-mcts --native-engine \
      --iterations 256 --max-depth-rounds 4 --k 24 \
      --t-class 0.5 --t-pos 0.3 \
      --games 32 --workers 16 \
      --search-mode "$2" --pos-pin "$3"
}

run "① joint（参照）"        joint    argmax
run "② class-only"           class-only argmax
run "③ pos-only(argmax)"     pos-only argmax
run "④ pos-only(playable)"   pos-only playable
echo "############ 动作类/位置搜索拆分消融完成 ############"
