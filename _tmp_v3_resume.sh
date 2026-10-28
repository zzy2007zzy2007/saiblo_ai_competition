#!/usr/bin/env bash
# V3：断电断点续跑。
#   stage1：正常跑 2 局（seed 901/902）。
#   stage2：手动把 seed902.log 的 verdict=engine 改成 verdict=timeout（模拟"那局是坏的/断电"），
#           再带 --resume ⇒ 期望"已完成 1 个、待跑 1 个"，只跑 seed902。
#   stage3：手动删掉 seed901.log（模拟"日志没落盘"），再 --resume ⇒ 期望只跑 seed901
#           （这条同时验证"日志是唯一权威，_progress.json 说 ok 也不算"）。
export PYTHONIOENCODING=utf-8
TINY='AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=16;AZAI_DEPTH=2;AZAI_K=8;AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off'
FLAT="$TINY;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000"
MCTS="$TINY;AZAI_SEARCH=mcts"
OUT=match_results/retry_resume_test/v3_resume
PY=D:/anaconda3/envs/pytorch-gpu/python.exe

echo "########## stage1：正常跑 2 局 ##########"
bash code/run_logged.sh v3_resume_s1_normal $PY -u code/test_match/ab_search_modes.py --seeds 901 902 --jobs 1 --out-dir $OUT --flat-env="$FLAT" --mcts-env="$MCTS"

echo "########## 人为把 seed902 改成 verdict=timeout ##########"
$PY -c "
import pathlib
p = pathlib.Path('$OUT/seed902.log')
t = p.read_text(encoding='utf-8')
t2 = t.replace('verdict=engine', 'verdict=timeout')
assert t2 != t, 'RESULT 行里没找到 verdict=engine'
p.write_text(t2, encoding='utf-8')
print('[corrupt]', p, '-> verdict=timeout')
"
grep -h 'RESULT seed=' $OUT/seed902.log | tail -1

echo "########## stage2：--resume 只补那一局 ##########"
bash code/run_logged.sh v3_resume_s2_resume $PY -u code/test_match/ab_search_modes.py --seeds 901 902 --jobs 1 --out-dir $OUT --resume --flat-env="$FLAT" --mcts-env="$MCTS"

echo "########## 人为删掉 seed901.log（模拟日志没落盘）##########"
rm -f $OUT/seed901.log

echo "########## stage3：--resume 只补被删的那一局 ##########"
bash code/run_logged.sh v3_resume_s3_missing $PY -u code/test_match/ab_search_modes.py --seeds 901 902 --jobs 1 --out-dir $OUT --resume --flat-env="$FLAT" --mcts-env="$MCTS"

echo "########## 最终 _progress.json ##########"
cat $OUT/_progress.json
