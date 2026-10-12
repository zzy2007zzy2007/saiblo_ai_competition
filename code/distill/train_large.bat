@echo off
cd /d D:\2026智能体大赛_新2
call D:\anaconda3\Scripts\conda.bat activate pytorch-gpu
set PYTHONUNBUFFERED=1
python -u code/distill/train_large_balanced.py > distill_data_v2\train_large.log 2>&1
