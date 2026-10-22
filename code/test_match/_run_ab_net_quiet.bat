@echo off
rem 由 Windows 计划任务调用：完全脱离 DSH 的进程树，工具调用中断也不会被连带杀死。
cd /d D:\2026智能体大赛_新2_distill
"D:\Program Files\Git\bin\bash.exe" code/run_logged.sh ab_mc_net_vs_quiet_upg3 bash code/test_match/_ab_mc_net_vs_quiet.sh
