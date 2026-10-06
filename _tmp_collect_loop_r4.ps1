# _tmp_collect_loop_r4.ps1 —— 第 4 轮闭环重训的**数据采集**：
#   用**部署配置**（mc_quiet + MC_EVERY=5 + ROOTONLY=1 + HORIZON=256）自对弈 300 局（8 workers）
#   ⇒ 这是"我们现在真正在用的策略"的分布（项目史上所有重训都基于"不建塔"的旧分布）。
#   seed 基数 5001 ⇒ 实际种子 50,010,000+ ⇒ **与判据 seed（7..70 / 1000..9927）完全不重叠**（章程③）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
& $runner -Name loop_r4_collect300 -CommandLine "$Py -u code/my_ai/az_intent/az_selfplay.py --checkpoint checkpoints/posnet_A_k5_m32.pt --games 300 --workers 8 --seed 5001 --iterations 256 --max-depth-rounds 4 --k 24 --sample-mult 15 --search-mode pos-only --skip-single-candidate --pos-pin mc_quiet --mc-horizon 256 --mc-every 5 --mc-root-only 1 --native-engine --out-dir training_history/vprior/data_loop_r4_300"
Write-Host "[loop_r4] 采集完成 $(Get-Date -Format 'HH:mm:ss')"
