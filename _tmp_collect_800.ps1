# _tmp_collect_800.ps1 —— 第 10 轮：**规模检查用的自对弈数据**（部署配置，800 局）
#   目的（唯一未试过的"量"）：第 5 轮的"类头学不成状态依赖（logit std 0.023）"是不是**样本量问题**？
#   76 局 → 800 局（约 10×）。种子基数 6001 ⇒ 实际种子 60,010,000+ ⇒ **与判据 seed 完全不重叠**（章程③）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
& $runner -Name loop_scale_collect800 -CommandLine "$Py -u code/my_ai/az_intent/az_selfplay.py --checkpoint checkpoints/posnet_A_k5_m32.pt --games 800 --workers 14 --seed 6001 --iterations 256 --max-depth-rounds 4 --k 24 --sample-mult 15 --search-mode pos-only --skip-single-candidate --pos-pin mc_quiet --mc-horizon 256 --mc-every 5 --mc-root-only 1 --native-engine --out-dir training_history/vprior/data_scale_800"
Write-Host "[scale] 采集完成 $(Get-Date -Format 'HH:mm:ss')"
