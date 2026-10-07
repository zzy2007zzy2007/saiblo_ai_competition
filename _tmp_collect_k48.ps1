# _tmp_collect_k48.ps1 —— 预注册 `docs/prereg_20261013_distill_k48_cells.md` 第 1 步：**教师采集**
#   部署配置 + K=48/ITERS=512 自对弈 80 局（种子 70,010,000+，与判据 seed 不重叠）
#   产出 = training_history/vprior/data_k48_80/*.pkl（含每决策每头的候选格子访问分布）
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
& $runner -Name collect_k48_80 -CommandLine "$Py -u code/my_ai/az_intent/az_selfplay.py --checkpoint checkpoints/posnet_A_k5_m32.pt --games 80 --workers 8 --seed 7001 --iterations 512 --max-depth-rounds 4 --k 48 --sample-mult 15 --search-mode pos-only --skip-single-candidate --pos-pin mc_quiet --mc-horizon 256 --mc-every 5 --mc-root-only 1 --native-engine --out-dir training_history/vprior/data_k48_80"
Write-Host "[collect_k48] done $(Get-Date -Format 'HH:mm:ss')"
