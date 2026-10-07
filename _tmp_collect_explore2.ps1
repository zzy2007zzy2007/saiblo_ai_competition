# _tmp_collect_explore.ps1 —— 计划/预注册 `docs/plan_20261015_explore_awr_class.md` 第 1 步：**探索采集**
#   部署配置（mc_quiet/EVERY=5/ROOTONLY=1、K=24/ITERS=256）+ **30% 随机类钉**
#   种子 80,010,000+（与判据 seed 不重叠）；产出 training_history/vprior/data_explore_60/*.pkl
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
& $runner -Name collect_explore_60 -CommandLine "$Py -u code/my_ai/az_intent/az_selfplay.py --checkpoint checkpoints/posnet_A_k5_m32.pt --games 60 --workers 8 --seed 8001 --iterations 256 --max-depth-rounds 4 --k 24 --sample-mult 15 --search-mode pos-only --skip-single-candidate --pos-pin mc_quiet --mc-horizon 256 --mc-every 5 --mc-root-only 1 --class-pin-random-prob 0.08 --native-engine --out-dir training_history/vprior/data_explore_60"
Write-Host "[collect_explore] done $(Get-Date -Format 'HH:mm:ss')"

