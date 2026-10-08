# _tmp_nlr_nd.ps1 —— 预注册 `docs/prereg_20261018_nlr_churn_ablation.md`
#   NL-R + AZAI_NO_DOWNGRADE=1（一次只动一个开关），固定段 128 局
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'; $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
$env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
$env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
$env:AZAI_FORCE_UPG = '0'; $env:AZAI_PROBE_UPG = '0'
$env:AZAI_BAN_CLASSES = ''; $env:AZAI_BAN_OPS = ''
$env:AZAI_SLOW_DECISION_SEC = '30'
# NL-R 的三个开关
$env:AZAI_MC_NO_LIGHT = '1'; $env:AZAI_MC_UPG = '1'; $env:AZAI_MC_TOPK = '3'
# ★ 本臂唯一新增：执行层剔除降级
$env:AZAI_NO_DOWNGRADE = '1'
$tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
& "$Repo/code/run_logged.ps1" -Name NLRND_fixed128 -CommandLine "$Py -u _tmp_ladder.py --tag=NLRND_fixed128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=NLRND_fixed128 --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/NLRND_fixed128/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[NLRND] done $(Get-Date -Format 'HH:mm:ss')"
