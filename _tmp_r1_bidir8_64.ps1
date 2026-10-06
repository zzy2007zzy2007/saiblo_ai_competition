# _tmp_r1_bidir8_64.ps1 —— rollout 阶梯 R1：**K=8 双向**，固定段前缀 seed 7..38（64 局）
#   预注册：docs/prereg_20261006_rollout_ladder.md（筛选 n=64、晋级线 = R0 前缀 +8pp）
#   R0 同前缀读数 = 0.6562（32/32 对，来自已有 M6cROOT_128 日志）⇒ 晋级线 = 0.7362
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Tag = 'R1_bidir8_64',
    [int]$K = 8
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

$seeds = 7..38
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[$Tag] seed 数=$($seeds.Count)，局数=$(2*$seeds.Count)；K=$K"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = "$K"
$env:AZAI_SLOW_DECISION_SEC = '30'

& $runner -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=7-38 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[$Tag] 完成 $(Get-Date -Format 'HH:mm:ss')"
