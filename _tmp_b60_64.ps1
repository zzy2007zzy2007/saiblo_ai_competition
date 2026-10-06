# _tmp_d_nodown_64.ps1 —— 预注册 `docs/prereg_20261007_build_gate.md` 的诊断臂：
#   **建塔花钱门槛**（AZAI_BUILD_MIN_COINS=90：金币<60 不建塔），64 局（前缀 7..38）
#   对照：M6cROOT_128 同前缀 = 0.6562；判据（诊断臂）：≥0.7362 = 投资轴为真 / ≤0.6062 = 该轴关闭
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Tag = 'B60_buildgate_64'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
$seeds = 7..38
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[$Tag] seed 数=$($seeds.Count)，局数=64；强制升级诊断臂"

$env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
$env:AZAI_PROBE_UPG = '0'
$env:AZAI_FORCE_UPG = '0'
$env:AZAI_NO_DOWNGRADE = '0'
$env:AZAI_MC_EVERY = '5'
$env:AZAI_LIGHT_MIN_COINS = '0'
$env:AZAI_BUILD_MIN_COINS = '60'
$env:AZAI_SLOW_DECISION_SEC = '30'

& $runner -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=7-38 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[$Tag] 完成 $(Get-Date -Format 'HH:mm:ss')"


