# _tmp_search_up.ps1 —— 预注册 `docs/prereg_20261011_search_up.md`
#   K48（K=48/ITERS=512，4× 搜索）与 K96（K=96/ITERS=1024，16×），各 64 局（固定段前缀 7..38）
#   对照 = M6cROOT_128 同前缀 0.6562（K=24/ITERS=256）
#   ⚠️ 路径从 $PSScriptRoot 推导（本仓库路径含中文，硬编码会被编码问题弄坏）
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
$tokens = ((7..38 | ForEach-Object { "$_", "${_}r" }) -join ' ')

function Run-Arm([string]$Name, [string]$K, [string]$Iters) {
    $env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS = $Iters; $env:AZAI_DEPTH = '4'
    $env:AZAI_K = $K; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
    $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
    $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_SLOW_DECISION_SEC = '30'
    Write-Host "[$Name] K=$K ITERS=$Iters start $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $Name -CommandLine "$Py -u _tmp_ladder.py --tag=$Name --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$Name --seeds=7-38 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Name/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}

Run-Arm 'S_k48_64' '48' '512'
Run-Arm 'S_k96_64' '96' '1024'
Write-Host "[search-up] all done $(Get-Date -Format 'HH:mm:ss')"
