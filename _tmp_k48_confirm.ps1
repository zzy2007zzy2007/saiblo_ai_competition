# _tmp_k48_confirm.ps1 —— 预注册 `docs/prereg_20261011_search_up.md` 的**确认阶段**
#   S1 = 固定段后半 seed 39..70（64 局，对照 M6cROOT_128 同段 = 0.8125）
#   S2 = 冻结随机段（128 局，对照 M6cROOT_rand128 = 0.7031）
#   配置 = K48/ITERS=512（其余同 M6c-ROOT）。两个都为正且合并 ≥ +3pp 才算"方向可信"。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Which = 'S1'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

$env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '512'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '48'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
$env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
$env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
$env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
$env:AZAI_SLOW_DECISION_SEC = '30'

if ($Which -eq 'S1') {
    $tokens = ((39..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & $runner -Name K48_conf_S1 -CommandLine "$Py -u _tmp_ladder.py --tag=K48_conf_S1 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=K48_conf_S1 --seeds=39-70 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/K48_conf_S1/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
} else {
    $spec = (Get-Content -Raw -Encoding UTF8 "$Repo/training_history/vprior/_rand_seeds_spec.txt").Trim()
    $tokens = (($spec -split ',') | ForEach-Object { $_, "${_}r" }) -join ' '
    & $runner -Name K48_conf_S2 -CommandLine "$Py -u _tmp_ladder.py --tag=K48_conf_S2 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=K48_conf_S2 --seeds="$spec" 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/K48_conf_S2/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[K48 confirm $Which] done $(Get-Date -Format 'HH:mm:ss')"
