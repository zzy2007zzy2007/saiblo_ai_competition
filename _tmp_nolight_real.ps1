# _tmp_nolight_real.ps1 —— 预注册 `docs/prereg_20261018_mc_nolight_real.md`
#   NL-R 臂：真实对局 + AZAI_MC_NO_LIGHT=1 + MC_UPG=1 + MC_TOPK=3
#   判据尺度：固定段 128 局（7..70）+ 随机段 128 局。对照 = M6cROOT_128 / M6cROOT_rand128。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

function Set-Cfg {
    $env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'; $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
    $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_MARGIN = '0'
    $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_BAN_CLASSES = ''; $env:AZAI_BAN_OPS = ''      # ★ 真实对局：闪电不被禁
    $env:AZAI_SLOW_DECISION_SEC = '30'
    # ★ 本臂的三个开关
    $env:AZAI_MC_NO_LIGHT = '1'; $env:AZAI_MC_UPG = '1'; $env:AZAI_MC_TOPK = '3'
}

Set-Cfg
$fixed = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[NL-R] 固定段 128 局 start $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name NLR_fixed128 -CommandLine "$Py -u _tmp_ladder.py --tag=NLR_fixed128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $fixed"
$o1 = & $Py -u code/test_match/analyze_paired.py --tag=NLR_fixed128 --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/NLR_fixed128/_analysis.txt", $o1, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o1

Set-Cfg
$spec = (Get-Content -Raw -Encoding UTF8 "$Repo/training_history/vprior/_rand_seeds_spec.txt").Trim()
$rand = (($spec -split ',') | ForEach-Object { $_, "${_}r" }) -join ' '
Write-Host "[NL-R] 随机段 128 局 start $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name NLR_rand128 -CommandLine "$Py -u _tmp_ladder.py --tag=NLR_rand128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $rand"
$o2 = & $Py -u code/test_match/analyze_paired.py --tag=NLR_rand128 --seeds="$spec" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/NLR_rand128/_analysis.txt", $o2, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o2
Write-Host "[NL-R] done $(Get-Date -Format 'HH:mm:ss')"
