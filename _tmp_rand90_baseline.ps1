# _tmp_rand90_baseline.ps1 —— 第二里程碑第 1 轮：**M6c-ROOT 在随机段的基线**（128 局）
#   判据/配置/成本见 docs/prereg_20261006_90_milestone.md §1.1（先写死，再跑）
#   随机段 seed = 冻结的 64 个（§2）⇒ 每个 seed 双向（N / Nr）= 128 局
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 60
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[rand90] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|az_selfplay|train_') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Write-Host "[rand90] 开工 $(Get-Date -Format 'HH:mm:ss')"

# 冻结的随机段（docs/prereg_20261006_90_milestone.md §2；不许改）
$seeds = @(1000,1091,1188,1479,1526,1621,1630,1660,1662,1982,2031,2316,2332,2407,2598,2600,
           2765,2908,2918,3017,3098,3199,3604,3678,3721,3778,3992,4229,4431,4475,4567,4717,
           4734,4820,4855,5004,5216,5473,5695,5729,5806,6054,6420,6755,6846,7038,7343,7730,
           7992,7993,8145,8500,8564,8936,9106,9125,9128,9169,9352,9358,9577,9597,9693,9927)
if ($seeds.Count -ne 64) { Write-Host "[rand90] !! seed 数=$($seeds.Count)（应为 64）"; exit 1 }
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[rand90] seed 数=$($seeds.Count)，局数=128；tag=M6cROOT_rand128"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_SLOW_DECISION_SEC = '30'

& $runner -Name M6cROOT_rand128 -CommandLine "$Py -u _tmp_ladder.py --tag=M6cROOT_rand128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
# ⚠️ 随机段 seed **不连续** ⇒ 必须传逗号列表（`1000-9927` 会被展开成 1.7 万个 seed，整批误判 INVALID）
$out = & $Py -u code/test_match/analyze_paired.py --tag=M6cROOT_rand128 --seeds=1000,1091,1188,1479,1526,1621,1630,1660,1662,1982,2031,2316,2332,2407,2598,2600,2765,2908,2918,3017,3098,3199,3604,3678,3721,3778,3992,4229,4431,4475,4567,4717,4734,4820,4855,5004,5216,5473,5695,5729,5806,6054,6420,6755,6846,7038,7343,7730,7992,7993,8145,8500,8564,8936,9106,9125,9128,9169,9352,9358,9577,9597,9693,9927 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/M6cROOT_rand128/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[rand90] 完成 $(Get-Date -Format 'HH:mm:ss')"
