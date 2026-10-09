# _tmp_s1c_random.ps1 —— S1c 第②段（随机段）：与 G1 完全同配置，跑项目已冻结的 64 个随机 seed ×双向 = 128 局
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Ckpt = 'checkpoints/grafted_sdk_A.pt'
)
Set-Location $Repo
$env:AZAI_CKPT = $Ckpt
$env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
$env:AZAI_MODE='pos-only'; $env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'
$env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
$env:AZAI_POSPIN='mc_quiet'; $env:AZAI_TCLASS='0.5'
$env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'
$env:AZAI_MC_SCORE=''; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
$env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_ROLLOUT_CLASSES='4'; $env:AZAI_MC_NO_LIGHT='0'
$env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'; $env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'
$env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'
$env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
$env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
$env:AZAI_SLOW_DECISION_SEC='60'
$env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'; $env:AI_BAN_LIGHTNING='1'
$seeds = @(1000,1091,1188,1479,1526,1621,1630,1660,1662,1982,2031,2316,2332,2407,2598,2600,2765,2908,2918,3017,3098,3199,3604,3678,3721,3778,3992,4229,4431,4475,4567,4717,4734,4820,4855,5004,5216,5473,5695,5729,5806,6054,6420,6755,6846,7038,7343,7730,7992,7993,8145,8500,8564,8936,9106,9125,9128,9169,9352,9358,9577,9597,9693,9927)
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
$seedSpec = ($seeds -join ',')
Write-Host ("=== S1C 随机段 start {0}  seeds={1} 个" -f (Get-Date -Format 'HH:mm:ss'), $seeds.Count)
& "$Repo/code/run_logged.ps1" -Name S1C_G1_rand128 -CommandLine "$Py -u _tmp_ladder.py --tag=S1C_G1_rand128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=S1C_G1_rand128 --seeds=$seedSpec 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/S1C_G1_rand128/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
$out -split "`n" | Select-String -Pattern '配对胜率|完整配对数|有效性门槛|判词来源' | ForEach-Object { "  " + $_.Line.Trim() }
Write-Host ("=== S1C 随机段 done {0}" -f (Get-Date -Format 'HH:mm:ss'))
