# _tmp_repro_m6c.ps1 —— 零漂移回归：与 20261006_120438_M6cROOT_128 的 cmd.txt/env 逐字相同，只换 tag
param([string]$Repo = $PSScriptRoot, [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe')
Set-Location $Repo
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_DEPTH = '4'; $env:AZAI_ITERS = '256'; $env:AZAI_K = '24'
$env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MODE = 'pos-only'; $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_POSPRIOR = 'off'
$env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_TPOS = '1.0'
$env:AZAI_TRACE = '1'; $env:AZAI_VERIFY = '1'
# 我这十几轮加过的所有开关，全部显式置"关"，确保与当初一致
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_UPG = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'
$env:AZAI_MC_CELLS = '0'; $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
$env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
$env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
$env:AZAI_SLOW_DECISION_SEC = '20'
$tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
& "$Repo/code/run_logged.ps1" -Name REPRO_m6c_128 -CommandLine "$Py -u _tmp_ladder.py --tag=REPRO_m6c_128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
Write-Host "[repro] done $(Get-Date -Format 'HH:mm:ss')"
