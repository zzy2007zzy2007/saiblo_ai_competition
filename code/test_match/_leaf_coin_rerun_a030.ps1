# Re-run leaf_coin linear a0.30 (the 6 timed-out games; --resume skips the 2 done).
# WMI Win32_Process Create detached. PURE ASCII.
param([int]$Jobs = 2)
$ErrorActionPreference = 'Continue'
$env:PATH = 'D:\Program Files\Git\usr\bin;D:\Program Files\Git\bin;D:\Program Files\Git\cmd;' + $env:PATH
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONIOENCODING = 'utf-8'
$env:ALPHAS = '0.30'
$env:SEEDS = '100 101 102 103'
$env:JOBS = "$Jobs"
$env:MODE = 'linear'
$Bash = 'D:\Program Files\Git\usr\bin\bash.exe'
& $Bash code/run_logged.sh leaf_coin_a030_rerun bash code/test_match/_leaf_coin_alpha_scan.sh *>> '_tmp_leaf_coin_a030_rerun.log'
