# Launch Q-net bias A/B experiment detached (via WMI Win32_Process Create) so it
# survives the agent turn. Keep this file PURE ASCII.
param(
    [string]$Mode = 'classprior',   # classprior | s2mix | combo
    [string]$Seeds = '100 101',
    [int]$Jobs = 2,
    [string]$Qnets = 'terminal kgeo50',
    [string]$W = '0.3 1.0',
    [string]$Tag = ''
)
$ErrorActionPreference = 'Continue'
# WMI-created process inherits a minimal PATH (no git-bash coreutils). Add them so
# bash's dirname/date/grep/sed/tee/git all resolve.
$env:PATH = 'D:\Program Files\Git\usr\bin;D:\Program Files\Git\bin;D:\Program Files\Git\cmd;' + $env:PATH
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
if ($Mode -eq '__test__') {
    Set-Content -Path (Join-Path $Repo '_tmp_qnet_launch_test.txt') -Value "ok repo=$Repo"
    exit 0
}
$env:PYTHONIOENCODING = 'utf-8'
$env:SEEDS = $Seeds
$env:JOBS = "$Jobs"
$env:QNETS = $Qnets
$env:RETRY = '1'
$env:GAME_TIMEOUT_S = '10800'
$Bash = 'D:\Program Files\Git\usr\bin\bash.exe'
$marker = Join-Path $Repo ('_tmp_qnet_launch_' + $Mode + '.log')
Add-Content -Path $marker -Value "$(Get-Date -Format 'HH:mm:ss') [launch] mode=$Mode seeds=$Seeds jobs=$Jobs qnets=$Qnets"
if ($Mode -eq 'classprior') {
    $name = if ($Tag) { $Tag } else { 'qnet_classprior_ab_small' }
    & $Bash code/run_logged.sh $name bash code/test_match/_qnet_classprior_ab.sh *>> $marker
} elseif ($Mode -eq 'combo') {
    $env:W = $W
    $name = if ($Tag) { $Tag } else { 'qnet_combo_ab_small' }
    & $Bash code/run_logged.sh $name bash code/test_match/_qnet_combo_ab.sh *>> $marker
} else {
    $env:W = $W
    $name = if ($Tag) { $Tag } else { 'qnet_s2mix_ab_small' }
    & $Bash code/run_logged.sh $name bash code/test_match/_qnet_s2mix_ab.sh *>> $marker
}
Add-Content -Path $marker -Value "$(Get-Date -Format 'HH:mm:ss') [launch] DONE mode=$Mode exit=$LASTEXITCODE"
