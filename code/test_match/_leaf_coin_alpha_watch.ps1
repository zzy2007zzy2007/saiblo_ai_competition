# Watch the leaf_coin_alpha scan (6 alphas x 8 games = 48 engine games) and
# auto-run the analysis when done. Launched detached via WMI Win32_Process Create
# so it survives the agent turn.
# NOTE: keep this file PURE ASCII (no non-ASCII chars) - Windows PowerShell 5.1
# reads no-BOM files as ANSI and mangles UTF-8 comments, which breaks parsing.
param(
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Total = 48,
    [int]$PollSec = 300,
    [int]$MaxMin = 900
)
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONIOENCODING = 'utf-8'
$outRoot = Join-Path $Repo 'match_results\leaf_coin_alpha'
$log = Join-Path $outRoot '_watch.log'
$t0 = Get-Date

function Count-Engine {
    $n = 0
    Get-ChildItem (Join-Path $outRoot 'a*\seed*_p*.log') -ErrorAction SilentlyContinue | ForEach-Object {
        if (Select-String -Path $_.FullName -Pattern "verdict=engine" -Quiet) { $n++ }
    }
    return $n
}

Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [watch] start, waiting for $Total engine games"
while ($true) {
    $done = Count-Engine
    $el = [int]((Get-Date) - $t0).TotalMinutes
    Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] engine=$done/$Total elapsed=${el}min"
    if ($done -ge $Total) { break }
    if ($el -ge $MaxMin) {
        Add-Content -Path $log -Value "[watch] hit ${MaxMin}min cap, running analysis on whatever is present"
        break
    }
    Start-Sleep -Seconds $PollSec
}

$analysis = Join-Path $outRoot '_analysis.txt'
$jsonOut = Join-Path $outRoot '_summary.json'
$cmdline = "$Py -u code/test_match/analyze_leaf_coin_alpha.py --alphas 0,0.02,0.037,0.07,0.15,0.30 --seeds 100-103 --json `"$jsonOut`" > `"$analysis`" 2>&1"
& cmd.exe /d /c $cmdline
Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] analysis done -> $analysis"
Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] ALL_DONE"
