# Watch the shape_pin vs pin+flat A/B (16 games) and auto-run the analysis when done.
# Launched detached via WMI Win32_Process Create so it survives the agent turn.
# NOTE: keep this file PURE ASCII (no non-ASCII chars) - Windows PowerShell 5.1 reads
# no-BOM files as ANSI and mangles UTF-8 comments, which breaks parsing.
param(
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Total = 16,
    [int]$PollSec = 300,
    [int]$MaxMin = 600
)
$ErrorActionPreference = 'Continue'
# Repo root = two levels above this script (code/test_match/.. -> repo).
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONIOENCODING = 'utf-8'
$outDir = Join-Path $Repo 'match_results\shape_pin_ab'
$log = Join-Path $outDir '_watch.log'
$t0 = Get-Date

function Count-Engine {
    $n = 0
    Get-ChildItem (Join-Path $outDir 'seed*.log') -ErrorAction SilentlyContinue | ForEach-Object {
        if (Select-String -Path $_.FullName -Pattern "RESULT seed=.*verdict=engine" -Quiet) { $n++ }
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

$analysis = Join-Path $outDir '_analysis.txt'
$jsonOut = Join-Path $outDir '_summary.json'
$cmdline = "$Py -u code/test_match/analyze_shape_pin_ab.py --seeds 100-107 --json `"$jsonOut`" > `"$analysis`" 2>&1"
& cmd.exe /d /c $cmdline
Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] analysis done -> $analysis"
