# Watch the Q-net combo arms (3 arms x 4 games = 12 engine games) and auto-run
# the analysis when done. WMI Win32_Process Create detached. PURE ASCII.
param(
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Total = 12,
    [int]$PollSec = 300,
    [int]$MaxMin = 600
)
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONIOENCODING = 'utf-8'

$arms = @(
    @{ dir = 'match_results/qnet_combo_ab/combo';        tag = 'qcombo_combo';        label = 'combo(classprior+s2mix-w0.3)' },
    @{ dir = 'match_results/qnet_combo_ab/steps48prior'; tag = 'qcombo_steps48prior'; label = 'steps48prior(CAND_STEPS=48+classprior)' },
    @{ dir = 'match_results/qnet_combo_ab/steps48mix';   tag = 'qcombo_steps48mix';   label = 'steps48mix(CAND_STEPS=48+s2mix-w0.3)' }
)
$log = 'match_results/_qnet_combo_watch.log'
$t0 = Get-Date

function Count-Engine {
    $n = 0
    foreach ($a in $arms) {
        Get-ChildItem (Join-Path $a.dir 'seed*_p*.log') -ErrorAction SilentlyContinue | ForEach-Object {
            if (Select-String -Path $_.FullName -Pattern 'verdict=engine' -Quiet) { $n++ }
        }
    }
    return $n
}

Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] start, waiting for $Total engine games"
while ($true) {
    $done = Count-Engine
    $el = [int]((Get-Date) - $t0).TotalMinutes
    Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] engine=$done/$Total elapsed=${el}min"
    if ($done -ge $Total) { break }
    if ($el -ge $MaxMin) {
        Add-Content -Path $log -Value "[watch] hit ${MaxMin}min cap, analyzing whatever is present"
        break
    }
    Start-Sleep -Seconds $PollSec
}

foreach ($a in $arms) {
    $out = Join-Path $Repo ($a.dir + '/_analysis.txt')
    $json = Join-Path $Repo ($a.dir + '/summary.json')
    $cmd = "$Py -u code/test_match/analyze_qnet_prior_ab.py --out-dir $($a.dir) --stderr-tag $($a.tag) --label-a `"$($a.label)`" --seeds 100-101 --json `"$json`" > `"$out`" 2>&1"
    & cmd.exe /d /c $cmd
    Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] analyzed $($a.dir) -> $out"
}
Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') [watch] ALL_DONE"
