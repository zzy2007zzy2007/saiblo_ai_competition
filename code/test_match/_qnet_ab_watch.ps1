# Watch the Q-net bias A/B small batch (6 arms x 4 games = 24 engine games) and
# auto-run the analysis when done. Launched detached via WMI Win32_Process Create.
# PURE ASCII.
param(
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Total = 24,
    [int]$PollSec = 300,
    [int]$MaxMin = 600
)
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONIOENCODING = 'utf-8'

$arms = @(
    @{ dir = 'match_results/qnet_classprior_ab/terminal'; tag = 'qcp_terminal';  label = 'A(classprior-term)' },
    @{ dir = 'match_results/qnet_classprior_ab/kgeo50';   tag = 'qcp_kgeo50';    label = 'A(classprior-kgeo50)' },
    @{ dir = 'match_results/qnet_s2mix_ab/terminal_w0.3'; tag = 'qsm_terminal_w0.3'; label = 'B(s2mix-term-w0.3)' },
    @{ dir = 'match_results/qnet_s2mix_ab/terminal_w1.0'; tag = 'qsm_terminal_w1.0'; label = 'B(s2mix-term-w1.0)' },
    @{ dir = 'match_results/qnet_s2mix_ab/kgeo50_w0.3';   tag = 'qsm_kgeo50_w0.3';  label = 'B(s2mix-kgeo50-w0.3)' },
    @{ dir = 'match_results/qnet_s2mix_ab/kgeo50_w1.0';   tag = 'qsm_kgeo50_w1.0';  label = 'B(s2mix-kgeo50-w1.0)' }
)
$log = 'match_results/_qnet_ab_watch.log'
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
