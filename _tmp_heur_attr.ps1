# _tmp_heur_attr.ps1 —— 教师归因消融（章程②：手写启发式只作**诊断臂**）
#   ai0 = 忠实教师（启发式打分的 argmax）；ai1 = 官方 example（同一套打分、但跳过第 0 个束）
#   两臂：HEUR_SKIP_FIRST=0（忠实教师） / =1（模仿 example 的选择规则）
#   子博弈设定：双方都禁闪电（AI_BAN_LIGHTNING=1）。jobs=4（与正在跑的判据并存，总进程 ≤16）。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$env:AI_BAN_LIGHTNING = '1'
$env:AZAI_SLOW_DECISION_SEC = '60'

function Report-Arm([string]$tag) {
    foreach ($seeds in @('7-70','7-38','39-70')) {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$seeds 2>&1 | Out-String
        if ($seeds -eq '7-70') { [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false))) }
        $p = ($out -split "`n" | Select-String -Pattern '配对胜率' | Select-Object -First 1)
        Write-Host ("  [report] {0} seeds={1} | {2}" -f $tag, $seeds, ($(if ($p) { $p.Line.Trim() } else { 'p_hat=?' })))
    }
    Write-Host ("  [mech] " + (& $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1))
}

function Run-Arm([string]$name, [string]$skipFirst) {
    $env:HEUR_SKIP_FIRST = $skipFirst
    $tag = "HEUR_$name"
    Write-Host ("=== 教师臂 {0} (HEUR_SKIP_FIRST={1}) start {2}" -f $name, $skipFirst, (Get-Date -Format 'HH:mm:ss'))
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=4 --ai0=code/test_match/heur_pkg/main.py --ai1=code/test_match/ex_pkg/main.py $tokens"
    Report-Arm $tag
    Write-Host ("=== 教师臂 {0} done {1}" -f $name, (Get-Date -Format 'HH:mm:ss'))
}

Run-Arm 'H0_argmax'   '0'
Run-Arm 'H1_skiptop'  '1'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
