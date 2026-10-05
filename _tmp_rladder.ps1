# _tmp_rladder.ps1 —— 预注册 `docs/prereg_20261005_reserve_ladder.md` 的储备门槛阶梯
#   R120 / R180(对照) / R240 / R360，各在 **seed 71..102（32 局）** 上筛（**不碰判据列表 7..70**）
#   决策规则：筛选最高者若比对照高 ≥0.08 ⇒ 对胜者跑 128 局验收（seed 7..70）
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[rl] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[rl] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'reserve'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'

$tokens = ((71..102) | ForEach-Object { "$_", "${_}r" }) -join ' '
$res = [ordered]@{}
foreach ($rv in 120, 180, 240, 360) {
    $env:AZAI_RESERVE = "$rv"
    $tag = "R${rv}_s71_102"
    Write-Host "[rl] == $tag 开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=71-102 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
    $m = [regex]::Match($out, '配对胜率 p̂ = ([0-9.]+)')
    if ($m.Success) { $res[$rv] = [double]$m.Groups[1].Value }
}

Write-Host "[rl] 筛选结果: $($res | Out-String)"
$base = 180
$best = ($res.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1)
Write-Host "[rl] best=$($best.Key) p=$($best.Value)  control(180)=$($res[180])"
if ($best.Key -ne $base -and $best.Value -ge ($res[$base] + 0.08)) {
    $env:AZAI_RESERVE = "$($best.Key)"
    $tag = "R$($best.Key)_128"
    Write-Host "[rl] 胜者回判据列表验收: $tag（seed 7..70）$(Get-Date -Format 'HH:mm:ss')"
    $t128 = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $t128"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-70 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
} else {
    Write-Host "[rl] 未达 +0.08 ⇒ 保留储备 180，本阶梯只记档。"
}
Write-Host "[rl] 完成 $(Get-Date -Format 'HH:mm:ss')"
