# _tmp_after_delta.ps1 —— 依序接续：`DELTA_old`（第二个 delta 臂）→ 刷屏数据重训链（`_tmp_spamfix.ps1`）
#
#   为什么这样排：刷屏链的 `joint` 配置单局 ~40 min、60 局要 5h+，会把"1.5–2h 就能出判据读数"的 delta 臂堵住。
#   所以顺序改成：**delta 臂先出读数，刷屏链在其后自动接上**（全部仍然自动化，机器不空转）。
#
#   守卫：等 `DELTA_vsearch` 的 128 局跑满（或分析文件出现）⇒ 跑 `DELTA_old` ⇒ 再无条件接刷屏链。
param(
    [string]$Repo = $PSScriptRoot,
    [int]$GuardTimeoutMin = 240
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

function Wait-Tag([string]$pattern, [int]$need) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) {
        $r = Get-ChildItem training_history/runs -Directory -ErrorAction SilentlyContinue |
             Where-Object Name -like $pattern | Sort-Object Name -Descending | Select-Object -First 1
        if ($r) {
            $n = (Select-String -Path (Join-Path $r.FullName 'output.log') -Pattern 'rc=' -ErrorAction SilentlyContinue |
                  Measure-Object).Count
            if ($n -ge $need) { return $true }
        }
        Start-Sleep -Seconds 60
    }
    return $false
}

Write-Host ("[after-delta] 等 DELTA_vsearch 跑满 128 局… {0}" -f (Get-Date -Format 'HH:mm:ss'))
$ok = Wait-Tag '*DELTA_vsearch*' 128
Write-Host ("[after-delta] DELTA_vsearch 结束（跑满={0}）{1} ⇒ 跑 DELTA_old" -f $ok, (Get-Date -Format 'HH:mm:ss'))

& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'DELTA_old' `
    -Ckpt 'checkpoints/grafted_sdk_A.pt' -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' `
    -McEvery 1 -McScore 'delta' -Opp 'baseline' -SeedSpec '7-70' -Jobs 8 -Repo $Repo
Write-Host ("[after-delta] DELTA_old 跑完 {0} ⇒ 接刷屏数据重训链" -f (Get-Date -Format 'HH:mm:ss'))

& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_spamfix.ps1" -Repo $Repo
Write-Host ("[after-delta] 全部结束 {0}" -f (Get-Date -Format 'HH:mm:ss'))
