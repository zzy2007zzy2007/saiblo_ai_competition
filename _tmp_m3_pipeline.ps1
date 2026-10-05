# _tmp_m3_pipeline.ps1 —— M3：新数据（100 局类探索）→ ingest → Q 头训练 → **离线识别性判据** →（若像样）16 局快读数
#   预注册：docs/prereg_20261005_exploration_identification.md
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m3] 等采集结束…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m3] 开工 $(Get-Date -Format 'HH:mm:ss')"

& $runner -Name M3_ingest -CommandLine "$Py -u code/my_ai/az_intent/ingest_selfplay_classes.py --src training_history/vprior/data_sp_exp100 --dst training_history/vprior/data_sp_exp100_exec --label executed"
& $runner -Name M3_qhead_train -CommandLine "$Py -u code/my_ai/az_intent/train_q_head.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_exp100_exec --out training_history/vprior/qhead_M3.pt --epochs 30 --class-weight-power 0.5"

Write-Host "[m3] == 离线识别性判据（新数据 + 新 Q 头）"
$chk = & $Py -u code/my_ai/az_intent/check_identification.py --data training_history/vprior/data_sp_exp100_exec --q training_history/vprior/qhead_M3.pt --limit-games 20 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_m3_ident.txt", $chk,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $chk

Write-Host "[m3] 完成 $(Get-Date -Format 'HH:mm:ss')"
