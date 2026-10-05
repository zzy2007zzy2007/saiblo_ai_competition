# _tmp_m3b_pipeline.ps1 —— M3b（预注册 §7）：50% 类探索 60 局 → ingest → Q 训练(power 0) → 离线判据 P-M3a + P-M3c
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m3b] 等采集结束…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m3b] 开工 $(Get-Date -Format 'HH:mm:ss')"

& $runner -Name M3b_ingest -CommandLine "$Py -u code/my_ai/az_intent/ingest_selfplay_classes.py --src training_history/vprior/data_sp_exp50_60 --dst training_history/vprior/data_sp_exp50_60_exec --label executed"
& $runner -Name M3b_qhead_train -CommandLine "$Py -u code/my_ai/az_intent/train_q_head.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_exp50_60_exec --out training_history/vprior/qhead_M3b.pt --epochs 30 --class-weight-power 0.0"

$chk = & $Py -u code/my_ai/az_intent/check_identification.py --data training_history/vprior/data_sp_exp50_60_exec --q training_history/vprior/qhead_M3b.pt --coin-idx 22 --limit-games 20 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_m3b_ident.txt", $chk,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $chk

$probe = & $Py -u code/my_ai/az_intent/probe_q_deployment.py --data training_history/vprior/data_pin002_100 --q training_history/vprior/qhead_M3b.pt --limit-games 12 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_m3b_probe.txt", $probe,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $probe
Write-Host "[m3b] 完成 $(Get-Date -Format 'HH:mm:ss')"
