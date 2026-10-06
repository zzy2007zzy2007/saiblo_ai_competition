# _tmp_b3_progress.ps1 —— 一次性进度探针（B3 基线 128 局）
# 用法: powershell -NoProfile -File _tmp_b3_progress.ps1 [-SleepSec 580]
# 为什么存在：单次前台命令上限 600s（实测），所以用"睡一段 + 报一次"的方式等待长跑实验，
# 每次调用返回一行紧凑状态（避免刷屏、也避免上下文被进度噪声吃掉）。
param([int]$SleepSec = 0)

if ($SleepSec -gt 0) { Start-Sleep -Seconds $SleepSec }

$Repo = 'D:/2026智能体大赛_新2'
Set-Location $Repo
$d = 'match_results/ladder_logs/B3_baseline_rulev4_128'
$files = @(Get-ChildItem $d -File -ErrorAction SilentlyContinue)
$n = @($files | Where-Object { Select-String -Path $_.FullName -Pattern '^  RESULT' -Quiet }).Count
$start = [datetime]'2026-10-05 01:29:37'
$el = ((Get-Date) - $start).TotalMinutes
$rate = if ($el -gt 0) { $n / $el } else { 0 }
$eta = if ($rate -gt 0) { (Get-Date).AddMinutes((128 - $n) / $rate).ToString('HH:mm') } else { '?' }
$dirs = @(Get-ChildItem training_history/runs -Directory | Where-Object Name -like '*B3_baseline*')
$sumLine = ''
if ($dirs.Count -gt 0) {
    $tail = @(Get-Content -Encoding UTF8 "$($dirs[0].FullName)/output.log" -Tail 40 -ErrorAction SilentlyContinue)
    $hit = $tail | Select-String -Pattern '均值 =|无效局|僵局嫌疑|先手侧胜负'
    if ($hit) { $sumLine = ($hit | ForEach-Object { $_.Line.Trim() }) -join ' | ' }
}
"完成 {0}/128  已跑 {1:N0}min  速率 {2:N2}局/min  ETA {3}  时间 {4}" -f `
    $n, $el, $rate, $eta, (Get-Date -Format 'HH:mm:ss')
if ($sumLine) { "汇总: $sumLine" }
