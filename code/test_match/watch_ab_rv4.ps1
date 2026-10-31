# 等待 rule_v4 A/B 队列跑完（每 5 分钟打一条进度，跑满 48 局或进程消失就退出）
# 用法：pwsh -File code/test_match/watch_ab_rv4.ps1 [-Target 48] [-IntervalSec 300]
param(
  [int]$Target = 48,
  [int]$IntervalSec = 300,
  [string]$Dir = "match_results\value_ab_rv4\AB_RV4_FULL"
)
$ErrorActionPreference = "Stop"
Set-Location "D:\2026智能体大赛_新2_distill"
$start = Get-Date
while ($true) {
  $hits = @(Select-String -Path "$Dir\*\*.log" -Pattern "^ *RESULT" -ErrorAction SilentlyContinue)
  $n = $hits.Count
  $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
             Where-Object { $_.CommandLine -match "match_via_sdk|az_bridge_ai|rv4_pkg" }).Count
  $npz = @(Get-ChildItem training_history\collect_v2_s1 -Filter *.npz -ErrorAction SilentlyContinue).Count
  $times = @(Get-Content "$Dir\_times.tsv" -ErrorAction SilentlyContinue)
  $avg = if ($times.Count -gt 0) {
    [math]::Round((($times | ForEach-Object { [double](($_ -split "`t")[2]) }) | Measure-Object -Average).Average / 60, 1)
  } else { 0 }
  $elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
  "{0}  RESULT={1}/{2}  match进程={3}  采集={4}/256  已跑{5}min  均局{6}min" -f `
    (Get-Date -Format "HH:mm:ss"), $n, $Target, $procs, $npz, $elapsed, $avg
  if ($n -ge $Target) { "== 队列完成：$n/$Target ==" ; break }
  if ($procs -eq 0) { "== 队列进程消失（可能已停或异常），RESULT=$n/$Target ==" ; break }
  Start-Sleep -Seconds $IntervalSec
}
