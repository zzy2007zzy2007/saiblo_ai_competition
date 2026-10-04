# _tmp_c1_after_baseline.ps1 —— 等 B3 基线跑完后，**按预注册**自动跑"配置阶梯"的第一段：
#   ① A1 成本闸门（1 局）→ ② A1 筛查（32 局 = seed 7..22 镜像）→ ③ 判据统计
#
# 依据：`docs/prereg_20261005_search_config_ladder.md`（§2 臂定义、§3 成本闸门、§4 判据）
#   A1 = 历史标准配置：AZAI_K=24 AZAI_SAMPLE_MULT=15 AZAI_MODE=pos-only AZAI_SKIP1=1
#        （其余沿用钉死值；ckpt 不动）
# ⚠️ 本脚本**只做预注册里已经写死的事**：不选臂、不改判据、不做任何"成功"宣称。
#    它的输出只是"读数"，结论由 agent 写进 experiment_log / goal_register。
# A2（厚 joint）**故意不在这里自动跑**：自检说它 ~9–16 min/局，等 A1 的读数出来再决定要不要花。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$GateMaxMin = 6,          # 预注册 §3：A1 > 6 min/局 ⇒ 停
    [int]$WaitTimeoutMin = 420
)

Set-Location $Repo

# ---- 钉死协议（章程 §2）+ A1 的四个变量 ----
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
$env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'

$runner = "$Repo/code/run_logged.ps1"
$seeds32 = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
$seeds16 = ((7..14) | ForEach-Object { "$_", "${_}r" }) -join ' '

function Latest-RunDir([string]$name) {
    @(Get-ChildItem "$Repo/training_history/runs" -Directory |
        Where-Object { $_.Name -like "*_$name" } | Sort-Object Name -Descending)[0].FullName
}

# ---------- 0) 等基线跑完 ----------
Write-Host "[c1] 等 B3 基线退出（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin)
$saw = $false
while ((Get-Date) -lt $deadline) {
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*_tmp_ladder.py*' })
    if ($running.Count -gt 0) { $saw = $true }
    elseif ($saw) { Write-Host "[c1] 基线已退出 $(Get-Date -Format 'HH:mm:ss')"; break }
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 15

# ---------- 1) 成本闸门：1 局 ----------
Write-Host "[c1] A1 成本闸门（token 7）…"
& $runner -Name C1_a1_gate -CommandLine "$Py -u _tmp_ladder.py --tag=C1_a1_gate --jobs=1 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 7"
$gateDir = Latest-RunDir 'C1_a1_gate'
$gateLog = Get-Content -Encoding UTF8 "$gateDir/output.log" -ErrorAction SilentlyContinue
$mins = $null
foreach ($ln in $gateLog) { if ($ln -match '^\[7\] rc=(-?\d+)\s+([\d.]+) min') { $mins = [double]$Matches[2] } }
$gateInvalid = @($gateLog | Select-String -Pattern 'winner=INVALID' -Quiet).Count -gt 0
Write-Host "[c1] 闸门用时 = $mins min（阈值 $GateMaxMin）; INVALID=$gateInvalid"
if ($null -eq $mins -or $mins -gt $GateMaxMin -or $gateInvalid) {
    Write-Host "[c1] ⛔ 闸门未过（太慢或无效局）⇒ 按预注册 §3 停止，不跑筛查；等 agent 判断。"
    exit 0
}

# ---------- 2) A1 筛查：32 局（seed 7..22 镜像）----------
Write-Host "[c1] A1 筛查 32 局… $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name C1_a1_hist32 -CommandLine "$Py -u _tmp_ladder.py --tag=C1_a1_hist32 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $seeds32"

# ---------- 3) 判据统计 ----------
Write-Host "[c1] 判据统计…"
$out = & $Py -u code/test_match/analyze_paired.py --tag=C1_a1_hist32 --seeds=7-22 `
    --json="match_results/ladder_logs/C1_a1_hist32/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/C1_a1_hist32/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[c1] 完成 $(Get-Date -Format 'HH:mm:ss')  —— A2（厚 joint）等 agent 决定"
