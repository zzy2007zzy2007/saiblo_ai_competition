# _tmp_q2_pipeline.ps1 —— 候选 Q2 全自动流程（预注册 docs/prereg_20261005_valuehead_rv4data.md）
#
#   等机器空闲 → ① 128 局 A1 批 + 开判据分布 dump（tag Q2_collect128）
#              → ② 判据统计（它同时是 A1 在 128 局的**重测**）
#              → ③ ingest（裸观测 → 带标签 pkl）+ 覆盖率硬读数
#              → ④ 建缓存 + 只训价值头 → ⑤ 早停检查（与对照逐张量相同则停）
#              → ⑥ A1 配置 32 局快读数 → ⑦ 判据统计
# 只执行预注册里写死的事；不做选择或结论。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 240
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

# ---------- 0) 等机器空闲（连续两次都没有 az_selfplay / ladder 进程） ----------
Write-Host "[q2] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -like '*az_selfplay.py*' -or $_.CommandLine -like '*_tmp_ladder.py*') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[q2] 开工 $(Get-Date -Format 'HH:mm:ss')"

# ---------- 1) 128 局 A1 批 + dump ----------
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_DUMP_RAW_DIR = 'training_history/vprior/dump_rv4_128'

$tokens128 = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name Q2_collect128 -CommandLine "$Py -u _tmp_ladder.py --tag=Q2_collect128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens128"

# ---------- 2) 判据统计（同时是 A1 在 128 局的重测） ----------
$out0 = & $Py -u code/test_match/analyze_paired.py --tag=Q2_collect128 --seeds=7-70 `
    --json="match_results/ladder_logs/Q2_collect128/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/Q2_collect128/_analysis.txt", $out0,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out0

# ---------- 3) ingest + 覆盖率 ----------
$ing = & $Py -u code/my_ai/az_intent/ingest_bridge_dump.py `
    --raw-dir training_history/vprior/dump_rv4_128 --ladder-tag Q2_collect128 `
    --out-dir training_history/vprior/data_vs_rv4 2>&1 | Out-String
Write-Host $ing
$cov = & $Py -u code/test_match/check_tower_coverage.py training_history/vprior/data_vs_rv4 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_q2_coverage.txt", $cov,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $cov

# ---------- 4) 建缓存 + 只训价值头 ----------
& $runner -Name Q2_cache_build -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_vs_rv4 --cache training_history/vprior/vcache_q2 --build-cache-only"
& $runner -Name Q2_valtrain -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --cache training_history/vprior/vcache_q2 --epochs 4 --lr 3e-4 --label-mode terminal --freeze-bn --out training_history/vprior/posnet_Q2_valrv4.pt"

# ---------- 5) 早停检查 ----------
$same = & $Py -u code/test_match/same_ckpt.py --a training_history/vprior/posnet_A_k5_m32.pt --b training_history/vprior/posnet_Q2_valrv4.pt 2>&1 | Out-String
Write-Host "[q2] same_ckpt: $($same.Trim())"
if ($same -match 'IDENTICAL') {
    Write-Host "[q2] ⛔ 产物与对照逐张量相同（早停回 epoch -1）⇒ 按构造为 null，不跑对局。"
    exit 0
}

# ---------- 6) 快读数：A1 配置 + 新 ckpt，seed 7..22 ----------
$env:AZAI_CKPT = 'training_history/vprior/posnet_Q2_valrv4.pt'
$tokens32 = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name Q2_valrv4_32 -CommandLine "$Py -u _tmp_ladder.py --tag=Q2_valrv4_32 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens32"

$out = & $Py -u code/test_match/analyze_paired.py --tag=Q2_valrv4_32 --seeds=7-22 `
    --json="match_results/ladder_logs/Q2_valrv4_32/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/Q2_valrv4_32/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[q2] 完成 $(Get-Date -Format 'HH:mm:ss')"
