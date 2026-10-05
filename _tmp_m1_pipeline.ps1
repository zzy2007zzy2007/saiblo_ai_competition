# _tmp_m1_pipeline.ps1 —— 方法侧 M1（预注册 docs/prereg_20261005_distill_reserve_to_policy.md）
#   等机器空闲 → ① 自对弈 60 局（**开储备规则**，采集"规则会怎么选类"的数据，seed 50000+）
#              → ② ingest（pkl -> chosen_cls/class_mask）
#              → ③ 只训 class_state（纯蒸馏 CE：--beta 1e6）
#              → ④ 快读数：**关掉规则**（POSPIN=argmax）+ 新类头，seed 7..22（32 局）
#              → ⑤ 判据统计
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 240
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m1] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m1] 开工 $(Get-Date -Format 'HH:mm:ss')"

# ① 自对弈采集（**带储备规则**）
$cl = "$Py -u code/my_ai/az_intent/az_selfplay.py --checkpoint training_history/vprior/posnet_A_k5_m32.pt " +
      "--games 60 --workers 8 --seed 5 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 1.0 " +
      "--k 24 --sample-mult 15 --search-mode pos-only --skip-single-candidate " +
      "--pos-pin reserve --reserve-coins 180 --native-engine " +
      "--out-dir training_history/vprior/data_sp_reserve60"
& $runner -Name M1_collect60_reserve -CommandLine $cl

# ② ingest
$ing = & $Py -u code/my_ai/az_intent/ingest_selfplay_classes.py `
    --src training_history/vprior/data_sp_reserve60 --dst training_history/vprior/data_sp_reserve60_cls 2>&1 | Out-String
Write-Host $ing

# ③ 只训类头（纯蒸馏：beta 极大 ⇒ 权重趋均匀的 CE）
& $runner -Name M1_distill_train -CommandLine "$Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_reserve60_cls --epochs 20 --lr 1e-3 --beta 1000000 --out training_history/vprior/posnet_M1_distill.pt"

# ④ 快读数：**POSPIN=argmax ⇒ 部署里没有任何手写规则**
$env:AZAI_CKPT = 'training_history/vprior/posnet_M1_distill.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M1_distill_32'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-22 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m1] 完成 $(Get-Date -Format 'HH:mm:ss')"
