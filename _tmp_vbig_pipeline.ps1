# _tmp_vbig_pipeline.ps1 —— 按用户 2026-10-05 的经验（"100 局大概率不够，200-400 局勉强能用"）
# 做"价值头数据规模"实验：
#   等机器空闲 → ① 采 400 局 vs rule_v4（A1 配置、开判据分布 dump；seed 71..270 双向，**非判据 seed**）
#              → ② 判据统计（这 400 局本身也是一次大样本 A1 重测，但**不作为判据读数**：seed 不在钉死列表里）
#              → ③ ingest → ④ 建缓存 → ⑤ 训价值头（lr 1e-4，12 epoch，只训 value_state）
#              → ⑥ 与对照比较（相同则停）→ ⑦ A1 配置 32 局快读数（seed 7..22）+ 判据统计
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[vbig] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -like '*_tmp_ladder.py*' -or $_.CommandLine -like '*az_selfplay.py*') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[vbig] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_RESERVE = '90'; $env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '0'
$env:AZAI_CLASSPIN_PROB = '0'
$env:AZAI_DUMP_RAW_DIR = 'training_history/vprior/dump_rv4_big'

$tokens = ((71..270) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name V_BIG_collect400 -CommandLine "$Py -u _tmp_ladder.py --tag=V_BIG_collect400 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"

# ingest（标签从 ladder 日志的 base_hp 反推）
$ing = & $Py -u code/my_ai/az_intent/ingest_bridge_dump.py `
    --raw-dir training_history/vprior/dump_rv4_big --ladder-tag V_BIG_collect400 `
    --out-dir training_history/vprior/data_vs_rv4_big 2>&1 | Out-String
Write-Host $ing
$cov = & $Py -u code/test_match/check_tower_coverage.py training_history/vprior/data_vs_rv4_big 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_vbig_coverage.txt", $cov,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $cov

& $runner -Name V_BIG_cache -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_vs_rv4_big --cache training_history/vprior/vcache_vbig --build-cache-only"
& $runner -Name V_BIG_valtrain -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --cache training_history/vprior/vcache_vbig --epochs 12 --lr 1e-4 --label-mode terminal --freeze-bn --out training_history/vprior/posnet_VBIG_valrv4.pt"

$same = & $Py -u code/test_match/same_ckpt.py --a training_history/vprior/posnet_A_k5_m32.pt --b training_history/vprior/posnet_VBIG_valrv4.pt 2>&1 | Out-String
Write-Host "[vbig] same_ckpt: $($same.Trim())"
if ($same -match 'IDENTICAL') { Write-Host "[vbig] ⛔ 与对照逐张量相同 ⇒ 按构造为 null，不跑对局。"; exit 0 }

$env:AZAI_CKPT = 'training_history/vprior/posnet_VBIG_valrv4.pt'
$t32 = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name VBIG_valrv4_32 -CommandLine "$Py -u _tmp_ladder.py --tag=VBIG_valrv4_32 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $t32"

$out = & $Py -u code/test_match/analyze_paired.py --tag=VBIG_valrv4_32 --seeds=7-22 `
    --json="match_results/ladder_logs/VBIG_valrv4_32/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/VBIG_valrv4_32/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[vbig] 完成 $(Get-Date -Format 'HH:mm:ss')"
