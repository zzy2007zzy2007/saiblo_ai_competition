# _tmp_m1c.ps1 —— M1c：修掉 M1b 的**权重被约掉**的 bug，重跑（masked 部署 + executed 标签 + 开方反频率权重）
#   M1b 的教训（如实记）：`per_sample` 除以了 `cw.sum(dim=1)` ⇒ 权重整体约掉 ⇒ 学成"恒 HOLD"、
#   整局 0 操作、16 对全败（p̂=0.0000）。本脚本用修好的 trainer 重跑，并第一次同时打开：
#     * 决策机制 `pos_pin=masked`（合法类含 HOLD 上取 argmax）
#     * 标签 `--label executed`（无落点记 HOLD）
#     * `--class-weight auto --class-weight-power 0.5`（开方反频率，温和）
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m1c] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay|train_class_outcome') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m1c] 开工 $(Get-Date -Format 'HH:mm:ss')"

& $runner -Name M1c_train -CommandLine "$Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_reserve60_exec --epochs 20 --lr 1e-3 --beta 1000000 --class-weight auto --class-weight-power 0.5 --out training_history/vprior/posnet_M1c.pt"

$env:AZAI_CKPT = 'training_history/vprior/posnet_M1c.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'masked'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M1c_masked_32'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-22 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m1c] 完成 $(Get-Date -Format 'HH:mm:ss')"
