# _tmp_m1b.ps1 —— 方法侧 M1b：**修决策机制（masked）+ 修标签（executed）+ 类频率反加权** → 关规则部署
#   依据（同轮实测）：
#   * M1：蒸馏只改类头（跨局面 std 0.024→0.178），但 `pos_pin=argmax` 部署行为逐字不变
#     （原因：训练目标 99.3% 是闪电 ⇒ argmax 恒闪电；而 `playable` 又会崩成"见缝就花"）。
#   * ⇒ 修两处：① **决策机制** `pos_pin=masked`（在**合法类（含 HOLD）**上取 argmax；已冒烟：
#     与 A1 行为一致 ⇒ 无回归，且"HOLD"重新成为策略可选项）；
#     ② **标签口径** `--label executed`（没落点记 HOLD 23；实测分布 HOLD 96.4% / 闪电 2.9% / 花钱 0.8%）
#     ⇒ 必须配 **`--class-weight auto`**（反频率），否则 CE 只会学到"恒 HOLD"。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m1b] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay|train_class_outcome') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m1b] 开工 $(Get-Date -Format 'HH:mm:ss')"

& $runner -Name M1b_train -CommandLine "$Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_reserve60_exec --epochs 20 --lr 1e-3 --beta 1000000 --class-weight auto --out training_history/vprior/posnet_M1b_masked.pt"

$env:AZAI_CKPT = 'training_history/vprior/posnet_M1b_masked.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'masked'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M1b_masked_32'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-22 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m1b] 完成 $(Get-Date -Format 'HH:mm:ss')"
