# _tmp_delta_arm.ps1 —— 排队跑 **`MC_SCORE=delta`**（rollout + 价值网差分）的**判据尺度**臂
#
#   依据：
#   - 前史（2026-10-08，prereg_20261008_mc_value_delta.md）在**旧设定**里测过，判读是「**机制反向**」（建塔数没降）；
#   - 在本子博弈里**只跑过 `value_only`**（0.3438，有害）⇒ **`delta` 未测**；
#   - 新发现：**固定段的失分集中在长局**（短 0.790 / 长 0.576），而"安静 rollout"的偏置在长局最严重；
#     `delta` 的设计目的正是「**只修掉'安静 rollout 看不见花钱代价'的短视**」（potential-based shaping Φ=V）
#     ⇒ 所以它是"长局病灶"的**机制对口候选**。
#
#   我的预测（先写下，事后对照）：
#   - **胜负**：偏**中性到有害**（它与已被证伪的 `value_only` 共享"V 差分"项；前史还出现"机制反向"）；
#   - **机制**：**建塔数应下降**、长局胜率应改善 ⇒ 这比分数更重要（若机制没动 ⇒ 即使分数好看也不作结论）。
#
#   守卫：等 `C1SPAM_collect` 的 ladder 跑完（60 局）腾出槽位 ⇒ 先跑「搜索值网」版，再跑「旧网」版。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$GuardTimeoutMin = 480,
    [int]$Jobs = 6
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo
$t0 = Get-Date
Write-Host ("[delta] 等 C1SPAM_collect 采集完成… {0}" -f (Get-Date -Format 'HH:mm:ss'))
while (((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) {
    $r = Get-ChildItem training_history/runs -Directory | Where-Object Name -like '*C1SPAM_collect*' | Sort-Object Name -Descending | Select-Object -First 1
    if ($r) {
        $done = (Select-String -Path (Join-Path $r.FullName 'output.log') -Pattern 'rc=' | Measure-Object).Count
        if ($done -ge 60) { Write-Host "[delta] 采集已完成（$done 局）⇒ 开始"; break }
    }
    Start-Sleep -Seconds 60
}
foreach ($ck in @('checkpoints/grafted_sdk_A_vsearch.pt', 'checkpoints/grafted_sdk_A.pt')) {
    $name = 'DELTA_' + (($ck -split '/')[-1] -replace 'grafted_sdk_A_?', '' -replace '\.pt$', '')
    if ($name -eq 'DELTA_') { $name = 'DELTA_old' }
    Write-Host ("[delta] 跑 {0}（ckpt={1}，MC_SCORE=delta，MC_EVERY=1，vs 冻结基线，128 局，jobs={2}）{3}" -f $name, $ck, $Jobs, (Get-Date -Format 'HH:mm:ss'))
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name $name `
        -Ckpt $ck -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' -McEvery 1 -McScore 'delta' `
        -Opp 'baseline' -SeedSpec '7-70' -Jobs $Jobs -Repo $Repo
    Write-Host ("[delta] {0} 完成 {1}" -f $name, (Get-Date -Format 'HH:mm:ss'))
}
Write-Host "[delta] 两个 delta 臂跑完；对照：MC1=0.6797、MC1+搜索值网=0.7188（固定段）"
