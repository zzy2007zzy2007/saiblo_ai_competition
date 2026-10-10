# _tmp_c1_after_spamfix.ps1 —— 刷屏链之后的**最后一段自动化**：冒烟通过 ⇒ 自动跑 C1 的缩短版判据臂
#
#   背景：`_tmp_spamfix.ps1` 的最后一站是 **C1c 冒烟**（`joint + SKIP1=0 + CLASS_FEAS_MASK=1`，
#   类/位来自 phold10、**叶值换成"用刷屏数据重训过的网"**）。若冒烟**不退化**，就值得跑判据。
#
#   ⚠️ 但 C1 的**完整**判据（4 段 128 局）在 `joint` 配置下是**天级**机时 ⇒ 这里先跑**缩短版**：
#      固定段 **seed 7..38（32 对 = 64 局）**、8 线程、`MC_HORIZON=256`（已核实：开局 245 回合到真终局）。
#      **口径**：这是**缩短版**⇒ 只作"方向性读数"，**不构成任何成功声明**；且必须与**同段同协议**的对照比：
#      `MC1` 在该段的筛查读数 ≈ 0.9375（vs example）/ 未测（vs 基线）⇒ 因此本臂的对照用**固定段 7..38 上的 MC1**（若未测则如实说明）。
#
#   守卫：等 `C1c_spamfix` 出现、且**没有** "[SKIP] C1c_spamfix 冒烟即退化" 字样。
param(
    [string]$Repo = $PSScriptRoot,
    [int]$GuardTimeoutMin = 600
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo
$t0 = Get-Date
Write-Host ("[c1] 等 C1c_spamfix 冒烟结果… {0}" -f (Get-Date -Format 'HH:mm:ss'))
$smokeOk = $false
while (((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) {
    $d = 'match_results/ladder_logs/SMK_AB_C1c_spamfix_vs_baseline'
    if (Test-Path $d) {
        $txt = (Get-ChildItem $d -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object { Get-Content -Raw -Encoding UTF8 $_.FullName }) -join "`n"
        if ($txt -match 'RESULT') {                     # 至少有一局出结果
            if ($txt -match 'SKIP') { Write-Host "[c1] 冒烟退化（日志含 SKIP 标记）⇒ 不跑判据臂"; exit 0 }
            $smokeOk = $true; break
        }
    }
    Start-Sleep -Seconds 60
}
if (-not $smokeOk) { Write-Host "[c1] 守卫超时/未拿到冒烟结果 ⇒ 不跑判据臂（如实记录）"; exit 0 }
Write-Host ("[c1] 冒烟未退化 {0} ⇒ 跑 C1 缩短版判据臂（seed 7..38 = 32 对，8 线程）" -f (Get-Date -Format 'HH:mm:ss'))
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'C1c_crit32' `
    -Ckpt 'checkpoints/grafted_phold10_vspam.pt' -Mode 'joint' -Pin 'argmax' -Tclass '0.5' `
    -K 24 -Skip1 0 -ClassFeasMask 1 -Opp 'baseline' -SeedSpec '7-38' -Jobs 8 -Repo $Repo
Write-Host ("[c1] 跑完 {0}；⚠️ 缩短版读数 ⇒ 只作方向性判断，不构成成功声明" -f (Get-Date -Format 'HH:mm:ss'))
