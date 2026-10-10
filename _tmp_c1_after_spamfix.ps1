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
Write-Host ("[c1] 第一道守卫：等刷屏链产出 ckpt `checkpoints/grafted_phold10_vspam.pt`… {0}" -f (Get-Date -Format 'HH:mm:ss'))
while ((-not (Test-Path 'checkpoints/grafted_phold10_vspam.pt')) -and ((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) { Start-Sleep -Seconds 60 }
if (-not (Test-Path 'checkpoints/grafted_phold10_vspam.pt')) { Write-Host "[c1] 超时未见 ckpt ⇒ 不跑判据臂（如实记录）"; exit 0 }
Write-Host ("[c1] ckpt 已就绪 {0} ⇒ 再看冒烟（只看**本次之后**修改过的日志，避免陈旧目录误判）" -f (Get-Date -Format 'HH:mm:ss'))
$smokeStart = Get-Date
$smokeOk = $false
while (((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) {
    $d = 'match_results/ladder_logs/SMK_AB_C1c_spamfix_vs_baseline'
    if (Test-Path $d) {
        $txt = (Get-ChildItem $d -Filter '*.log' -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt $smokeStart } |
                ForEach-Object { Get-Content -Raw -Encoding UTF8 $_.FullName }) -join "`n"
        if (-not $txt) { Start-Sleep -Seconds 60; continue }   # 本次还没写日志 ⇒ 继续等（不拿陈旧日志判）
        if ($txt -match 'RESULT') {                     # 至少有一局出结果
            if ($txt -match 'SKIP') { Write-Host "[c1] 冒烟退化（日志含 SKIP 标记）⇒ 不跑判据臂"; exit 0 }
            $smokeOk = $true; break
        }
    }
    Start-Sleep -Seconds 60
}
if (-not $smokeOk) { Write-Host "[c1] 守卫超时/未拿到冒烟结果 ⇒ 不跑判据臂（如实记录）"; exit 0 }
Write-Host ("[c1] 冒烟未退化 {0} ⇒ 跑 C1 缩短版判据臂（seed 7..38 = 32 对，8 线程）" -f (Get-Date -Format 'HH:mm:ss'))
# ⚠️ 2026-10-09 实测：`K=24 + SAMPLE_MULT=15` 在 joint 模式下**单决策可达 ~30 min**（不可行）⇒
#    本臂用**廉价搜索**（K=4 / ITERS=32 / SAMPLE_MULT=2）以保证跑得完；
#    ⇒ **代价**：这是"**缩短版 + 廉价搜索版**"，其强度只是**下界**，**绝不能**当作部署候选的成绩（已在输出里明确标注）。
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'C1c_crit32' `
    -Ckpt 'checkpoints/grafted_phold10_vspam.pt' -Mode 'joint' -Pin 'argmax' -Tclass '0.5' `
    -K 4 -Iters 32 -SampleMult 2 -Skip1 0 -ClassFeasMask 1 -Opp 'baseline' -SeedSpec '7-38' -Jobs 8 -Repo $Repo
Write-Host ("[c1] 跑完 {0}；⚠️ 缩短版 + 廉价搜索版 ⇒ 强度只是下界，只作方向性判断，绝不构成成功声明" -f (Get-Date -Format 'HH:mm:ss'))
