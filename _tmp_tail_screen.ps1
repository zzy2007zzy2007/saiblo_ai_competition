# _tmp_tail_screen.ps1 —— 按预注册跑 **MC_TAIL 筛查矩阵**（`docs/prereg_20261009_goal2_mc_tail_screen.md`）
#
#   7 臂（全 MC1 口径、opp=example、seed 7..38 = 32 对、jobs=4）：
#     T256(对照) / T128 / T64 / T32  ⇒ 无尾（应随 horizon 缩短而掉分，若不掉 ⇒ 推翻"必须补尾"）
#     T64V / T32V（MIX=1.0）、T64M（MIX=0.5）⇒ 补尾
#   每臂必报：min/局、p̂、无效+缺失、执先/执后、机制。**只作方向性判断，不构成成功声明。**
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Jobs = 4
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo
$arms = @(
    @{ n = 'TAIL_T256'; h = 256; tail = '0'; mix = '1.0' },
    @{ n = 'TAIL_T128'; h = 128; tail = '0'; mix = '1.0' },
    @{ n = 'TAIL_T64';  h = 64;  tail = '0'; mix = '1.0' },
    @{ n = 'TAIL_T32';  h = 32;  tail = '0'; mix = '1.0' },
    @{ n = 'TAIL_T64V'; h = 64;  tail = '1'; mix = '1.0' },
    @{ n = 'TAIL_T32V'; h = 32;  tail = '1'; mix = '1.0' },
    @{ n = 'TAIL_T64M'; h = 64;  tail = '1'; mix = '0.5' }
)
foreach ($a in $arms) {
    Write-Host ("[tail] 跑 {0}（MC_HORIZON={1} TAIL={2} MIX={3}）{4}" -f $a.n, $a.h, $a.tail, $a.mix, (Get-Date -Format 'HH:mm:ss'))
    # ⚠️ 教训（本轮已踩两次）：`-ExtraEnv @{...}` **不能经命令行传给 -File 脚本**（会被字符串化 ⇒ 绑定失败）。
    #    驱动**不管理** AZAI_MC_TAIL / _MIX ⇒ 直接在**父进程**设 env 即可（子进程继承）。
    $env:AZAI_MC_TAIL     = $(if ($a.tail -eq '1') { 'value' } else { '' })
    $env:AZAI_MC_TAIL_MIX = $a.mix
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name $a.n `
        -Ckpt 'checkpoints/grafted_sdk_A_vsearch.pt' -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' `
        -McEvery 1 -McHorizon $a.h -McScore '' -Opp 'example' -SeedSpec '7-38' -Jobs $Jobs -Repo $Repo 2>&1 |
        Select-Object -Last 6
    Write-Host ("[tail] {0} 完成 {1}" -f $a.n, (Get-Date -Format 'HH:mm:ss'))
}
Write-Host "[tail] 全部 7 臂结束；对照 T256（= 现状）与 T64V/T32V（补尾）"
