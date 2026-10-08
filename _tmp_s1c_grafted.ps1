# _tmp_s1c_grafted.ps1 —— S1c：评估"蒸馏策略 + 嫁接 ckpt"在双方禁闪电子博弈里的表现
#   三臂（都指向嫁接后的 ckpt）：
#     G1 = pos-only + mc_quiet + 宽菜单（与 S0=0.3828 直接可比）
#     G2 = pos-only + masked + 窄菜单（类头在合法类上 argmax ⇒ 检验"类头现在有信号了吗"）
#     G3 = joint + masked + TCLASS=1.0（"新游戏"模式：类由头分布采样 + 搜位置）
#   每臂先 2 局冒烟，我方建塔 ≤0.5/局 ⇒ 判退化、跳过（省机时）。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Ckpt = 'checkpoints/grafted_sdk_A.pt'
)
Set-Location $Repo
function Set-Base {
    $env:AZAI_CKPT = $Ckpt
    $env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
    $env:AZAI_MODE='pos-only'; $env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'
    $env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
    $env:AZAI_POSPIN='mc_quiet'; $env:AZAI_TCLASS='0.5'
    $env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'
    $env:AZAI_MC_SCORE=''; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
    $env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_ROLLOUT_CLASSES='4'; $env:AZAI_MC_NO_LIGHT='0'
    $env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'; $env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'
    $env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'
    $env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
    $env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
    $env:AZAI_SLOW_DECISION_SEC='60'
    $env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'; $env:AI_BAN_LIGHTNING='1'
}
function Our-Builds([string]$tag) {
    $line = & $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1
    if (-not $line) { return -1.0 }
    $m = [regex]::Match($line, '^\S+\s+\d+\s+\S+\s+([\d.]+)/')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return -1.0
}
function Report-Arm([string]$tag) {
    foreach ($seeds in @('7-70','7-38','39-70')) {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$seeds 2>&1 | Out-String
        if ($seeds -eq '7-70') { [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false))) }
        $p = ($out -split "`n" | Select-String -Pattern '配对胜率' | Select-Object -First 1)
        Write-Host ("  [report] {0} seeds={1} | {2}" -f $tag, $seeds, ($(if ($p) { $p.Line.Trim() } else { 'p_hat=?' })))
    }
    Write-Host ("  [mech] " + (& $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1))
}
function Run-Arm([string]$name, [string]$mode, [string]$pin, [string]$tclass, [string]$menu) {
    Set-Base
    $env:AZAI_MODE = $mode; $env:AZAI_POSPIN = $pin; $env:AZAI_TCLASS = $tclass
    if ($menu -eq 'narrow') { $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0' }
    $tag = "S1C_$name"
    Write-Host ("=== arm {0} (ckpt={1} mode={2} pin={3} tclass={4} menu={5}) smoke {6}" -f $name,$Ckpt,$mode,$pin,$tclass,$menu,(Get-Date -Format 'HH:mm:ss'))
    & $Py -u _tmp_ladder.py --tag="SMK_$tag" --jobs=2 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py 7 7r 2>&1 | Out-Null
    $b = Our-Builds "SMK_$tag"
    Write-Host ("  [smoke] 我方建塔/局 = {0}" -f $b)
    if ($b -le 0.5) { Write-Host ("  [SKIP] {0} 冒烟即退化 ⇒ 不跑 128 局" -f $name); return }
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py $tokens"
    Report-Arm $tag
    Write-Host ("=== arm {0} done {1}" -f $name,(Get-Date -Format 'HH:mm:ss'))
}
Run-Arm 'G1_posonly_wide' 'pos-only' 'mc_quiet' '0.5' 'wide'
Run-Arm 'G2_posonly_masked' 'pos-only' 'masked' '0.5' 'narrow'
Run-Arm 'G3_joint_masked' 'joint' 'masked' '1.0' 'narrow'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
