# _tmp_subgame_pospin2.ps1 —— 修正版：C0 用 MC_EVERY=5；主角改 MODE=joint（类由头分布采样）
#   ⚠️ 每个臂先跑 2 局冒烟，若"我方建塔=0"（退化）则**跳过该臂并记录**，不浪费 128 局。
param([string]$Repo = $PSScriptRoot, [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe')
Set-Location $Repo
function Set-Base {
    $env:AZAI_CKPT='checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
    $env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'
    $env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
    $env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'; $env:AZAI_MC_SCORE=''
    $env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
    $env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
    $env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'; $env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'
    $env:AZAI_MC_NO_LIGHT='0'; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
    $env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_ROLLOUT_CLASSES='4'; $env:AZAI_SLOW_DECISION_SEC='60'
    $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0'
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
function Run-Arm([string]$name, [string]$mode, [string]$pin, [string]$tclass) {
    Set-Base
    $env:AZAI_MODE = $mode; $env:AZAI_POSPIN = $pin; $env:AZAI_TCLASS = $tclass
    $tag = "PS2_$name"
    Write-Host ("=== arm {0} (mode={1} pin={2} tclass={3}) smoke {4}" -f $name,$mode,$pin,$tclass,(Get-Date -Format 'HH:mm:ss'))
    & $Py -u _tmp_ladder.py --tag="SMK_$tag" --jobs=2 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py 7 7r 2>&1 | Out-Null
    $b = Our-Builds "SMK_$tag"
    Write-Host ("  [smoke] 我方建塔/局 = {0}" -f $b)
    if ($b -le 0.5) { Write-Host ("  [SKIP] {0} 冒烟即退化（我方不出招） ⇒ 不跑 128 局" -f $name); return }
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py $tokens"
    Report-Arm $tag
    Write-Host ("=== arm {0} done {1}" -f $name,(Get-Date -Format 'HH:mm:ss'))
}
Run-Arm 'C0r_quiet_narrow' 'pos-only' 'mc_quiet' '0.5'
Run-Arm 'J1_joint'         'joint'    'masked'   '0.5'
Run-Arm 'J2_joint_t1'      'joint'    'masked'   '1.0'
Run-Arm 'J3_joint_argmax'  'joint'    'argmax'   '1.0'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
