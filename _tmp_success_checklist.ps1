# _tmp_success_checklist.ps1 —— 候选若在**固定段**≥阈值，自动走完剩下的判据流程
#
#   设计要点（避免"手工拼命令"引入不一致）：
#     随机段**不重新手写配置**，而是从候选自己的 run 归档 `training_history/runs/*_<Tag>/` 里
#     读回 `env.txt`（全部 AZAI_*）与 `cmd.txt`（`--ai0/--ai1/--jobs`），**原样复用**，只把
#     `--tag` 换成 `<Tag>_rand`、把 token 换成随机段的 64 个 seed（documents/…/random_segment_seeds.txt）。
#     ⇒ 这样"两段"的唯一差异就是 seed，符合判据里"两段都须 ≥0.80"的口径。
#
#   流程：
#     1. 读固定段 `_analysis.txt` 的 p̂；<阈值 ⇒ 直接停（不浪费机时）；
#     2. 跑随机段（128 局）；
#     3. 跑机械门 `goal_gate_vs_baseline.py`（含"两侧配置指纹必须不同"核对）；
#     4. 打印提醒：**还需无上下文独立验证**（该步必须由不知情的验证者做，不能由本脚本自证）。
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [double]$Threshold = 0.80,
    [string]$RandomSeedsFile = 'docs/random_segment_seeds.txt',
    [int]$Jobs = 8
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

# ── 1. 固定段是否过阈值 ──
$fx = "match_results/ladder_logs/$Tag/_analysis.txt"
if (-not (Test-Path $fx)) { Write-Host "[chk] 找不到固定段分析 $fx ⇒ 停"; exit 2 }
$m = Select-String -Path $fx -Pattern '配对胜率 p̂ = ([0-9.]+)' | Select-Object -First 1
$phat = if ($m) { [double]$m.Matches[0].Groups[1].Value } else { -1 }
Write-Host ("[chk] {0} 固定段 p̂ = {1}（阈值 {2}）" -f $Tag, $phat, $Threshold)
if ($phat -lt $Threshold) { Write-Host "[chk] 未过阈值 ⇒ 按规则不跑随机段（如实记'未达标'）。"; exit 0 }

# ── 2. 复用候选自己的配置跑随机段 ──
$run = Get-ChildItem training_history/runs -Directory | Where-Object Name -like "*_$Tag" | Sort-Object Name -Descending | Select-Object -First 1
if (-not $run) { Write-Host "[chk] 找不到 run 归档（无法复用配置）⇒ 停"; exit 3 }
$cmd = Get-Content -Raw -Encoding UTF8 (Join-Path $run.FullName 'cmd.txt')
$ai0 = ([regex]::Match($cmd, '--ai0=(\S+)')).Groups[1].Value
$ai1 = ([regex]::Match($cmd, '--ai1=(\S+)')).Groups[1].Value
if (-not $ai0 -or -not $ai1) { Write-Host "[chk] cmd.txt 里读不到 ai0/ai1 ⇒ 停"; exit 4 }
foreach ($line in Get-Content -Encoding UTF8 (Join-Path $run.FullName 'env.txt')) {
    if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
}
$seeds = ((Get-Content -Encoding UTF8 $RandomSeedsFile | Where-Object { $_ -and -not $_.StartsWith('#') } | Select-Object -Last 1) -split ',') |
    ForEach-Object { $_.Trim() } | Where-Object { $_ }
$tokens = ($seeds | ForEach-Object { $_, "${_}r" }) -join ' '
Write-Host ("[chk] 随机段：{0} 个 seed ⇒ {1} 局；ai0={2} ai1={3}" -f $seeds.Count, (2 * $seeds.Count), $ai0, $ai1)
& "$Repo/code/run_logged.ps1" -Name "${Tag}_rand" -CommandLine "$Py -u _tmp_ladder.py --tag=${Tag}_rand --jobs=$Jobs --ai0=$ai0 --ai1=$ai1 $tokens"

# ── 3. 机械门（两段 + 防作弊指纹核对）──
& $Py -u code/test_match/goal_gate_vs_baseline.py --tag $Tag --seeds-fixed=7-70 --seeds-random=($seeds -join ',') --threshold $Threshold
Write-Host "[chk] ⚠️ 若门通过：**还必须**由**不知情的独立验证者**复核（不能由本脚本或我自己自证）——见 docs/task_ruleV4_90_charter.md。"
