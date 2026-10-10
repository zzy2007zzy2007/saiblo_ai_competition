# _tmp_spamfix.ps1 —— "补缺失数据"路线：用**刷屏局面**定向重训价值网，再重跑路 C 的机制冒烟
#
#   动机（第 60 轮复盘）：路 C 的 C1 冒烟里，`joint` 配置退化成"建塔刷屏/僵持"（实测 269 回合 / 塔 184）。
#   我原先归因"缺学习型停止模型 ⇒ 要真 RL"，但更具体、且可修的是：
#     **我们的自对弈数据是双方都用基线配置采的 ⇒ 数据里从来没有"刷屏"这种局面**
#     ⇒ 价值网没机会学到刷屏的代价；而那局 C1 冒烟里**刷屏的一方确实是输的一方**（有 RESULT 可查）。
#   因此：用 C1 的配置、**对冻结基线**采一批"刷屏对局"，用**终局胜负**当目标重训价值网（只换 value_state），
#   再重跑 C1 冒烟看刷屏是否消失。**这不是 RL，是补一类缺失的训练数据。**
#
#   流程：守卫(等控制臂 MC1_rand_ctrl) → 采集 30 seed/60 局(ai0=C1b配置+双dump, ai1=冻结基线)
#         → ingest(终局标签) → pkl→npz → 用**终局目标**训 AntWarNetwork 价值网 → 嫁接(phold10 的类/位 + 新价值)
#         → 自动重跑 C1 冒烟(机制闸门：回合数/塔数是否回归正常)
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Lo = 10061, [int]$Hi = 10090,      # 训练专用 seed 段（全新，避开判据/随机/未见段）
    [string]$Tag = 'C1SPAM_collect',
    [int]$GuardTimeoutMin = 180
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo
$guard = 'match_results/ladder_logs/MC1_rand_ctrl/_analysis.txt'
$t0 = Get-Date
Write-Host ("[spamfix] 等控制臂 MC1_rand_ctrl 完成… {0}" -f (Get-Date -Format 'HH:mm:ss'))
while (-not (Test-Path $guard) -and ((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) { Start-Sleep -Seconds 45 }
Write-Host ("[spamfix] 守卫就绪/超时 {0} ⇒ 开始采集（C1b 配置 vs 冻结基线）" -f (Get-Date -Format 'HH:mm:ss'))

New-Item -ItemType Directory -Force -Path "data/c1spam_obs/$Tag", "data/c1spam_policy/$Tag" | Out-Null
# 1) 采集：ai0 = 我们的桥（joint + 不短路 + 可行性掩蔽 + 双 dump），ai1 = base_pkg（冻结基线）
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name $Tag `
    -Ckpt 'checkpoints/grafted_phold10_vsearch.pt' -Mode 'joint' -Pin 'argmax' -Tclass '0.5' `
    -K 24 -Skip1 0 -ClassFeasMask 1 -Opp 'baseline' -SeedSpec "$Lo-$Hi" -Jobs 8 -Repo $Repo `
    -ExtraEnv @{ AZAI_DUMP_RAW_DIR = "data/c1spam_obs/$Tag"; AZAI_AZ_DUMP = "data/c1spam_policy/$Tag" }
Write-Host ("[spamfix] 采集完成 {0}" -f (Get-Date -Format 'HH:mm:ss'))

# 2) ingest（从 ladder 日志取**终局标签**）→ 3) pkl→npz
& $Py -u code/my_ai/az_intent/ingest_bridge_dump.py --raw-dir "data/c1spam_obs/$Tag" --ladder-tag $Tag --out-dir 'data/c1spam_labeled' 2>&1 | Select-Object -Last 3
& $Py -u code/my_ai/az_intent/pkl_to_value_npz.py --src 'data/c1spam_labeled' --dst 'data/c1spam_npz' 2>&1 | Select-Object -Last 2

# 4) 训练（**终局目标**；同架构；按局切分；只替换 value_state）
& $Py -u code/my_ai/train_value_antwar.py --data-dirs data/c1spam_npz --base checkpoints/grafted_sdk_A.pt `
    --out checkpoints/value_c1spam.pt --epochs 30 --batch-size 256 --lr 1e-3 --val-games 6 2>&1 | Select-Object -Last 3

# 5) 嫁接：phold10 的 class/pos + 新价值（graft_value.py 三方核对）
& $Py -u code/tools/graft_value.py --policy checkpoints/grafted_phold10.pt --value checkpoints/value_c1spam.pt --out checkpoints/grafted_phold10_vspam.pt 2>&1 | Select-Object -Last 3

# 6) 重跑 C1 冒烟（机制闸门）
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'C1c_spamfix' `
    -Ckpt 'checkpoints/grafted_phold10_vspam.pt' -Mode 'joint' -Pin 'argmax' -Tclass '0.5' `
    -K 24 -Skip1 0 -ClassFeasMask 1 -Opp 'baseline' -SeedSpec '3-4' -Jobs 2 -Repo $Repo
Write-Host ("[spamfix] 全流程结束 {0} —— 看 C1c 冒烟的机制行：回合数/塔数是否回归正常" -f (Get-Date -Format 'HH:mm:ss'))
