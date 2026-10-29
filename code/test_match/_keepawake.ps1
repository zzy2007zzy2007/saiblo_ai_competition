# Windows 保活（防止「低功耗待机 / Modern Standby」把长实验冻住）
#
# 为什么需要这个（2026-10-29 踩到的真事）：
#   本机是 **S0 低功耗待机（Modern Standby）** 系统，`powercfg /change standby-timeout-ac 0`
#   并不能阻止它进 S0 待机。00:01 起的 `PIN_K2_VS_RU` 第一轮跑到 02:17 时系统进入待机，
#   07:52 才被唤醒 —— 期间**所有对局进程都被冻住 5.5 小时**：
#     · 亚军自己的日志里出现 `elapsed_us=235139862`（235 s）和 `elapsed_us=7551388975`（**2.1 h**）
#       —— 那不是它的思考时间，是墙钟被冻住的时间；
#     · bridge 的读包超时（`MVS_TIMEOUT_SECONDS=1800`）在冻结期间到期 ⇒
#       `TimeoutError: timed out reading ai_cpp_lure_v4` ⇒ `verdict=timeout` / `winner=INVALID`
#       ⇒ 那 4 局**全部作废**；
#     · 决策耗时的 `max` 被污染成上万秒（p50/p95 还好），可行性读数也不可信。
#
# 做法：`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)`
#   每 30 s 刷一次；本进程活着期间系统**不会**进待机。进程一退出（Ctrl-C / Stop-Process）
#   约束自动解除，**不改任何系统设置**（比改电源方案可逆）。
#
# 用法（当后台任务跑，实验结束就杀掉它）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File code/test_match/_keepawake.ps1
#
# 自检：启动后跑 `powercfg /requests`（需要管理员），PROCESS 一行里应当出现 powershell.exe。

Add-Type -Name KeepAwakeNative -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

$ES_CONTINUOUS       = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED  = [uint32]"0x00000001"
$ES_DISPLAY_REQUIRED = [uint32]"0x00000002"
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED

Write-Output ("[keepawake] start {0}  flags=0x{1:X8}  pid={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $flags, $PID)
$n = 0
try {
    while ($true) {
        $r = [Win32.KeepAwakeNative]::SetThreadExecutionState($flags)
        $n++
        if ($n % 20 -eq 1) {
            Write-Output ("[keepawake] tick={0} {1}  ret=0x{2:X8}" -f $n, (Get-Date -Format 'HH:mm:ss'), $r)
        }
        Start-Sleep -Seconds 30
    }
}
finally {
    # 退出时显式清掉约束（ES_CONTINUOUS 不传 ⇒ 恢复系统默认行为）
    [void][Win32.KeepAwakeNative]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Output ("[keepawake] stop {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
}
