# _tmp_memsample.ps1 —— 只做采样，不启动任何东西（上一版把 ladder 起挂了）
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [int]$Minutes = 45,
    [int]$SampleSec = 15,
    [string]$Csv = 'training_history/vprior/eng_mem8_samples.csv'
)
Set-Location $Repo
"t_sec,pid,role,workingset_mb,private_mb" | Set-Content -Encoding UTF8 $Csv
$t0 = Get-Date
$peak = 0
while (((Get-Date) - $t0).TotalMinutes -lt $Minutes) {
    Start-Sleep -Seconds $SampleSec
    $el = [int]((Get-Date) - $t0).TotalSeconds
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
    $tot = 0; $n = 0
    foreach ($p in $procs) {
        $cmd = [string]$p.CommandLine
        if ($cmd -notmatch 'az_bridge_ai|_tmp_ladder|rv4_pkg') { continue }
        $role = if ($cmd -match 'az_bridge_ai') { 'us' } elseif ($cmd -match '_tmp_ladder') { 'ladder' } else { 'opp' }
        $ws = [math]::Round($p.WorkingSetSize / 1MB, 1)
        $pv = [math]::Round($p.PrivatePageCount / 1MB, 1)
        "$el,$($p.ProcessId),$role,$ws,$pv" | Add-Content -Encoding UTF8 $Csv
        $tot += $ws; $n++
    }
    if ($tot -gt $peak) { $peak = $tot }
    Write-Host ("[{0}] 进程={1} 合计WS={2:N0} MB 峰值={3:N0} MB" -f (Get-Date -Format 'HH:mm:ss'), $n, $tot, $peak)
}
Write-Host "[mem] 采样结束 $(Get-Date -Format 'HH:mm:ss') 峰值=$([math]::Round($peak)) MB"
