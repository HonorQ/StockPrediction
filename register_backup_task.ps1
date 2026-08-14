<#
  Register a scheduled task to run backup.ps1 daily at 08:00
  Usage (admin recommended):
    powershell -ExecutionPolicy Bypass -File register_backup_task.ps1
#>
$TaskName = "StockPrediction_BackupDB"
$ProjectDir = "C:\Users\EDY\Desktop\StockPrediction"
$PowerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$ScriptArg = "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\backup.ps1`""

$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $ScriptArg -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At 08:00
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "[INFO] Existing task found, removing: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Description "Daily stock_history.db backup to GitHub Releases (08:00)" `
    -Trigger $Trigger `
    -Action $Action `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null

Write-Host "[OK] Registered task: $TaskName (daily 08:00)" -ForegroundColor Green