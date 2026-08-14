<#
  Register Windows scheduled tasks for the stock prediction system
  premarket: every day 07:00  python run_daily.py --premarket
  intraday : every day 09:25  python run_daily.py --intraday
  afternoon: every day 14:00  python run_daily.py --afternoon

  Run as admin:
    powershell -ExecutionPolicy Bypass -File register_tasks.ps1
#>
$TaskNamePrefix = "StockPrediction"
$ProjectDir = "C:\Users\EDY\Desktop\StockPrediction"

# Locate Python
$PythonPaths = @(
    "C:\Users\EDY\AppData\Local\Programs\Python\Python313\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe"
)
$PythonExe = $null
foreach ($p in $PythonPaths) {
    if (Test-Path $p) {
        $PythonExe = $p
        break
    }
}
if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found. Set \$PythonExe manually and retry." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Using Python: $PythonExe" -ForegroundColor Green

$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $IsAdmin) {
    Write-Host "[WARN] Admin recommended (task may fail to create otherwise)" -ForegroundColor Yellow
    $choice = Read-Host "Continue? (y/n)"
    if ($choice -ne 'y') { exit }
}

# Define tasks
$Tasks = @(
    @{
        Name = "${TaskNamePrefix}_Premarket"
        Description = "Stock prediction - premarket full compute (daily 07:00)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 07:00
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --premarket" -WorkingDirectory $ProjectDir
    },
    @{
        Name = "${TaskNamePrefix}_Intraday"
        Description = "Stock prediction - intraday quick fix (daily 09:25)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 09:25
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --intraday" -WorkingDirectory $ProjectDir
    },
    @{
        Name = "${TaskNamePrefix}_Afternoon"
        Description = "Stock prediction - afternoon deep fix (daily 14:00, tail close buy decision)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 14:00
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --afternoon" -WorkingDirectory $ProjectDir
    }
)

# Register tasks
foreach ($task in $Tasks) {
    $existing = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  [WARN] Task exists, updating: $($task.Name)" -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
    }

    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    Register-ScheduledTask -TaskName $task.Name `
        -Description $task.Description `
        -Trigger $task.Trigger `
        -Action $task.Action `
        -Principal $principal `
        -Settings $settings `
        -Force | Out-Null

    Write-Host "  [OK] Registered: $($task.Name) ($($task.Description))" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Done! Tasks:" -ForegroundColor Cyan
Write-Host "  Premarket: daily 07:00" -ForegroundColor Cyan
Write-Host "  Intraday : daily 09:25" -ForegroundColor Cyan
Write-Host "  Afternoon: daily 14:00" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Manual test:" -ForegroundColor Yellow
Write-Host "  python run_daily.py --premarket" -ForegroundColor White
Write-Host "  python run_daily.py --intraday" -ForegroundColor White
Write-Host "  python run_daily.py --afternoon" -ForegroundColor White