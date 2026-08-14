<#
自动注册 Windows 计划任务
  盘前模式: 每天 7:00  python run_daily.py --premarket
  盘中模式: 每天 9:25  python run_daily.py --intraday

管理员权限执行：
  powershell -ExecutionPolicy Bypass -File register_tasks.ps1
#>

$TaskNamePrefix = "StockPrediction"
$ProjectDir = "C:\Users\EDY\Desktop\StockPrediction"

# 找 Python
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
    Write-Host "❌ 找不到 Python，请手动指定 \$PythonExe 路径后重试" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 使用 Python: $PythonExe" -ForegroundColor Green

# 检查是否有管理员权限
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $IsAdmin) {
    Write-Host "⚠️  建议以管理员身份运行此脚本（否则计划任务可能无法创建）" -ForegroundColor Yellow
    Write-Host "   请右键 PowerShell → 以管理员身份运行 → 再执行本脚本" -ForegroundColor Yellow
    $choice = Read-Host "是否继续？(y/n)"
    if ($choice -ne 'y') { exit }
}

# 定义任务
$Tasks = @(
    @{
        Name = "${TaskNamePrefix}_Premarket"
        Description = "股票预测系统 - 盘前全量计算 (每天7:00)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 07:00
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --premarket" -WorkingDirectory $ProjectDir
    },
    @{
        Name = "${TaskNamePrefix}_Intraday"
        Description = "股票预测系统 - 盘中快速修正 (每天9:25)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 09:25
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --intraday" -WorkingDirectory $ProjectDir
    },
    @{
        Name = "${TaskNamePrefix}_Afternoon"
        Description = "股票预测系统 - 午后深度修正 (每天14:00，尾盘买入决策)"
        Trigger = New-ScheduledTaskTrigger -Daily -At 14:00
        Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run_daily.py --afternoon" -WorkingDirectory $ProjectDir
    }
)

# 注册任务
foreach ($task in $Tasks) {
    $existing = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  ⚠️  任务已存在，正在更新: $($task.Name)" -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
    }

    # 创建任务（使用当前用户权限，登录才运行）
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)

    Register-ScheduledTask -TaskName $task.Name `
        -Description $task.Description `
        -Trigger $task.Trigger `
        -Action $task.Action `
        -Principal $principal `
        -Settings $settings `
        -Force | Out-Null

    Write-Host "  ✅ 已注册: $($task.Name) ($($task.Description))" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  注册完成！" -ForegroundColor Cyan
Write-Host "  盘前模式: 每天  7:00 自动运行" -ForegroundColor Cyan
Write-Host "  盘中模式: 每天  9:25 自动运行" -ForegroundColor Cyan
Write-Host "  午后模式: 每天 13:00 自动运行" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "如需手动测试运行：" -ForegroundColor Yellow
Write-Host "  python run_daily.py --premarket" -ForegroundColor White
Write-Host "  python run_daily.py --intraday" -ForegroundColor White
Write-Host "  python run_daily.py --afternoon" -ForegroundColor White
Write-Host ""
Write-Host "如需删除任务：" -ForegroundColor Yellow
Write-Host '  Unregister-ScheduledTask -TaskName "StockPrediction_*" -Confirm:$false' -ForegroundColor White
