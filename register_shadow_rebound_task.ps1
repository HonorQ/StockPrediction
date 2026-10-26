<#
  注册 72.5 档超跌反弹影子追踪计划任务: 每日 16:40

  用途: oversold_rebound=72.5 档(温和双跌)是全市场面板终审出的唯一正超额
        档位, 但生产推荐从未选中过它。本任务每日收盘后用东财/新浪快照
        独立重算 72.5 档清单并追踪前向收益, 为"影子运行"累积实盘外推样本,
        与生产推荐/评分链路完全隔离(独立库 data/shadow_rebound.db)。

  设计约束:
    - 16:40 触发: 避开 15:45 BackfillReturn 与 16:20 补抓; 且快照源需收盘后
      数据(脚本自带 15:10 前拒绝运行的守卫)
    - 执行上限 30 分钟: 实测全流程约 2~4 分钟(快照 15~25s + 重算 30~60s)
    - MultipleInstances IgnoreNew: 不叠加
    - 幂等: 先删后插当日 picks, 前向收益每次重算覆写(可自愈数据修订)
    - 快照源: 东财失败自动回退新浪(2026-10-16 实测东财被拒时新浪可用)

  用法(管理员):
    powershell -ExecutionPolicy Bypass -File register_shadow_rebound_task.ps1

  手动运行一次:
    python -u scripts/shadow_rebound_track.py
#>
$TaskName = "StockPrediction_ShadowRebound"
$ProjectDir = "C:\Users\EDY\Desktop\StockPrediction"

# 与其他任务一致: 经 bat 包装, 由 bat 负责日志重定向与 --no-pause
$Action = New-ScheduledTaskAction `
    -Execute "$ProjectDir\run_shadow_rebound.bat" `
    -Argument "--no-pause" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger -Daily -At "16:40"

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "[INFO] 任务已存在, 先移除: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Description "72.5档超跌反弹影子追踪(每日16:40收盘后, 独立于生产推荐)" `
    -Trigger $Trigger `
    -Action $Action `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null

Write-Host "[OK] 已注册: $TaskName (每日 16:40)" -ForegroundColor Green
Write-Host "     动作: $ProjectDir\run_shadow_rebound.bat --no-pause" -ForegroundColor Gray
Write-Host "     日志: logs\shadow_rebound.log" -ForegroundColor Gray
