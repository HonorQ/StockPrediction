<#
  注册 K线补抓计划任务: 每 4 小时重试一次 scripts/backfill_missing_klines.py

  用途: baostock 限流恢复后自动补齐 fetch_1year_data 漏掉的 K线缓存
        (2026-10-15 首跑 5404 只仅成功 422 只)。缓存补齐后才能重算
        reports/market_ic_verdict.json。

  设计约束:
    - 复用上午 12:20 起每 4 小时的节奏(12:20/16:20/20:20/00:20/04:20/08:20),
      与其余 baostock 使用者保持缓冲:
        15:45 BackfillReturn / 07:00 premarket / 14:40 afternoon
      60 分钟执行上限下, 各触发点最晚结束时间为 13:20/17:20/21:20/01:20/
      05:20/09:20, 均不与上述时点重叠
    - ExecutionTimeLimit 60 分钟: 实测补抓约 51 只/分钟, 单轮可补 3000 只左右,
      避免过长占用 baostock(AGENTS.md 规则7: 全局 socket 不支持并发)
    - MultipleInstances IgnoreNew: 本任务自身不叠加
    - 幂等: 已有缓存的股票直接跳过, 补满后单次运行数秒内结束
    - baostock 不可用时子进程 30 秒内超时退出, 不会长时间空转
    - baostock 限流是波动的(2026-10-15 实测: 15时限流 -> 16:42 恢复 ->
      18:18 又限流), 故用重试而非单次拉取

  用法(管理员):
    powershell -ExecutionPolicy Bypass -File register_backfill_klines_task.ps1

  补满后如需手动触发一次:
    python -u scripts/backfill_missing_klines.py
#>
$TaskName = "StockPrediction_BackfillKlines"
$ProjectDir = "C:\Users\EDY\Desktop\StockPrediction"

# 与 StockPrediction_WeeklyMonitor / MonthlyBacktest 一致: 经 bat 包装,
# 由 bat 负责日志重定向与 --no-pause(无人值守时 pause 会挂死到超时)
$Action = New-ScheduledTaskAction `
    -Execute "$ProjectDir\run_backfill_klines.bat" `
    -Argument "--no-pause" `
    -WorkingDirectory $ProjectDir

# 每 4 小时重试, 持续 7 天; 起点 12:20 以错开 15:45/07:00/14:40 的 baostock 占用
$Trigger = New-ScheduledTaskTrigger `
    -Once -At "12:20" `
    -RepetitionInterval (New-TimeSpan -Hours 4) `
    -RepetitionDuration (New-TimeSpan -Days 7)

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60)

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "[INFO] 任务已存在, 先移除: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Description "补齐 baostock K线缓存(每4小时重试, 限流恢复后自动续跑)" `
    -Trigger $Trigger `
    -Action $Action `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null

Write-Host "[OK] 已注册: $TaskName (每4小时, 12:20 起, 持续7天)" -ForegroundColor Green
Write-Host "     动作: $ProjectDir\run_backfill_klines.bat --no-pause" -ForegroundColor Gray
Write-Host "     日志: logs\backfill_klines.log" -ForegroundColor Gray
