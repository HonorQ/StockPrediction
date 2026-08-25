# 研究流水线：A补K线缓存 → B逐月回测矩阵(基线+反转) → C收尾任务
# 由计划任务在独立会话执行，不受终端关闭影响。断点安全(已有缓存自动跳过)。
$ErrorActionPreference = "Continue"
Set-Location C:\Users\EDY\Desktop\StockPrediction
$env:NO_PROXY = "*"
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Step($name) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$stamp] === $name ==="
}

# ============ A. 补齐全市场1年K线缓存 ============
Step "A: fetch_1year_data (补K线缓存)"
python -u scripts\fetch_1year_data.py *> logs\pipe_fetch1year.log
Step "A done, exit=$LASTEXITCODE"

# ============ B. 逐月回测矩阵 ============
$windows = @(
    @('2026-01-05', '2026-01-23'),
    @('2026-03-02', '2026-03-31'),
    @('2026-05-06', '2026-05-29'),
    @('2026-08-03', '2026-08-22')
)
foreach ($w in $windows) {
    $s = $w[0]; $e = $w[1]
    Step "B: baseline $s ~ $e"
    python -u run_backtest.py --start $s --end $e *> "logs\pipe_base_$s.log"
    Start-Sleep -Seconds 30
    Step "B: invert $s ~ $e"
    python -u run_backtest.py --invert-signals --start $s --end $e *> "logs\pipe_inv_$s.log"
    Start-Sleep -Seconds 60
}

# ============ C. 收尾任务 ============
Step "C: evaluate_recs backfill"
python -u evaluate_recs.py --backfill *> logs\pipe_eval_recs.log
Step "C: rolling IC validation"
python -u rolling_ic_weight_validation.py *> logs\pipe_roll_ic.log
Step "C: cross-window report"
python -u scripts\cross_window_analysis.py *> logs\pipe_cross_window.log
# train_ml 放最后(依赖最新快照，且耗时波动大)
Step "C: train_ml"
python -u train_ml.py *> logs\pipe_train_ml.log

Step "PIPELINE ALL DONE"
"DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File logs\research_pipeline_done.txt -Encoding utf8
