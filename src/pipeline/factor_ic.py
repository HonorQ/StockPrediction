# -*- coding: utf-8 -*-
"""因子IC评估(手动)

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger

logger = get_logger()

from src.pipeline.common import (  # noqa: F401
    _DualOutput, setup_logging, write_summary, _prefetch_history_and_flow,
    save_cache, load_cache, _log_source_health, _cleanup_cache,
)


def run_factor_ic(config):
    """因子IC评估: 从数据库加载历史因子数据，计算IC/ICIR，输出权重建议"""
    from src.engine.ml_selector import FactorOptimizer

    lookback = config.get("factor_ic", {}).get("lookback_days", 60)
    forward = config.get("factor_ic", {}).get("forward_window", 3)
    weights = config.get("weights", {})

    print("=" * 60)
    print("  因子IC评估")
    print(f"  回溯天数: {lookback}天 | 未来窗口: {forward}天")
    print("=" * 60)
    print()

    optimizer = FactorOptimizer()
    factor_data = optimizer.load_from_database(
        lookback_days=lookback, forward_window=forward
    )

    if factor_data.empty or "future_return" not in factor_data.columns:
        print("  ⚠️ 数据库历史数据不足，无法计算因子IC")
        print("  💡 提示：需要至少运行几次盘前模式积累数据")
        print(f"  📊 当前可用数据: {factor_data.shape if hasattr(factor_data, 'shape') else '0'}")
        return

    print(f"  📊 加载因子截面数据: {len(factor_data)} 条")
    print()

    ic = optimizer.calculate_ic(factor_data, return_col="future_return")
    if not ic:
        print("  ⚠️ 无法计算因子IC")
        return

    # 按日计算ICIR（标准口径：mean(IC_daily) / std(IC_daily)）
    ic_by_day = optimizer.calculate_ic_by_day(factor_data, return_col="future_return")

    print("  📈 因子IC排名（前15）：")
    print("  " + "-" * 60)
    print(f"  {'排名':>4} {'因子':<20} {'IC值':>10} {'ICIR':>10} {'样本日':>8} {'强度':>6}")
    print("  " + "-" * 60)
    sorted_ic = sorted(ic.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, ic_val) in enumerate(sorted_ic[:15], 1):
        daily_ics = ic_by_day.get(name, [])
        icir = optimizer.calculate_icir(daily_ics)
        intensity = "🔵 强" if abs(ic_val) > 0.05 else ("🟡 中" if abs(ic_val) > 0.03 else "⚪ 弱")
        print(f"  {i:>4} {name:<20} {ic_val:>+10.4f} {icir:>10.2f} {len(daily_ics):>8} {intensity:>6}")
    print()

    high_corr = optimizer.analyze_factor_correlation(factor_data)
    if not high_corr.empty and len(high_corr) > 0:
        print("  ⚡ 高度相关因子对（>0.7）：")
        for _, row in high_corr.iterrows():
            print(f"      {row['factor1']} ↔ {row['factor2']}: {row['correlation']:.3f}")

    print()
    opt_weights = optimizer.optimize_weights(factor_data, return_col="future_return")
    if opt_weights:
        print("  🎯 优化权重建议（对比当前权重）：")
        print(f"  {'因子':<20} {'当前权重':>10} {'优化权重':>10} {'调整':>10}")
        print("  " + "-" * 50)
        for name in weights:
            cur = weights.get(name, 0)
            opt = opt_weights.get(name, 0)
            delta = opt - cur
            print(f"  {name:<20} {cur:>10.3f} {opt:>10.4f} {delta:>+10.4f}")
        print()

    report = optimizer.generate_weight_report()
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "factor_ic")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir,
                               f"factor_ic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ 因子IC报告已保存: {report_path}")


# ======================================================================
# 交易日判断
# ======================================================================
