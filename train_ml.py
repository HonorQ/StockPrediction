"""
机器学习训练入口 - 从 factor_snapshots 加载真实因子截面 + recommendations 收益标签
→ 训练模型 → 优化权重

训练数据源：data/stock_history.db 的 factor_snapshots 表（生产 scorer 每日写入）
标签来源：recommendations 表的 d3_return（T+3 真实收益，已回填）
这样保证训练/推理特征分布一致，消除 TechnicalFactors 单独训练导致的漂移。
"""
from src.utils.log import setup_logging, get_logger
setup_logging()
logger = get_logger()

import os
import sys
import sqlite3
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import StockDatabase
from src.engine.ml_selector import MLStockSelector, FactorOptimizer


def collect_training_data():
    """从 factor_snapshots + recommendations 加载训练数据并训练。"""
    db = StockDatabase()
    ml = MLStockSelector()
    optimizer = FactorOptimizer()

    logger.info('%s', '=' * 60)
    logger.info('  机器学习数据收集 & 模型训练（数据源：factor_snapshots）')

    # 直接 SQL JOIN：factor_snapshots ⋈ recommendations(trade_date+code → d3_return)
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT fs.trade_date, fs.code, fs.factors, r.d3_return
            FROM factor_snapshots fs
            JOIN recommendations r
              ON fs.trade_date = r.trade_date AND fs.code = r.code
            WHERE r.d3_return IS NOT NULL
            ORDER BY fs.trade_date ASC
        """).fetchall()

    if not rows:
        logger.warning('❌ 无训练数据（factor_snapshots 与 recommendations JOIN 为空）')
        return

    logger.info(f'  原始样本数: {len(rows)}')
    logger.info(f'  日期范围: {rows[0]["trade_date"]} ~ {rows[-1]["trade_date"]}')

    # 构造训练记录
    all_records = []
    factor_data_rows = []
    skipped = 0
    for row in rows:
        factors = row["factors"]
        if isinstance(factors, str):
            try:
                factors = json.loads(factors)
            except Exception:
                skipped += 1
                continue
        if not isinstance(factors, dict) or not factors:
            skipped += 1
            continue

        future_return = float(row["d3_return"])
        date_str = str(row["trade_date"])[:10]

        all_records.append({
            "factors": factors,
            "future_return": future_return,
            "stock_code": row["code"],
            "date": date_str,
        })

        # 因子优化数据（展开 factors 到列）
        row_data = {"future_return": future_return, "stock_code": row["code"], "date": date_str}
        row_data.update(factors)
        factor_data_rows.append(row_data)

    if skipped > 0:
        logger.info(f'  跳过无效记录: {skipped} 条')

    logger.info(f'  有效训练样本: {len(all_records)} 条')
    if len(all_records) < 100:
        logger.info('❌ 训练数据不足 100 条，跳过训练')
        return

    # --- 训练ML模型 ----
    logger.info('\n--- 训练机器学习模型 ---')
    df = ml.prepare_training_data(all_records)
    if df.empty:
        logger.warning('❌ 数据准备失败')
        return

    # 提取日期列表用于时序验证（已按日期升序）
    dates = [r.get("date", "") for r in all_records]
    result = ml.train(df, dates=dates)
    logger.info('\n训练结果:')
    logger.info(f"  模型类型: {result.get('model_type', 'unknown')}")
    logger.info(f"  验证方式: {result.get('validation', 'random_split')}")
    logger.info(f"  准确率: {result['accuracy']:.1%}")
    logger.info(f"  精确率: {result['precision']:.1%}")
    logger.info(f"  召回率: {result['recall']:.1%}")
    logger.info(f"  F1分数: {result['f1_score']:.3f}")
    logger.info(f"  训练样本: {result['train_samples']}")
    logger.info(f"  测试样本: {result['test_samples']}")
    logger.info('\nTop10重要特征:')
    for name, imp in result["top_features"]:
        logger.info(f'  {name}: {imp:.1%}')

    # 特征重要性报告
    report_path = os.path.join(ROOT, "reports", "training", "feature_importance.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 特征重要性报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"模型类型: {result.get('model_type', 'unknown')}\n")
        f.write(f"验证方式: {result.get('validation', 'random_split')}\n\n")
        f.write("| 排名 | 特征 | 重要性 |\n")
        f.write("|------|------|--------|\n")
        for i, (name, imp) in enumerate(result["top_features"], 1):
            f.write(f"| {i} | {name} | {imp:.2%} |\n")

    logger.info(f"\n✅ 模型已保存: {result['model_saved']}")
    logger.info(f'✅ 特征重要性报告: {report_path}')

    # 多因子组合优化 ----
    logger.info('\n--- 多因子组合优化 ---')
    factor_df = pd.DataFrame(factor_data_rows)

    # 过滤无效列
    for col in factor_df.columns:
        if factor_df[col].dtype == object:
            if col not in ["stock_code", "date"]:
                factor_df[col] = pd.to_numeric(factor_df[col], errors="coerce")
        factor_df[col] = factor_df[col].fillna(0) if col != "stock_code" else factor_df[col]

    # 同时把 date 列重命名为 trade_date 以便 IC 计算按日分组
    if "date" in factor_df.columns:
        factor_df["trade_date"] = factor_df["date"]

    ic_result = optimizer.calculate_ic(factor_df)
    logger.info('\n因子IC排名（Top10，按日分组ICIR）:')
    ic_by_day = optimizer.calculate_ic_by_day(factor_df)
    sorted_ic = sorted(ic_result.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, ic_val) in enumerate(sorted_ic[:10], 1):
        daily_ics = ic_by_day.get(name, [])
        icir = optimizer.calculate_icir(daily_ics)
        logger.info(f'  {i}. {name}: IC={ic_val:+.4f} ICIR={icir:+.2f} 样本日={len(daily_ics)}')
    # 权重优化
    weights = optimizer.optimize_weights(factor_df)
    if weights:
        logger.info('\n优化权重（Top10）:')
        for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f'  {name}: {w:.2%}')

    # 因子相关性
    corr_pairs = optimizer.analyze_factor_correlation(factor_df)
    if not corr_pairs.empty:
        logger.info(f'\n高度相关因子对（>0.7）: {len(corr_pairs)}对')
        for _, row in corr_pairs.head(5).iterrows():
            logger.info(f"  {row['factor1']} ↔ {row['factor2']}: {row['correlation']:.3f}")
        # 保存优化报告
    opt_report = optimizer.generate_weight_report()
    opt_path = os.path.join(ROOT, "reports", "training", "factor_optimization.md")
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    with open(opt_path, "w", encoding="utf-8") as f:
        f.write(opt_report)
    logger.info(f'\n✅ 因子优化报告: {opt_path}')


if __name__ == "__main__":
    collect_training_data()
