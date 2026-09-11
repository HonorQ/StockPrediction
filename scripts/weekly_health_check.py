# -*- coding: utf-8 -*-
"""每周胜率监控 - ashare_specific评分体系健康检查

每周运行一次, 检查:
1. ashare_specific 月度胜率是否>45%
2. 最近30天推荐胜率
3. 评分分布是否正常
4. 异常告警

注意: 只统计实盘推荐(premarket/afternoon), 排除回测推荐(backtest/backtest_invert/feedback_training)

用法: python scripts/weekly_health_check.py
"""
import os, json, sqlite3
from datetime import datetime

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
DB = os.path.join(ROOT, "data", "stock_history.db")
REPORTS = os.path.join(ROOT, "reports")

# 阈值
WINRATE_THRESHOLD = 45.0  # 低于此值触发告警
ALERT_CONSECUTIVE_MONTHS = 2  # 连续N个月低于阈值触发审查

# 实盘模式（排除回测模式）
LIVE_MODES = "('premarket', 'afternoon')"


def main():
    print("=" * 60)
    print("每周健康检查 - ashare_specific评分体系")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    alerts = []

    # 1. 最近30天推荐胜率
    print("\n1. 最近30天实盘推荐胜率")
    cur.execute(f"""
        SELECT COUNT(*) total,
               SUM(CASE WHEN d5_return > 0 THEN 1 ELSE 0 END) wins,
               AVG(d5_return) avg_d5,
               AVG(d3_return) avg_d3
        FROM recommendations
        WHERE mode IN {LIVE_MODES} AND d5_return IS NOT NULL
          AND trade_date >= date('now', '-30 days')
    """)
    r = cur.fetchone()
    if r and r["total"] > 0:
        wr = r["wins"] / r["total"] * 100 if r["wins"] else 0
        status = "✅" if wr >= WINRATE_THRESHOLD else "❌"
        print(f"  {status} 总推荐: {r['total']}  胜率: {wr:.1f}%  平均d5: {r['avg_d5']:.2f}%")
        if wr < WINRATE_THRESHOLD:
            alerts.append(f"最近30天胜率{wr:.1f}%低于阈值{WINRATE_THRESHOLD}%")
    else:
        print("  ⚠️ 最近30天无推荐数据")

    # 2. 最近7天
    print("\n2. 最近7天实盘推荐胜率")
    cur.execute(f"""
        SELECT COUNT(*) total,
               SUM(CASE WHEN d5_return > 0 THEN 1 ELSE 0 END) wins,
               AVG(d5_return) avg_d5
        FROM recommendations
        WHERE mode IN {LIVE_MODES} AND d5_return IS NOT NULL
          AND trade_date >= date('now', '-7 days')
    """)
    r = cur.fetchone()
    if r and r["total"] > 0:
        wr = r["wins"] / r["total"] * 100 if r["wins"] else 0
        status = "✅" if wr >= WINRATE_THRESHOLD else "❌"
        print(f"  {status} 总推荐: {r['total']}  胜率: {wr:.1f}%  平均d5: {r['avg_d5']:.2f}%")
    else:
        print("  ⚠️ 最近7天无推荐数据")

    # 3. 月度胜率趋势
    print("\n3. 月度实盘推荐胜率趋势")
    cur.execute(f"""
        SELECT substr(trade_date, 1, 7) month,
               COUNT(*) total,
               SUM(CASE WHEN d5_return > 0 THEN 1 ELSE 0 END) wins,
               AVG(d5_return) avg_d5
        FROM recommendations
        WHERE mode IN {LIVE_MODES} AND d5_return IS NOT NULL
          AND trade_date >= date('now', '-6 months')
        GROUP BY month
        ORDER BY month
    """)
    rows = cur.fetchall()
    low_months = 0
    for r in rows:
        wr = r["wins"] / r["total"] * 100 if r["wins"] else 0
        status = "✅" if wr >= WINRATE_THRESHOLD else "❌"
        print(f"  {status} {r['month']}: 胜率={wr:.1f}%  平均d5={r['avg_d5']:.2f}%  推荐={r['total']}")
        if wr < WINRATE_THRESHOLD:
            low_months += 1

    if low_months >= ALERT_CONSECUTIVE_MONTHS:
        alerts.append(f"最近{len(rows)}月中有{low_months}月胜率低于{WINRATE_THRESHOLD}%, 需审查因子")

    # 4. 评分分布
    print("\n4. 最近一天实盘推荐评分分布")
    cur.execute(f"""
        SELECT score, COUNT(*) cnt
        FROM recommendations
        WHERE mode IN {LIVE_MODES}
          AND trade_date = (SELECT MAX(trade_date) FROM recommendations WHERE mode IN {LIVE_MODES})
        GROUP BY score
        ORDER BY score DESC
    """)
    rows = cur.fetchall()
    if rows:
        total = sum(r["cnt"] for r in rows)
        print(f"  样本: {total}只")
        for r in rows[:10]:
            print(f"    分数{r['score']:.0f}: {r['cnt']}只")
    else:
        print("  ⚠️ 无评分数据")

    # 5. 告警汇总
    print(f"\n{'='*60}")
    if alerts:
        print(f"⚠️ 告警 ({len(alerts)}条):")
        for a in alerts:
            print(f"  - {a}")
        print("\n建议: 检查 ashare_specific 因子是否失效, 是否需要调整评分体系")
    else:
        print("✅ 全部正常, 无告警")

    # 保存检查结果
    out = {
        "check_time": datetime.now().isoformat(),
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }
    outp = os.path.join(REPORTS, "weekly_health_check.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")

    conn.close()


if __name__ == "__main__":
    main()
