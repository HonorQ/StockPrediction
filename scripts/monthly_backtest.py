# -*- coding: utf-8 -*-
"""月度回测验证 - ashare_specific组合模拟

每月运行一次，验证ashare_specific因子的有效性。

输出: reports/monthly_backtest.json + 资金曲线图

用法: python scripts/monthly_backtest.py [--top-n 30] [--hold 5]
"""
import os, sys, json, time
import numpy as np, pandas as pd
import sqlite3
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.log import get_logger

logger = get_logger()

DB = os.path.join(ROOT, "data", "stock_history.db")
REPORTS = os.path.join(ROOT, "reports")

# 成本参数
COST_ROUNDTRIP_PCT = 0.36  # 双边成本(佣金万2.5x2+印花税0.05+滑点)


def max_drawdown(curve):
    """计算最大回撤"""
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / peak * 100
    return float(dd.max())


def load_monthly_data(months=3):
    """加载最近N个月的股票数据"""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
    # 获取有足够数据的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt
        FROM kline_daily
        WHERE trade_date >= date('now', '-120 days')
        GROUP BY code
        HAVING cnt >= 60
        ORDER BY cnt DESC
        LIMIT 100
    """)
    codes = [r[0] for r in cur.fetchall()]
    
    all_data = {}
    for code in codes:
        try:
            df = pd.read_sql_query(
                """SELECT trade_date AS date, open, high, low, close, volume, amount, turnover
                   FROM kline_daily WHERE code=? ORDER BY trade_date DESC LIMIT 120""",
                conn, params=(code,)
            )
            df = df.iloc[::-1].reset_index(drop=True)
            if len(df) >= 60:
                all_data[code] = df
        except:
            continue
    conn.close()
    return all_data


def compute_ashare_scores(all_data):
    """计算所有股票的ashare_specific评分"""
    from src.factors.ashare_specific import AShareCharacteristics
    
    ashare = AShareCharacteristics()
    
    # 收集每日截面数据
    daily_records = []
    for code, df in all_data.items():
        if "close" not in df.columns:
            continue
        close = df["close"]
        df["fwd5"] = close.shift(-5) / close - 1
        
        for i in range(60, len(df) - 5):
            date = str(df["date"].iloc[i])[:10]
            window = df.iloc[max(0, i-60):i+1].copy()
            
            try:
                factors = ashare.calculate_all(window, code=code, name="")
                score = factors.get("ashare_specific_score", 50)
                fwd5 = df["fwd5"].iloc[i] * 100 if pd.notna(df["fwd5"].iloc[i]) else None
                
                if fwd5 is not None:
                    daily_records.append({
                        "date": date,
                        "code": code,
                        "ashare_specific": score,
                        "fwd5": fwd5,
                    })
            except:
                continue
    
    return pd.DataFrame(daily_records) if daily_records else None


def simulate(panel, top_n=30, hold=5):
    """模拟交易"""
    dates = sorted(panel["date"].unique())
    rebalance_days = dates[30::hold]  # 前30日预热（减少预热期要求）
    
    segments = []
    for d in rebalance_days:
        g = panel[panel["date"] == d].dropna(subset=["ashare_specific", "fwd5"])
        if len(g) < 50:
            continue
        
        # 选Top-N
        top = g.nlargest(top_n, "ashare_specific")
        seg_ret = float(top["fwd5"].mean())
        bench_ret = float(g["fwd5"].mean())
        
        segments.append({
            "date": d,
            "gross": seg_ret,
            "net": seg_ret - COST_ROUNDTRIP_PCT,
            "bench": bench_ret,
        })
    
    return segments


def compute_metrics(segments, hold):
    """计算绩效指标"""
    if not segments:
        return None
    
    net_rets = [s["net"] for s in segments]
    bench_rets = [s["bench"] for s in segments]
    
    # 组合资金曲线
    curve = [1_000_000]
    for r in net_rets:
        curve.append(curve[-1] * (1 + r / 100))
    curve = np.array(curve)
    
    # 基准资金曲线
    bench_curve = [1_000_000]
    for r in bench_rets:
        bench_curve.append(bench_curve[-1] * (1 + r / 100))
    bench_curve = np.array(bench_curve)
    
    # 计算指标
    total_return = (curve[-1] / curve[0] - 1) * 100
    n_years = len(segments) * hold / 252
    annual_return = ((curve[-1] / curve[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    mdd = max_drawdown(curve)
    
    arr = np.array(net_rets) / 100
    sharpe = (arr.mean() / arr.std() * (252 / hold) ** 0.5) if len(arr) > 2 and arr.std() > 0 else 0
    winrate = float((arr > 0).mean() * 100)
    
    # 超额收益
    bench_total = (bench_curve[-1] / bench_curve[0] - 1) * 100
    excess = total_return - bench_total
    
    return {
        "segments": len(segments),
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(mdd, 2),
        "sharpe": round(sharpe, 2),
        "winrate": round(winrate, 1),
        "excess_return": round(excess, 2),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--hold", type=int, default=5)
    args = ap.parse_args()
    
    t0 = time.time()
    print("=" * 60)
    print("月度回测验证 - ashare_specific")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"配置: Top{args.top_n}, 持有{args.hold}日")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    all_data = load_monthly_data()
    print(f"   有效数据: {len(all_data)} 只")
    
    # 2. 计算评分
    print("\n2. 计算ashare_specific评分...")
    panel = compute_ashare_scores(all_data)
    if panel is None or panel.empty:
        print("   无有效数据")
        return
    
    print(f"   面板数据: {len(panel)} 行, {panel['date'].nunique()} 天")
    
    # 3. 模拟交易
    print("\n3. 模拟交易...")
    segments = simulate(panel, args.top_n, args.hold)
    if not segments:
        print("   无有效分段")
        return
    
    # 4. 计算指标
    print("\n4. 计算绩效指标...")
    metrics = compute_metrics(segments, args.hold)
    
    print(f"\n{'指标':<20}{'数值':<15}")
    print("-" * 35)
    print(f"{'分段数':<20}{metrics['segments']}")
    print(f"{'总收益':<20}{metrics['total_return']:+.2f}%")
    print(f"{'年化收益':<20}{metrics['annual_return']:+.2f}%")
    print(f"{'最大回撤':<20}{metrics['max_drawdown']:.2f}%")
    print(f"{'夏普比率':<20}{metrics['sharpe']:.2f}")
    print(f"{'胜率':<20}{metrics['winrate']:.1f}%")
    print(f"{'超额收益':<20}{metrics['excess_return']:+.2f}%")
    
    # 5. 检查告警
    alerts = []
    if metrics["winrate"] < 45:
        alerts.append(f"胜率{metrics['winrate']:.1f}%低于45%阈值")
    if metrics["max_drawdown"] > 25:
        alerts.append(f"最大回撤{metrics['max_drawdown']:.1f}%超过25%阈值")
    
    # 6. 保存报告
    out = {
        "generated_at": datetime.now().isoformat(),
        "config": {"top_n": args.top_n, "hold": args.hold},
        "metrics": metrics,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }
    outp = os.path.join(REPORTS, "monthly_backtest.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n完成! 耗时{elapsed:.0f}秒")
    print(f"报告已保存: {outp}")
    
    if alerts:
        print(f"\n⚠️ 告警 ({len(alerts)}条):")
        for a in alerts:
            print(f"   - {a}")


if __name__ == "__main__":
    main()
