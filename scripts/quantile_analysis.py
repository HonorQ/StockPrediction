# -*- coding: utf-8 -*-
"""十分位分层分析 - ashare_specific因子区分度验证

每月运行一次，验证ashare_specific因子的区分度。

输出: reports/quantile_analysis.json

用法: python scripts/quantile_analysis.py
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


def load_data():
    """加载数据"""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
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


def compute_scores(all_data):
    """计算评分"""
    from src.factors.ashare_specific import AShareCharacteristics
    
    ashare = AShareCharacteristics()
    
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


def quantile_analysis(panel, n_quantiles=10):
    """十分位分层分析"""
    results = []
    
    for date, g in panel.groupby("date"):
        if len(g) < n_quantiles * 5:
            continue
        
        # 按ashare_specific评分分十分位
        g = g.copy()
        g["quantile"] = pd.qcut(g["ashare_specific"].rank(method="first"), 
                                n_quantiles, labels=False)
        
        for q in range(n_quantiles):
            qg = g[g["quantile"] == q]
            if len(qg) > 0:
                results.append({
                    "date": date,
                    "quantile": q,
                    "fwd5": qg["fwd5"].mean(),
                    "count": len(qg),
                })
    
    return pd.DataFrame(results)


def compute_quantile_stats(qdf, n_quantiles=10):
    """计算各分位统计"""
    stats = []
    
    for q in range(n_quantiles):
        qg = qdf[qdf["quantile"] == q]
        if len(qg) == 0:
            continue
        
        mean_ret = qg["fwd5"].mean()
        std_ret = qg["fwd5"].std()
        count = qg["count"].sum()
        
        stats.append({
            "quantile": q,
            "mean_return": round(float(mean_ret), 4),
            "std_return": round(float(std_ret), 4),
            "count": int(count),
        })
    
    return stats


def main():
    t0 = time.time()
    print("=" * 60)
    print("十分位分层分析 - ashare_specific")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    all_data = load_data()
    print(f"   有效数据: {len(all_data)} 只")
    
    # 2. 计算评分
    print("\n2. 计算ashare_specific评分...")
    panel = compute_scores(all_data)
    if panel is None or panel.empty:
        print("   无有效数据")
        return
    
    print(f"   面板数据: {len(panel)} 行, {panel['date'].nunique()} 天")
    
    # 3. 十分位分析
    print("\n3. 十分位分层分析...")
    qdf = quantile_analysis(panel)
    if qdf.empty:
        print("   无有效分层")
        return
    
    stats = compute_quantile_stats(qdf)
    
    # 4. 打印结果
    print(f"\n{'分位':<8}{'平均收益':<12}{'标准差':<12}{'样本数':<10}")
    print("-" * 42)
    
    for s in stats:
        q = s["quantile"]
        mean = s["mean_return"]
        std = s["std_return"]
        count = s["count"]
        print(f"Q{q:<7}{mean:<+12.4f}{std:<12.4f}{count}")
    
    # 5. 计算多空价差
    if len(stats) >= 2:
        q1 = stats[0]["mean_return"]
        q10 = stats[-1]["mean_return"]
        spread = q10 - q1
        print(f"\nQ10-Q1 价差: {spread:+.4f}%")
        
        # 判定
        if spread > 0.05:
            print("判定: ✅ 因子有效（正向区分度）")
        elif spread > 0:
            print("判定: ⚠️ 因子弱（弱正向区分度）")
        else:
            print("判定: ❌ 因子无效（无区分度或反向）")
    else:
        spread = 0
    
    # 6. 保存报告
    out = {
        "generated_at": datetime.now().isoformat(),
        "quantiles": stats,
        "spread": round(float(spread), 4),
        "status": "ok",
    }
    outp = os.path.join(REPORTS, "quantile_analysis.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n完成! 耗时{elapsed:.0f}秒")
    print(f"报告已保存: {outp}")


if __name__ == "__main__":
    main()
