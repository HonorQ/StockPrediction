# -*- coding: utf-8 -*-
"""ashare_specific 子因子IC分析

逐日计算 ashare_specific 的5个子因子，与 T+5 收益率做 Rank IC。
输出: reports/ashare_subfactor_ic.json

用法: python scripts/ashare_subfactor_ic.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.log import get_logger
from src.factors.ashare_specific import AShareCharacteristics

logger = get_logger()


def rank_ic(group, factor_col, ret_col):
    """截面 Rank IC"""
    valid = group[[factor_col, ret_col]].dropna()
    if len(valid) < 5 or valid[ret_col].nunique() < 2:
        return np.nan
    return valid[factor_col].rank().corr(valid[ret_col].rank())


def main():
    t0 = time.time()
    print("=" * 60)
    print("ashare_specific 子因子IC分析")
    print("=" * 60)

    # 1. 加载股票列表(从数据库)
    import sqlite3
    db_path = os.path.join(ROOT, "data", "stock_history.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 获取有足够数据的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt 
        FROM kline_daily 
        WHERE trade_date >= date('now', '-180 days')
        GROUP BY code 
        HAVING cnt >= 60
        ORDER BY cnt DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        print("数据库中无足够数据的股票")
        return
    
    codes = [r[0] for r in rows]
    print(f"股票池: {len(codes)} 只 (按数据量排序)")

    # 2. 获取历史数据(直接从数据库)
    import sqlite3
    db_path = os.path.join(ROOT, "data", "stock_history.db")
    lookback_days = 120  # 回看120天
    all_data = {}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 获取所有有足够数据的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt 
        FROM kline_daily 
        GROUP BY code 
        HAVING cnt >= 60
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    codes = [r[0] for r in rows]
    print(f"股票池: {len(codes)} 只")
    
    # 加载数据
    for code in codes:
        try:
            df = pd.read_sql_query(
                """SELECT trade_date AS date, open, high, low, close, volume, amount, turnover
                   FROM kline_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?""",
                conn, params=(code, lookback_days)
            )
            df = df.iloc[::-1].reset_index(drop=True)
            if len(df) >= 60:
                all_data[code] = df
        except Exception as e:
            continue
    conn.close()
    
    print(f"有效数据: {len(all_data)} 只")

    # 3. 逐日计算子因子
    ashare = AShareCharacteristics()
    sub_factors = [
        "mean_reversion_score",  # 均值回归评分（取反后）
        "volatility_score",      # 波动率评分
        "turnover_ratio",        # 换手率异动
        "activity_score",        # 涨跌停活跃度评分
        "position_score",        # 价格位置评分
    ]
    
    # 收集每日截面数据
    daily_records = []
    for code, df in all_data.items():
        close_col = "close"
        if close_col not in df.columns:
            continue
        close = df[close_col]
        
        # 计算T+5收益率
        df["fwd5"] = close.shift(-5) / close - 1
        
        # 逐日计算子因子(取最近60天)
        for i in range(60, len(df) - 5):
            date = str(df["date"].iloc[i])[:10]
            window = df.iloc[max(0, i-60):i+1].copy()
            
            try:
                factors = ashare.calculate_all(window, code=code, name="")
                record = {
                    "date": date,
                    "code": code,
                    "fwd5": df["fwd5"].iloc[i] * 100,  # 百分比
                }
                for sf in sub_factors:
                    record[sf] = factors.get(sf, np.nan)
                daily_records.append(record)
            except Exception as e:
                continue
    
    if not daily_records:
        print("无有效记录")
        return
    
    panel = pd.DataFrame(daily_records)
    print(f"面板数据: {len(panel)} 行 / {panel['date'].nunique()} 天")

    # 4. 计算各子因子的 Rank IC
    print(f"\n{'='*60}")
    print("子因子 IC 统计 (T+5)")
    print(f"{'='*60}")
    
    results = {}
    for sf in sub_factors:
        ic_series = []
        for date, g in panel.groupby("date"):
            ic = rank_ic(g, sf, "fwd5")
            if not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            print(f"  {sf}: 无有效IC")
            continue
        
        ic_arr = np.array(ic_series)
        mean_ic = ic_arr.mean()
        std_ic = ic_arr.std()
        t_val = mean_ic / std_ic * np.sqrt(len(ic_arr)) if std_ic > 0 else 0
        pos_ratio = (ic_arr > 0).mean()
        
        results[sf] = {
            "ic_mean": round(float(mean_ic), 4),
            "ic_std": round(float(std_ic), 4),
            "t_value": round(float(t_val), 2),
            "positive_ratio": round(float(pos_ratio), 3),
            "samples": len(ic_arr),
        }
        
        # 判断
        if abs(mean_ic) > 0.03:
            strength = "有效"
        elif abs(mean_ic) > 0.01:
            strength = "弱"
        else:
            strength = "噪声"
        
        direction = "正向" if mean_ic > 0 else "负向"
        print(f"  {sf}:")
        print(f"    IC均值: {mean_ic:+.4f} | ICIR: {t_val:+.2f} | 正向占比: {pos_ratio:.1%}")
        print(f"    判定: {direction}{strength} ({'保留' if mean_ic > 0 and pos_ratio > 0.55 else '建议砍掉或取反'})")

    # 5. 保存
    out = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidates_count": len(codes),
        "valid_data_count": len(all_data),
        "panel_rows": len(panel),
        "panel_days": int(panel["date"].nunique()),
        "factors": results,
        "recommendation": {},
    }
    
    # 生成建议
    for sf, stats in results.items():
        if stats["ic_mean"] > 0 and stats["positive_ratio"] > 0.55:
            out["recommendation"][sf] = "保留(正向有效)"
        elif stats["ic_mean"] < -0.01:
            out["recommendation"][sf] = "砍掉(负向)"
        elif abs(stats["ic_mean"]) < 0.01:
            out["recommendation"][sf] = "可选(噪声)"
        else:
            out["recommendation"][sf] = "保留(弱正向)"
    
    outp = os.path.join(ROOT, "reports", "ashare_subfactor_ic.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完成! 耗时 {elapsed:.0f}秒")
    print(f"报告已保存: {outp}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
