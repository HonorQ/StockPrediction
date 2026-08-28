# -*- coding: utf-8 -*-
"""每周IC监控 - ashare_specific子因子IC衰减检测

每周运行一次，计算4个子因子的Rank IC，检测因子衰减。

告警条件：
1. IC均值连续4周<0.02 → 告警
2. IC均值连续4周转负 → 告警

用法: python scripts/weekly_ic_monitor.py
输出: reports/weekly_ic.json + reports/weekly_ic_history.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
import sqlite3
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.log import get_logger
from src.factors.ashare_specific import AShareCharacteristics

logger = get_logger()

DB = os.path.join(ROOT, "data", "stock_history.db")
REPORTS = os.path.join(ROOT, "reports")
HISTORY_PATH = os.path.join(REPORTS, "weekly_ic_history.json")

# 子因子列表
SUB_FACTORS = [
    "mean_reversion_score",  # 均值回归（取反后）
    "volatility_score",      # 波动率
    "turnover_ratio",        # 换手率异动
    "position_score",        # 价格位置
]

# 告警阈值
IC_MIN_THRESHOLD = 0.02
IC_WARN_WEEKS = 4


def rank_ic(group, factor_col, ret_col):
    """截面 Rank IC"""
    valid = group[[factor_col, ret_col]].dropna()
    if len(valid) < 10 or valid[ret_col].nunique() < 2:
        return np.nan
    return valid[factor_col].rank().corr(valid[ret_col].rank())


def load_weekly_data(lookback_days=30):
    """加载最近30天的股票数据"""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
    # 获取有足够数据的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt
        FROM kline_daily
        GROUP BY code
        HAVING cnt >= 60
        ORDER BY cnt DESC
        LIMIT 100
    """)
    codes = [r[0] for r in cur.fetchall()]
    
    all_data = {}
    max_date = None
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
                # 记录最大日期
                if "date" in df.columns:
                    code_max = df["date"].max()
                    if max_date is None or code_max > max_date:
                        max_date = code_max
        except:
            continue
    conn.close()
    return all_data, max_date


def compute_weekly_ic(all_data):
    """计算本周各子因子IC"""
    ashare = AShareCharacteristics()
    
    # 收集面板数据
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
                record = {"date": date, "code": code, "fwd5": df["fwd5"].iloc[i] * 100}
                for sf in SUB_FACTORS:
                    record[sf] = factors.get(sf, np.nan)
                daily_records.append(record)
            except:
                continue
    
    if not daily_records:
        return None
    
    panel = pd.DataFrame(daily_records)
    
    # 计算各子因子IC
    results = {}
    for sf in SUB_FACTORS:
        ic_series = []
        for date, g in panel.groupby("date"):
            ic = rank_ic(g, sf, "fwd5")
            if not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
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
    
    return results


def load_history():
    """加载历史IC记录"""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def check_alerts(history):
    """检查告警条件"""
    alerts = []
    
    if len(history) < IC_WARN_WEEKS:
        return alerts
    
    recent = history[-IC_WARN_WEEKS:]
    
    for sf in SUB_FACTORS:
        # 检查连续4周IC<0.02
        low_ic = all(
            r.get("factors", {}).get(sf, {}).get("ic_mean", 0) < IC_MIN_THRESHOLD
            for r in recent
        )
        if low_ic:
            alerts.append(f"{sf} 连续{IC_WARN_WEEKS}周IC<{IC_MIN_THRESHOLD}，可能衰减")
        
        # 检查连续4周转负
        negative_ic = all(
            r.get("factors", {}).get(sf, {}).get("ic_mean", 0) < 0
            for r in recent
        )
        if negative_ic:
            alerts.append(f"{sf} 连续{IC_WARN_WEEKS}周IC为负，因子失效")
    
    return alerts


def main():
    t0 = time.time()
    print("=" * 60)
    print("每周IC监控 - ashare_specific子因子")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    all_data, data_max_date = load_weekly_data(lookback_days=90)
    print(f"   有效数据: {len(all_data)} 只")
    print(f"   数据截止日: {data_max_date}")
    
    # 2. 计算本周IC
    print("\n2. 计算本周IC...")
    factors = compute_weekly_ic(all_data)
    if not factors:
        print("   无有效数据")
        return
    
    # 3. 打印结果
    print("\n3. 本周IC统计:")
    print("-" * 60)
    print(f"{'子因子':<25}{'IC均值':<10}{'ICIR':<10}{'正向占比':<10}{'判定':<10}")
    print("-" * 60)
    
    for sf, stats in factors.items():
        ic = stats["ic_mean"]
        t = stats["t_value"]
        pos = stats["positive_ratio"]
        
        if ic > 0.03:
            status = "✅ 有效"
        elif ic > 0.01:
            status = "⚠️ 弱"
        elif ic > 0:
            status = "⚠️ 弱正"
        else:
            status = "❌ 负"
        
        print(f"{sf:<25}{ic:<+10.4f}{t:<+10.2f}{pos:<10.1%}{status}")
    
    # 4. 更新历史
    print("\n4. 更新历史记录...")
    history = load_history()
    record = {
        "check_time": datetime.now().isoformat(),
        "factors": factors,
    }
    history.append(record)
    
    # 只保留最近12周
    if len(history) > 12:
        history = history[-12:]
    
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # 5. 检查告警
    print("\n5. 检查告警...")
    alerts = check_alerts(history)
    
    if alerts:
        print(f"\n⚠️ 告警 ({len(alerts)}条):")
        for a in alerts:
            print(f"   - {a}")
    else:
        print("\n✅ 无告警")
    
    # 6. 保存本周报告
    out = {
        "check_time": datetime.now().isoformat(),
        "data_max_date": data_max_date,
        "factors": factors,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }
    outp = os.path.join(REPORTS, "weekly_ic.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n完成! 耗时{elapsed:.0f}秒")
    print(f"报告已保存: {outp}")
    print(f"历史已保存: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
