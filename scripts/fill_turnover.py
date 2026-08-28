# -*- coding: utf-8 -*-
"""补充数据库中缺失的换手率数据

从baostock批量获取换手率，更新kline_daily表的turnover列。

用法: python scripts/fill_turnover.py
"""
import os, sys, time, pickle
import sqlite3
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.log import get_logger

logger = get_logger()

# baostock超时设置
BAOSTOCK_TIMEOUT = 10


def fill_turnover():
    """批量补充换手率数据"""
    import baostock as bs
    
    db_path = os.path.join(ROOT, "data", "stock_history.db")
    cache_dir = os.path.join(ROOT, "cache", "turnover")
    os.makedirs(cache_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 获取所有有K线数据的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt, 
               MIN(trade_date) as min_date, MAX(trade_date) as max_date
        FROM kline_daily 
        GROUP BY code
        ORDER BY cnt DESC
    """)
    stocks = cur.fetchall()
    print(f"数据库中股票数: {len(stocks)}")
    
    # 统计
    total_updated = 0
    total_skipped = 0
    total_failed = 0
    
    # 登录baostock
    try:
        bs.login()
        print("baostock登录成功")
    except Exception as e:
        print(f"baostock登录失败: {e}")
        return
    
    start_time = time.time()
    
    for i, (code, cnt, min_date, max_date) in enumerate(stocks):
        # 检查是否需要补充
        cur.execute("""
            SELECT COUNT(*) FROM kline_daily 
            WHERE code=? AND turnover IS NOT NULL
        """, (code,))
        existing = cur.fetchone()[0]
        
        if existing > 0:
            total_skipped += 1
            continue
        
        # 从baostock获取换手率
        try:
            # 转换代码格式
            if code.startswith(("6", "5", "9")):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"
            
            # 获取数据
            rs = bs.query_history_k_data_plus(
                bs_code, "date,turn",
                start_date=min_date,
                end_date=max_date,
                frequency="d", adjustflag="2"
            )
            
            turn_map = {}
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if len(row) >= 2 and row[1] not in (None, ""):
                    try:
                        turn_map[row[0]] = float(row[1])
                    except (TypeError, ValueError):
                        pass
            
            # 更新数据库
            if turn_map:
                updated = 0
                for date, turn in turn_map.items():
                    cur.execute("""
                        UPDATE kline_daily SET turnover=? 
                        WHERE code=? AND trade_date=?
                    """, (turn, code, date))
                    if cur.rowcount > 0:
                        updated += 1
                
                conn.commit()
                total_updated += 1
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(stocks)}] {code}: 更新{updated}条 | "
                          f"累计更新{total_updated}/跳过{total_skipped}/失败{total_failed} | "
                          f"耗时{elapsed:.0f}秒")
            else:
                total_failed += 1
        
        except Exception as e:
            total_failed += 1
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(stocks)}] {code}: 失败 {e}")
        
        # 避免请求过快
        time.sleep(0.1)
    
    bs.logout()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n完成! 耗时{elapsed:.0f}秒")
    print(f"更新: {total_updated}只 | 跳过: {total_skipped}只 | 失败: {total_failed}只")


if __name__ == "__main__":
    fill_turnover()
