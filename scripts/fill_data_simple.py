# -*- coding: utf-8 -*-
"""补充数据库中缺失的换手率和涨跌幅数据（简化版）

从baostock批量获取turnover和pctChg，更新kline_daily表中的NULL值。

用法: python scripts/fill_data_simple.py
"""
import os, sys, time
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "data", "stock_history.db")


def fill_data():
    """批量补充数据"""
    import baostock as bs
    
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
    # 获取所有有空值的股票
    cur.execute("""
        SELECT code, COUNT(*) as cnt
        FROM kline_daily
        WHERE turnover IS NULL OR pct_chg IS NULL
        GROUP BY code
        ORDER BY cnt DESC
    """)
    stocks = cur.fetchall()
    print(f"需要补充数据的股票: {len(stocks)}只")
    
    # 登录baostock
    bs.login()
    print("baostock登录成功")
    
    start_time = time.time()
    total_updated = 0
    
    for i, (code, cnt) in enumerate(stocks):
        # 转换代码格式
        if code.startswith(("6", "5", "9")):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"
        
        # 获取数据
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, "date,turn,pctChg",
                start_date="2025-01-01",
                end_date="2026-12-31",
                frequency="d", adjustflag="2"
            )
            
            data_map = {}
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if len(row) >= 3:
                    date = row[0]
                    turn = None
                    pctchg = None
                    try:
                        turn = float(row[1]) if row[1] else None
                    except:
                        pass
                    try:
                        pctchg = float(row[2]) if row[2] else None
                    except:
                        pass
                    data_map[date] = (turn, pctchg)
            
            # 更新数据库
            updated = 0
            for date, (turn, pctchg) in data_map.items():
                if turn is not None:
                    cur.execute("""
                        UPDATE kline_daily SET turnover=? 
                        WHERE code=? AND trade_date=? AND turnover IS NULL
                    """, (turn, code, date))
                    updated += cur.rowcount
                
                if pctchg is not None:
                    cur.execute("""
                        UPDATE kline_daily SET pct_chg=? 
                        WHERE code=? AND trade_date=? AND pct_chg IS NULL
                    """, (pctchg, code, date))
                    updated += cur.rowcount
            
            conn.commit()
            total_updated += updated
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[{i+1}/{len(stocks)}] {code}: 更新{updated}条 | 累计{total_updated}条 | 耗时{elapsed:.0f}秒")
        
        except Exception as e:
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(stocks)}] {code}: 失败 {e}")
        
        time.sleep(0.1)
    
    bs.logout()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n完成! 耗时{elapsed:.0f}秒，共更新{total_updated}条")


if __name__ == "__main__":
    fill_data()
