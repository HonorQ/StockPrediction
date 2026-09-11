# -*- coding: utf-8 -*-
"""补充数据库中缺失的换手率和涨跌幅数据

从baostock批量获取turnover和pctChg，更新kline_daily表。

用法: python scripts/fill_turnover_pctchg.py
"""
import os, sys, time
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.log import get_logger

logger = get_logger()

# baostock超时设置
BAOSTOCK_TIMEOUT = 10


def fill_data():
    """批量补充换手率和涨跌幅数据"""
    import baostock as bs
    
    db_path = os.path.join(ROOT, "data", "stock_history.db")
    
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
    total_turnover_updated = 0
    total_pctchg_updated = 0
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
        # 检查需要补充的字段
        cur.execute("""
            SELECT COUNT(*) FROM kline_daily 
            WHERE code=? AND turnover IS NOT NULL
        """, (code,))
        existing_turnover = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) FROM kline_daily 
            WHERE code=? AND pct_chg IS NOT NULL
        """, (code,))
        existing_pctchg = cur.fetchone()[0]
        
        # 检查空值数量
        cur.execute("""
            SELECT COUNT(*) FROM kline_daily 
            WHERE code=? AND turnover IS NULL
        """, (code,))
        null_turnover = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) FROM kline_daily 
            WHERE code=? AND pct_chg IS NULL
        """, (code,))
        null_pctchg = cur.fetchone()[0]
        
        # 如果两个字段都没有空值，跳过
        if null_turnover == 0 and null_pctchg == 0:
            continue
        
        # 从baostock获取数据
        try:
            # 转换代码格式
            if code.startswith(("6", "5", "9")):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"
            
            # 获取turnover和pctChg
            fields = []
            if existing_turnover == 0:
                fields.append("turn")
            if existing_pctchg == 0:
                fields.append("pctChg")
            
            if not fields:
                continue
            
            rs = bs.query_history_k_data_plus(
                bs_code, ",".join(fields),
                start_date=min_date,
                end_date=max_date,
                frequency="d", adjustflag="2"
            )
            
            data_map = {}
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if len(row) >= 1 + len(fields):
                    date = row[0]
                    vals = {}
                    for j, f in enumerate(fields):
                        try:
                            vals[f] = float(row[j + 1]) if row[j + 1] not in (None, "") else None
                        except:
                            vals[f] = None
                    data_map[date] = vals
            
            # 更新数据库
            turnover_updated = 0
            pctchg_updated = 0
            
            for date, vals in data_map.items():
                if "turn" in vals and vals["turn"] is not None and null_turnover > 0:
                    cur.execute("""
                        UPDATE kline_daily SET turnover=? 
                        WHERE code=? AND trade_date=? AND turnover IS NULL
                    """, (vals["turn"], code, date))
                    if cur.rowcount > 0:
                        turnover_updated += 1
                
                if "pctChg" in vals and vals["pctChg"] is not None and null_pctchg > 0:
                    cur.execute("""
                        UPDATE kline_daily SET pct_chg=? 
                        WHERE code=? AND trade_date=? AND pct_chg IS NULL
                    """, (vals["pctChg"], code, date))
                    if cur.rowcount > 0:
                        pctchg_updated += 1
            
            conn.commit()
            
            if turnover_updated > 0:
                total_turnover_updated += 1
            if pctchg_updated > 0:
                total_pctchg_updated += 1
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[{i+1}/{len(stocks)}] {code}: turnover更新{turnover_updated}条, pctchg更新{pctchg_updated}条 | "
                      f"累计 turnover={total_turnover_updated}/pctchg={total_pctchg_updated}/失败={total_failed} | "
                      f"耗时{elapsed:.0f}秒")
        
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
    print(f"turnover更新: {total_turnover_updated}只")
    print(f"pctchg更新: {total_pctchg_updated}只")
    print(f"失败: {total_failed}只")


if __name__ == "__main__":
    fill_data()
