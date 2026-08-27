"""
回填 factor_snapshots 的 sub_relative_strength 因子。

背景：反馈训练链路曾漏传 market_df，导致 relative_strength 恒为默认 50。
本脚本用沪深300（sh.000300）作为市场基准，重算历史 snapshots 的相对强弱因子。

幂等：只更新 sub_relative_strength 为 NULL/50 的行；可重复运行，不覆盖真实值。
用法：
    python scripts/backfill_relative_strength.py [--limit N]   # --limit 调试用
"""
import logging
import os
import sys
import sqlite3
import argparse
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "stock_history.db")
START = "2024-07-01"  # 预留 60+ 交易日回看


def bs_code(code: str) -> str:
    if "." in code:
        return code
    if code.startswith(("sh", "sz", "bj")):
        return code[:2] + "." + code[2:]
    return ("sh." + code) if code.startswith("6") else ("sz." + code)


def fetch_all(code_bs: str):
    """拉取全历史日K，返回 {date: row(list, 8字段)}"""
    import baostock as bs
    rs = bs.query_history_k_data_plus(
        code_bs, "date,close,volume,turn,pctChg,open,high,low",
        start_date=START, end_date="2026-12-31", frequency="d", adjustflag="3")
    rows = {}
    while (rs.error_code == "0") and rs.next():
        r = rs.get_row_data()
        if r and r[0]:
            rows[r[0]] = r
    return rows


def main(limit):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from src.engine.backtest_factors import baostock_rows_to_dataframe
    from src.factors.relative_strength import RelativeStrength
    from src.engine.scorer import StockScorer

    import baostock as bs
    bs.login()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. 拉指数K线（市场基准）
    print("拉取沪深300指数K线...")
    market_rows = fetch_all("sh.000300")
    print(f"  指数 {len(market_rows)} 个交易日")

    def market_df(date):
        m = [r for r in market_rows.values() if r[0] <= date]
        if len(m) < 20:
            return None
        return baostock_rows_to_dataframe(m)

    # 2. 待回填快照
    cur.execute("""SELECT trade_date, code, sub_scores FROM factor_snapshots
                   WHERE sub_relative_strength IS NULL OR sub_relative_strength=50
                   ORDER BY trade_date""")
    snaps = cur.fetchall()
    if limit:
        snaps = snaps[:limit]
    print(f"待回填：{len(snaps)} 条")

    # 3. 按股票拉K线（缓存）
    codes = list(OrderedDict((s["code"], None) for s in snaps).keys())
    klines = {}
    for i, code in enumerate(codes):
        klines[code] = fetch_all(bs_code(code))
        if (i + 1) % 50 == 0 or i == len(codes) - 1:
            print(f"  K线加载 {i+1}/{len(codes)}")
    print(f"  K线加载完成: {len(klines)} 只")

    # 4. 重算并更新
    rs_calc = RelativeStrength()
    scorer = StockScorer()
    updated = 0
    for s in snaps:
        date, code = s["trade_date"], s["code"]
        rows = klines.get(code)
        if not rows:
            continue
        day_rows = [r for r in rows.values() if r[0] <= date]
        if len(day_rows) < 20:
            continue
        try:
            df = baostock_rows_to_dataframe(day_rows)
            factors = rs_calc.calculate_all(df, market_df=market_df(date))
            sub = scorer._score_relative_strength(factors)
        except Exception as e:
            logging.warning("捕获异常已忽略(scripts/backfill_relative_strength.py:102): %s", e)
            continue
        if not (0 <= sub <= 100):
            continue

        # 更新独立列 + JSON（保持一致）
        cur.execute("""UPDATE factor_snapshots SET sub_relative_strength=?
                       WHERE trade_date=? AND code=? AND (sub_relative_strength IS NULL OR sub_relative_strength=50)""",
                    (sub, date, code))
        if cur.rowcount == 0:
            continue
        if s["sub_scores"]:
            import json
            try:
                js = json.loads(s["sub_scores"])
            except Exception as e:
                logging.warning("捕获异常已忽略(scripts/backfill_relative_strength.py:117): %s", e)
                js = None
            if isinstance(js, dict):
                js["relative_strength"] = sub
                cur.execute("UPDATE factor_snapshots SET sub_scores=? WHERE trade_date=? AND code=?",
                            (json.dumps(js, ensure_ascii=False), date, code))
        updated += 1
        if updated % 500 == 0:
            print(f"  已更新 {updated} 条...")

    conn.commit()
    print(f"\n完成：更新 {updated} 条 sub_relative_strength")
    bs.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回填 relative_strength")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试）")
    args = parser.parse_args()
    main(args.limit)
