"""
回填 recommendations 的 T+N 收益（d1/d3/d5/d10/d14）。

用途：实盘 premarket 推荐（每天 1-2 条）的收益回填一直不足（IC 无法评估实盘）。
run_daily 内置的自动回填受"自然日 check_date 窗口"限制，对实盘推荐回填效率低。

用法（建议每日收盘后 15:30 后运行）：
    python scripts/backfill_recommendation_returns.py
    python scripts/backfill_recommendation_returns.py --mode live

--mode: 指定回填哪些 mode 的推荐
    premarket  仅盘前(默认, 保持既有计划任务行为)
    afternoon  仅午后
    live       premarket + afternoon(纯实盘, 推荐用于日常回填)
    all        全部(含 backtest/feedback_training 的模拟数据, 慎用)
    --all 等价于 --mode all

幂等：只更新当前为 NULL 的字段；已回填的保持原值（不覆盖，除非 --overwrite）。
安全：UPDATE/幂等查询均带 mode 条件。同一 (trade_date, code) 可能同时存在
    premarket 与 afternoon 两条记录(已有实例: 2026-09-18 000338), 缺 mode
    条件会把两行一并改写, 造成串写污染(2026-10-15 修复)。
"""
import logging
import os
import sqlite3
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "stock_history.db")

WIN = {1: "d1_return", 3: "d3_return", 5: "d5_return", 10: "d10_return", 14: "d14_return"}

# 纯实盘模式: 不含 backtest/backtest_invert/feedback_training 的模拟收益
LIVE_MODES = ("premarket", "afternoon")


def bs_code(code: str) -> str:
    if "." in code:
        return code
    if code.startswith(("sh", "sz", "bj")):
        return code[:2] + "." + code[2:]
    return ("sh." + code) if code.startswith("6") else ("sz." + code)


def fetch_daily(code_bs: str):
    """拉取 2025-07-01 至今日K，返回 {date: close}"""
    import baostock as bs
    rs = bs.query_history_k_data_plus(
        code_bs, "date,close", start_date="2025-07-01", end_date="2026-12-31",
        frequency="d", adjustflag="3")
    out = {}
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        if row[0] and row[1]:
            out[row[0]] = float(row[1])
    return out


def backfill(mode: str, overwrite: bool = False):
    import baostock as bs
    bs.login()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 取待回填推荐（按日期正序，先回填早的）
    # 必须带回 mode: 同一 (trade_date, code) 可能有 premarket/afternoon 两条记录,
    # 用 (trade_date, code) 定位会一次命中多行, 导致跨模式串写
    if mode == "all":
        cur.execute("SELECT trade_date, mode, code, current_price FROM recommendations ORDER BY trade_date")
    elif mode == "live":
        cur.execute(
            "SELECT trade_date, mode, code, current_price FROM recommendations "
            "WHERE mode IN (?, ?) ORDER BY trade_date",
            LIVE_MODES,
        )
    else:
        cur.execute("SELECT trade_date, mode, code, current_price FROM recommendations WHERE mode=? ORDER BY trade_date", (mode,))
    recs = cur.fetchall()
    print(f"共 {len(recs)} 条推荐待检查 (mode={mode})")

    # 缓存每只股票的日K（避免重复拉取）
    kline_cache = {}

    updated = 0
    for trade_date, rec_mode, code, entry in recs:
        if not entry or entry <= 0:
            continue
        code_bs = bs_code(code)
        if code_bs not in kline_cache:
            kline_cache[code_bs] = fetch_daily(code_bs)
        daily = kline_cache[code_bs]

        # 定位 trade_date 在K线中的交易日索引
        trading_days = sorted(daily.keys())
        try:
            base_idx = trading_days.index(trade_date)
        except ValueError as e:
            logging.warning("捕获异常已忽略(scripts/backfill_recommendation_returns.py:75): %s", e)
            continue  # 该日无行情（周末/停牌），跳过

        for n_days, field in WIN.items():
            exit_idx = base_idx + n_days
            if exit_idx >= len(trading_days):
                continue  # T+N 未到
            exit_date = trading_days[exit_idx]
            exit_price = daily[exit_date]
            if not exit_price or exit_price <= 0:
                continue
            # 幂等：已有值且非覆盖则跳过（同带 mode 条件，避免读到他行的值）
            cur.execute(f"SELECT {field} FROM recommendations WHERE trade_date=? AND code=? AND mode=?",
                        (trade_date, code, rec_mode))
            existing = cur.fetchone()[0]
            if existing is not None and not overwrite:
                continue
            ret = round((exit_price / entry - 1) * 100, 2)
            cur.execute(f"UPDATE recommendations SET {field}=? WHERE trade_date=? AND code=? AND mode=?",
                        (ret, trade_date, code, rec_mode))
            updated += 1
            print(f"  ✅ {trade_date} {code} {field}(T+{n_days}) = {ret}%")

    conn.commit()
    conn.close()
    print(f"\n回填完成：更新 {updated} 个字段")
    bs.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回填推荐收益")
    parser.add_argument("--all", action="store_true", help="等价 --mode all（含模拟数据，慎用）")
    parser.add_argument(
        "--mode",
        default="premarket",
        choices=["premarket", "afternoon", "live", "all"],
        help="回填哪些 mode: premarket(默认)/afternoon/live(两者纯实盘)/all(全部含模拟数据)",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已回填的值（默认幂等）")
    args = parser.parse_args()
    backfill("all" if args.all else args.mode, overwrite=args.overwrite)
