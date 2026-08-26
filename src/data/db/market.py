# -*- coding: utf-8 -*-
"""K线 / 集合竞价快照 相关表操作。

由 database.py 机械拆分(AGENTS.md 技术债偿还): 方法体零改动。
"""
import sqlite3
from datetime import datetime, timedelta



class MarketMixin:
    """见模块 docstring。"""

    def save_auction_snapshot(self, trade_date: str, rows: list) -> int:
        """批量落库竞价快照。rows: [{code,name,open_price,open_pct,volume_ratio,turnover,amount}]
        同日同码幂等覆盖(唯一索引兜底)。返回写入行数。"""
        if not rows:
            return 0
        with self._connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO auction_snapshots
                   (trade_date, code, name, open_price, open_pct,
                    volume_ratio, turnover, amount)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(trade_date,
                  r.get("code", ""), r.get("name", ""),
                  r.get("open_price"), r.get("open_pct"),
                  r.get("volume_ratio"), r.get("turnover"), r.get("amount"))
                 for r in rows])
            return len(rows)


    def get_auction_snapshots(self, days: int = 120) -> list:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT * FROM auction_snapshots WHERE trade_date >= ?"
                " ORDER BY trade_date, volume_ratio DESC", (cutoff,)).fetchall()
        return [dict(r) for r in rows]


    def save_klines(self, code: str, df) -> int:
        """批量幂等写入日K线。df 列名兼容中文(akshare)/英文两种风格。
        返回写入行数；缺少日期/收盘列时返回0。"""
        if df is None or df.empty:
            return 0
        rename = {c: self._KLINE_COL_MAP[c] for c in df.columns if c in self._KLINE_COL_MAP}
        std = df.rename(columns=rename)
        for col in ("trade_date", "close"):
            if col not in std.columns:
                return 0
        rows = []
        for _, r in std.iterrows():
            try:
                d = str(r["trade_date"])[:10]
                close_v = float(r.get("close") or 0)
                if not d or close_v <= 0:
                    continue
                rows.append((
                    code, d,
                    float(r.get("open") or 0), float(r.get("high") or 0),
                    float(r.get("low") or 0), close_v,
                    float(r.get("volume") or 0), float(r.get("amount") or 0),
                ))
            except (TypeError, ValueError):
                continue
        if not rows:
            return 0
        with self._connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO kline_daily
                   (code, trade_date, open, high, low, close, volume, amount)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)


    def get_klines(self, code: str, days: int = 250):
        """读取个股日K线(按日期升序)，返回标准英文列 DataFrame 或空表"""
        import pandas as pd
        with self._connection() as conn:
            df = pd.read_sql_query(
                """SELECT trade_date AS date, open, high, low, close, volume, amount
                   FROM kline_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?""",
                conn, params=(code, days),
            )
        return df.iloc[::-1].reset_index(drop=True)

