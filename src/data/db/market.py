# -*- coding: utf-8 -*-
"""K线 / 集合竞价快照 相关表操作。

由 database.py 机械拆分(AGENTS.md 技术债偿还): 方法体零改动。
"""
from datetime import datetime, timedelta

# K线列名映射: 中文(akshare) → 英文(标准)
_KLINE_COL_MAP = {
    "日期": "trade_date", "date": "trade_date",
    "开盘": "open", "open": "open",
    "最高": "high", "high": "high",
    "最低": "low", "low": "low",
    "收盘": "close", "close": "close",
    "成交量": "volume", "volume": "volume",
    "成交额": "amount", "amount": "amount",
    "换手率": "turnover", "turnover": "turnover",
    "涨跌幅": "pct_chg", "pctChg": "pct_chg",
}



class MarketMixin:
    """见模块 docstring。"""

    def _create_ml_auction_tables(self, conn):
        """建表: ML 版本/影子评分 + 日K线/竞价快照域。由 _init_db 调用, 接收已开启 PRAGMA 的连接。"""

        # 日K线落库：ML_TRAINING_PLAN 关键缺口#5——原始K线此前不落库，
        # 导致无法离线做需要重算因子的全市场回测。盘前预取后幂等写入。
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kline_daily (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                turnover REAL,
                pct_chg REAL,
                PRIMARY KEY (code, trade_date)
            )
        """)
        # 兼容旧表：ALTER TABLE 补 turnover/pct_chg 列（已存在则静默忽略）
        for col, typ in [("turnover", "REAL"), ("pct_chg", "REAL")]:
            try:
                conn.execute(f"ALTER TABLE kline_daily ADD COLUMN {col} {typ}")
            except Exception:
                pass  # 列已存在

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_daily(trade_date)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                model_type TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                train_samples INTEGER,
                test_samples INTEGER,
                top_features TEXT,
                model_path TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 影子 ML 评分表：影子模式下 ML 模型对候选股的预测结果，与主推荐隔离
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_shadow_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT DEFAULT '',
                ml_score REAL NOT NULL,
                ml_probability REAL DEFAULT 0.5,
                ml_prediction TEXT DEFAULT 'neutral',
                existing_score REAL DEFAULT 50,
                existing_rank INTEGER DEFAULT 0,
                ml_rank INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mss_date ON ml_shadow_scores(trade_date)
        """)

        # 集合竞价快照：9:25 盘中模式落库，为竞价强度因子积累 IC 样本
        # (量比=竞价强度核心字段; open_pct 9:25时刻的涨跌幅即开盘涨幅)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auction_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT DEFAULT '',
                open_price REAL,
                open_pct REAL,
                volume_ratio REAL,
                turnover REAL,
                amount REAL,
                captured_at TEXT DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_auction_date_code
            ON auction_snapshots(trade_date, code)
        """)

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
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._fetch_all(
            "SELECT * FROM auction_snapshots WHERE trade_date >= ?"
            " ORDER BY trade_date, volume_ratio DESC", (cutoff,)
        )


    def save_klines(self, code: str, df) -> int:
        """批量幂等写入日K线。df 列名兼容中文(akshare)/英文两种风格。
        返回写入行数；缺少日期/收盘列时返回0。"""
        if df is None or df.empty:
            return 0
        rename = {c: _KLINE_COL_MAP[c] for c in df.columns if c in _KLINE_COL_MAP}
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
                    float(r.get("turnover") or 0),
                    float(r.get("pct_chg") or 0),
                ))
            except (TypeError, ValueError):
                continue
        if not rows:
            return 0
        with self._connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO kline_daily
                   (code, trade_date, open, high, low, close, volume, amount, turnover, pct_chg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)


    def get_klines(self, code: str, days: int = 250):
        """读取个股日K线(按日期升序)，返回标准英文列 DataFrame 或空表"""
        import pandas as pd
        with self._connection() as conn:
            df = pd.read_sql_query(
                """SELECT trade_date AS date, open, high, low, close, volume, amount,
                          turnover, pct_chg
                   FROM kline_daily WHERE code=? ORDER BY trade_date DESC LIMIT ?""",
                conn, params=(code, days),
            )
        return df.iloc[::-1].reset_index(drop=True)

