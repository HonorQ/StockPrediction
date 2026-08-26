# -*- coding: utf-8 -*-
"""数据库基类: 连接管理/建表/迁移。"""
from src.data.database_const import SUB_SCORE_DIMS
import os
import sqlite3
from contextlib import contextmanager

from src.data.database_const import DB_PATH


class BaseStockDB:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._migrate_schema()

    @contextmanager
    def _connection(self):
        """连接上下文管理器：事务提交/回滚 + 显式关闭连接。

        Python sqlite3 的 `with conn:` 只管理事务、不关闭连接，原先每个方法
        裸用导致大量短命连接依赖 GC 回收。commit 遇到并发写锁
        (OperationalError: database is locked) 时按指数退避重试。
        """
        import time
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            for attempt in range(3):
                try:
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < 2:
                        time.sleep(0.1 * (2 ** attempt))
                        continue
                    raise
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    score REAL NOT NULL,
                    current_price REAL,
                    rank INTEGER DEFAULT 1,
                    mode TEXT DEFAULT 'premarket',
                    regime TEXT,
                    d1_return REAL,
                    d3_return REAL,
                    d5_return REAL,
                    d10_return REAL,
                    factors TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(trade_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rec_code ON recommendations(code)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    score REAL,
                    sub_scores TEXT,
                    factors TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_factor_date ON factor_snapshots(trade_date)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    total_return REAL,
                    annual_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_loss_ratio REAL,
                    total_trades INTEGER,
                    calmar_ratio REAL,
                    params TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # 持仓跟踪表：跟踪每只推荐股票的完整交易生命周期
            # 与 recommendations 表互补：recommendations 记录 d1/d3/d5/d10 收益统计；
            # paper_trades 模拟真实交易（止损/止盈/到期平仓），用于评估策略实盘表现
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    score REAL,
                    entry_price REAL,
                    entry_date TEXT,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    return_pct REAL,
                    pnl REAL,
                    position_pct REAL,
                    shares INTEGER,
                    holding_days INTEGER,
                    max_drawdown_pct REAL,
                    max_profit_pct REAL,
                    trade_config TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pt_status ON paper_trades(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pt_code_date ON paper_trades(code, trade_date)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_attribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    score REAL,
                    actual_return REAL,
                    is_failure INTEGER DEFAULT 0,
                    return_horizon INTEGER,
                    factor_results TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # 旧库迁移：补 return_horizon 列(记录归因所用收益窗口, d14/d10/d5/d3/d1)
            try:
                conn.execute(
                    "ALTER TABLE factor_attribution ADD COLUMN return_horizon INTEGER")
            except sqlite3.OperationalError:
                pass  # 列已存在
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_attr_date ON factor_attribution(trade_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_attr_failure ON factor_attribution(is_failure)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS ic_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    ic_value REAL NOT NULL,
                    window_days INTEGER DEFAULT 0,
                    sample_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ic_date ON ic_history(trade_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ic_factor ON ic_history(factor_name)
            """)

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
                    PRIMARY KEY (code, trade_date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_daily(trade_date)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    old_weight REAL,
                    new_weight REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wt_date ON weight_history(trade_date)
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

            # 个股因子权重表：每只股票独立维护 21 维因子权重
            # 与 weight_history（全局 factor-level）互补：本表是 per-stock 维度
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_factor_weights (
                    code TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    weight REAL NOT NULL,
                    samples INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (code, factor_name)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sfw_code ON stock_factor_weights(code)
            """)

            # 个股权重变更历史（追踪每次调整的原因和幅度）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    old_weight REAL,
                    new_weight REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_swh_code_date ON stock_weight_history(code, trade_date)
            """)

            # 行业因子权重表：同行业股票共享归因数据，解决个股归因稀疏问题
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_factor_weights (
                    industry_name TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    weight REAL NOT NULL,
                    samples INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (industry_name, factor_name)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sector_industry ON sector_factor_weights(industry_name)
            """)

            # 行业权重的变更历史
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    industry_name TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    old_weight REAL,
                    new_weight REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sector_wh_industry ON sector_weight_history(industry_name, trade_date)
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

    def _migrate_schema(self):
        """幂等迁移：把 18 维子评分提升为 factor_snapshots 的独立列，并补建复合索引。
        旧数据不回填（保持 sub_scores JSON 可读），新写入同时落独立列 + JSON。
        同时为 recommendations 表添加 d14_return 列（T+14 反馈训练主信号）。
        """
        with self._connection() as conn:
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(factor_snapshots)").fetchall()}

            # 添加 18 个 sub_* 列
            for dim in SUB_SCORE_DIMS:
                col = f"sub_{dim}"
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE factor_snapshots ADD COLUMN {col} REAL")

            # 补建复合索引（用于排名、IC 截面筛选）
            indexes = conn.execute("PRAGMA index_list(factor_snapshots)").fetchall()
            existing_idx = {row[1] for row in indexes}
            if "idx_factor_date_score" not in existing_idx:
                conn.execute("CREATE INDEX idx_factor_date_score ON factor_snapshots(trade_date, score)")
            if "idx_factor_code_date" not in existing_idx:
                conn.execute("CREATE INDEX idx_factor_code_date ON factor_snapshots(code, trade_date)")

            # recommendations 表添加 d14_return 列（T+14 反馈训练主信号）
            rec_cols = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "d14_return" not in rec_cols:
                conn.execute("ALTER TABLE recommendations ADD COLUMN d14_return REAL")
            # recommendations 表添加 regime 列（大盘择时结果，用于 regime 分层归因）
            if "regime" not in rec_cols:
                conn.execute("ALTER TABLE recommendations ADD COLUMN regime TEXT")

            # (trade_date, code, mode) 唯一约束：save_recommendation 的
            # SELECT-then-INSERT 模拟 upsert 在并发下有竞态(双 INSERT 重复行)，
            # 唯一索引兜底 + INSERT OR IGNORE 保证原子性。
            # 已有重复数据时创建会失败——先去重再建索引
            dup = conn.execute("""
                SELECT COUNT(*) FROM (
                    SELECT trade_date, code, mode FROM recommendations
                    GROUP BY trade_date, code, mode HAVING COUNT(*) > 1
                )""").fetchone()[0]
            if dup == 0:
                conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_rec_date_code_mode ON recommendations(trade_date, code, mode)""")

            # factor_attribution 补 code 索引：get_recent_attribution_for_stock
            # 按 code+trade_date 查询，此前只有 date/failure 索引 → 全表扫描
            attr_idx = {row[1] for row in conn.execute("PRAGMA index_list(factor_attribution)").fetchall()}
            if "idx_attr_code" not in attr_idx:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_attr_code ON factor_attribution(code)")

