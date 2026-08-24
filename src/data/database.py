"""
SQLite数据库 - 历史因子截面 + 推荐追踪 + 回测记录
"""
import sqlite3
import json
from contextlib import contextmanager
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型 → JSON 序列化"""
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super().default(o)


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "stock_history.db")


# 24 维子评分维度（与 StockScorer.get_sub_scores 完全一致）
# 用于 factor_snapshots 表的独立列，便于 SQL 直接过滤/排序/聚合，无需解析 JSON
SUB_SCORE_DIMS = [
    "trend", "strategy", "chip", "fundamental", "institutional",
    "sector_rotation", "oversold_rebound", "news", "relative_strength",
    "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
    "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
    "policy_event", "national_team", "capital_behavior", "moat",
    "holder_concentration", "circulate_holder",
]


class StockDatabase:
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

    def save_recommendation(self, trade_date: str, code: str, name: str,
                            score: float, price: float, rank: int = 1,
                            mode: str = "premarket", factors: dict = None,
                            regime: str = None):
        with self._connection() as conn:
            # 去重键为 (trade_date, code, mode)：同一股票同日允许盘前/午后各留一条，
            # 这样才能分别回填收益、对比"盘前推荐 vs 午后推荐"的历史准确率。
            # (旧逻辑按 (trade_date, code) 去重导致午后覆盖盘前记录，两者永远无法对比)
            existing = conn.execute(
                "SELECT id FROM recommendations WHERE trade_date=? AND code=? AND mode=?",
                (trade_date, code, mode)
            ).fetchone()

            if existing:
                # regime 为空时保留已有值（intraday/afternoon 覆盖写不丢失 premarket 的 regime）
                if regime is None:
                    regime = conn.execute(
                        "SELECT regime FROM recommendations WHERE id=?", (existing[0],)
                    ).fetchone()[0]
                conn.execute(
                    """UPDATE recommendations
                       SET name=?, score=?, current_price=?, rank=?, factors=?, regime=?
                       WHERE trade_date=? AND code=? AND mode=?""",
                    (name, score, price, rank,
                     json.dumps(factors, ensure_ascii=False, cls=_NumpyEncoder) if factors else None,
                     regime, trade_date, code, mode)
                )
            else:
                conn.execute(
                    """INSERT INTO recommendations
                       (trade_date, code, name, score, current_price, rank, mode, factors, regime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_date, code, name, score, price, rank, mode,
                     json.dumps(factors, ensure_ascii=False, cls=_NumpyEncoder) if factors else None,
                     regime)
                )

    def update_performance(self, trade_date: str, code: str,
                           d1: float = None, d3: float = None,
                           d5: float = None, d10: float = None,
                           d14: float = None, mode: str = None):
        """回填 T+N 收益。
        mode 提供时精确更新该 mode 的行(盘前/午后入场价不同，收益必须分开算)；
        None 时保持旧行为(不限 mode，同日同码多行会被写成相同值——仅兼容旧调用)。"""
        updates = []
        params = []
        for k, v in [("d1_return", d1), ("d3_return", d3),
                      ("d5_return", d5), ("d10_return", d10),
                      ("d14_return", d14)]:
            if v is not None:
                updates.append(f"{k}=?")
                params.append(v)
        if not updates:
            return
        where = "trade_date=? AND code=?"
        params.extend([trade_date, code])
        if mode:
            where += " AND mode=?"
            params.append(mode)
        with self._connection() as conn:
            conn.execute(
                f"UPDATE recommendations SET {', '.join(updates)} WHERE {where}",
                params
            )

    def get_recent_recommendations(self, days: int = 30) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM recommendations WHERE trade_date >= ?
                   ORDER BY trade_date DESC, rank ASC""", (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recommendation_history(self, code: str, days: int = 60) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM recommendations WHERE code=? AND trade_date >= ?
                   ORDER BY trade_date DESC""", (code, cutoff)
            ).fetchall()
        return [dict(r) for r in rows]

    def save_factor_snapshot(self, trade_date: str, code: str, name: str,
                             score: float, sub_scores: dict, factors: dict):
        # 同时写入 18 维独立列 + JSON（独立列供 SQL 过滤/排序，JSON 保留完整结构供向后兼容）
        sub_cols = {f"sub_{dim}": float(sub_scores.get(dim, 50)) for dim in SUB_SCORE_DIMS}
        col_names = ["trade_date", "code", "name", "score", "sub_scores", "factors"] + list(sub_cols.keys())
        placeholders = ", ".join(["?"] * len(col_names))
        values = [
            trade_date, code, name, score,
            json.dumps(sub_scores, ensure_ascii=False, cls=_NumpyEncoder) if sub_scores else None,
            json.dumps(factors, ensure_ascii=False, cls=_NumpyEncoder) if factors else None,
        ] + list(sub_cols.values())
        sql = f"INSERT INTO factor_snapshots ({', '.join(col_names)}) VALUES ({placeholders})"
        with self._connection() as conn:
            conn.execute(sql, values)

    def save_factor_snapshots_batch(self, items: list):
        if not items:
            return
        sub_cols = {f"sub_{dim}" for dim in SUB_SCORE_DIMS}
        col_names = ["trade_date", "code", "name", "score", "sub_scores", "factors"]
        values_list = []
        for item in items:
            sub_scores = item.get("sub_scores", {})
            factors = item.get("factors", {})
            cols = {f"sub_{dim}": float(sub_scores.get(dim, 50)) for dim in SUB_SCORE_DIMS}
            values = [
                item["trade_date"], item["code"], item.get("name", ""),
                item.get("score", 50),
                json.dumps(sub_scores, ensure_ascii=False, cls=_NumpyEncoder) if sub_scores else None,
                json.dumps(factors, ensure_ascii=False, cls=_NumpyEncoder) if factors else None,
            ] + list(cols.values())
            values_list.append(values)
        full_cols = col_names + sorted(sub_cols)
        placeholders = ", ".join(["?"] * len(full_cols))
        sql = f"INSERT INTO factor_snapshots ({', '.join(full_cols)}) VALUES ({placeholders})"
        with self._connection() as conn:
            conn.executemany(sql, values_list)

    def get_factor_snapshot_by_code_date(self, trade_date: str,
                                          code: str) -> Optional[Dict]:
        """精确查询某只股票某天的因子快照，返回解析后的 sub_scores dict。
        用于反馈训练归因（避免 query_factor_snapshots 的 limit/排序导致漏查）。
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM factor_snapshots
                   WHERE trade_date=? AND code=?
                   ORDER BY id DESC LIMIT 1""",
                (trade_date, code)
            ).fetchall()
        if not rows:
            return None
        r = dict(rows[0])
        sub = r.get("sub_scores")
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except Exception:
                sub = {}
        if not isinstance(sub, dict) or not sub:
            # fallback: 从独立列 sub_* 还原
            sub = {}
            for dim in SUB_SCORE_DIMS:
                col = f"sub_{dim}"
                v = r.get(col)
                if v is not None:
                    sub[dim] = v
        return sub

    def get_factor_snapshot_raw_factors(self, code: str, trade_date: str) -> Optional[Dict]:
        """获取某只股票某天的原始因子字典（factors JSON字段）。
        用于 ML 训练时获取原始因子特征，而非24维子评分。
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT factors FROM factor_snapshots
                   WHERE trade_date=? AND code=?
                   ORDER BY id DESC LIMIT 1""",
                (trade_date, code)
            ).fetchall()
        if not rows:
            return None
        r = dict(rows[0])
        factors = r.get("factors")
        if isinstance(factors, str):
            try:
                factors = json.loads(factors)
            except Exception:
                return None
        if isinstance(factors, dict) and "z_scores" not in factors:
            return factors
        return None

    def get_factor_history(self, days: int = 365) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM factor_snapshots WHERE trade_date >= ?
                   ORDER BY trade_date DESC""", (cutoff,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            # 优先用独立列 sub_* 还原 sub_scores（新数据）；
            # 缺失时回退到 JSON sub_scores（旧数据）
            sub_from_cols = {}
            for dim in SUB_SCORE_DIMS:
                col = f"sub_{dim}"
                v = d.get(col)
                if v is not None:
                    sub_from_cols[dim] = v
            if sub_from_cols:
                d["sub_scores"] = sub_from_cols
            elif d.get("sub_scores"):
                d["sub_scores"] = json.loads(d["sub_scores"])
            else:
                d["sub_scores"] = {}
            if d.get("factors"):
                d["factors"] = json.loads(d["factors"])
            result.append(d)
        return result

    def query_factor_snapshots(self, days: int = 60, min_score: float = None,
                                dim_filters: Dict[str, float] = None,
                                limit: int = None) -> List[Dict]:
        """按维度筛选因子快照（直接走 SQL 列，无需解析 JSON）。

        Args:
            days: 回溯天数
            min_score: 综合分下限
            dim_filters: {dim: min_value}，例如 {"trend": 70} 等价于 sub_trend >= 70
            limit: 返回条数上限
        """
        sql = "SELECT * FROM factor_snapshots WHERE trade_date >= ?"
        params = [(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")]
        if min_score is not None:
            sql += " AND score >= ?"
            params.append(min_score)
        if dim_filters:
            for dim, val in dim_filters.items():
                if dim in SUB_SCORE_DIMS:
                    sql += f" AND sub_{dim} >= ?"
                    params.append(val)
        sql += " ORDER BY trade_date DESC, score DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_sub_score_panel(self, days: int = 60, today: str = None,
                            return_col: str = "d3_return") -> List[Dict]:
        """批量加载 24 维 sub_* 独立列 + 推荐收益（JOIN recommendations）。

        与逐条 get_recommendation_history 不同：不受 30 天窗口限制，一次性取回
        factor_snapshots JOIN recommendations 的全样本，供 IC 分析 / 反馈调权使用。

        Args:
            days: 回溯天数（相对今天）。
            today: 截止日期 (YYYY-MM-DD)。None 时使用真实当前日期。
                   只取 trade_date <= today 的快照（避免回测未来函数）。
            return_col: 收益列，默认 d3_return。

        Returns:
            [{trade_date, code, <24 个 dim 名>, future_return}, ...]
            24 维列名为 dim（去 sub_ 前缀），future_return 即 return_col 值。
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        sub_cols = ", ".join(f"fs.sub_{dim} AS {dim}" for dim in SUB_SCORE_DIMS)
        sql = (
            "SELECT fs.trade_date, fs.code, " + sub_cols +
            f", r.{return_col} AS future_return"
            " FROM factor_snapshots fs"
            " INNER JOIN recommendations r"
            "   ON fs.trade_date = r.trade_date AND fs.code = r.code"
            # 因子快照产自盘前全量计算，只 JOIN 盘前推荐行：
            # 1) 语义一致；2) 同码同日存在 premarket/afternoon 两行时防止面板样本重复
            " WHERE fs.trade_date >= ? AND fs.trade_date <= ? AND r.mode='premarket'"
            f" AND r.{return_col} IS NOT NULL"
            " ORDER BY fs.trade_date"
        )
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (cutoff, today)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["trade_date"] = str(d["trade_date"])
            result.append(d)
        return result

    def get_trade_dates_with_data(self, days: int = 60) -> List[str]:
        with self._connection() as conn:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT DISTINCT trade_date FROM factor_snapshots
                   WHERE trade_date >= ? ORDER BY trade_date""", (cutoff,)
            ).fetchall()
        return [r[0] for r in rows]

    def save_backtest_result(self, metrics: dict):
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO backtest_results
                   (run_date, total_return, annual_return, sharpe_ratio,
                    max_drawdown, win_rate, profit_loss_ratio, total_trades,
                    calmar_ratio, params)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    metrics.get("total_return"),
                    metrics.get("annual_return"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("win_rate"),
                    metrics.get("profit_loss_ratio"),
                    metrics.get("total_trades"),
                    metrics.get("calmar_ratio"),
                    json.dumps({k: v for k, v in metrics.items()
                               if k not in ("total_return", "annual_return",
                                           "sharpe_ratio", "max_drawdown",
                                           "win_rate", "profit_loss_ratio",
                                           "total_trades", "calmar_ratio")},
                              ensure_ascii=False) if metrics else None
                )
            )

    def get_backtest_history(self, limit: int = 20) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # Paper trades（持仓跟踪）
    # ==================================================================
    def open_paper_trade(self, trade_date: str, code: str, name: str,
                         score: float, entry_price: float,
                         position_pct: float = None, shares: int = None,
                         trade_config: dict = None,
                         entry_date: str = None) -> int:
        """开仓：记录一笔模拟交易的入场信息。
        同一 trade_date + code 已存在记录时跳过（避免重复开仓）。
        注意：去重不能限制 status='open'，否则已平仓记录不参与去重，
        同一天重跑 premarket 会重复开仓+平仓，污染胜率统计。
        返回新记录 id（已存在时返回旧记录 id）。
        """
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM paper_trades WHERE trade_date=? AND code=?",
                (trade_date, code)
            ).fetchone()
            if existing:
                return existing[0]
            cur = conn.execute(
                """INSERT INTO paper_trades
                   (trade_date, code, name, score, entry_price, entry_date,
                    position_pct, shares, trade_config, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (trade_date, code, name, score, entry_price,
                 entry_date or trade_date,
                 position_pct, shares,
                 json.dumps(trade_config, ensure_ascii=False) if trade_config else None)
            )
            return cur.lastrowid

    def close_paper_trade(self, trade_id: int, exit_date: str, exit_price: float,
                          exit_reason: str, max_drawdown_pct: float = None,
                          max_profit_pct: float = None) -> bool:
        """平仓：填入出场信息并计算收益。
        exit_reason: take_profit / stop_loss / hold_expire / manual
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT entry_price, position_pct, shares FROM paper_trades WHERE id=? AND status='open'",
                (trade_id,)
            ).fetchone()
            if not row:
                return False
            entry_price = row["entry_price"] or 0
            position_pct = row["position_pct"] or 0
            shares = row["shares"] or 0
            if entry_price <= 0:
                return False
            return_pct = (exit_price / entry_price - 1) * 100
            # pnl 用入场时的资金（position_pct × 假设账户规模 10万）× 收益率
            # 若 shares 已知则用 shares × 价差
            if shares > 0:
                pnl = shares * (exit_price - entry_price)
            else:
                notional = 100000 * position_pct
                pnl = notional * return_pct / 100

            # 持仓天数
            holding_days = None
            try:
                entry_row = conn.execute(
                    "SELECT entry_date FROM paper_trades WHERE id=?", (trade_id,)
                ).fetchone()
                if entry_row and entry_row["entry_date"]:
                    d0 = datetime.strptime(entry_row["entry_date"], "%Y-%m-%d")
                    d1 = datetime.strptime(exit_date, "%Y-%m-%d")
                    holding_days = (d1 - d0).days
            except Exception:
                pass

            conn.execute(
                """UPDATE paper_trades
                   SET exit_date=?, exit_price=?, exit_reason=?, return_pct=?,
                       pnl=?, holding_days=?, max_drawdown_pct=?, max_profit_pct=?,
                       status='closed'
                   WHERE id=?""",
                (exit_date, exit_price, exit_reason, round(return_pct, 2),
                 round(pnl, 2), holding_days, max_drawdown_pct, max_profit_pct, trade_id)
            )
            return True

    def update_paper_trade_extremes(self, trade_id: int,
                                     max_drawdown_pct: float = None,
                                     max_profit_pct: float = None) -> None:
        """更新未平仓交易的极值（仅在更差/更好时覆盖）"""
        with self._connection() as conn:
            if max_drawdown_pct is not None:
                conn.execute(
                    """UPDATE paper_trades SET max_drawdown_pct=?
                       WHERE id=? AND (max_drawdown_pct IS NULL OR ? < max_drawdown_pct)""",
                    (max_drawdown_pct, trade_id, max_drawdown_pct)
                )
            if max_profit_pct is not None:
                conn.execute(
                    """UPDATE paper_trades SET max_profit_pct=?
                       WHERE id=? AND (max_profit_pct IS NULL OR ? > max_profit_pct)""",
                    (max_profit_pct, trade_id, max_profit_pct)
                )

    def update_trade_config(self, trade_id: int, config: dict) -> None:
        """更新 paper_trade 的 trade_config JSON（如 trailing stop 状态）"""
        with self._connection() as conn:
            conn.execute(
                "UPDATE paper_trades SET trade_config=? WHERE id=?",
                (json.dumps(config, ensure_ascii=False), trade_id)
            )

    def get_open_paper_trades(self) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE status='open' ORDER BY trade_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def has_open_position(self, code: str) -> bool:
        """检查某个股票是否已有未平仓的 paper_trade（避免重复建仓）"""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM paper_trades WHERE code=? AND status='open' LIMIT 1",
                (code,)
            ).fetchone()
        return row is not None

    def get_closed_paper_trades(self, days: int = 30) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM paper_trades
                   WHERE status='closed' AND trade_date >= ?
                   ORDER BY exit_date DESC""",
                (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_paper_trade_stats(self, days: int = 30) -> Dict:
        """统计最近 N 天已平仓交易的整体表现"""
        with self._connection() as conn:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT return_pct, pnl, exit_reason, holding_days
                   FROM paper_trades
                   WHERE status='closed' AND trade_date >= ?""",
                (cutoff,)
            ).fetchall()
        if not rows:
            return {"total": 0}
        returns = [r[0] or 0 for r in rows]
        pnls = [r[1] or 0 for r in rows]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        reasons = {}
        for r in rows:
            reasons[r[2] or "unknown"] = reasons.get(r[2] or "unknown", 0) + 1
        avg_hold = sum(r[3] or 0 for r in rows) / len(rows) if rows else 0
        return {
            "total": len(rows),
            "win_count": len(wins),
            "lose_count": len(losses),
            "win_rate": round(len(wins) / len(rows) * 100, 2) if rows else 0,
            "avg_return": round(sum(returns) / len(returns), 2) if returns else 0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_holding_days": round(avg_hold, 1),
            "exit_reasons": reasons,
        }

    # ==================================================================
    # 反馈机制：归因记录
    # ==================================================================
    # ==================================================================
    # 日K线落库（离线回测数据基建）
    # ==================================================================
    _KLINE_COL_MAP = {
        "日期": "trade_date", "date": "trade_date",
        "开盘": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "收盘": "close", "close": "close",
        "成交量": "volume", "volume": "volume",
        "成交额": "amount", "amount": "amount",
    }

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

    def save_attribution(self, trade_date: str, code: str, name: str,
                         score: float, actual_return: float,
                         factor_results: dict, is_failure: bool = False,
                         return_horizon: int = None):
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO factor_attribution
                   (trade_date, code, name, score, actual_return, is_failure,
                    return_horizon, factor_results)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_date, code, name, score, round(actual_return, 2),
                 int(is_failure), return_horizon,
                 json.dumps(factor_results, ensure_ascii=False, cls=_NumpyEncoder))
            )

    def get_attribution_history(self, days: int = 365,
                                only_failures: bool = False) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            sql = "SELECT * FROM factor_attribution WHERE trade_date >= ?"
            if only_failures:
                sql += " AND is_failure=1"
            sql += " ORDER BY trade_date DESC"
            rows = conn.execute(sql, (cutoff,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("factor_results"):
                d["factor_results"] = json.loads(d["factor_results"])
            result.append(d)
        return result

    def get_attribution_stats(self, days: int = 365) -> Dict:
        rows = self.get_attribution_history(days=days)
        if not rows:
            return {}
        failures = [r for r in rows if r.get("is_failure")]
        total = len(rows)

        factor_totals = {}
        for r in rows:
            for factor, result in (r.get("factor_results") or {}).items():
                if factor not in factor_totals:
                    factor_totals[factor] = {"correct": 0, "wrong": 0, "neutral": 0}
                key = result.get("result", "neutral")
                if key in factor_totals[factor]:
                    factor_totals[factor][key] += 1

        factor_accuracy = {}
        for factor, counts in factor_totals.items():
            total_pred = counts["correct"] + counts["wrong"]
            if total_pred > 0:
                factor_accuracy[factor] = {
                    "accuracy": round(counts["correct"] / total_pred, 3),
                    "correct": counts["correct"],
                    "wrong": counts["wrong"],
                }

        return {
            "total": total,
            "failures": len(failures),
            "failure_rate": round(len(failures) / total, 3) if total else 0,
            "avg_return": round(sum(r.get("actual_return", 0) for r in rows) / total, 2) if total else 0,
            "factor_accuracy": factor_accuracy,
        }

    # ==================================================================
    # 反馈机制：IC历史
    # ==================================================================
    def save_ic_record(self, trade_date: str, ic_values: Dict[str, float],
                       window_days: int = 0, sample_count: int = 0):
        with self._connection() as conn:
            for factor, ic in ic_values.items():
                conn.execute(
                    """INSERT INTO ic_history
                       (trade_date, factor_name, ic_value, window_days, sample_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (trade_date, factor, round(ic, 4), window_days, sample_count)
                )

    def get_ic_history(self, factor_name: str = None, days: int = 60) -> pd.DataFrame:
        with self._connection() as conn:
            if factor_name:
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                rows = conn.execute(
                    """SELECT trade_date, factor_name, ic_value
                       FROM ic_history
                       WHERE factor_name=? AND trade_date>=?
                       ORDER BY trade_date""", (factor_name, cutoff)
                ).fetchall()
            else:
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                rows = conn.execute(
                    """SELECT trade_date, factor_name, ic_value
                       FROM ic_history
                       WHERE trade_date>=?
                       ORDER BY trade_date, factor_name""", (cutoff,)
                ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=["trade_date", "factor_name", "ic_value"])

    def get_latest_ic(self, days: int = 14) -> Dict[str, float]:
        df = self.get_ic_history(days=days)
        if df.empty:
            return {}
        return df.groupby("factor_name")["ic_value"].mean().to_dict()

    # ==================================================================
    # 反馈机制：权重变更历史
    # ==================================================================
    def save_weight_update(self, trade_date: str, factor_name: str,
                           old_weight: float, new_weight: float, reason: str = ""):
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO weight_history
                   (trade_date, factor_name, old_weight, new_weight, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                (trade_date, factor_name, old_weight, new_weight, reason)
            )

    def get_weight_history(self, days: int = 90) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM weight_history WHERE trade_date >= ?
                   ORDER BY trade_date DESC""", (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_weights(self, today: str = None) -> Dict[str, float]:
        """获取截至 today 的每个因子最新权重快照（用于 scorer 运行时合并）。

        规则 §6：config.yaml 是静态先验，反馈循环的调权结果落库。
        scorer 实例化时调用本方法，把库里的最新调权结果叠加到静态权重上。
        回测时传 today，只引用截至该日的调权记录，避免权重前视。

        Args:
            today: 截止日期 (YYYY-MM-DD)。None 时使用真实当前日期。

        Returns:
            {factor_name: new_weight}，只返回有调权记录的因子。
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if today is None:
                today = datetime.now().strftime("%Y-%m-%d")
            # 子查询：每个 factor 最新一条记录
            rows = conn.execute(
                """SELECT factor_name, new_weight FROM weight_history wh
                   WHERE trade_date <= ?
                     AND id = (
                       SELECT MAX(id) FROM weight_history
                       WHERE factor_name = wh.factor_name AND trade_date <= ?
                     )""",
                (today, today)
            ).fetchall()
        return {r["factor_name"]: r["new_weight"] for r in rows}

    # ==================================================================
    # 反馈机制：模型版本追踪
    # ==================================================================
    def save_model_version(self, version: str, trained_at: str,
                           model_type: str, metrics: dict,
                           top_features: list, model_path: str):
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO model_versions
                   (version, trained_at, model_type, accuracy, precision,
                    recall, f1_score, train_samples, test_samples,
                    top_features, model_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (version, trained_at, model_type,
                 metrics.get("accuracy"), metrics.get("precision"),
                 metrics.get("recall"), metrics.get("f1_score"),
                 metrics.get("train_samples"), metrics.get("test_samples"),
                 json.dumps(top_features, ensure_ascii=False, cls=_NumpyEncoder) if top_features else None,
                 model_path)
            )

    def get_latest_model_version(self) -> Dict:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM model_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return {}
        d = dict(row)
        if d.get("top_features"):
            d["top_features"] = json.loads(d["top_features"])
        return d

    def get_model_history(self, limit: int = 10) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("factor_results"):
                d["factor_results"] = json.loads(d["factor_results"])
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # 行业因子权重
    # ------------------------------------------------------------------

    def get_sector_weights(self, industry_name: str) -> Dict[str, float]:
        """返回该行业的 {factor_name: weight} 字典。"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT factor_name, weight FROM sector_factor_weights WHERE industry_name=?",
                (industry_name,)
            ).fetchall()
        if not rows:
            return {}
        return {r["factor_name"]: r["weight"] for r in rows}

    def get_all_sector_weights(self) -> Dict[str, Dict[str, float]]:
        """批量加载所有行业的权重，返回 {industry_name: {factor: weight}}"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT industry_name, factor_name, weight FROM sector_factor_weights"
            ).fetchall()
        result: Dict[str, Dict[str, float]] = {}
        for r in rows:
            result.setdefault(r["industry_name"], {})[r["factor_name"]] = r["weight"]
        return result

    def update_sector_weight(self, industry_name: str, factor_name: str,
                             new_weight: float, samples: int,
                             accuracy: float, trade_date: str,
                             old_weight: float = None, reason: str = ""):
        """UPSERT 行业因子权重 + 写入变更历史。"""
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO sector_factor_weights
                   (industry_name, factor_name, weight, samples, accuracy, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(industry_name, factor_name) DO UPDATE SET
                       weight=excluded.weight,
                       samples=excluded.samples,
                       accuracy=excluded.accuracy,
                       last_updated=excluded.last_updated""",
                (industry_name, factor_name, round(new_weight, 6),
                 samples, round(accuracy, 4), trade_date)
            )
            if old_weight is not None and abs(new_weight - old_weight) >= 1e-6:
                conn.execute(
                    """INSERT INTO sector_weight_history
                       (trade_date, industry_name, factor_name, old_weight, new_weight, reason)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (trade_date, industry_name, factor_name,
                     round(old_weight, 6), round(new_weight, 6), reason)
                )

    # ==================================================================
    # 影子 ML 评分
    # ==================================================================

    def save_shadow_ml_scores(self, trade_date: str, scores: List[Dict]):
        """批量写入影子 ML 评分（每日覆盖写入）。"""
        with self._connection() as conn:
            conn.execute("DELETE FROM ml_shadow_scores WHERE trade_date=?", (trade_date,))
            for s in scores:
                conn.execute(
                    """INSERT INTO ml_shadow_scores
                       (trade_date, code, name, ml_score, ml_probability,
                        ml_prediction, existing_score, existing_rank, ml_rank)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_date, s.get("code", ""), s.get("name", ""),
                     s.get("ml_score", 50), s.get("ml_probability", 0.5),
                     s.get("ml_prediction", "neutral"),
                     s.get("existing_score", 50),
                     s.get("existing_rank", 0), s.get("ml_rank", 0))
                )

    def get_shadow_ml_scores(self, trade_date: str) -> List[Dict]:
        """获取某天的影子 ML 评分数据。"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ml_shadow_scores WHERE trade_date=? ORDER BY ml_rank",
                (trade_date,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_shadow_ml_history(self, days: int = 30) -> List[Dict]:
        """获取最近 N 天的影子 ML 评分历史。"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ml_shadow_scores WHERE trade_date >= ? ORDER BY trade_date, ml_rank",
                (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # 个股因子权重 (per-stock factor weights)
    # ==================================================================
    def get_stock_weights(self, code: str) -> Dict[str, float]:
        """返回该股票的 {factor_name: weight} 字典。
        若该股票没有任何记录，返回空 dict（调用方应 fallback 到全局权重）。
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT factor_name, weight FROM stock_factor_weights WHERE code=?",
                (code,)
            ).fetchall()
        if not rows:
            return {}
        return {r["factor_name"]: r["weight"] for r in rows}

    def get_stock_weight_stats(self, code: str) -> Dict[str, Dict]:
        """返回该股票每个因子的完整统计 {factor: {weight, samples, accuracy}}"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT factor_name, weight, samples, accuracy
                   FROM stock_factor_weights WHERE code=?""",
                (code,)
            ).fetchall()
        return {
            r["factor_name"]: {
                "weight": r["weight"],
                "samples": r["samples"],
                "accuracy": r["accuracy"],
            }
            for r in rows
        }

    def get_all_stock_weights(self) -> Dict[str, Dict[str, float]]:
        """批量加载所有股票的权重，返回 {code: {factor: weight}}。
        回测主循环用，避免 N+1 查询。
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT code, factor_name, weight FROM stock_factor_weights"
            ).fetchall()
        result: Dict[str, Dict[str, float]] = {}
        for r in rows:
            result.setdefault(r["code"], {})[r["factor_name"]] = r["weight"]
        return result

    def update_stock_weight(self, code: str, factor_name: str,
                            new_weight: float, samples: int,
                            accuracy: float, trade_date: str,
                            old_weight: float = None, reason: str = ""):
        """UPSERT 个股因子权重 + 写入变更历史。
        old_weight 若提供且不等于 new_weight，则写入 stock_weight_history。
        """
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO stock_factor_weights
                   (code, factor_name, weight, samples, accuracy, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(code, factor_name) DO UPDATE SET
                       weight=excluded.weight,
                       samples=excluded.samples,
                       accuracy=excluded.accuracy,
                       last_updated=excluded.last_updated""",
                (code, factor_name, round(new_weight, 6),
                 samples, round(accuracy, 4), trade_date)
            )
            if old_weight is not None and abs(new_weight - old_weight) >= 1e-6:
                conn.execute(
                    """INSERT INTO stock_weight_history
                       (trade_date, code, factor_name, old_weight, new_weight, reason)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (trade_date, code, factor_name,
                     round(old_weight, 6), round(new_weight, 6), reason)
                )

    def get_stock_weight_history(self, code: str, days: int = 90) -> List[Dict]:
        """获取该股票的权重变更历史"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM stock_weight_history
                   WHERE code=? AND trade_date >= ?
                   ORDER BY trade_date DESC, id DESC""",
                (code, cutoff)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_attribution_for_stock(self, code: str,
                                          days: int = 60) -> List[Dict]:
        """获取该股票最近 N 天的归因记录（供 StockWeightManager 调权用）"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT * FROM factor_attribution
                   WHERE code=? AND trade_date >= ?
                   ORDER BY trade_date DESC""",
                (code, cutoff)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("factor_results"):
                d["factor_results"] = json.loads(d["factor_results"])
            result.append(d)
        return result
