# -*- coding: utf-8 -*-
"""推荐记录 / 因子快照 / 回测结果 相关表操作。

由 database.py 机械拆分(AGENTS.md 技术债偿还): 方法体零改动。
"""
from typing import Dict, List, Optional
import sqlite3
import json
import logging
from datetime import datetime, timedelta

from src.data.database_const import SUB_SCORE_DIMS, _NumpyEncoder

logger = logging.getLogger(__name__)


class RecommendationsMixin:
    """见模块 docstring。"""

    def _create_recommendations_tables(self, conn):
        """建表: 推荐/因子快照/回测/归因域。由 _init_db 调用, 接收已开启 PRAGMA 的连接。"""

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
                # OR IGNORE：并发下唯一索引兜底，冲突行由后续 UPDATE 路径处理
                conn.execute(
                    """INSERT OR IGNORE INTO recommendations
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


    def get_recent_recommendations(self, days: int = 30,
                                   modes: List[str] = None) -> List[Dict]:
        """取最近 N 天推荐。
        modes: 提供时只返回指定模式(如报告回顾用 ("afternoon","premarket","feedback_training")
               排除回测行)；None=不过滤(旧行为)。"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            if modes:
                placeholders = ",".join("?" for _ in modes)
                rows = conn.execute(
                    f"""SELECT * FROM recommendations WHERE trade_date >= ?
                        AND (mode IN ({placeholders}) OR mode IS NULL)
                        ORDER BY trade_date DESC, rank ASC""", (cutoff, *modes)
                ).fetchall()
            else:
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
            except Exception as e:
                logger.warning(f"解析sub_scores失败: {e}")
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
            except Exception as e:
                logger.warning(f"解析factors失败: {e}")
                return None
        if isinstance(factors, dict) and "z_scores" not in factors:
            return factors
        return None


    def get_factor_history(self, days: int = 365) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self._fetch_all(
            """SELECT * FROM factor_snapshots WHERE trade_date >= ?
               ORDER BY trade_date DESC""", (cutoff,)
        )
        for d in rows:
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
        return rows


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
            # 因子快照产自盘前全量计算，只 JOIN 盘前类推荐行：
            # 1) 语义一致；2) 同码同日存在 premarket/afternoon 两行时防止面板样本重复。
            # mode 取值实况：'feedback_training'(历史5980条) / 'premarket'(现行) /
            # NULL(最早数据) 均属盘前口径；afternoon/backtest 排除
            " WHERE fs.trade_date >= ? AND fs.trade_date <= ?"
            " AND (r.mode IN ('premarket', 'feedback_training') OR r.mode IS NULL)"
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

