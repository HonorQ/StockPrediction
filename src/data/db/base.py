# -*- coding: utf-8 -*-
"""数据库基类: 连接管理/通用Repository方法/迁移。"""
import logging
from src.data.database_const import SUB_SCORE_DIMS
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from src.data.database_const import DB_PATH

logger = logging.getLogger(__name__)


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
            except Exception as e:
                logger.warning("[base.py:46] BaseStockDB/_connection 失败(忽略): %s", e)
            raise
        finally:
            conn.close()

    def _init_db(self):
        """建库: PRAGMA + 按表域分组建表/索引(子方法由各Mixin提供)。"""
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._create_recommendations_tables(conn)
            self._create_paper_trade_tables(conn)
            self._create_weight_ic_tables(conn)
            self._create_stock_weight_tables(conn)
            self._create_ml_auction_tables(conn)

    # === 通用Repository方法 ===

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """执行查询，返回单行字典。"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def _fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """执行查询，返回所有行的字典列表。"""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def _execute(self, sql: str, params: tuple = ()) -> int:
        """执行写操作，返回影响行数。"""
        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def _fetch_paginated(self, sql: str, params: tuple = (),
                         limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """分页查询。"""
        paginated_sql = f"{sql} LIMIT ? OFFSET ?"
        return self._fetch_all(paginated_sql, params + (limit, offset))

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
