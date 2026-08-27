# -*- coding: utf-8 -*-
"""纸面交易 / 个股权重 / 行业权重 / 影子ML 相关表操作。

由 database.py 机械拆分(AGENTS.md 技术债偿还): 方法体零改动。
"""
from typing import Dict, List
import sqlite3
import json
from datetime import datetime, timedelta



class PortfolioMixin:
    """见模块 docstring。"""

    def _create_paper_trade_tables(self, conn):
        """建表: 纸面交易域。由 _init_db 调用, 接收已开启 PRAGMA 的连接。"""

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
            except (ValueError, TypeError):
                pass  # entry_date 脏数据时保持 NULL

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
        return self._fetch_all(
            "SELECT * FROM paper_trades WHERE status='open' ORDER BY trade_date DESC"
        )


    def has_open_position(self, code: str) -> bool:
        """检查某个股票是否已有未平仓的 paper_trade（避免重复建仓）"""
        row = self._fetch_one(
            "SELECT 1 FROM paper_trades WHERE code=? AND status='open' LIMIT 1",
            (code,)
        )
        return row is not None


    def get_closed_paper_trades(self, days: int = 30) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._fetch_all(
            """SELECT * FROM paper_trades
               WHERE status='closed' AND trade_date >= ?
               ORDER BY exit_date DESC""",
            (cutoff,)
        )


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
        # NULL 收益(数据缺失/未回填)不能当 0 计入胜率分母——
        # 原实现 r[0] or 0 把 NULL 混入统计，胜率/均值口径失真
        valid = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] is not None]
        null_count = len(rows) - len(valid)
        returns = [r[0] for r in valid]
        pnls = [r[1] for r in valid]
        wins = [v for v in returns if v > 0]
        losses = [v for v in returns if v <= 0]
        reasons = {}
        for r in rows:
            reasons[r[2] or "unknown"] = reasons.get(r[2] or "unknown", 0) + 1
        avg_hold = sum(r[3] or 0 for r in rows) / len(rows) if rows else 0
        return {
            "total": len(rows),
            "valid_count": len(valid),
            "null_return_count": null_count,
            "win_count": len(wins),
            "lose_count": len(losses),
            "win_rate": round(len(wins) / len(valid) * 100, 2) if valid else 0,
            "avg_return": round(sum(returns) / len(valid), 2) if valid else 0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_holding_days": round(avg_hold, 1),
            "exit_reasons": reasons,
        }


    def get_stock_weights(self, code: str) -> Dict[str, float]:
        """返回该股票的 {factor_name: weight} 字典。
        若该股票没有任何记录，返回空 dict（调用方应 fallback 到全局权重）。
        """
        rows = self._fetch_all(
            "SELECT factor_name, weight FROM stock_factor_weights WHERE code=?",
            (code,)
        )
        if not rows:
            return {}
        return {r["factor_name"]: r["weight"] for r in rows}


    def get_stock_weight_stats(self, code: str) -> Dict[str, Dict]:
        """返回该股票每个因子的完整统计 {factor: {weight, samples, accuracy}}"""
        rows = self._fetch_all(
            """SELECT factor_name, weight, samples, accuracy
               FROM stock_factor_weights WHERE code=?""",
            (code,)
        )
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
        rows = self._fetch_all(
            "SELECT code, factor_name, weight FROM stock_factor_weights"
        )
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

