# -*- coding: utf-8 -*-
"""归因 / IC / 权重历史 / 模型版本 相关表操作。

由 database.py 机械拆分(AGENTS.md 技术债偿还): 方法体零改动。
"""
from typing import Dict, List
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta

from src.data.database_const import _NumpyEncoder


class FeedbackMixin:
    """见模块 docstring。"""

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

