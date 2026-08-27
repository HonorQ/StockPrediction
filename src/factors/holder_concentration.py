"""
股东人数/筹码集中度因子（基于新浪财务数据）
判断标准：股东总数减少=筹码集中(利好)，股东总数增加=筹码分散(利空)
"""
import logging
import pandas as pd
from typing import Dict
import os, json, time

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HolderConcentrationFactor:
    """股东人数/筹码集中度因子（季度数据，缓存7天）"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_file = os.path.join(ROOT, "cache", "holder_cache.json")
        self._load_cache()

    def _load_cache(self):
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            self._cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("[holder_concentration.py:36] HolderConcentrationFactor/_save_cache 失败(忽略): %s", e)

    def analyze(self, stock_code: str) -> Dict:
        result = {
            "holder_score": 50,
            "holder_trend": "neutral",
            "holder_quarter_change_pct": 0,
            "holder_yoy_change_pct": 0,
            "total_holders": 0,
            "avg_shares": 0,
        }
        try:
            clean_code = stock_code.lower().replace("sh", "").replace("sz", "").replace("bj", "").split(".")[0]
            cache_key = f"holder_{clean_code}"

            cached = self._cache.get(cache_key)
            if cached and time.time() - cached.get("ts", 0) < 86400 * 7:
                data = cached.get("data")
                if data:
                    return self._score_from_history(data)

            import akshare as ak
            df = ak.stock_main_stock_holder(stock=clean_code)
            if df is None or df.empty or "股东总数" not in df.columns:
                return result

            df["截至日期"] = pd.to_datetime(df["截至日期"])
            df = df.sort_values("截至日期", ascending=False)
            unique_dates = df["截至日期"].unique()
            history = []
            for d in unique_dates:
                rows = df[df["截至日期"] == d]
                holders = rows["股东总数"].dropna()
                avg = rows["平均持股数"].dropna()
                if not holders.empty:
                    history.append({
                        "date": str(d.date()),
                        "holders": int(holders.iloc[0]),
                        "avg_shares": int(avg.iloc[0]) if not avg.empty else 0,
                    })

            self._cache[cache_key] = {"ts": time.time(), "data": history}
            self._save_cache()
            return self._score_from_history(history)

        except Exception as e:
            logger.warning(f"分析股东集中度失败: {e}")
            return result

    def _score_from_history(self, history: list) -> Dict:
        result = {
            "holder_score": 50,
            "holder_trend": "neutral",
            "holder_quarter_change_pct": 0,
            "holder_yoy_change_pct": 0,
            "total_holders": 0,
            "avg_shares": 0,
        }
        if len(history) < 1:
            return result

        latest = history[0]
        result["total_holders"] = latest["holders"]
        result["avg_shares"] = latest["avg_shares"]

        if len(history) >= 2:
            prev = history[1]
            q_change = (latest["holders"] - prev["holders"]) / max(prev["holders"], 1) * 100
            result["holder_quarter_change_pct"] = round(q_change, 2)
        if len(history) >= 4:
            yoy = history[3]
            y_change = (latest["holders"] - yoy["holders"]) / max(yoy["holders"], 1) * 100
            result["holder_yoy_change_pct"] = round(y_change, 2)

        q_change = result["holder_quarter_change_pct"]
        if q_change < -5:
            result["holder_trend"] = "concentrating"
            result["holder_score"] = 80
        elif q_change < -2:
            result["holder_trend"] = "slightly_concentrating"
            result["holder_score"] = 65
        elif q_change > 5:
            result["holder_trend"] = "spreading"
            result["holder_score"] = 20
        elif q_change > 2:
            result["holder_trend"] = "slightly_spreading"
            result["holder_score"] = 35
        else:
            result["holder_trend"] = "stable"
            result["holder_score"] = 50

        if result["holder_yoy_change_pct"] < -10:
            result["holder_score"] = min(result["holder_score"] + 10, 100)

        return result
