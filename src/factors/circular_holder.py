"""
流通股东因子（基于新浪财务数据）
跟踪前十大流通股东变化，判断机构增/减持
"""
import logging
import pandas as pd
from typing import Dict
import os, json, time

logger = logging.getLogger(__name__)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CirculateHolderFactor:
    """流通股东跟踪因子（季度数据，缓存7天）"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_file = os.path.join(ROOT, "cache", "circulate_cache.json")
        self._load_cache()

    def _load_cache(self):
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, default=str)
        except Exception:
            pass

    def analyze(self, stock_code: str) -> Dict:
        result = {
            "circulate_score": 50,
            "inst_holding_change": "neutral",
            "northbound_change": "neutral",
            "top10_holding_pct": 0,
        }
        try:
            clean_code = stock_code.lower().replace("sh", "").replace("sz", "").replace("bj", "").split(".")[0]
            cache_key = f"circ_{clean_code}"

            cached = self._cache.get(cache_key)
            if cached and time.time() - cached.get("ts", 0) < 86400 * 7:
                return self._calc_score(cached.get("data", {}))

            import akshare as ak
            df = ak.stock_circulate_stock_holder(stock=clean_code)
            if df is None or df.empty:
                return result

            df["截止日期"] = pd.to_datetime(df["截止日期"])
            df = df.sort_values("截止日期", ascending=False)
            latest_date = df["截止日期"].iloc[0]
            current = df[df["截止日期"] == latest_date].copy()
            total_pct = current["占流通股比例"].sum() if "占流通股比例" in current.columns else 0

            data = {
                "top10_pct": float(total_pct),
                "northbound_pct": 0,
                "inst_names": current["股东名称"].tolist() if "股东名称" in current.columns else [],
            }

            northbound_mask = current["股东名称"].str.contains("香港中央结算|香港上海|渣打|汇丰|摩根|花旗", na=False)
            if northbound_mask.any():
                nb_row = current[northbound_mask]
                if "占流通股比例" in nb_row.columns:
                    data["northbound_pct"] = float(nb_row["占流通股比例"].sum())

            self._cache[cache_key] = {"ts": time.time(), "data": data}
            self._save_cache()
            return self._calc_score(data)

        except Exception as e:
            logger.debug("CirculateHolderFactor.analyze(%s) 失败: %s", stock_code, e)
            return result

    def _calc_score(self, data: dict) -> Dict:
        score = 50
        top10 = data.get("top10_pct", 0)
        northbound = data.get("northbound_pct", 0)

        if top10 > 60:
            score += 15
        elif top10 > 40:
            score += 8
        elif top10 < 20:
            score -= 10

        if northbound > 5:
            score += 10
        elif northbound > 2:
            score += 5
        elif northbound > 0:
            score += 2

        inst_names = data.get("inst_names", [])
        inst_known = sum(1 for n in inst_names if "基金" in n or "保险" in n or "社保" in n or "证金" in n or "汇金" in n)
        if inst_known >= 3:
            score += 10

        return {
            "circulate_score": max(0, min(100, score)),
            "inst_holding_change": "positive" if northbound > 2 else "neutral",
            "northbound_change": "positive" if northbound > 3 else "neutral",
            "top10_holding_pct": round(top10, 1),
        }
