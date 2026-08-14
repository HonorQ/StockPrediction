"""
因子归因引擎 — 推荐失败时定位是哪个因子预测错了

核心逻辑：
1. 每条推荐有22维子评分 sub_scores (0-100)（含护城河moat）
2. 每个因子 score>60=看多, <40=看空, 40-60=中性
3. 跟踪 T+N 后实际收益
4. 判断每个因子的预测是否正确，统计命中率/失败率
5. 过时输入权重调整 + ML重训练
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


BULLISH_THRESHOLD = 60   # >60 视为看多信号
BEARISH_THRESHOLD = 40   # <40 视为看空信号
FAIL_RETURN_THRESHOLD = -3.0  # 收益低于此值视为失败案例 (%)
# 新增：收益缓冲带，避免横盘震荡被误判
CORRECT_RETURN_MIN = 1.0    # 收益 > +1% 才算正确
WRONG_RETURN_MAX = -1.0     # 收益 < -1% 才算错误
                            # 中间 [-1%, +1%] → neutral


def classify_signal(score: float, factor_median: float = 50.0) -> str:
    """判断信号方向。
    使用相对强度：score 须比该股票所有因子的中位数高出/低出足够多才算明确信号。
    """
    deviation = score - factor_median
    if score > BULLISH_THRESHOLD and deviation > 5:
        return "bullish"
    elif score < BEARISH_THRESHOLD and deviation < -5:
        return "bearish"
    return "neutral"


def evaluate_prediction(signal: str, actual_return: float) -> str:
    """带缓冲带的归因评估。"""
    if signal == "neutral":
        return "neutral"
    if signal == "bullish":
        if actual_return > CORRECT_RETURN_MIN:
            return "correct"
        elif actual_return < WRONG_RETURN_MAX:
            return "wrong"
        else:
            return "neutral"  # 横盘不计数
    if signal == "bearish":
        if actual_return < -CORRECT_RETURN_MIN:
            return "correct"
        elif actual_return > -WRONG_RETURN_MAX:
            return "wrong"
        else:
            return "neutral"
    return "neutral"


class FactorAttribution:
    """单条推荐的因子归因"""

    def __init__(self, recommendation: Dict, actual_return: float):
        self.code = recommendation.get("code", "")
        self.name = recommendation.get("name", "")
        self.trade_date = recommendation.get("trade_date", "")
        self.score = recommendation.get("score", 0)
        self.actual_return = actual_return
        self.is_failure = actual_return < FAIL_RETURN_THRESHOLD

        sub_scores = recommendation.get("sub_scores", {})
        factor_median = 50.0
        if sub_scores:
            scores_list = [v for v in sub_scores.values() if isinstance(v, (int, float))]
            if scores_list:
                factor_median = sorted(scores_list)[len(scores_list) // 2]
        self.factor_results = {}
        for factor, score in sub_scores.items():
            signal = classify_signal(score, factor_median)
            result = evaluate_prediction(signal, actual_return)
            self.factor_results[factor] = {
                "score": score,
                "signal": signal,
                "result": result,
            }

    def get_wrong_factors(self) -> List[str]:
        return [
            f for f, r in self.factor_results.items()
            if r["result"] == "wrong"
        ]

    def get_correct_factors(self) -> List[str]:
        return [
            f for f, r in self.factor_results.items()
            if r["result"] == "correct"
        ]

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "name": self.name,
            "trade_date": self.trade_date,
            "score": self.score,
            "actual_return": round(self.actual_return, 2),
            "is_failure": self.is_failure,
            "factor_results": self.factor_results,
        }


class AttributionAggregator:
    """归因聚合器 — 跨时间统计每个因子的表现"""

    def __init__(self, records: List[Dict] = None):
        self.records = records or []

    def add_record(self, record: Dict):
        self.records.append(record)

    def get_factor_stats(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            for factor, result in record["factor_results"].items():
                rows.append({
                    "factor": factor,
                    "code": record["code"],
                    "date": record["trade_date"],
                    "score": result["score"],
                    "signal": result["signal"],
                    "result": result["result"],
                    "actual_return": record["actual_return"],
                    "is_failure": record["is_failure"],
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)

        stats = df.groupby("factor").agg(
            total_cases=("result", "count"),
            correct_count=("result", lambda x: (x == "correct").sum()),
            wrong_count=("result", lambda x: (x == "wrong").sum()),
            bullish_count=("signal", lambda x: (x == "bullish").sum()),
            bearish_count=("signal", lambda x: (x == "bearish").sum()),
            avg_score=("score", "mean"),
            avg_return=("actual_return", "mean"),
            failure_rate=("is_failure", "mean"),
        ).reset_index()

        stats["accuracy"] = stats["correct_count"] / (
            stats["correct_count"] + stats["wrong_count"]
        ).replace(0, np.nan)
        stats["wrong_rate"] = stats["wrong_count"] / (
            stats["correct_count"] + stats["wrong_count"]
        ).replace(0, np.nan)
        stats.fillna(0, inplace=True)
        stats["net_accuracy"] = stats["accuracy"] - stats["wrong_rate"]
        return stats.sort_values("accuracy")

    def get_top_wrong_factors(self, top_n: int = 5) -> List[str]:
        df = self.get_factor_stats()
        if df.empty:
            return []
        return df.nsmallest(top_n, "accuracy")["factor"].tolist()

    def get_top_correct_factors(self, top_n: int = 5) -> List[str]:
        df = self.get_factor_stats()
        if df.empty:
            return []
        return df.nlargest(top_n, "accuracy")["factor"].tolist()

    def summarize(self) -> str:
        df = self.get_factor_stats()
        if df.empty:
            return "暂无归因数据"
        lines = []
        lines.append(f"归因分析: {len(self.records)} 条推荐")
        failures = sum(1 for r in self.records if r.get("is_failure"))
        lines.append(f"失败案例: {failures} 条 (return<{FAIL_RETURN_THRESHOLD}%)")
        lines.append("")
        lines.append("因子准确率排名 (最差->最好):")
        for _, row in df.iterrows():
            bar = "█" * int(abs(row["accuracy"]) * 10)
            if row["total_cases"] >= 3:
                flag = " ⚠️" if row["accuracy"] < 0.4 else " ✅" if row["accuracy"] > 0.7 else ""
                lines.append(
                    f"  {row['factor']:20s} accuracy={row['accuracy']:.0%} "
                    f"({int(row['correct_count'])}/{int(row['correct_count']+row['wrong_count'])})"
                    f"{flag}"
                )
        return "\n".join(lines)
