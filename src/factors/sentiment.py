"""
情绪因子计算
"""
from typing import Dict


class SentimentFactors:
    """情绪因子计算器"""

    def calculate_all(self, market_sentiment: dict, stock_in_hot: bool = False) -> Dict[str, float]:
        """
        market_sentiment: 从 HotStocksFetcher.calculate_market_sentiment() 获取
        stock_in_hot: 该股票是否在同花顺热股榜
        """
        f = {}

        score = market_sentiment.get("emotion_score", 50)
        f["market_sentiment_score"] = score / 100  # 归一化到 [0, 1]

        limit_up = market_sentiment.get("limit_up", 0)
        f["limit_up_count"] = min(limit_up / 50, 1.0)  # 50家涨停为满分

        limit_down = market_sentiment.get("limit_down", 0)
        f["limit_down_count"] = min(limit_down / 20, 1.0)

        broken = market_sentiment.get("broken", 0)
        f["broken_count"] = min(broken / 20, 1.0)

        max_board = market_sentiment.get("max_board", 0)
        f["max_board_height"] = min(max_board / 5, 1.0)  # 5板为满分

        f["in_hot_list"] = 1.0 if stock_in_hot else 0.0

        # 情绪综合分
        emotion_level = market_sentiment.get("emotion_level", "")
        if "偏热" in emotion_level:
            f["emotion_level"] = 0.8
        elif "中性" in emotion_level:
            f["emotion_level"] = 0.5
        else:
            f["emotion_level"] = 0.2

        return f
