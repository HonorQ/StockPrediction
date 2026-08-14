"""
人气因子
股吧人气排名、人气变化、粉丝占比
"""
import pandas as pd
from typing import Dict, List


class PopularityFactors:
    """人气因子计算器"""

    def __init__(self):
        pass

    def calculate_all(self, hot_data: Dict, stock_code: str = None) -> Dict:
        """
        计算人气因子
        hot_data: 热点数据（含热股榜、人气榜等）
        stock_code: 股票代码
        """
        factors = {}

        # 热股榜排名
        factors.update(self._hot_rank_factors(hot_data, stock_code))

        # 涨停/跌停统计
        factors.update(self._limit_factors(hot_data))

        return factors

    def _hot_rank_factors(self, data: Dict, stock_code: str) -> Dict:
        """热股榜因子"""
        factors = {}

        hot_rank = data.get("hot_rank", 0)
        in_hot_list = data.get("in_hot_list", False)
        hot_change = data.get("hot_change", 0)

        # 热度排名评分
        if in_hot_list:
            if hot_rank <= 10:
                factors["hot_rank_score"] = 90
            elif hot_rank <= 30:
                factors["hot_rank_score"] = 80
            elif hot_rank <= 50:
                factors["hot_rank_score"] = 70
            else:
                factors["hot_rank_score"] = 60
        else:
            factors["hot_rank_score"] = 40

        # 排名变化
        if hot_change > 0:
            factors["hot_rising"] = 1  # 排名上升
        elif hot_change < 0:
            factors["hot_falling"] = 1  # 排名下降
        else:
            factors["hot_rising"] = 0
            factors["hot_falling"] = 0

        factors["hot_rank"] = hot_rank
        factors["hot_change"] = hot_change

        return factors

    def _limit_factors(self, data: Dict) -> Dict:
        """涨停/跌停统计因子"""
        factors = {}

        limit_up = data.get("limit_up", 0)
        limit_down = data.get("limit_down", 0)
        broken = data.get("broken", 0)
        max_board = data.get("max_board", 0)

        # 涨停家数评分
        if limit_up > 100:
            factors["limit_up_score"] = 90
        elif limit_up > 50:
            factors["limit_up_score"] = 70
        elif limit_up > 20:
            factors["limit_up_score"] = 60
        else:
            factors["limit_up_score"] = 40

        # 涨跌停比评分
        if limit_down > 0:
            ratio = limit_up / limit_down
        else:
            ratio = limit_up

        if ratio > 10:
            factors["limit_ratio_score"] = 90
        elif ratio > 5:
            factors["limit_ratio_score"] = 70
        else:
            factors["limit_ratio_score"] = 50

        # 连板高度评分
        if max_board >= 5:
            factors["board_height_score"] = 90
        elif max_board >= 3:
            factors["board_height_score"] = 70
        else:
            factors["board_height_score"] = 50

        factors["limit_up_count"] = limit_up
        factors["limit_down_count"] = limit_down
        factors["max_board_height"] = max_board

        return factors

    def get_popularity_score(self, factors: Dict) -> float:
        """计算人气综合评分 (0-100)"""
        score = 50

        # 热度排名
        hot_score = factors.get("hot_rank_score", 40)
        score += (hot_score - 50) * 0.3

        # 涨停家数
        limit_score = factors.get("limit_up_score", 50)
        score += (limit_score - 50) * 0.2

        # 涨跌停比
        ratio_score = factors.get("limit_ratio_score", 50)
        score += (ratio_score - 50) * 0.15

        # 连板高度
        board_score = factors.get("board_height_score", 50)
        score += (board_score - 50) * 0.15

        return max(0, min(100, score))
