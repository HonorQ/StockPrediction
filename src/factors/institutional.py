"""
机构持股因子
机构持股家数、持股比例、机构调研
"""
from typing import Dict


class InstitutionalFactors:
    """机构持股因子计算器"""

    def __init__(self):
        pass

    def calculate_all(self, stock_code: str, market_data: Dict = None) -> Dict:
        """
        计算机构持股因子
        stock_code: 股票代码
        market_data: 市场数据（含机构持股信息）
        """
        factors = {}

        # 从市场数据中提取机构信息
        if market_data:
            # 机构持股家数
            inst_count = market_data.get("机构持股家数", 0) or 0
            factors["inst_holding_count"] = inst_count

            # 机构持股比例
            inst_ratio = market_data.get("机构持股比例", 0) or 0
            factors["inst_holding_ratio"] = inst_ratio

            # 基金持股家数
            fund_count = market_data.get("基金持股家数", 0) or 0
            factors["fund_holding_count"] = fund_count

            # QFII持股
            qfii = market_data.get("QFII持股", 0) or 0
            factors["qfii_holding"] = qfii

            # 社保基金持股
            social_security = market_data.get("社保基金持股", 0) or 0
            factors["social_security_holding"] = social_security

            # 机构评级
            rating = market_data.get("机构评级", "")
            factors["inst_rating"] = self._parse_rating(rating)

            # 近期调研
            research = market_data.get("近期调研", 0) or 0
            factors["recent_research"] = research

        # 计算机构因子评分
        factors.update(self._calculate_scores(factors))

        return factors

    def _parse_rating(self, rating: str) -> int:
        """解析机构评级"""
        if not rating:
            return 0
        rating_map = {
            "买入": 5, "强烈推荐": 5, "强推": 5,
            "推荐": 4, "增持": 4, "优于大市": 4,
            "中性": 3, "持有": 3, "同步大市": 3,
            "减持": 2, "弱于大市": 2,
            "卖出": 1,
        }
        for key, val in rating_map.items():
            if key in str(rating):
                return val
        return 3

    def _calculate_scores(self, factors: Dict) -> Dict:
        """计算各项评分"""
        scores = {}

        # 机构持股家数评分
        count = factors.get("inst_holding_count", 0)
        if count >= 20:
            scores["inst_count_score"] = 90
        elif count >= 10:
            scores["inst_count_score"] = 75
        elif count >= 5:
            scores["inst_count_score"] = 60
        elif count > 0:
            scores["inst_count_score"] = 45
        else:
            scores["inst_count_score"] = 30

        # 机构持股比例评分
        ratio = factors.get("inst_holding_ratio", 0)
        if ratio >= 30:
            scores["inst_ratio_score"] = 90
        elif ratio >= 20:
            scores["inst_ratio_score"] = 75
        elif ratio >= 10:
            scores["inst_ratio_score"] = 60
        elif ratio > 0:
            scores["inst_ratio_score"] = 45
        else:
            scores["inst_ratio_score"] = 30

        # 机构评级评分
        rating = factors.get("inst_rating", 3)
        scores["inst_rating_score"] = rating * 20

        # 近期调研评分
        research = factors.get("recent_research", 0)
        if research >= 5:
            scores["research_score"] = 80
        elif research >= 3:
            scores["research_score"] = 65
        elif research > 0:
            scores["research_score"] = 55
        else:
            scores["research_score"] = 40

        # 综合机构得分
        scores["institutional_score"] = (
            scores.get("inst_count_score", 50) * 0.3 +
            scores.get("inst_ratio_score", 50) * 0.3 +
            scores.get("inst_rating_score", 50) * 0.2 +
            scores.get("research_score", 50) * 0.2
        )

        return scores

    def get_institutional_score(self, factors: Dict) -> float:
        """获取机构因子综合评分 (0-100)"""
        return factors.get("institutional_score", 50)
