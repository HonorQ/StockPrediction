"""
基本面因子
估值指标、盈利能力、成长能力、资本结构
"""
import pandas as pd
import numpy as np
from typing import Dict


class FundamentalFactors:
    """基本面因子计算器"""

    def __init__(self):
        pass

    def calculate_all(self, stock_data: Dict, financial_data: Dict = None) -> Dict:
        """
        计算所有基本面因子
        stock_data: 实时行情数据（含PE、PB、总市值等）
        financial_data: 财务数据（可选）
        """
        factors = {}

        # 估值因子
        factors.update(self._valuation_factors(stock_data))

        # 市值因子
        factors.update(self._market_cap_factors(stock_data))

        # 换手率因子
        factors.update(self._turnover_factors(stock_data))

        return factors

    def _valuation_factors(self, data: Dict) -> Dict:
        """估值因子"""
        factors = {}

        pe = data.get("市盈率-动态", data.get("pe", 0)) or 0
        pb = data.get("市净率", data.get("pb", 0)) or 0
        ps = data.get("市销率", data.get("ps", 0)) or 0

        # PE评分
        if pe > 0:
            if pe < 15:
                factors["pe_score"] = 90  # 低估
            elif pe < 25:
                factors["pe_score"] = 70  # 合理
            elif pe < 40:
                factors["pe_score"] = 50  # 偏高
            elif pe < 60:
                factors["pe_score"] = 30  # 高估
            else:
                factors["pe_score"] = 10  # 极度高估
        else:
            factors["pe_score"] = 20  # 亏损

        # PB评分
        if pb > 0:
            if pb < 1:
                factors["pb_score"] = 90  # 破净
            elif pb < 2:
                factors["pb_score"] = 70
            elif pb < 4:
                factors["pb_score"] = 50
            else:
                factors["pb_score"] = 30
        else:
            factors["pb_score"] = 20

        factors["pe"] = pe
        factors["pb"] = pb

        return factors

    def _market_cap_factors(self, data: Dict) -> Dict:
        """市值因子"""
        factors = {}

        total_mv = data.get("总市值", 0) or 0
        circ_mv = data.get("流通市值", 0) or 0

        mv_yi = total_mv / 1e8  # 转换为亿
        circ_yi = circ_mv / 1e8

        # 市值评分（中盘股最佳）
        if 50 <= mv_yi <= 200:
            factors["market_cap_score"] = 80
        elif 200 < mv_yi <= 500:
            factors["market_cap_score"] = 70
        elif 20 < mv_yi < 50:
            factors["market_cap_score"] = 60
        elif mv_yi > 500:
            factors["market_cap_score"] = 50
        else:
            factors["market_cap_score"] = 40  # 小盘股风险高

        # 流通比例
        if total_mv > 0:
            factors["float_ratio"] = round(circ_mv / total_mv, 2)
        else:
            factors["float_ratio"] = 0

        factors["total_mv_yi"] = round(mv_yi, 1)
        factors["circ_mv_yi"] = round(circ_yi, 1)

        return factors

    def _turnover_factors(self, data: Dict) -> Dict:
        """换手率因子"""
        factors = {}

        turnover = data.get("换手率", 0) or 0
        volume_ratio = data.get("量比", 0) or 0

        # 换手率评分
        if 2 <= turnover <= 8:
            factors["turnover_score"] = 80  # 活跃适中
        elif 8 < turnover <= 15:
            factors["turnover_score"] = 60  # 偏活跃
        elif turnover > 15:
            factors["turnover_score"] = 40  # 过度活跃
        elif turnover < 2:
            factors["turnover_score"] = 50  # 不活跃
        else:
            factors["turnover_score"] = 50

        # 量比评分
        if 1 <= volume_ratio <= 2:
            factors["volume_ratio_score"] = 80
        elif 2 < volume_ratio <= 5:
            factors["volume_ratio_score"] = 70
        elif volume_ratio > 5:
            factors["volume_ratio_score"] = 50
        else:
            factors["volume_ratio_score"] = 40

        factors["turnover"] = turnover
        factors["volume_ratio"] = volume_ratio

        return factors

    def get_fundamental_score(self, factors: Dict) -> float:
        """计算基本面综合评分 (0-100)"""
        score = 50

        # 估值得分
        pe_score = factors.get("pe_score", 50)
        pb_score = factors.get("pb_score", 50)
        score += (pe_score - 50) * 0.2
        score += (pb_score - 50) * 0.15

        # 市值得分
        cap_score = factors.get("market_cap_score", 50)
        score += (cap_score - 50) * 0.15

        # 换手率得分
        turnover_score = factors.get("turnover_score", 50)
        score += (turnover_score - 50) * 0.1

        return max(0, min(100, score))
