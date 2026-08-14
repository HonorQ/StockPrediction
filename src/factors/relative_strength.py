"""
相对强弱因子
个股相对于大盘、板块的强弱表现
"""
import pandas as pd
import numpy as np
from typing import Dict


class RelativeStrength:
    """相对强弱因子计算器"""

    def __init__(self):
        pass

    def calculate_all(self, stock_df: pd.DataFrame, market_df: pd.DataFrame = None,
                      sector_df: pd.DataFrame = None) -> Dict:
        """
        计算相对强弱因子
        stock_df: 个股历史数据
        market_df: 大盘历史数据（如沪深300）
        sector_df: 板块历史数据
        """
        factors = {}

        if stock_df is None or stock_df.empty or len(stock_df) < 20:
            return factors

        close = stock_df["收盘"] if "收盘" in stock_df.columns else stock_df["close"]

        # === 个股自身强弱 ===
        factors.update(self._price_strength(close))

        # === 相对大盘强弱 ===
        if market_df is not None and not market_df.empty:
            factors.update(self._relative_to_market(close, market_df))

        # === 相对板块强弱 ===
        if sector_df is not None and not sector_df.empty:
            factors.update(self._relative_to_sector(close, sector_df))

        return factors

    def _price_strength(self, close: pd.Series) -> Dict:
        """个股价格强度"""
        factors = {}

        # 多周期动量
        if len(close) >= 5:
            factors["momentum_5d"] = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2)
        if len(close) >= 10:
            factors["momentum_10d"] = round((close.iloc[-1] / close.iloc[-10] - 1) * 100, 2)
        if len(close) >= 20:
            factors["momentum_20d"] = round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 2)
        if len(close) >= 60:
            factors["momentum_60d"] = round((close.iloc[-1] / close.iloc[-60] - 1) * 100, 2)

        # 创新高/新低
        if len(close) >= 20:
            high_20 = close.tail(20).max()
            low_20 = close.tail(20).min()
            factors["near_20d_high"] = round(close.iloc[-1] / high_20, 4)
            factors["near_20d_low"] = round(close.iloc[-1] / low_20, 4)

        if len(close) >= 60:
            high_60 = close.tail(60).max()
            factors["near_60d_high"] = round(close.iloc[-1] / high_60, 4)

        # 价格位置（0-1，越高越接近最高点）
        if len(close) >= 60:
            high_60 = close.tail(60).max()
            low_60 = close.tail(60).min()
            if high_60 > low_60:
                factors["price_position"] = round((close.iloc[-1] - low_60) / (high_60 - low_60), 4)
            else:
                factors["price_position"] = 0.5

        # 强度评分
        factors["strength_score"] = self._calculate_strength_score(factors)

        return factors

    def _relative_to_market(self, stock_close: pd.Series, market_df: pd.DataFrame) -> Dict:
        """相对大盘强弱"""
        factors = {}

        market_close = market_df["收盘"] if "收盘" in market_df.columns else market_df["close"]

        # 确保长度一致
        min_len = min(len(stock_close), len(market_close))
        if min_len < 20:
            return factors

        stock_ret_5d = stock_close.iloc[-1] / stock_close.iloc[-min(5, min_len)] - 1
        market_ret_5d = market_close.iloc[-1] / market_close.iloc[-min(5, min_len)] - 1

        stock_ret_20d = stock_close.iloc[-1] / stock_close.iloc[-min(20, min_len)] - 1
        market_ret_20d = market_close.iloc[-1] / market_close.iloc[-min(20, min_len)] - 1

        # 相对强度（超额收益）
        factors["relative_strength_5d"] = round((stock_ret_5d - market_ret_5d) * 100, 2)
        factors["relative_strength_20d"] = round((stock_ret_20d - market_ret_20d) * 100, 2)

        # Beta系数估算
        if min_len >= 20:
            stock_returns = stock_close.pct_change().dropna().tail(20)
            market_returns = market_close.pct_change().dropna().tail(20)
            if len(stock_returns) > 0 and len(market_returns) > 0:
                correlation = stock_returns.corr(market_returns)
                stock_vol = stock_returns.std()
                market_vol = market_returns.std()
                if market_vol > 0:
                    beta = correlation * stock_vol / market_vol
                    factors["beta"] = round(beta, 4)

        return factors

    def _relative_to_sector(self, stock_close: pd.Series, sector_df: pd.DataFrame) -> Dict:
        """相对板块强弱"""
        factors = {}

        sector_close = sector_df["收盘"] if "收盘" in sector_df.columns else sector_df["close"]

        min_len = min(len(stock_close), len(sector_close))
        if min_len < 20:
            return factors

        stock_ret = stock_close.iloc[-1] / stock_close.iloc[-min(20, min_len)] - 1
        sector_ret = sector_close.iloc[-1] / sector_close.iloc[-min(20, min_len)] - 1

        factors["sector_relative_strength"] = round((stock_ret - sector_ret) * 100, 2)

        return factors

    def _calculate_strength_score(self, factors: Dict) -> float:
        """计算强度综合评分 (0-100)"""
        score = 50

        # 5日动量
        m5 = factors.get("momentum_5d", 0)
        if m5 > 5:
            score += 15
        elif m5 > 2:
            score += 10
        elif m5 > 0:
            score += 5
        elif m5 < -5:
            score -= 15
        elif m5 < -2:
            score -= 10

        # 20日动量
        m20 = factors.get("momentum_20d", 0)
        if m20 > 10:
            score += 15
        elif m20 > 5:
            score += 10
        elif m20 > 0:
            score += 5

        # 价格位置
        pos = factors.get("price_position", 0.5)
        if pos > 0.8:
            score += 10
        elif pos > 0.6:
            score += 5

        # 接近20日高点
        near_high = factors.get("near_20d_high", 0)
        if near_high > 0.98:
            score += 10

        return max(0, min(100, score))

    def get_relative_strength_score(self, factors: Dict) -> float:
        """获取相对强弱综合评分"""
        return factors.get("strength_score", 50)
