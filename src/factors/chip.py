"""
筹码分布因子
参考 InStock 项目实现
"""
import pandas as pd
import numpy as np
from typing import Dict


class ChipDistribution:
    """筹码分布计算器"""

    def __init__(self, precision: int = 100):
        """
        precision: 价格档位精度（越大越精确，计算越慢）
        """
        self.precision = precision

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        计算筹码分布
        返回: {
            profit_ratio: 获利比例,
            avg_cost: 平均成本,
            concentration_90: 90%筹码集中度,
            concentration_70: 70%筹码集中度,
            price_peak: 筹码峰值价格,
            support: 支撑位,
            pressure: 压力位,
        }
        """
        if df is None or df.empty or len(df) < 30:
            return self._empty_result()

        # 获取价格和成交量数据
        open_ = df["开盘"] if "开盘" in df.columns else df["open"]
        high = df["最高"] if "最高" in df.columns else df["high"]
        low = df["最低"] if "最低" in df.columns else df["low"]
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]
        turnover = df["换手率"] if "换手率" in df.columns else None

        # 价格范围
        price_min = low.min()
        price_max = high.max()

        if price_min == price_max:
            return self._empty_result()

        # 价格档位
        price_range = np.linspace(price_min, price_max, self.precision)

        # 初始化筹码分布
        chips = np.zeros(self.precision)

        # 三角形分布模型计算每日筹码分布
        for i in range(len(df)):
            o, h, l, c = open_.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
            vol = volume.iloc[i]

            if vol == 0 or h == l:
                continue

            # 每日平均价格（四价均值）
            avg_price = (o + h + l + c) / 4

            # 三角形分布参数
            low_idx = max(0, int((l - price_min) / (price_max - price_min) * (self.precision - 1)))
            high_idx = min(self.precision - 1, int((h - price_min) / (price_max - price_min) * (self.precision - 1)))
            avg_idx = int((avg_price - price_min) / (price_max - price_min) * (self.precision - 1))
            avg_idx = max(low_idx, min(high_idx, avg_idx))

            # 三角形分布
            if avg_idx > low_idx:
                for j in range(low_idx, avg_idx + 1):
                    chips[j] += vol * (j - low_idx) / (avg_idx - low_idx + 1)

            if high_idx > avg_idx:
                for j in range(avg_idx, high_idx + 1):
                    chips[j] += vol * (high_idx - j) / (high_idx - avg_idx + 1)

            # 换手率衰减（旧筹码按换手率衰减）
            if turnover is not None and i > 0:
                decay = 1 - turnover.iloc[i] / 100
                chips[:low_idx] *= decay

        # 归一化
        total_chips = chips.sum()
        if total_chips > 0:
            chips /= total_chips

        # 计算指标
        current_price = close.iloc[-1]

        # 获利比例（当前价格以下的筹码比例）
        current_idx = int((current_price - price_min) / (price_max - price_min) * (self.precision - 1))
        current_idx = max(0, min(self.precision - 1, current_idx))
        profit_ratio = chips[:current_idx + 1].sum()

        # 平均成本（加权平均价格）
        avg_cost = np.sum(price_range * chips)

        # 筹码集中度
        sorted_chips = np.sort(chips)[::-1]
        cumsum = np.cumsum(sorted_chips)

        # 90%筹码集中度
        idx_90 = np.searchsorted(cumsum, 0.9)
        price_90_low = price_range[np.where(chips >= sorted_chips[min(idx_90, self.precision - 1)])[0][0]]
        price_90_high = price_range[np.where(chips >= sorted_chips[min(idx_90, self.precision - 1)])[0][-1]]
        concentration_90 = (price_90_high - price_90_low) / avg_cost if avg_cost > 0 else 0

        # 70%筹码集中度
        idx_70 = np.searchsorted(cumsum, 0.7)
        price_70_low = price_range[np.where(chips >= sorted_chips[min(idx_70, self.precision - 1)])[0][0]]
        price_70_high = price_range[np.where(chips >= sorted_chips[min(idx_70, self.precision - 1)])[0][-1]]
        concentration_70 = (price_70_high - price_70_low) / avg_cost if avg_cost > 0 else 0

        # 筹码峰值（最大筹码对应的价格）
        peak_idx = np.argmax(chips)
        price_peak = price_range[peak_idx]

        # 支撑位和压力位
        # 支撑位：当前价格下方筹码最密集的价位
        below_current = chips[:current_idx + 1]
        if len(below_current) > 0 and below_current.sum() > 0:
            support_idx = np.argmax(below_current)
            support = price_range[support_idx]
        else:
            support = price_min

        # 压力位：当前价格上方筹码最密集的价位
        above_current = chips[current_idx:]
        if len(above_current) > 0 and above_current.sum() > 0:
            pressure_idx = np.argmax(above_current) + current_idx
            pressure = price_range[min(pressure_idx, self.precision - 1)]
        else:
            pressure = price_max

        return {
            "profit_ratio": round(profit_ratio, 4),
            "avg_cost": round(avg_cost, 2),
            "concentration_90": round(concentration_90, 4),
            "concentration_70": round(concentration_70, 4),
            "price_peak": round(price_peak, 2),
            "support": round(support, 2),
            "pressure": round(pressure, 2),
            "price_above_peak": 1 if current_price > price_peak else 0,
            "profit_ratio_high": 1 if profit_ratio > 0.7 else 0,
        }

    def get_chip_score(self, df: pd.DataFrame) -> float:
        """
        筹码分布综合评分（0-100）
        """
        result = self.calculate(df)
        if not result:
            return 50

        score = 50
        current_price = df["收盘"].iloc[-1] if "收盘" in df.columns else df["close"].iloc[-1]

        # 获利比例适中加分（50%-80%最佳）
        pr = result["profit_ratio"]
        if 0.5 <= pr <= 0.8:
            score += 15
        elif pr > 0.9:
            score -= 10  # 获利盘太多，抛压大

        # 筹码集中度加分（越集中越好）
        conc = result["concentration_70"]
        if conc < 0.1:
            score += 15
        elif conc < 0.2:
            score += 10

        # 价格在筹码峰上方加分
        if result["price_above_peak"]:
            score += 10

        # 接近支撑位加分
        if current_price > 0:
            dist_to_support = (current_price - result["support"]) / current_price
            if dist_to_support < 0.03:
                score += 10

        # 远离压力位加分
        if current_price > 0:
            dist_to_pressure = (result["pressure"] - current_price) / current_price
            if dist_to_pressure > 0.05:
                score += 5

        return max(0, min(100, score))

    def _empty_result(self) -> Dict:
        return {
            "profit_ratio": 0,
            "avg_cost": 0,
            "concentration_90": 0,
            "concentration_70": 0,
            "price_peak": 0,
            "support": 0,
            "pressure": 0,
            "price_above_peak": 0,
            "profit_ratio_high": 0,
        }
