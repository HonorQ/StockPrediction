"""
ATR动态止损 + 仓位管理

由 backtest.py 拆分(AGENTS.md 技术债偿还)。
"""
import pandas as pd
from typing import Dict



class ATRStopLoss:
    """ATR动态止损"""

    def __init__(self, atr_period: int = 14, multiplier: float = 2.0):
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["最高"] if "最高" in df.columns else df["high"]
        low = df["最低"] if "最低" in df.columns else df["low"]
        close = df["收盘"] if "收盘" in df.columns else df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        return atr

    def get_stop_loss(self, entry_price: float, atr: float, direction: str = "long") -> float:
        if direction == "long":
            stop_loss = entry_price - atr * self.multiplier
        else:
            stop_loss = entry_price + atr * self.multiplier
        return round(stop_loss, 2)

    def get_position_size(self, capital: float, risk_per_trade: float,
                         entry_price: float, stop_loss: float) -> int:
        risk_amount = capital * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        shares = int(risk_amount / risk_per_share / 100) * 100
        return max(100, shares)


class PositionManager:
    """仓位管理器"""

    def __init__(self, max_position_pct: float = 0.25, min_position_pct: float = 0.05,
                 max_stop_loss_pct: float = -0.08, volatility_adaptive: bool = True,
                 volatility_factor: float = 1.5):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_stop_loss_pct = max_stop_loss_pct  # ATR止损空间上限(小数)
        self.volatility_adaptive = volatility_adaptive  # 是否启用波动率自适应止损
        self.volatility_factor = volatility_factor  # 波动率调整系数

    def calculate_position(self, score: float, capital: float, price: float) -> Dict:
        if score >= 85:
            position_pct = self.max_position_pct
            risk_level = "高"
        elif score >= 75:
            position_pct = 0.20
            risk_level = "中高"
        elif score >= 65:
            position_pct = 0.15
            risk_level = "中"
        elif score >= 55:
            position_pct = 0.10
            risk_level = "中低"
        else:
            position_pct = self.min_position_pct
            risk_level = "低"

        position_amount = capital * position_pct
        shares = int(position_amount / price / 100) * 100

        return {
            "position_pct": position_pct,
            "position_amount": round(position_amount, 2),
            "shares": max(100, shares),
            "risk_level": risk_level,
        }

    def calculate_stop_loss(self, entry_price: float, atr: float, score: float,
                           volatility_ratio: float = 1.0) -> Dict:
        """
        计算动态止损
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            score: 综合评分
            volatility_ratio: 波动率比率 (当前波动率/历史平均波动率)
        """
        if score >= 85:
            multiplier = 2.5
        elif score >= 75:
            multiplier = 2.0
        elif score >= 65:
            multiplier = 1.5
        else:
            multiplier = 1.0

        # ATR 缺失(历史数据不足/获取失败)时退回最大固定百分比止损。
        # 原实现 atr=0 → stop_loss==entry_price → 开仓当日即被"止损"平仓
        if not atr or atr <= 0:
            fallback_pct = self.max_stop_loss_pct or -0.08
            return {
                "stop_loss": round(entry_price * (1 + fallback_pct), 2),
                "stop_loss_pct": round(fallback_pct * 100, 2),
                "atr_multiplier": 0,
                "volatility_ratio": 0,
            }

        # 波动率自适应调整：高波动率时放宽止损，低波动率时收紧止损
        if self.volatility_adaptive and volatility_ratio > 0:
            # 波动率翻倍，止损放宽 factor 倍
            volatility_adjustment = 1 + (volatility_ratio - 1) * (self.volatility_factor - 1)
            multiplier *= volatility_adjustment

        stop_loss = entry_price - atr * multiplier
        stop_loss_pct = (stop_loss / entry_price - 1) * 100

        # 止损空间上限:防止高价股/高ATR票止损被设得过宽(如协创数据-11.4%被穿透到-22.8%)
        max_sl_pct = self.max_stop_loss_pct * 100  # 转百分数,如 -8.0
        if stop_loss_pct < max_sl_pct:
            stop_loss_pct = max_sl_pct
            stop_loss = round(entry_price * (1 + self.max_stop_loss_pct), 2)

        return {
            "stop_loss": round(stop_loss, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "atr_multiplier": round(multiplier, 2),
            "volatility_ratio": round(volatility_ratio, 2),
        }

    def calculate_take_profit(self, entry_price: float, atr: float, score: float) -> Dict:
        if score >= 85:
            multiplier = 4.0
        elif score >= 75:
            multiplier = 3.0
        elif score >= 65:
            multiplier = 2.5
        else:
            multiplier = 2.0

        # ATR 缺失时退回固定百分比止盈(与止损兜底同因：atr=0 → 止盈价==入场价)
        if not atr or atr <= 0:
            fallback_pct = 10.0
            return {
                "take_profit": round(entry_price * (1 + fallback_pct / 100), 2),
                "take_profit_pct": fallback_pct,
                "atr_multiplier": 0,
            }

        take_profit = entry_price + atr * multiplier
        take_profit_pct = (take_profit / entry_price - 1) * 100

        return {
            "take_profit": round(take_profit, 2),
            "take_profit_pct": round(take_profit_pct, 2),
            "atr_multiplier": multiplier,
        }
