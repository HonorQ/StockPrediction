"""
技术因子计算（趋势 + 动量 + 形态）
"""
import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import safe_divide, calculate_rsi, calculate_atr


class TechnicalFactors:
    """技术因子计算器"""

    def __init__(self, df: pd.DataFrame, config: dict = None):
        """
        df: 个股历史日线数据（需至少60天）
        config: 技术参数配置
        """
        self.df = df.copy()
        self.cfg = config or {}
        self._precompute()

    def _precompute(self):
        """预计算常用指标"""
        c = self.df["收盘"]
        h = self.df["最高"]
        l = self.df["最低"]

        # 均线
        self.ma5 = c.rolling(5).mean()
        self.ma10 = c.rolling(10).mean()
        self.ma20 = c.rolling(20).mean()
        self.ma60 = c.rolling(60).mean()

        # EMA
        self.ema12 = c.ewm(span=12, adjust=False).mean()
        self.ema26 = c.ewm(span=26, adjust=False).mean()

        # MACD
        self.dif = self.ema12 - self.ema26
        self.dea = self.dif.ewm(span=9, adjust=False).mean()
        self.macd = 2 * (self.dif - self.dea)

        # RSI
        self.rsi6 = calculate_rsi(c, 6)
        self.rsi14 = calculate_rsi(c, 14)

        # 布林带
        self.boll_mid = c.rolling(20).mean()
        self.boll_std = c.rolling(20).std()
        self.boll_upper = self.boll_mid + 2 * self.boll_std
        self.boll_lower = self.boll_mid - 2 * self.boll_std

        # ATR
        self.atr14 = calculate_atr(h, l, c, 14)

        # 20日最高/最低
        self.high_20 = h.rolling(20).max()
        self.low_20 = l.rolling(20).min()

    # ------------------------------------------------------------------
    def calculate_all(self) -> Dict[str, float]:
        """计算全部技术因子"""
        if self.df.empty or len(self.df) < 20:
            return self._empty_factors()

        f = {}

        # ---- MACD ----
        f["macd_golden_cross"] = self._macd_cross()
        f["macd_above_zero"] = 1.0 if self.dif.iloc[-1] > 0 else 0.0
        f["macd_histogram"] = self._normalize(self.macd.iloc[-1], self.macd.tail(60))

        # ---- RSI ----
        f["rsi_6"] = self.rsi6.iloc[-1] if not np.isnan(self.rsi6.iloc[-1]) else 50.0
        f["rsi_14"] = self.rsi14.iloc[-1] if not np.isnan(self.rsi14.iloc[-1]) else 50.0

        # ---- 均线 ----
        f["ma_bullish"] = self._ma_bullish()
        f["price_vs_ma20"] = safe_divide(
            self.df["收盘"].iloc[-1] - self.ma20.iloc[-1],
            self.ma20.iloc[-1],
        )

        # ---- 布林带 ----
        f["boll_position"] = self._boll_position()
        f["boll_width"] = self._boll_width()

        # ---- 突破信号 ----
        f["breakout_20d_high"] = self._breakout_high(20)
        f["breakout_60d_high"] = self._breakout_high(60)

        # ---- 量价 ----
        f["volume_price_up"] = self._volume_price_up()
        f["volume_ratio_5d"] = self._volume_ratio(5)

        # ---- 动量衰减 ----
        f["momentum_decay"] = self._momentum_decay(5)

        # ---- 波动率 ----
        f["atr_14"] = self.atr14.iloc[-1] if not np.isnan(self.atr14.iloc[-1]) else 0
        f["volatility_20d"] = self._volatility(20)

        # ---- K线形态 ----
        f["candle_pattern"] = self._candle_pattern()

        return f

    # ------------------------------------------------------------------
    # 动量
    # ------------------------------------------------------------------
    def _momentum(self, days: int) -> float:
        if len(self.df) < days + 1:
            return 0
        now = self.df["收盘"].iloc[-1]
        prev = self.df["收盘"].iloc[-days - 1]
        return safe_divide(now - prev, prev)

    # ------------------------------------------------------------------
    # 动量衰减（加速=正值可能见顶，减速=负值可能见底）
    # ------------------------------------------------------------------
    def _momentum_decay(self, period: int = 5) -> float:
        if len(self.df) < 2 * period + 2:
            return 0.0
        c = self.df["收盘"].values
        recent = c[-1] / c[-period - 1] - 1
        historical = c[-period - 1] / c[-2 * period - 1] - 1
        return max(-1.0, min(1.0, recent - historical))

    # ------------------------------------------------------------------
    # MACD 金叉/死叉
    # ------------------------------------------------------------------
    def _macd_cross(self) -> float:
        """1=金叉, -1=死叉, 0=无"""
        if len(self.dif) < 2:
            return 0
        if self.dif.iloc[-1] > self.dea.iloc[-1] and self.dif.iloc[-2] <= self.dea.iloc[-2]:
            return 1.0
        if self.dif.iloc[-1] < self.dea.iloc[-1] and self.dif.iloc[-2] >= self.dea.iloc[-2]:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # 均线多头排列
    # ------------------------------------------------------------------
    def _ma_bullish(self) -> float:
        """MA5 > MA10 > MA20 则为1"""
        if any(np.isnan(v) for v in [self.ma5.iloc[-1], self.ma10.iloc[-1], self.ma20.iloc[-1]]):
            return 0
        if self.ma5.iloc[-1] > self.ma10.iloc[-1] > self.ma20.iloc[-1]:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # 布林带位置 (0=下轨, 1=上轨)
    # ------------------------------------------------------------------
    def _boll_position(self) -> float:
        upper = self.boll_upper.iloc[-1]
        lower = self.boll_lower.iloc[-1]
        if np.isnan(upper) or np.isnan(lower) or upper == lower:
            return 0.5
        return max(0, min(1, (self.df["收盘"].iloc[-1] - lower) / (upper - lower)))

    def _boll_width(self) -> float:
        """布林带宽度（越小越可能变盘）"""
        upper = self.boll_upper.iloc[-1]
        lower = self.boll_lower.iloc[-1]
        mid = self.boll_mid.iloc[-1]
        if np.isnan(mid) or mid == 0:
            return 0
        return safe_divide(upper - lower, mid)

    # ------------------------------------------------------------------
    # 突破信号
    # ------------------------------------------------------------------
    def _breakout_high(self, period: int) -> float:
        """突破N日新高"""
        if period == 20:
            high = self.high_20.iloc[-2] if len(self.high_20) > 1 else 0
        else:
            high = self.df["最高"].rolling(period).max().iloc[-2] if len(self.df) > period else 0
        return 1.0 if self.df["收盘"].iloc[-1] >= high else 0.0

    # ------------------------------------------------------------------
    # 量价关系
    # ------------------------------------------------------------------
    def _volume_price_up(self) -> float:
        """量价齐升"""
        if len(self.df) < 3:
            return 0
        price_up = self.df["收盘"].iloc[-1] > self.df["收盘"].iloc[-2]
        vol_up = self.df["成交量"].iloc[-1] > self.df["成交量"].iloc[-2] * 1.2
        return 1.0 if (price_up and vol_up) else 0.0

    def _volume_ratio(self, days: int = 5) -> float:
        """今日成交量 / N日均量"""
        if len(self.df) < days + 1:
            return 1.0
        avg_vol = self.df["成交量"].iloc[-(days + 1):-1].mean()
        return safe_divide(self.df["成交量"].iloc[-1], avg_vol)

    # ------------------------------------------------------------------
    # 波动率
    # ------------------------------------------------------------------
    def _volatility(self, days: int = 20) -> float:
        returns = self.df["收盘"].pct_change().tail(days)
        return returns.std() if not returns.empty else 0

    # ------------------------------------------------------------------
    # K线形态打分
    # ------------------------------------------------------------------
    def _candle_pattern(self) -> float:
        """
        简单K线形态评分
        1=强势（大阳线）, 0.5=中性, 0=弱势（大阴线）
        """
        if len(self.df) < 2:
            return 0.5
        row = self.df.iloc[-1]
        body = row["收盘"] - row["开盘"]
        total = row["最高"] - row["最低"]
        if total == 0:
            return 0.5
        ratio = body / total
        if ratio > 0.6:
            return 1.0   # 大阳线
        elif ratio < -0.6:
            return 0.0   # 大阴线
        return 0.5

    # ------------------------------------------------------------------
    def _normalize(self, value, series) -> float:
        """将值标准化到 [-1, 1]"""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return 0
        return max(-1, min(1, (value - mn) / (mx - mn) * 2 - 1))

    def _empty_factors(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "macd_golden_cross", "macd_above_zero", "macd_histogram",
            "rsi_6", "rsi_14",
            "ma_bullish", "price_vs_ma20",
            "boll_position", "boll_width",
            "breakout_20d_high", "breakout_60d_high",
            "volume_price_up", "volume_ratio_5d",
            "momentum_decay",
            "atr_14", "volatility_20d",
            "candle_pattern",
        ]}
