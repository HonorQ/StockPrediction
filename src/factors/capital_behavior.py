"""
资金行为模式检测因子

检测主力资金的行为模式：吸筹/拉升/出货/洗盘
基于量价关系和价格形态分析
"""
import pandas as pd
import numpy as np
from typing import Dict


class CapitalBehaviorFactor:
    """
    资金行为模式检测

    核心逻辑：主力操作股票有四个阶段
    1. 吸筹 (Accumulation): 低位震荡，量能温和放大，偶尔试盘
    2. 拉升 (Markup): 放量突破，连续阳线，趋势向上
    3. 出货 (Distribution): 高位震荡，量价背离，频繁高开低走
    4. 洗盘 (Shakeout): 急跌后快速收回，清洗浮筹
    """

    def __init__(self, config=None):
        self.config = config or {}

    def calculate(self, history: pd.DataFrame, **kwargs) -> Dict:
        """
        计算资金行为模式

        Args:
            history: 历史K线数据 (OHLCV)

        Returns:
            capital_behavior: 模式名称 (accumulation/markup/distribution/shakeout/neutral)
            capital_behavior_confidence: 置信度 (0-1)
            capital_behavior_signals: 信号列表
            flow_continuity: 资金流向连续性 (-1到1)
        """
        if history is None or len(history) < 20:
            return {
                "capital_behavior": "neutral",
                "capital_behavior_confidence": 0.0,
                "capital_behavior_signals": [],
                "flow_continuity": 0.0,
            }

        try:
            signals = []
            scores = {
                "accumulation": 0.0,
                "markup": 0.0,
                "distribution": 0.0,
                "shakeout": 0.0,
            }

            # 列名兼容：东财/新浪源返回中文列名，baostock 返回英文列名
            close = (history["收盘"] if "收盘" in history.columns else history["close"]).values
            high = (history["最高"] if "最高" in history.columns else history["high"]).values
            low = (history["最低"] if "最低" in history.columns else history["low"]).values
            volume = (history["成交量"] if "成交量" in history.columns else history["volume"]).values
            if "开盘" in history.columns:
                open_price = history["开盘"].values
            elif "open" in history.columns:
                open_price = history["open"].values
            else:
                open_price = close

            # === 1. 量价背离检测 ===
            vol_price_corr = self._volume_price_correlation(close, volume)
            if vol_price_corr > 0.3:
                signals.append("量价齐升(吸筹/拉升信号)")
                scores["accumulation"] += 0.15
                scores["markup"] += 0.15
            elif vol_price_corr < -0.3:
                signals.append("量价背离(出货信号)")
                scores["distribution"] += 0.2

            # === 2. 成交量变化模式 ===
            vol_pattern = self._volume_pattern(volume, close=close)
            if vol_pattern == "gradual_increase":
                signals.append("成交量温和放大(吸筹特征)")
                scores["accumulation"] += 0.2
            elif vol_pattern == "sudden_spike":
                signals.append("成交量突变(可能拉升或出货)")
                scores["markup"] += 0.1
                scores["distribution"] += 0.1
            elif vol_pattern == "decline_at_high":
                signals.append("高位缩量(出货特征)")
                scores["distribution"] += 0.15

            # === 3. 价格形态分析 ===
            price_pattern = self._price_pattern(close, high, low)
            if price_pattern == "consolidation_low":
                signals.append("低位震荡整理(吸筹形态)")
                scores["accumulation"] += 0.2
            elif price_pattern == "breakout_up":
                signals.append("向上突破(拉升形态)")
                scores["markup"] += 0.25
            elif price_pattern == "high_volatility":
                signals.append("高位剧烈波动(出货形态)")
                scores["distribution"] += 0.2
            elif price_pattern == "sharp_drop_recover":
                signals.append("急跌后快速收回(洗盘形态)")
                scores["shakeout"] += 0.3

            # === 4. 高开低走检测 ===
            gap_down_count = self._count_high_open_low_close(open_price, close, high)
            if gap_down_count >= 3:
                signals.append(f"近期{gap_down_count}次高开低走(出货嫌疑)")
                scores["distribution"] += 0.15

            # === 5. 下影线检测（洗盘/吸筹） ===
            lower_shadow_ratio = self._lower_shadow_ratio(close, low, open_price)
            if lower_shadow_ratio > 0.3:
                signals.append("频繁长下影线(主力护盘/吸筹)")
                scores["accumulation"] += 0.1
                scores["shakeout"] += 0.1

            # === 6. 趋势强度 ===
            trend = self._trend_strength(close)
            if trend > 0.5:
                scores["markup"] += 0.1
            elif trend < -0.5:
                scores["distribution"] += 0.05

            # === 综合判断 ===
            max_behavior = max(scores, key=scores.get)
            max_score = scores[max_behavior]

            if max_score < 0.1:
                behavior = "neutral"
                confidence = 0.0
            else:
                behavior = max_behavior
                confidence = min(max_score, 1.0)

            # 资金流向连续性
            flow = self._flow_continuity(close, volume)

            return {
                "capital_behavior": behavior,
                "capital_behavior_confidence": round(confidence, 3),
                "capital_behavior_signals": signals,
                "flow_continuity": round(flow, 3),
            }

        except Exception as e:
            return {
                "capital_behavior": "neutral",
                "capital_behavior_confidence": 0.0,
                "capital_behavior_signals": [f"计算异常: {e}"],
                "flow_continuity": 0.0,
            }

    def _volume_price_correlation(self, close, volume, period=10):
        """量价相关性"""
        if len(close) < period:
            return 0.0
        price_changes = np.diff(close[-period:])
        vol_changes = np.diff(volume[-period:])
        if np.std(price_changes) == 0 or np.std(vol_changes) == 0:
            return 0.0
        return np.corrcoef(price_changes, vol_changes)[0, 1]

    def _volume_pattern(self, volume, close=None, period=10):
        """成交量变化模式"""
        if len(volume) < period:
            return "unknown"

        recent_vol = volume[-period:]
        avg_vol = np.mean(volume[-30:]) if len(volume) >= 30 else np.mean(volume)

        # 温和放大：最近5天均量 > 前5天均量，但不超3倍
        if len(recent_vol) >= 5:
            last_5 = np.mean(recent_vol[-5:])
            prev_5 = np.mean(recent_vol[:5])
            if prev_5 > 0:
                ratio = last_5 / prev_5
                if 1.2 < ratio < 3.0:
                    return "gradual_increase"
                elif ratio > 3.0:
                    return "sudden_spike"

        # 高位缩量（需要 close 数据判断是否在高位）
        if close is not None and len(close) >= 20:
            if recent_vol[-1] < avg_vol * 0.6 and np.mean(close[-5:]) > np.mean(close[-20:]):
                return "decline_at_high"

        return "normal"

    def _price_pattern(self, close, high, low, period=20):
        """价格形态"""
        if len(close) < period:
            return "unknown"

        recent_close = close[-period:]
        recent_high = high[-period:]
        recent_low = low[-period:]

        # 低位震荡：价格在低位窄幅波动
        price_range = (np.max(recent_high) - np.min(recent_low)) / np.mean(recent_close)
        if price_range < 0.1:
            return "consolidation_low"

        # 向上突破：最近5天创20日新高
        if np.max(close[-5:]) >= np.max(recent_high):
            return "breakout_up"

        # 高位剧烈波动
        if price_range > 0.15 and np.std(recent_close[-5:]) > np.std(recent_close) * 1.5:
            return "high_volatility"

        # 急跌后快速收回
        min_idx = np.argmin(recent_low)
        if min_idx < period - 3:  # 低点不是最近
            if close[-1] > recent_close[min_idx] * 1.03:  # 收回超过3%
                return "sharp_drop_recover"

        return "normal"

    def _count_high_open_low_close(self, open_p, close, high, period=10):
        """统计高开低走次数"""
        count = 0
        for i in range(-period, 0):
            if open_p[i] > close[i] and (high[i] - open_p[i]) > (open_p[i] - close[i]) * 0.5:
                count += 1
        return count

    def _lower_shadow_ratio(self, close, low, open_p, period=10):
        """长下影线比例"""
        count = 0
        for i in range(-period, 0):
            body = abs(close[i] - open_p[i])
            lower_shadow = min(close[i], open_p[i]) - low[i]
            if body > 0 and lower_shadow > body * 2:
                count += 1
        return count / period

    def _trend_strength(self, close, period=20):
        """趋势强度 (-1到1)"""
        if len(close) < period:
            return 0.0
        x = np.arange(period)
        y = close[-period:]
        if np.std(y) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return np.clip(slope / (np.mean(y) * 0.01), -1, 1)

    def _flow_continuity(self, close, volume, period=10):
        """资金流向连续性"""
        if len(close) < period:
            return 0.0

        flow = 0.0
        for i in range(-period, 0):
            if close[i] > close[i - 1]:
                flow += volume[i]  # 上涨放量视为流入
            elif close[i] < close[i - 1]:
                flow -= volume[i]  # 下跌放量视为流出

        # 归一化到 -1 ~ 1
        total_vol = np.sum(volume[-period:])
        if total_vol == 0:
            return 0.0
        return np.clip(flow / total_vol, -1, 1)
