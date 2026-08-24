"""
国家队因子

检测国家队（汇金/证金/社保）的可能持仓和操作迹象
基于价格稳定性和异常量能分析
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class NationalTeamFactor:
    """
    国家队因子

    核心逻辑：国家队操作特征
    1. 在市场恐慌时逆势买入
    2. 重点持仓大市值蓝筹股
    3. 维护市场稳定，减少波动
    4. 长期持有，换手率低
    """

    # 国家队重仓的行业（历史偏好）
    PREFERRED_SECTORS = {
        "银行", "保险", "证券", "石油", "石化", "电力", "煤炭",
        "钢铁", "有色金属", "交通运输", "建筑", "通信",
    }

    # 国家队重仓的市值阈值（亿元）
    LARGE_CAP_THRESHOLD = 500

    def __init__(self, config=None):
        self.config = config or {}

    def calculate(
        self,
        history: pd.DataFrame,
        stock_info: Optional[Dict] = None,
        market_index: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict:
        """
        计算国家队因子

        Args:
            history: 历史K线
            stock_info: 股票基本信息（行业、市值等）
            market_index: 大盘指数（用于对比）

        Returns:
            national_team_score: 国家队关联度 (0-1)
            national_team_holding: 是否疑似持仓 (bool)
            national_team_change: 持仓变化信号 (-1到1)
            national_team_signals: 信号列表
        """
        if history is None or len(history) < 30:
            return {
                "national_team_score": 0.0,
                "national_team_holding": False,
                "national_team_change": 0.0,
                "national_team_signals": ["数据不足"],
            }

        try:
            signals = []
            score = 0.0

            # 列名兼容：东财/新浪源返回中文列名，baostock 返回英文列名
            close = (history["收盘"] if "收盘" in history.columns else history["close"]).values
            high = (history["最高"] if "最高" in history.columns else history["high"]).values
            low = (history["最低"] if "最低" in history.columns else history["low"]).values
            volume = (history["成交量"] if "成交量" in history.columns else history["volume"]).values

            # === 1. 大盘下跌时的抗跌性 ===
            if market_index is not None and len(market_index) >= len(history):
                idx_col = "收盘" if "收盘" in market_index.columns else "close"
                idx_close = market_index[idx_col].values[-len(close):]
                resilience = self._market_resilience(close, idx_close)
                if resilience > 0.3:
                    signals.append("大盘下跌时明显抗跌")
                    score += 0.25
                elif resilience > 0.1:
                    signals.append("大盘下跌时较为抗跌")
                    score += 0.15

            # === 2. 关键价位护盘行为 ===
            support_defense = self._support_defense(close, low, volume)
            if support_defense > 0:
                signals.append(f"关键价位护盘({support_defense:.0f}次)")
                score += min(support_defense * 0.1, 0.2)

            # === 3. 尾盘异常拉升 ===
            late_strength = self._late_session_strength(close, high, low)
            if late_strength > 0.3:
                signals.append("尾盘异常拉升(疑似国家队)")
                score += 0.2

            # === 4. 成交量异常稳定 ===
            vol_stability = self._volume_stability(volume)
            if vol_stability > 0.7:
                signals.append("成交量异常稳定(机构控盘)")
                score += 0.15

            # === 5. 波动率异常低 ===
            volatility = self._low_volatility(close)
            if volatility < 0.015:
                signals.append("波动率异常低(疑似护盘)")
                score += 0.1

            # === 6. 行业和市值判断 ===
            if stock_info:
                sector = stock_info.get("行业", "")
                total_mv = stock_info.get("总市值", 0)

                if sector in self.PREFERRED_SECTORS:
                    score += 0.1
                    signals.append(f"行业匹配({sector})")

                if total_mv and total_mv > self.LARGE_CAP_THRESHOLD * 1e8:
                    score += 0.1
                    signals.append("大市值蓝筹")

            # === 综合判断 ===
            score = min(score, 1.0)
            holding = score > 0.3

            # 持仓变化信号
            change = self._position_change_signal(close, volume)

            return {
                "national_team_score": round(score, 3),
                "national_team_holding": holding,
                "national_team_change": round(change, 3),
                "national_team_signals": signals,
            }

        except Exception as e:
            return {
                "national_team_score": 0.0,
                "national_team_holding": False,
                "national_team_change": 0.0,
                "national_team_signals": [f"计算异常: {e}"],
            }

    def _market_resilience(self, stock_close, idx_close, period=20):
        """大盘下跌时的抗跌性"""
        if len(stock_close) < period or len(idx_close) < period:
            return 0.0

        stock_ret = np.diff(stock_close[-period:]) / stock_close[-period:-1]
        idx_ret = np.diff(idx_close[-period:]) / idx_close[-period:-1]

        # 找出大盘下跌的日子
        down_days = idx_ret < -0.01
        if np.sum(down_days) < 3:
            return 0.0

        # 计算这些日子股票的相对表现
        stock_on_down = stock_ret[down_days]
        idx_on_down = idx_ret[down_days]

        # 抗跌性 = 股票跌幅 - 大盘跌幅（越正越抗跌）
        resilience = np.mean(idx_on_down - stock_on_down)
        return min(max(resilience * 10, -1), 1)  # 放大并限制范围

    def _support_defense(self, close, low, volume, lookback=60):
        """关键价位护盘检测"""
        if len(close) < lookback:
            return 0

        # 找出近期的关键支撑位（最近60天的低点）
        support_level = np.min(low[-lookback:])
        close[-1]
        threshold = support_level * 1.02  # 距支撑位2%以内

        # 检测触及支撑位后反弹的次数
        defense_count = 0
        for i in range(-10, 0):
            if low[i] <= threshold and close[i] > low[i]:  # 下影线触及后收回
                defense_count += 1

        return defense_count

    def _late_session_strength(self, close, high, low, period=10):
        """尾盘异常拉升检测"""
        if len(close) < period:
            return 0.0

        # 尾盘拉升特征：收盘价接近最高价
        strength_count = 0
        for i in range(-period, 0):
            if high[i] > low[i]:
                # 收盘价在当日区间的上半部分
                position = (close[i] - low[i]) / (high[i] - low[i])
                if position > 0.8:  # 收盘接近最高
                    strength_count += 1

        return strength_count / period

    def _volume_stability(self, volume, period=20):
        """成交量稳定性"""
        if len(volume) < period:
            return 0.0

        recent_vol = volume[-period:]
        mean_vol = np.mean(recent_vol)
        if mean_vol == 0:
            return 0.0

        cv = np.std(recent_vol) / mean_vol  # 变异系数
        # 变异系数越小越稳定
        return max(0, 1 - cv)

    def _low_volatility(self, close, period=20):
        """波动率"""
        if len(close) < period:
            return 0.5

        returns = np.diff(close[-period:]) / close[-period:-1]
        return np.std(returns)

    def _position_change_signal(self, close, volume, short=5, long=20):
        """持仓变化信号"""
        if len(close) < long:
            return 0.0

        # 短期 vs 长期量价关系
        short_ret = (close[-1] - close[-short]) / close[-short]
        short_vol = np.mean(volume[-short:])
        long_vol = np.mean(volume[-long:])

        # 量能变化
        vol_change = (short_vol - long_vol) / long_vol if long_vol > 0 else 0

        # 价涨量增=加仓，价跌量增=减仓
        if short_ret > 0 and vol_change > 0.2:
            return 0.5  # 疑似加仓
        elif short_ret < 0 and vol_change > 0.3:
            return -0.5  # 疑似减仓
        return 0.0
