"""
A股特色因子
政策市、3000点保卫战、游资行为、均值回归、情绪周期
"""
import pandas as pd
import numpy as np
from typing import Dict

from src.utils.helpers import get_limit_pct, load_score_tiers


class AShareCharacteristics:
    """A股特色因子"""

    def __init__(self):
        pass

    def calculate_all(self, df: pd.DataFrame, market_sentiment: Dict = None,
                      code: str = "", name: str = "") -> Dict:
        """
        计算A股特色因子
        df: 个股历史数据
        market_sentiment: 市场情绪数据
        
        子因子IC分析结论(2026-08-28, 全市场102只股票):
        - position_score: IC=+0.0888, ICIR=+1.98 → 正向有效, 权重45%
        - mean_reversion_score: IC=+0.0357(取反后) → 正向有效, 权重30%
        - turnover_ratio: IC=-0.0083 → 取反使用, 权重15%
        - volatility_score: IC=+0.0218 → 正向弱, 权重10%
        """
        factors = {}

        limit_pct = get_limit_pct(code, name)

        # 1. 均值回归因子（IC分析显示负向有效，取反使用）
        raw_mean_rev = self._mean_reversion_factor(df)
        factors.update(raw_mean_rev)
        if "mean_reversion_score" in factors:
            # 取反：原分数越高=超买越严重=应该减分
            factors["mean_reversion_score"] = 100 - factors["mean_reversion_score"]
            # 同步修改信号方向
            orig_signal = factors.get("reversion_signal", "neutral")
            if orig_signal == "strong_buy":
                factors["reversion_signal"] = "strong_sell"
            elif orig_signal == "buy":
                factors["reversion_signal"] = "sell"
            elif orig_signal == "strong_sell":
                factors["reversion_signal"] = "strong_buy"
            elif orig_signal == "sell":
                factors["reversion_signal"] = "buy"

        # 2. 波动率因子
        factors.update(self._volatility_factor(df))

        # 3. 换手率异动（IC负向，取反使用）
        raw_turnover = self._turnover_anomaly(df)
        factors.update(raw_turnover)
        if "turnover_ratio" in factors:
            # 取反：原ratio越高=换手率异常高=应该减分
            # turnover_ratio是当前/20日均值，>1表示放量
            # IC为负说明放量不好，取反后放量扣分
            factors["turnover_ratio_score"] = 100 - min(factors.get("turnover_ratio", 1) * 30, 50)

        # 4. 涨跌停统计（pct_chg数据缺失，暂不计算）
        # factors.update(self._limit_stats(df, limit_pct=limit_pct))

        # 5. 价格位置（IC正向有效）
        factors.update(self._price_position_factor(df))

        # === 加权合成最终ashare_specific评分 ===
        pos_score = factors.get("position_score", 50)
        mr_score = factors.get("mean_reversion_score", 50)
        turn_score = factors.get("turnover_ratio_score", 50)
        vol_score = factors.get("volatility_score", 50)
        
        # 加权平均: position(45%) + mean_reversion(30%) + turnover(15%) + volatility(10%)
        final_score = pos_score * 0.45 + mr_score * 0.30 + turn_score * 0.15 + vol_score * 0.10
        factors["ashare_specific_score"] = round(final_score, 1)

        return factors

    def _mean_reversion_factor(self, df: pd.DataFrame) -> Dict:
        """
        均值回归因子
        A股震荡市：涨多了会跌，跌多了会涨
        """
        factors = {}

        close = df["收盘"] if "收盘" in df.columns else df["close"]

        if len(close) < 20:
            return factors

        current = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

        # 偏离度（偏离均线越多，回归概率越大）
        deviation_20 = (current / ma20 - 1) * 100
        deviation_60 = (current / ma60 - 1) * 100 if len(close) >= 60 else 0

        factors["deviation_20d"] = round(deviation_20, 2)
        factors["deviation_60d"] = round(deviation_60, 2)

        # 均值回归评分
        if deviation_20 < -10:
            factors["mean_reversion_score"] = 90  # 严重超卖，回归概率高
            factors["reversion_signal"] = "strong_buy"
        elif deviation_20 < -5:
            factors["mean_reversion_score"] = 75
            factors["reversion_signal"] = "buy"
        elif deviation_20 > 10:
            factors["mean_reversion_score"] = 10  # 严重超买，回归下跌概率高
            factors["reversion_signal"] = "strong_sell"
        elif deviation_20 > 5:
            factors["mean_reversion_score"] = 25
            factors["reversion_signal"] = "sell"
        else:
            factors["mean_reversion_score"] = 50
            factors["reversion_signal"] = "neutral"

        return factors

    def _volatility_factor(self, df: pd.DataFrame) -> Dict:
        """
        波动率因子
        A股波动大，适合高抛低吸
        """
        factors = {}

        close = df["收盘"] if "收盘" in df.columns else df["close"]

        if len(close) < 20:
            return factors

        # 计算波动率
        returns = close.pct_change().dropna()
        volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100
        volatility_60d = returns.tail(60).std() * np.sqrt(252) * 100 if len(returns) >= 60 else volatility_20d

        factors["volatility_20d"] = round(volatility_20d, 2)
        factors["volatility_60d"] = round(volatility_60d, 2)

        # 波动率评级
        if volatility_20d > 60:
            factors["volatility_level"] = "extreme_high"
            factors["volatility_score"] = 20  # 高波动风险大
        elif volatility_20d > 40:
            factors["volatility_level"] = "high"
            factors["volatility_score"] = 40
        elif volatility_20d > 25:
            factors["volatility_level"] = "medium"
            factors["volatility_score"] = 60
        else:
            factors["volatility_level"] = "low"
            factors["volatility_score"] = 80  # 低波动更安全

        return factors

    def _turnover_anomaly(self, df: pd.DataFrame) -> Dict:
        """
        换手率异动
        A股游资喜欢高换手率的股票
        """
        factors = {}

        if "换手率" not in df.columns and "turnover" not in df.columns:
            return factors

        turnover = df["换手率"] if "换手率" in df.columns else df["turnover"]

        if len(turnover) < 10:
            return factors

        current_turnover = turnover.iloc[-1]
        if pd.isna(current_turnover):
            return factors

        avg_turnover = turnover.tail(20).mean()
        if pd.isna(avg_turnover) or avg_turnover <= 0:
            return factors

        turnover_ratio = current_turnover / avg_turnover

        factors["current_turnover"] = round(float(current_turnover), 2)
        factors["turnover_ratio"] = round(float(turnover_ratio), 2)

        # 换手率异动信号
        if turnover_ratio > 3:
            factors["turnover_anomaly"] = "extreme_high"
            factors["turnover_signal"] = "游资高度关注"
        elif turnover_ratio > 2:
            factors["turnover_anomaly"] = "high"
            factors["turnover_signal"] = "资金流入加速"
        elif turnover_ratio < 0.5:
            factors["turnover_anomaly"] = "low"
            factors["turnover_signal"] = "交投清淡"
        else:
            factors["turnover_anomaly"] = "normal"

        return factors

    def _limit_stats(self, df: pd.DataFrame, limit_pct: float = 10.0) -> Dict:
        """
        涨跌停统计
        A股特色：涨跌停板制度（按板块口径 10/20/30cm）
        """
        factors = {}

        if "涨跌幅" not in df.columns and "pct_chg" not in df.columns:
            return factors

        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df["pct_chg"]

        if len(change_pct) < 20:
            return factors

        recent_20 = change_pct.tail(20)

        # 统计涨跌停次数（阈值 = 板块幅度 - 0.5，兼容盘中未精确封板）
        limit_up_threshold = limit_pct - 0.5
        limit_up_count = sum(1 for x in recent_20 if x > limit_up_threshold)
        limit_down_count = sum(1 for x in recent_20 if x < -limit_up_threshold)
        large_up_count = sum(1 for x in recent_20 if x > 5)
        large_down_count = sum(1 for x in recent_20 if x < -5)

        factors["limit_up_20d"] = limit_up_count
        factors["limit_down_20d"] = limit_down_count
        factors["large_up_20d"] = large_up_count
        factors["large_down_20d"] = large_down_count

        # 活跃度评分
        if limit_up_count > 2:
            factors["activity_score"] = 90  # 活跃度高
        elif large_up_count > 5:
            factors["activity_score"] = 70
        elif limit_down_count > 2:
            factors["activity_score"] = 20  # 风险高
        else:
            factors["activity_score"] = 50

        return factors

    def _price_position_factor(self, df: pd.DataFrame) -> Dict:
        """
        价格位置因子
        A股3000点附近特殊处理
        """
        factors = {}

        close = df["收盘"] if "收盘" in df.columns else df["close"]

        if len(close) < 60:
            return factors

        current = close.iloc[-1]
        high_250 = close.tail(250).max() if len(close) >= 250 else close.max()
        low_250 = close.tail(250).min() if len(close) >= 250 else close.min()

        # 价格位置（0-1）
        if high_250 > low_250:
            price_position = (current - low_250) / (high_250 - low_250)
        else:
            price_position = 0.5

        factors["price_position_250d"] = round(price_position, 4)

        # 价格位置评分（底部买入，顶部卖出）
        if price_position < 0.2:
            factors["position_score"] = 90  # 底部区域
            factors["position_signal"] = "底部区域（可分批建仓）"
        elif price_position < 0.4:
            factors["position_score"] = 70
            factors["position_signal"] = "相对低位"
        elif price_position > 0.8:
            factors["position_score"] = 20  # 顶部区域
            factors["position_signal"] = "顶部区域（注意风险）"
        elif price_position > 0.6:
            factors["position_score"] = 40
            factors["position_signal"] = "相对高位"
        else:
            factors["position_score"] = 50

        return factors


class SentimentCycle:
    """情绪周期因子"""

    def __init__(self, config: dict = None):
        self._tiers = load_score_tiers(config)

    def analyze(self, market_sentiment: Dict, df: pd.DataFrame = None) -> Dict:
        """
        分析情绪周期
        A股情绪周期：冰点→修复→亢奋→恐慌→冰点
        """
        result = {
            "cycle_phase": "unknown",
            "cycle_score": 50,
            "suggestion": "",
        }

        if market_sentiment is None:
            return result

        t = self._tiers
        emotion_score = market_sentiment.get("emotion_score", 50)
        limit_up = market_sentiment.get("limit_up", 0)
        limit_down = market_sentiment.get("limit_down", 0)

        # 判断情绪周期
        if emotion_score < t["sentiment_ice"]:
            result["cycle_phase"] = "冰点"
            result["cycle_score"] = 90  # 冰点是买入机会
            result["suggestion"] = "市场情绪冰点，可逢低布局"
        elif emotion_score < t["sentiment_low"]:
            result["cycle_phase"] = "低迷"
            result["cycle_score"] = 70
            result["suggestion"] = "市场低迷，适合左侧布局"
        elif emotion_score > t["sentiment_hot"]:
            result["cycle_phase"] = "亢奋"
            result["cycle_score"] = 10  # 亢奋是卖出机会
            result["suggestion"] = "市场亢奋，注意风险"
        elif emotion_score > t["sentiment_warm"]:
            result["cycle_phase"] = "回暖"
            result["cycle_score"] = 60
            result["suggestion"] = "市场回暖，可适度参与"
        else:
            result["cycle_phase"] = "中性"
            result["cycle_score"] = 50
            result["suggestion"] = "市场中性，观望为主"

        # 涨跌停比修正
        if limit_down > limit_up * 0.5:
            result["cycle_score"] -= 10
            result["suggestion"] += "（跌停较多，风险偏高）"
        elif limit_up > limit_down * 10:
            result["cycle_score"] += 5
            result["suggestion"] += "（涨停较多，情绪偏热）"

        result["cycle_score"] = max(0, min(100, result["cycle_score"]))

        return result


class HotMoneyTracker:
    """游资追踪器"""

    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, hot_data: Dict = None,
                code: str = "", name: str = "") -> Dict:
        """
        分析游资行为
        游资特点：追涨杀跌、快进快出、喜欢打板
        """
        result = {
            "hot_money_score": 50,
            "signals": [],
        }

        if df is None or df.empty or len(df) < 10:
            return result

        limit_pct = get_limit_pct(code, name)

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        df["成交量"] if "成交量" in df.columns else df["volume"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        score = 50
        signals = []

        # 1. 连板检测（游资最爱，按板块口径）
        if len(change_pct) >= 3:
            recent_3d = change_pct.tail(3)
            if all(x > limit_pct - 1 for x in recent_3d):
                signals.append("3连板（游资接力）")
                score += 20
            elif all(x > 5 for x in recent_3d):
                signals.append("3连阳（强势）")
                score += 10

        # 2. 涨停板敢死队模式（盘中冲高未封住：接近涨停但未到）
        if not change_pct.empty:
            today_change = change_pct.iloc[-1]
            if limit_pct - 2 < today_change < limit_pct - 0.2:
                signals.append("涨停板博弈（敢死队模式）")
                score += 15

        # 3. 天地板检测（高风险）
        if len(df) >= 1:
            row = df.iloc[-1]
            high = row.get("最高", 0)
            low = row.get("最低", 0)
            row.get("收盘", 0)
            if high > 0 and low > 0:
                amplitude = (high - low) / low * 100
                if amplitude > 15:
                    signals.append(f"振幅{amplitude:.0f}%（天地板风险）")
                    score -= 20

        # 4. 尾盘异动（游资偷袭）
        if len(df) >= 2:
            today_close = close.iloc[-1]
            today_open = df.iloc[-1].get("开盘", 0)
            yesterday_close = close.iloc[-2]
            if today_open > 0 and yesterday_close > 0:
                # 尾盘拉升（收盘价远高于开盘价）
                if today_close > today_open * 1.05 and change_pct.iloc[-1] > 3:
                    signals.append("尾盘异动（疑似游资偷袭）")
                    score += 10

        result["hot_money_score"] = max(0, min(100, score))
        result["signals"] = signals

        return result
