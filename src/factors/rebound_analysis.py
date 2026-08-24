"""
反弹持续性 + 次日预判因子
判断反弹是一日游还是持续修复，预判次日走势
"""
import pandas as pd
from typing import Dict


class ReboundSustainability:
    """反弹持续性分析器"""

    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, sector_rotation: Dict = None) -> Dict:
        """
        分析反弹持续性
        df: 个股历史数据
        sector_rotation: 板块轮动数据
        """
        result = {
            "sustainability_score": 50,
            "rebound_type": "unknown",
            "next_day_bias": "neutral",
            "confidence": 0,
            "signals": [],
            "risk_factors": [],
        }

        if df is None or df.empty or len(df) < 20:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        score = 50
        signals = []
        risk_factors = []

        # ============================================================
        # 1. 反弹类型判断
        # ============================================================
        rebound_type = self._identify_rebound_type(close, volume, change_pct)
        result["rebound_type"] = rebound_type["type"]
        score += rebound_type["score_delta"]
        signals.extend(rebound_type["signals"])

        # ============================================================
        # 2. 量价配合分析
        # ============================================================
        volume_price = self._analyze_volume_price(close, volume, change_pct)
        score += volume_price["score_delta"]
        signals.extend(volume_price["signals"])
        risk_factors.extend(volume_price["risk_factors"])

        # ============================================================
        # 3. 反弹力度评估
        # ============================================================
        rebound_strength = self._evaluate_rebound_strength(close, change_pct)
        score += rebound_strength["score_delta"]
        signals.extend(rebound_strength["signals"])

        # ============================================================
        # 4. 套牢盘压力
        # ============================================================
        pressure = self._analyze_pressure(close)
        score += pressure["score_delta"]
        risk_factors.extend(pressure["risk_factors"])

        # ============================================================
        # 5. 市场环境加成
        # ============================================================
        if sector_rotation:
            market_score = self._market_environment_score(sector_rotation)
            score += market_score["score_delta"]
            signals.extend(market_score["signals"])

        # ============================================================
        # 6. 历史模式匹配
        # ============================================================
        pattern = self._match_historical_pattern(close, change_pct)
        score += pattern["score_delta"]
        signals.extend(pattern["signals"])
        risk_factors.extend(pattern["risk_factors"])

        # 计算最终得分和置信度
        result["sustainability_score"] = max(0, min(100, score))
        result["signals"] = signals
        result["risk_factors"] = risk_factors
        result["confidence"] = len(signals) * 10 + len(risk_factors) * (-5)
        result["confidence"] = max(0, min(100, result["confidence"]))

        # 次日偏向判断
        if score >= 70:
            result["next_day_bias"] = "bullish"
        elif score >= 55:
            result["next_day_bias"] = "slightly_bullish"
        elif score <= 30:
            result["next_day_bias"] = "bearish"
        elif score <= 45:
            result["next_day_bias"] = "slightly_bearish"
        else:
            result["next_day_bias"] = "neutral"

        return result

    def _identify_rebound_type(self, close: pd.Series, volume: pd.Series,
                                change_pct: pd.Series) -> Dict:
        """
        识别反弹类型
        - 技术性反弹：跌多了自然反弹，持续性差
        - 资金驱动型反弹：有增量资金入场，持续性强
        - 情绪修复型反弹：恐慌后情绪修复，中等持续性
        """
        result = {"type": "unknown", "score_delta": 0, "signals": []}

        if len(close) < 10:
            return result

        # 计算近期走势
        ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        ret_10d = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) >= 11 else 0
        today_change = change_pct.iloc[-1] if not change_pct.empty else 0

        # 量能变化
        vol_ratio = volume.iloc[-1] / volume.iloc[-6:-1].mean() if len(volume) >= 6 else 1

        # 类型判断
        if ret_10d < -15 and today_change > 3 and vol_ratio > 2:
            # 大跌后放量大涨：资金驱动型
            result["type"] = "capital_driven"
            result["score_delta"] = 15
            result["signals"].append("资金驱动型反弹（放量大涨）")
        elif ret_10d < -10 and today_change > 2:
            # 跌后反弹：情绪修复型
            result["type"] = "emotion_repair"
            result["score_delta"] = 8
            result["signals"].append("情绪修复型反弹")
        elif ret_5d < -5 and today_change > 1:
            # 小幅反弹：技术性反弹
            result["type"] = "technical"
            result["score_delta"] = 3
            result["signals"].append("技术性反弹（持续性存疑）")
        else:
            result["type"] = "continuation"
            result["score_delta"] = 5
            result["signals"].append("趋势延续")

        return result

    def _analyze_volume_price(self, close: pd.Series, volume: pd.Series,
                               change_pct: pd.Series) -> Dict:
        """
        量价配合分析
        真反弹：价涨量增，价跌量缩
        假反弹：价涨量缩，或放量滞涨
        """
        result = {"score_delta": 0, "signals": [], "risk_factors": []}

        if len(close) < 5 or len(volume) < 5:
            return result

        today_change = change_pct.iloc[-1] if not change_pct.empty else 0
        today_vol = volume.iloc[-1]
        avg_vol_5 = volume.iloc[-6:-1].mean()

        vol_ratio = today_vol / avg_vol_5 if avg_vol_5 > 0 else 1

        # 价涨量增（健康）
        if today_change > 0 and vol_ratio > 1.2:
            result["score_delta"] += 10
            result["signals"].append("价涨量增（健康反弹）")
        # 价涨量缩（不健康）
        elif today_change > 0 and vol_ratio < 0.8:
            result["score_delta"] -= 10
            result["risk_factors"].append("价涨量缩（反弹力度弱）")
        # 放量滞涨（危险）
        elif abs(today_change) < 1 and vol_ratio > 2:
            result["score_delta"] -= 15
            result["risk_factors"].append("放量滞涨（主力出货？）")
        # 缩量下跌（调整）
        elif today_change < 0 and vol_ratio < 0.7:
            result["score_delta"] += 5
            result["signals"].append("缩量调整（良性）")

        # 连续放量
        if len(volume) >= 3:
            recent_vols = volume.iloc[-3:].values
            if recent_vols[0] < recent_vols[1] < recent_vols[2]:
                result["score_delta"] += 5
                result["signals"].append("连续放量（资金持续流入）")
            elif recent_vols[0] > recent_vols[1] > recent_vols[2]:
                result["risk_factors"].append("连续缩量（资金退潮）")

        return result

    def _evaluate_rebound_strength(self, close: pd.Series, change_pct: pd.Series) -> Dict:
        """评估反弹力度"""
        result = {"score_delta": 0, "signals": []}

        if len(close) < 5:
            return result

        today_change = change_pct.iloc[-1] if not change_pct.empty else 0

        # 反弹力度
        if today_change > 5:
            result["score_delta"] += 12
            result["signals"].append(f"强势反弹（+{today_change:.1f}%）")
        elif today_change > 3:
            result["score_delta"] += 8
            result["signals"].append(f"有力反弹（+{today_change:.1f}%）")
        elif today_change > 1:
            result["score_delta"] += 4
            result["signals"].append(f"温和反弹（+{today_change:.1f}%）")

        # 修复前日跌幅的比例
        if len(change_pct) >= 2:
            prev_change = change_pct.iloc[-2]
            if prev_change < -3 and today_change > 0:
                repair_ratio = today_change / abs(prev_change)
                if repair_ratio > 0.8:
                    result["score_delta"] += 8
                    result["signals"].append(f"修复{repair_ratio*100:.0f}%前日跌幅")
                elif repair_ratio > 0.5:
                    result["score_delta"] += 4

        return result

    def _analyze_pressure(self, close: pd.Series) -> Dict:
        """分析套牢盘压力"""
        result = {"score_delta": 0, "risk_factors": []}

        if len(close) < 20:
            return result

        current = close.iloc[-1]
        high_20 = close.tail(20).max()
        high_60 = close.tail(60).max() if len(close) >= 60 else high_20

        # 距离套牢区
        dist_from_20d_high = (current / high_20 - 1) * 100
        dist_from_60d_high = (current / high_60 - 1) * 100

        # 套牢盘压力大（先判重度再判轻度：原 elif 顺序颠倒导致 <-10 重罚分支永不可达）
        if dist_from_20d_high < -10:
            result["score_delta"] -= 12
            result["risk_factors"].append(f"距20日高点{dist_from_20d_high:.1f}%（套牢盘很重）")
        elif dist_from_20d_high < -5:
            result["score_delta"] -= 8
            result["risk_factors"].append(f"距20日高点{dist_from_20d_high:.1f}%（套牢盘重）")

        if dist_from_60d_high < -15:
            result["score_delta"] -= 5
            result["risk_factors"].append(f"距60日高点{dist_from_60d_high:.1f}%")

        return result

    def _market_environment_score(self, rotation_data: Dict) -> Dict:
        """市场环境加成"""
        result = {"score_delta": 0, "signals": []}

        # 板块轮动信号
        signal = rotation_data.get("rotation_signal", "neutral")
        if signal == "tech_strong":
            result["score_delta"] += 8
            result["signals"].append("科技板块整体强势")
        elif signal == "oversold_rebound":
            result["score_delta"] += 10
            result["signals"].append("板块超跌反弹中")

        # 超跌反弹信号
        if rotation_data.get("oversold_rebound"):
            intensity = rotation_data.get("rebound_intensity", 0)
            result["score_delta"] += min(intensity // 10, 8)
            result["signals"].append("市场超跌反弹信号")

        return result

    def _match_historical_pattern(self, close: pd.Series, change_pct: pd.Series) -> Dict:
        """
        历史模式匹配
        匹配类似情况下的后续走势
        """
        result = {"score_delta": 0, "signals": [], "risk_factors": []}

        if len(close) < 30:
            return result

        change_pct.iloc[-1] if not change_pct.empty else 0

        # 查找历史相似模式
        # 条件：前10日跌>8%，今日涨>2%
        for i in range(20, len(close) - 5):
            # 历史上的类似反弹
            past_ret_10d = (close.iloc[i-10] / close.iloc[i-1] - 1) * 100
            past_today_change = change_pct.iloc[i]

            if past_ret_10d < -8 and past_today_change > 2:
                # 找到类似情况，看后续5日走势
                future_5d = (close.iloc[i+5] / close.iloc[i] - 1) * 100
                if future_5d > 3:
                    result["score_delta"] += 5
                    result["signals"].append("历史类似反弹后5日上涨")
                    break
                elif future_5d < -3:
                    result["risk_factors"].append("历史类似反弹后5日下跌")
                    break

        return result


class NextDayPredictor:
    """次日走势预判"""

    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame, sustainability: Dict,
                sector_rotation: Dict = None) -> Dict:
        """
        预判次日走势
        返回: {
            bias: bullish/bearish/neutral,
            gap_probability: 高开/低开概率,
            expected_range: 预期振幅,
            trading_suggestion: 操作建议,
        }
        """
        result = {
            "bias": "neutral",
            "gap_open_probability": 0,
            "expected_range": 0,
            "trading_suggestion": "观望",
            "reasons": [],
        }

        if df is None or df.empty or len(df) < 10:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        sustainability_score = sustainability.get("sustainability_score", 50)
        sustainability.get("rebound_type", "unknown")

        # ============================================================
        # 1. 次日偏向
        # ============================================================
        if sustainability_score >= 70:
            result["bias"] = "bullish"
            result["reasons"].append("反弹持续性强")
        elif sustainability_score >= 55:
            result["bias"] = "slightly_bullish"
            result["reasons"].append("反弹有一定持续性")
        elif sustainability_score <= 30:
            result["bias"] = "bearish"
            result["reasons"].append("反弹持续性弱")
        elif sustainability_score <= 45:
            result["bias"] = "slightly_bearish"
            result["reasons"].append("反弹力度不足")

        # ============================================================
        # 2. 高开/低开概率
        # ============================================================
        today_change = change_pct.iloc[-1] if not change_pct.empty else 0

        # 大涨后高开概率高
        if today_change > 5:
            result["gap_open_probability"] = 70
            result["reasons"].append("今日大涨，次日高开概率70%")
        elif today_change > 3:
            result["gap_open_probability"] = 60
            result["reasons"].append("今日上涨，次日高开概率60%")
        elif today_change < -3:
            result["gap_open_probability"] = 30
            result["reasons"].append("今日下跌，次日低开概率70%")

        # ============================================================
        # 3. 预期振幅
        # ============================================================
        if len(close) >= 20:
            recent_volatility = change_pct.tail(10).std()
            result["expected_range"] = round(recent_volatility, 2)

        # ============================================================
        # 4. 操作建议
        # ============================================================
        if sustainability_score >= 70 and today_change > 2:
            result["trading_suggestion"] = "可考虑介入，次日高开获利了结"
        elif sustainability_score >= 55 and today_change > 1:
            result["trading_suggestion"] = "轻仓试探，设好止损"
        elif sustainability_score <= 35:
            result["trading_suggestion"] = "观望为主，不宜追高"
        elif today_change > 5:
            result["trading_suggestion"] = "涨幅过大，警惕高开低走"
        else:
            result["trading_suggestion"] = "等待更明确信号"

        # ============================================================
        # 5. 风险提醒
        # ============================================================
        risk_factors = sustainability.get("risk_factors", [])
        if risk_factors:
            result["reasons"].extend([f"风险: {r}" for r in risk_factors[:2]])

        return result
