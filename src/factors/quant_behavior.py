"""
量化行为识别模块
识别A股量化交易特征：开盘秒板、尾盘闪崩、盘中异动、大单撤单、集合竞价异动
"""
import pandas as pd
from typing import Dict

from src.utils.helpers import get_limit_pct, limit_up_price as calc_limit_up, limit_down_price as calc_limit_down


class QuantBehaviorDetector:
    """量化行为检测器"""

    def __init__(self):
        # 量化行为特征
        self.quant_patterns = {
            "flash_crash": "闪崩",
            "instant_limit": "秒板",
            "tail_manipulation": "尾盘操控",
            "auction_anomaly": "集合竞价异动",
            "large_order_spoof": "大单诱多/诱空",
            "momentum_chasing": "动量追逐",
        }

    def detect_all(self, df: pd.DataFrame, realtime: Dict = None) -> Dict:
        """
        检测所有量化行为特征
        df: 历史数据
        realtime: 实时数据
        """
        result = {
            "quant_detected": False,
            "quant_score": 50,
            "quant_signals": [],
            "quant_pattern": "unknown",
            "quant_risk": "low",
        }

        signals = []
        score = 50

        # 板块口径：按代码前缀/ST标记确定涨跌停幅度(10/20/30/5%)
        realtime = realtime or {}
        stock_code = str(realtime.get("代码", "") or "")
        stock_name = str(realtime.get("名称", "") or "")

        # 1. 闪崩检测（巨量下跌）
        flash_crash = self._detect_flash_crash(df)
        if flash_crash["detected"]:
            signals.append(flash_crash["signal"])
            score -= 15

        # 2. 秒板检测（开盘秒涨停）
        instant_limit = self._detect_instant_limit(df, stock_code, stock_name)
        if instant_limit["detected"]:
            signals.append(instant_limit["signal"])
            score += 10 if instant_limit["type"] == "up" else -10

        # 3. 尾盘操控检测
        tail_manip = self._detect_tail_manipulation(df)
        if tail_manip["detected"]:
            signals.append(tail_manip["signal"])
            score -= 5

        # 4. 集合竞价异动检测
        auction_anomaly = self._detect_auction_anomaly(realtime)
        if auction_anomaly["detected"]:
            signals.append(auction_anomaly["signal"])
            score -= 10

        # 5. 大单诱多/诱空检测
        large_order = self._detect_large_order_spoof(df)
        if large_order["detected"]:
            signals.append(large_order["signal"])
            score -= 12

        # 6. 动量追逐检测（量化特征）
        momentum = self._detect_momentum_chasing(df)
        if momentum["detected"]:
            signals.append(momentum["signal"])
            score -= 8

        # 7. 分时异动检测（几分钟内巨量）
        intraday = self._detect_intraday_anomaly(df)
        if intraday["detected"]:
            signals.append(intraday["signal"])
            score -= 10

        # 8. 涨停板质量检测
        limit_quality = self._detect_limit_quality(df, stock_code, stock_name)
        if limit_quality["detected"]:
            signals.append(limit_quality["signal"])
            score -= 8 if limit_quality["type"] == "weak" else 5

        # 综合判断
        if signals:
            result["quant_detected"] = True
            result["quant_signals"] = signals
            result["quant_score"] = max(0, min(100, score))

            # 风险等级
            if len(signals) >= 3:
                result["quant_risk"] = "high"
                result["quant_pattern"] = "complex"
            elif len(signals) >= 2:
                result["quant_risk"] = "medium"
            else:
                result["quant_risk"] = "low"

        return result

    def _detect_flash_crash(self, df: pd.DataFrame) -> Dict:
        """
        闪崩检测
        特征：几分钟内巨量下跌，通常-5%以上
        """
        result = {"detected": False, "signal": "", "type": "down"}

        if df is None or len(df) < 5:
            return result

        # 计算日内波动
        high = df["最高"] if "最高" in df.columns else df.get("high", pd.Series())
        low = df["最低"] if "最低" in df.columns else df.get("low", pd.Series())
        close = df["收盘"] if "收盘" in df.columns else df.get("close", pd.Series())
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        if len(change_pct) < 5:
            return result

        # 近5日有单日跌幅>5%
        recent_5d = change_pct.tail(5)
        for val in recent_5d:
            if val < -5:
                result["detected"] = True
                result["signal"] = f"闪崩（近5日跌幅{val:.1f}%）"
                return result

        # 近5日有单日振幅>15%
        if len(high) >= 5 and len(low) >= 5 and len(close) >= 5:
            recent_high = high.tail(5)
            recent_low = low.tail(5)
            recent_close = close.tail(5)
            for i in range(len(recent_close)):
                if recent_low.iloc[i] > 0:
                    amplitude = (recent_high.iloc[i] - recent_low.iloc[i]) / recent_low.iloc[i] * 100
                    if amplitude > 15:
                        result["detected"] = True
                        result["signal"] = f"巨幅震荡（振幅{amplitude:.0f}%）"
                        return result

        return result

    def _detect_instant_limit(self, df: pd.DataFrame, code: str = "", name: str = "") -> Dict:
        """
        秒板检测
        特征：开盘即涨停/跌停，量化抢筹/出货
        """
        result = {"detected": False, "signal": "", "type": "unknown"}

        if df is None or df.empty:
            return result

        today = df.iloc[-1]
        open_price = today.get("开盘", 0)
        close_price = today.get("收盘", 0)
        high = today.get("最高", 0)
        low = today.get("最低", 0)
        pre_close = today.get("昨收", 0)

        if pre_close <= 0 or open_price <= 0:
            return result

        # 开盘价就是涨停/跌停（按板块口径，原硬编码1.1/0.9对20cm/30cm/ST全部误判）
        limit_pct = get_limit_pct(code, name)
        limit_up_price = calc_limit_up(pre_close, limit_pct)
        limit_down_price = calc_limit_down(pre_close, limit_pct)

        if abs(open_price - limit_up_price) / limit_up_price < 0.001:
            result["detected"] = True
            result["signal"] = "开盘秒涨停（量化抢筹）"
            result["type"] = "up"
        elif abs(open_price - limit_down_price) / limit_down_price < 0.001:
            result["detected"] = True
            result["signal"] = "开盘秒跌停（量化出货）"
            result["type"] = "down"

        return result

    def _detect_tail_manipulation(self, df: pd.DataFrame) -> Dict:
        """
        尾盘操控检测
        特征：尾盘30分钟大幅拉升/砸盘
        """
        result = {"detected": False, "signal": ""}

        if df is None or len(df) < 2:
            return result

        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        today_close = today.get("收盘", 0)
        today_high = today.get("最高", 0)
        today_low = today.get("最低", 0)
        yesterday_close = yesterday.get("收盘", 0)

        if yesterday_close <= 0 or today_close <= 0:
            return result

        # 收盘价接近最高价（尾盘拉升）
        if today_high > 0:
            close_to_high = (today_close - today_low) / (today_high - today_low) if today_high > today_low else 0.5
            if close_to_high > 0.9:
                # 尾盘拉升幅度
                tail拉升 = (today_close / yesterday_close - 1) * 100
                if tail拉升 > 3:
                    result["detected"] = True
                    result["signal"] = f"尾盘拉升{tail拉升:.1f}%（疑似量化操控）"
                elif tail拉升 < -3:
                    result["detected"] = True
                    result["signal"] = f"尾盘砸盘{tail拉升:.1f}%（疑似量化出货）"

        return result

    def _detect_auction_anomaly(self, realtime: Dict = None) -> Dict:
        """
        集合竞价异动检测
        特征：集合竞价阶段大幅高开/低开
        """
        result = {"detected": False, "signal": ""}

        if realtime is None:
            return result

        # 从实时数据获取集合竞价信息
        open_price = realtime.get("open", 0)
        price = realtime.get("price", 0)
        pre_close = realtime.get("pre_close", 0)

        if pre_close <= 0 or open_price <= 0:
            return result

        # 高开/低开幅度
        gap = (open_price / pre_close - 1) * 100

        if gap > 5:
            result["detected"] = True
            result["signal"] = f"集合竞价高开{gap:.1f}%（量化抢筹）"
        elif gap < -5:
            result["detected"] = True
            result["signal"] = f"集合竞价低开{gap:.1f}%（量化出货）"

        return result

    def _detect_large_order_spoof(self, df: pd.DataFrame) -> Dict:
        """
        大单诱多/诱空检测
        特征：巨量成交但价格不动（对倒）
        """
        result = {"detected": False, "signal": ""}

        if df is None or len(df) < 10:
            return result

        volume = df["成交量"] if "成交量" in df.columns else df.get("volume", pd.Series())
        close = df["收盘"] if "收盘" in df.columns else df.get("close", pd.Series())
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        if len(volume) < 10 or len(close) < 10:
            return result

        # 近10日平均成交量
        avg_volume_10d = volume.tail(10).mean()
        today_volume = volume.iloc[-1]

        # 量比
        volume_ratio = today_volume / avg_volume_10d if avg_volume_10d > 0 else 1

        # 今日涨跌幅
        today_change = change_pct.iloc[-1] if len(change_pct) > 0 else 0

        # 巨量但价格不动（对倒嫌疑）
        if volume_ratio > 5 and abs(today_change) < 2:
            result["detected"] = True
            result["signal"] = f"巨量横盘（量比{volume_ratio:.1f}，疑似对倒）"

        # 巨量涨/跌（可能是量化对倒）
        if volume_ratio > 8:
            result["detected"] = True
            if today_change > 3:
                result["signal"] = f"巨量拉升（量比{volume_ratio:.1f}，量化对倒嫌疑）"
            elif today_change < -3:
                result["signal"] = f"巨量下跌（量比{volume_ratio:.1f}，量化砸盘嫌疑）"

        return result

    def _detect_momentum_chasing(self, df: pd.DataFrame) -> Dict:
        """
        动量追逐检测
        特征：连续同方向运动（量化动量策略）
        """
        result = {"detected": False, "signal": ""}

        if df is None or len(df) < 5:
            return result

        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        if len(change_pct) < 5:
            return result

        recent_5d = change_pct.tail(5)

        # 连续5天同方向
        up_count = sum(1 for x in recent_5d if x > 0)
        down_count = sum(1 for x in recent_5d if x < 0)

        if up_count == 5:
            result["detected"] = True
            result["signal"] = "5连阳（量化动量追逐）"
        elif down_count == 5:
            result["detected"] = True
            result["signal"] = "5连阴（量化动量出货）"

        return result

    def _detect_intraday_anomaly(self, df: pd.DataFrame) -> Dict:
        """
        分时异动检测
        特征：几分钟内巨量涨跌
        """
        result = {"detected": False, "signal": ""}

        if df is None or df.empty:
            return result

        today = df.iloc[-1]
        change_pct = today.get("涨跌幅", 0)
        volume = today.get("成交量", 0)

        if len(df) < 10:
            return result

        # 今日成交量占比
        total_volume = df["成交量"].tail(10).sum() if "成交量" in df.columns else df.get("volume", pd.Series()).tail(10).sum()
        if total_volume > 0:
            volume_ratio = volume / total_volume
            # 今日成交量占比>50%且涨跌幅大
            if volume_ratio > 0.5 and abs(change_pct) > 5:
                result["detected"] = True
                if change_pct > 0:
                    result["signal"] = f"分时巨量拉升（占比{volume_ratio*100:.0f}%，量化行为）"
                else:
                    result["signal"] = f"分时巨量下跌（占比{volume_ratio*100:.0f}%，量化砸盘）"

        return result

    def _detect_limit_quality(self, df: pd.DataFrame, code: str = "", name: str = "") -> Dict:
        """
        涨停板质量检测
        特征：涨停后反复开板（量化博弈）
        """
        result = {"detected": False, "signal": "", "type": "unknown"}

        if df is None or df.empty:
            return result

        today = df.iloc[-1]
        change_pct = today.get("涨跌幅", 0)
        high = today.get("最高", 0)
        close = today.get("收盘", 0)
        pre_close = today.get("昨收", 0)

        if pre_close <= 0:
            return result

        limit_pct = get_limit_pct(code, name)
        limit_up_price = calc_limit_up(pre_close, limit_pct)

        # 涨停但没封住（阈值按板块口径缩放：20cm股的"接近涨停"标准更宽松）
        near_limit_change = limit_pct - 1
        if change_pct > near_limit_change and abs(close - limit_up_price) / limit_up_price > 0.005:
            result["detected"] = True
            result["signal"] = "涨停未封住（量化博弈）"
            result["type"] = "weak"

        return result
