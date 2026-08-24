"""
预期透支 + 利好兑现因子
识别"买预期卖事实"、利好落地反跌、获利盘出逃
"""
import pandas as pd
from typing import Dict, List

from src.utils.helpers import get_limit_pct


class ExpectationPricing:
    """预期透支分析器"""

    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, news_list: List[Dict] = None,
                stock_code: str = None) -> Dict:
        """
        分析预期透支程度
        df: 个股历史数据
        news_list: 新闻列表
        stock_code: 股票代码
        """
        result = {
            "expectation_score": 50,
            "overvalued": False,
            "reasons": [],
            "risk_level": "medium",
        }

        if df is None or df.empty or len(df) < 30:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]

        score = 50
        reasons = []

        # ============================================================
        # 1. 公告前涨幅（是否已提前上涨）
        # ============================================================
        announcement_analysis = self._analyze_pre_announcement_rise(close, change_pct)
        score += announcement_analysis["score_delta"]
        reasons.extend(announcement_analysis["reasons"])

        # ============================================================
        # 2. 获利盘压力
        # ============================================================
        profit_pressure = self._analyze_profit_pressure(close)
        score += profit_pressure["score_delta"]
        reasons.extend(profit_pressure["reasons"])

        # ============================================================
        # 3. 量价背离
        # ============================================================
        divergence = self._analyze_volume_price_divergence(close, volume, change_pct)
        score += divergence["score_delta"]
        reasons.extend(divergence["reasons"])

        # ============================================================
        # 4. 技术超买
        # ============================================================
        overbought = self._analyze_overbought(close, change_pct)
        score += overbought["score_delta"]
        reasons.extend(overbought["reasons"])

        # ============================================================
        # 5. 新闻利好已发布检测
        # ============================================================
        if news_list and stock_code:
            news_impact = self._analyze_news_timing(news_list, stock_code, close)
            score += news_impact["score_delta"]
            reasons.extend(news_impact["reasons"])

        result["expectation_score"] = max(0, min(100, score))
        result["reasons"] = reasons
        result["overvalued"] = score < 32
        result["risk_level"] = "high" if score < 25 else ("low" if score > 70 else "medium")

        return result

    def _analyze_pre_announcement_rise(self, close: pd.Series, change_pct: pd.Series) -> Dict:
        """分析公告前涨幅"""
        result = {"score_delta": 0, "reasons": []}

        if len(close) < 30:
            return result

        # 近20日涨幅
        ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        # 近60日涨幅
        ret_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0

        # 20日涨幅过大，预期透支
        if ret_20d > 45:
            result["score_delta"] -= 25
            result["reasons"].append(f"20日涨{ret_20d:.0f}%（预期严重透支）")
        elif ret_20d > 30:
            result["score_delta"] -= 15
            result["reasons"].append(f"20日涨{ret_20d:.0f}%（预期透支）")
        elif ret_20d > 15:
            result["score_delta"] -= 5
            result["reasons"].append(f"20日涨{ret_20d:.0f}%")

        if ret_60d > 80:
            result["score_delta"] -= 15
            result["reasons"].append(f"60日涨{ret_60d:.0f}%（累计涨幅巨大）")

        return result

    def _analyze_profit_pressure(self, close: pd.Series) -> Dict:
        """分析获利盘压力"""
        result = {"score_delta": 0, "reasons": []}

        if len(close) < 60:
            return result

        current = close.iloc[-1]
        low_60 = close.tail(60).min()

        # 获利盘比例（当前价vs60日最低价）
        profit_ratio = (current / low_60 - 1) * 100

        if profit_ratio > 60:
            result["score_delta"] -= 20
            result["reasons"].append(f"获利盘{profit_ratio:.0f}%（出逃压力大）")
        elif profit_ratio > 40:
            result["score_delta"] -= 10
            result["reasons"].append(f"获利盘{profit_ratio:.0f}%（有一定压力）")

        # 距离60日高点
        high_60 = close.tail(60).max()
        if current >= high_60 * 0.98:
            result["score_delta"] -= 10
            result["reasons"].append("处于高位（创新高后易回落）")

        return result

    def _analyze_volume_price_divergence(self, close: pd.Series, volume: pd.Series,
                                          change_pct: pd.Series) -> Dict:
        """分析量价背离"""
        result = {"score_delta": 0, "reasons": []}

        if len(close) < 10 or len(volume) < 10:
            return result

        # 价格创新高但成交量萎缩（顶背离）
        price_high = close.iloc[-1] >= close.tail(10).max() * 0.98
        vol_declining = volume.iloc[-1] < volume.tail(10).mean() * 0.8

        if price_high and vol_declining:
            result["score_delta"] -= 15
            result["reasons"].append("量价顶背离（价涨量缩）")

        # 连续上涨但量能递减
        if len(close) >= 5:
            price_rising = all(close.iloc[-i] > close.iloc[-i-1] for i in range(1, 4))
            vol_declining连续 = all(volume.iloc[-i] < volume.iloc[-i-1] for i in range(1, 4))
            if price_rising and vol_declining连续:
                result["score_delta"] -= 10
                result["reasons"].append("连续上涨但量能递减（上涨乏力）")

        return result

    def _analyze_overbought(self, close: pd.Series, change_pct: pd.Series) -> Dict:
        """分析技术超买"""
        result = {"score_delta": 0, "reasons": []}

        if len(close) < 14:
            return result

        # RSI超买
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        if not rsi.isna().iloc[-1]:
            rsi_val = rsi.iloc[-1]
            if rsi_val > 90:
                result["score_delta"] -= 15
                result["reasons"].append(f"RSI={rsi_val:.0f}（严重超买）")
            elif rsi_val > 82:
                result["score_delta"] -= 8
                result["reasons"].append(f"RSI={rsi_val:.0f}（超买）")

        # 连续上涨天数
        if len(change_pct) >= 5:
            up_days = sum(1 for i in range(-5, 0) if change_pct.iloc[i] > 0)
            if up_days >= 5:
                result["score_delta"] -= 5
                result["reasons"].append(f"连涨{up_days}天（获利盘累积）")

        return result

    def _analyze_news_timing(self, news_list: List[Dict], stock_code: str,
                              close: pd.Series) -> Dict:
        """分析新闻发布时间"""
        result = {"score_delta": 0, "reasons": []}

        # 查找该股票的业绩公告类新闻
        positive_keywords = ["业绩", "净利润", "增长", "超预期", "扭亏", "分红", "高送转"]
        company_news = []

        for news in news_list:
            content = news.get("content", "")
            if stock_code in content or any(k in content for k in positive_keywords):
                company_news.append(news)

        if company_news:
            # 检查是否已有利好公告
            result["reasons"].append("检测到业绩/利好公告")
            # 如果股价已大涨，说明预期已透支
            if len(close) >= 5:
                ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
                if ret_5d > 15:
                    result["score_delta"] -= 20
                    result["reasons"].append("利好公告前已大涨15%+（买预期卖事实）")
                elif ret_5d > 8:
                    result["score_delta"] -= 10
                    result["reasons"].append("利好公告前已上涨（预期部分透支）")

        return result


class NewsEventAnalyzer:
    """新闻事件分析器"""

    def __init__(self):
        # 利好落地变利空的典型模式
        self.pricing_patterns = {
            "earnings_beat": ["业绩超预期", "净利润增长", "扭亏为盈"],
            "dividend": ["高送转", "分红", "派息"],
            "contract": ["中标", "签约", "订单"],
            "policy": ["利好", "政策支持", "补贴"],
        }

    def analyze_event_pricing(self, news_list: List[Dict], stock_code: str,
                               df: pd.DataFrame) -> Dict:
        """
        分析新闻事件是否已被定价
        """
        result = {
            "event_priced": False,
            "pricing_degree": 0,
            "signals": [],
            "suggestion": "",
        }

        if df is None or df.empty or len(df) < 20:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        # 查找相关新闻
        relevant_news = self._find_relevant_news(news_list, stock_code)

        if not relevant_news:
            return result

        # 分析新闻类型
        event_type = self._classify_news_type(relevant_news)

        # 检查股价是否已反映
        if len(close) >= 20:
            ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100

            if ret_20d > 20 and event_type:
                result["event_priced"] = True
                result["pricing_degree"] = min(ret_20d / 2, 100)
                result["signals"].append(f"{event_type}类利好，20日已涨{ret_20d:.0f}%")
                result["suggestion"] = "利好可能已充分定价，警惕高开低走"
            elif ret_20d > 10 and event_type:
                result["event_priced"] = True
                result["pricing_degree"] = ret_20d
                result["signals"].append(f"{event_type}类利好，20日已涨{ret_20d:.0f}%")
                result["suggestion"] = "利好部分定价，谨慎追高"

        return result

    def _find_relevant_news(self, news_list: List[Dict], stock_code: str) -> List[Dict]:
        """查找相关新闻"""
        relevant = []
        for news in news_list:
            content = news.get("content", "")
            if stock_code in content:
                relevant.append(news)
        return relevant[:5]

    def _classify_news_type(self, news_list: List[Dict]) -> str:
        """分类新闻类型"""
        all_content = " ".join([n.get("content", "") for n in news_list])

        for pattern_name, keywords in self.pricing_patterns.items():
            if any(k in all_content for k in keywords):
                return pattern_name

        return "unknown"


class ProfitTakingDetector:
    """获利盘出逃检测器"""

    def __init__(self):
        pass

    def detect(self, df: pd.DataFrame, code: str = "", name: str = "") -> Dict:
        """
        检测获利盘出逃信号
        """
        result = {
            "detected": False,
            "intensity": 0,
            "signals": [],
        }

        if df is None or df.empty or len(df) < 20:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        signals = []
        intensity = 0

        # ============================================================
        # 1. 高位放量阴线
        # ============================================================
        if len(close) >= 5:
            today_change = change_pct.iloc[-1] if not change_pct.empty else 0
            today_vol = volume.iloc[-1]
            avg_vol = volume.iloc[-6:-1].mean()
            high_pos = close.iloc[-1] >= close.tail(20).max() * 0.95

            if today_change < -3 and today_vol > avg_vol * 1.5 and high_pos:
                signals.append("高位放量阴线（主力出货）")
                intensity += 30

        # ============================================================
        # 2. 冲高回落（上影线长）
        # ============================================================
        if len(df) >= 1:
            row = df.iloc[-1]
            o = row.get("开盘", 0)
            h = row.get("最高", 0)
            c = row.get("收盘", 0)
            row.get("最低", 0)

            if h > 0 and o > 0:
                upper_shadow = h - max(o, c)
                body = abs(c - o)
                if upper_shadow > body * 2 and c < h * 0.97:
                    signals.append("冲高回落（上影线长）")
                    intensity += 20

        # ============================================================
        # 3. 涨停板被砸开（按板块口径：接近涨停但未封住）
        # ============================================================
        if not change_pct.empty:
            today_change = change_pct.iloc[-1]
            limit_pct = get_limit_pct(code, name)
            if limit_pct - 2 < today_change < limit_pct - 0.2:
                signals.append("涨停板被砸开（抛压重）")
                intensity += 25

        # ============================================================
        # 4. 连续缩量阴跌
        # ============================================================
        if len(change_pct) >= 5:
            recent = change_pct.tail(5)
            down_days = sum(1 for x in recent if x < 0)
            if down_days >= 4:
                signals.append(f"近5日{down_days}天下跌（阴跌出货）")
                intensity += 15

        # ============================================================
        # 5. 量价背离加剧
        # ============================================================
        if len(close) >= 10 and len(volume) >= 10:
            # 价格横盘但成交量放大（主力对倒出货）
            price_range = (close.tail(10).max() - close.tail(10).min()) / close.tail(10).mean()
            vol_increase = volume.tail(5).mean() / volume.tail(10).head(5).mean()

            if price_range < 0.03 and vol_increase > 1.5:
                signals.append("横盘放量（对倒出货嫌疑）")
                intensity += 20

        result["detected"] = len(signals) >= 2
        result["intensity"] = min(intensity, 100)
        result["signals"] = signals

        return result
