"""
板块轮动 + 超跌反弹因子
识别板块强弱切换、超跌修复行情
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class SectorRotation:
    """板块轮动检测器"""

    def __init__(self):
        # 科技类板块关键词
        self.tech_sectors = [
            "芯片", "半导体", "人工智能", "AI", "算力", "光刻机",
            "消费电子", "苹果", "华为", "5G", "6G", "通信",
            "软件", "信创", "云计算", "大数据", "网络安全",
            "机器人", "自动驾驶", "智能汽车", "无人机",
            "光伏", "锂电", "新能源", "储能", "风电",
            "军工", "航空航天", "船舶", "核电",
            "医药", "创新药", "医疗器械", "CRO", "生物",
        ]

        # 周期类板块关键词
        self.cyclical_sectors = [
            "钢铁", "煤炭", "有色", "铜", "铝", "锂", "稀土",
            "化工", "石化", "石油", "天然气",
            "银行", "保险", "券商", "金融",
            "地产", "房地产", "基建", "建材",
            "消费", "白酒", "食品", "家电", "汽车",
        ]

    def analyze_sector_rotation(self, hot_df: pd.DataFrame, news_list: List[Dict]) -> Dict:
        """
        分析板块轮动
        返回: {
            strong_sectors: 强势板块列表,
            weak_sectors: 弱势板块列表,
            rotation_signal: 轮动信号,
            tech_strength: 科技板块强度,
            oversold_rebound: 超跌反弹信号,
        }
        """
        result = {
            "strong_sectors": [],
            "weak_sectors": [],
            "rotation_signal": "neutral",
            "tech_strength": 0,
            "oversold_rebound": False,
            "rebound_sectors": [],
        }

        if hot_df is None or hot_df.empty:
            return result

        # 提取热门概念
        hot_concepts = self._extract_hot_concepts(hot_df, news_list)

        # 分析科技板块热度
        tech_hot = [c for c in hot_concepts if any(k in c for k in self.tech_sectors)]
        cyclical_hot = [c for c in hot_concepts if any(k in c for k in self.cyclical_sectors)]

        result["tech_hot_count"] = len(tech_hot)
        result["cyclical_hot_count"] = len(cyclical_hot)

        # 科技板块强度（0-100）
        if len(hot_concepts) > 0:
            result["tech_strength"] = round(len(tech_hot) / len(hot_concepts) * 100, 1)

        # 检测超跌反弹
        rebound_info = self._detect_oversold_rebound(hot_concepts, news_list)
        result["oversold_rebound"] = rebound_info["detected"]
        result["rebound_sectors"] = rebound_info["sectors"]
        result["rebound_intensity"] = rebound_info["intensity"]

        # 判断轮动信号
        if len(tech_hot) > len(cyclical_hot) * 2:
            result["rotation_signal"] = "tech_strong"
            result["strong_sectors"] = tech_hot[:5]
        elif len(cyclical_hot) > len(tech_hot) * 2:
            result["rotation_signal"] = "cyclical_strong"
            result["strong_sectors"] = cyclical_hot[:5]
        else:
            result["rotation_signal"] = "balanced"

        return result

    def _extract_hot_concepts(self, hot_df: pd.DataFrame, news_list: List[Dict]) -> List[str]:
        """提取热门概念"""
        concepts = []

        # 从热股榜提取
        if "所属行业" in hot_df.columns:
            concepts.extend(hot_df["所属行业"].dropna().tolist())
        if "所属概念" in hot_df.columns:
            concepts.extend(hot_df["所属概念"].dropna().tolist())

        # 从新闻提取板块关键词
        for news in news_list[:20]:
            content = news.get("content", "")
            for keyword in self.tech_sectors + self.cyclical_sectors:
                if keyword in content and keyword not in concepts:
                    concepts.append(keyword)

        return list(set(concepts))

    def _detect_oversold_rebound(self, hot_concepts: List[str], news_list: List[Dict]) -> Dict:
        """
        检测超跌反弹信号
        条件：
        1. 近期有大跌报道
        2. 今日有反弹/修复报道
        3. 相关板块出现在热点中
        """
        result = {"detected": False, "sectors": [], "intensity": 0}

        # 检查新闻中的超跌反弹关键词
        rebound_keywords = ["反弹", "修复", "回暖", "回升", "拉升", "大涨", "强势"]
        oversold_keywords = ["大跌", "暴跌", "回调", "重挫", "杀跌", "崩盘", "恐慌"]

        rebound_news = []
        oversold_news = []

        for news in news_list[:30]:
            content = news.get("content", "")
            if any(k in content for k in rebound_keywords):
                rebound_news.append(content)
            if any(k in content for k in oversold_keywords):
                oversold_news.append(content)

        # 如果同时有超跌和反弹新闻，可能是修复行情
        if len(oversold_news) > 0 and len(rebound_news) > 0:
            # 检查具体板块
            for sector in self.tech_sectors:
                sector_in_rebound = any(sector in n for n in rebound_news)
                sector_in_oversold = any(sector in n for n in oversold_news)

                if sector_in_rebound and sector_in_oversold:
                    result["sectors"].append(sector)

            if len(result["sectors"]) > 0:
                result["detected"] = True
                result["intensity"] = min(len(result["sectors"]) * 20, 100)

        return result

    def get_rotation_score(self, rotation_data: Dict) -> float:
        """计算板块轮动评分 (0-100)"""
        score = 50

        # 科技板块强度
        tech_strength = rotation_data.get("tech_strength", 50)
        score += (tech_strength - 50) * 0.3

        # 超跌反弹信号
        if rotation_data.get("oversold_rebound"):
            intensity = rotation_data.get("rebound_intensity", 50)
            score += intensity * 0.3

        # 轮动信号
        signal = rotation_data.get("rotation_signal", "neutral")
        if signal == "tech_strong":
            score += 10
        elif signal == "cyclical_strong":
            score -= 5

        return max(0, min(100, score))


class OversoldReboundStrategy:
    """超跌反弹策略"""

    def __init__(self):
        pass

    def check_oversold_rebound(self, df: pd.DataFrame) -> Dict:
        """
        检测个股超跌反弹信号
        返回: {
            detected: bool,
            signals: list,
            score: float,
        }
        """
        result = {"detected": False, "signals": [], "score": 0}

        if df is None or df.empty or len(df) < 20:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())

        score = 0
        signals = []

        # 1. 近5日累计跌幅 > 10%
        if len(close) >= 6:
            ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if ret_5d < -10:
                signals.append(f"5日跌{ret_5d:.1f}%")
                score += 20
            elif ret_5d < -5:
                signals.append(f"5日跌{ret_5d:.1f}%")
                score += 10

        # 2. 近10日累计跌幅 > 15%
        if len(close) >= 11:
            ret_10d = (close.iloc[-1] / close.iloc[-11] - 1) * 100
            if ret_10d < -15:
                signals.append(f"10日跌{ret_10d:.1f}%")
                score += 25
            elif ret_10d < -8:
                signals.append(f"10日跌{ret_10d:.1f}%")
                score += 15

        # 3. RSI超卖（<30）
        if len(close) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            if not rsi.isna().iloc[-1]:
                rsi_val = rsi.iloc[-1]
                if rsi_val < 30:
                    signals.append(f"RSI={rsi_val:.0f}（超卖）")
                    score += 20
                elif rsi_val < 40:
                    signals.append(f"RSI={rsi_val:.0f}")
                    score += 10

        # 4. 今日放量反弹（涨幅>3%且放量）
        if len(df) >= 5:
            today_change = change_pct.iloc[-1] if not change_pct.empty else 0
            if today_change > 3:
                vol_col = "成交量" if "成交量" in df.columns else "volume"
                avg_vol = df[vol_col].iloc[-6:-1].mean()
                if avg_vol > 0 and df[vol_col].iloc[-1] / avg_vol > 1.5:
                    signals.append(f"放量反弹+{today_change:.1f}%")
                    score += 25

        # 5. 阳包阴（今日阳线包裹昨日阴线）
        if len(df) >= 2:
            o1, c1 = df.iloc[-2].get("开盘", 0), df.iloc[-2].get("收盘", 0)
            o2, c2 = df.iloc[-1].get("开盘", 0), df.iloc[-1].get("收盘", 0)
            if c1 < o1 and c2 > o2:  # 昨阴今阳
                if o2 <= c1 and c2 >= o1:  # 阳包阴
                    signals.append("阳包阴")
                    score += 20

        # 6. 触及布林带下轨后反弹
        if len(close) >= 20:
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            lower_band = ma20 - 2 * std20
            if close.iloc[-2] <= lower_band.iloc[-2] and close.iloc[-1] > close.iloc[-2]:
                signals.append("触及布林下轨后反弹")
                score += 15

        result["detected"] = len(signals) >= 2
        result["signals"] = signals
        result["score"] = min(score, 100)

        return result
