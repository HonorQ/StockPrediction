"""
政策事件因子 — 整合宏观事件检测+新闻政策分析+政策传导链映射

核心逻辑：
  1. 检测宏观事件（LPR/M2/PMI/CPI变化）
  2. 从新闻中识别政策类事件
  3. 通过政策传导链映射到受影响行业/个股
  4. 综合评分：政策影响力度×时效性×行业匹配度
"""
import os
import json
import math
from typing import Dict, List
from datetime import datetime

from src.factors.macro_event import MacroEventDetector


class PolicyEventFactor:
    """政策事件因子计算器"""

    # 政策新闻关键词（高优先级，比普通新闻权重高）
    POLICY_KEYWORDS = [
        "央行", "国务院", "证监会", "银保监会", "发改委", "工信部",
        "降准", "降息", "LPR", "MLF", "逆回购", "再贷款",
        "政治局会议", "中央经济工作会议", "两会", "政府工作报告",
        "补贴", "减税", "降费", "退税", "专项债",
        "制裁", "关税", "贸易战", "实体清单",
        "注册制", "IPO", "退市", "减持新规",
    ]

    # 重大事件关键词（最高优先级）
    MAJOR_EVENT_KEYWORDS = [
        "突发", "重磅", "紧急", "紧急通知", "紧急会议",
        "降准", "降息", "暂停IPO", "印花税",
        "战争", "冲突", "制裁", "禁运",
    ]

    def __init__(self):
        self.macro_detector = MacroEventDetector()
        self.policy_chain = self._load_policy_chain()
        self._macro_events_detected = False

    def _load_policy_chain(self) -> dict:
        """加载政策传导链映射"""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "policy_chain.json"
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ----------------------------------------------------------------
    # 主入口：计算政策事件对单只股票的影响
    # ----------------------------------------------------------------
    def calculate(self, stock_code: str, news_list: List[Dict],
                  stock_sector: str = "", stock_concepts: List[str] = None,
                  stock_name: str = "") -> Dict:
        """
        计算政策事件因子

        参数：
            stock_code: 股票代码
            news_list: 全局新闻列表
            stock_sector: 所属行业（可选）
            stock_concepts: 所属概念板块列表（可选）
            stock_name: 股票名称（可选，用于直接匹配）

        返回：
            {
                "policy_score": 0-100,
                "policy_direction": "positive"/"negative"/"neutral",
                "macro_impact": {...宏观事件影响...},
                "news_policy_impact": {...新闻政策影响...},
                "events": [...相关事件列表...],
                "signals": [...描述信号...]
            }
        """
        result = {
            "policy_score": 50,
            "policy_direction": "neutral",
            "macro_impact": {},
            "news_policy_impact": {},
            "events": [],
            "signals": [],
        }

        # 1. 宏观事件影响（全局性）
        macro_result = self._calculate_macro_impact(stock_code, stock_sector, stock_concepts)
        result["macro_impact"] = macro_result
        result["policy_score"] += (macro_result.get("impact_score", 50) - 50) * 0.4
        result["events"].extend(macro_result.get("events", []))
        result["signals"].extend(macro_result.get("signals", []))

        # 2. 新闻政策影响（个股相关）
        news_result = self._calculate_news_policy_impact(
            stock_code, news_list, stock_sector, stock_concepts, stock_name
        )
        result["news_policy_impact"] = news_result
        result["policy_score"] += (news_result.get("score", 50) - 50) * 0.6
        result["events"].extend(news_result.get("events", []))
        result["signals"].extend(news_result.get("signals", []))

        # 3. 综合方向判断
        result["policy_score"] = max(0, min(100, result["policy_score"]))
        if result["policy_score"] > 55:
            result["policy_direction"] = "positive"
        elif result["policy_score"] < 45:
            result["policy_direction"] = "negative"

        return result

    # ----------------------------------------------------------------
    # 宏观事件影响
    # ----------------------------------------------------------------
    def _calculate_macro_impact(self, stock_code: str, stock_sector: str,
                                 stock_concepts: List[str] = None) -> Dict:
        """计算宏观事件对个股的影响"""
        if not self._macro_events_detected:
            self.macro_detector.detect_all()
            self._macro_events_detected = True

        return self.macro_detector.get_stock_impact(
            stock_code, stock_sector, stock_concepts
        )

    # ----------------------------------------------------------------
    # 新闻政策影响
    # ----------------------------------------------------------------
    def _calculate_news_policy_impact(self, stock_code: str, news_list: List[Dict],
                                       stock_sector: str, stock_concepts: List[str],
                                       stock_name: str) -> Dict:
        """计算新闻政策对个股的影响"""
        result = {
            "score": 50,
            "events": [],
            "signals": [],
        }

        if not news_list:
            return result

        # 筛选政策相关新闻
        policy_news = []
        major_news = []

        for news in news_list:
            content = news.get("content", "") + " " + news.get("title", "")

            # 检查是否是政策新闻
            if any(kw in content for kw in self.POLICY_KEYWORDS):
                policy_news.append(news)

            # 检查是否是重大事件
            if any(kw in content for kw in self.MAJOR_EVENT_KEYWORDS):
                major_news.append(news)

        # 处理重大事件（最高优先级）
        for news in major_news:
            content = news.get("content", "")
            time_weight = self._time_weight(news, half_life_hours=12)

            # 直接提到该股
            if stock_code in content or (stock_name and stock_name in content):
                if any(kw in content for kw in ["利好", "突破", "增长", "超预期"]):
                    result["score"] += 20 * time_weight
                    result["signals"].append(f"重大利好事件提及{stock_name}")
                    result["events"].append({
                        "type": "major_positive",
                        "title": content[:50],
                        "impact": "high",
                    })
                elif any(kw in content for kw in ["利空", "下跌", "暴雷", "处罚", "减持"]):
                    result["score"] -= 20 * time_weight
                    result["signals"].append(f"重大利空事件提及{stock_name}")
                    result["events"].append({
                        "type": "major_negative",
                        "title": content[:50],
                        "impact": "high",
                    })

        # 处理政策新闻（通过传导链映射）
        for news in policy_news:
            content = news.get("content", "")
            time_weight = self._time_weight(news, half_life_hours=24)

            # 通过传导链匹配行业
            matched_sectors = self._match_policy_to_sectors(content)

            # 检查该股是否在受影响行业中
            if stock_sector and any(s in stock_sector for s in matched_sectors):
                # 判断政策方向
                if any(kw in content for kw in ["降准", "降息", "补贴", "减税", "支持", "利好"]):
                    result["score"] += 10 * time_weight
                    result["signals"].append(f"政策利好传导至{stock_sector}行业")
                elif any(kw in content for kw in ["加息", "收紧", "制裁", "限制", "利空"]):
                    result["score"] -= 10 * time_weight
                    result["signals"].append(f"政策利空传导至{stock_sector}行业")

            # 直接提到该股
            if stock_code in content or (stock_name and stock_name in content):
                if any(kw in content for kw in ["利好", "突破", "增长", "超预期", "获批", "中标"]):
                    result["score"] += 15 * time_weight
                    result["signals"].append(f"政策新闻直接利好{stock_name}")
                elif any(kw in content for kw in ["利空", "下跌", "暴雷", "处罚", "减持", "退市"]):
                    result["score"] -= 15 * time_weight
                    result["signals"].append(f"政策新闻直接利空{stock_name}")

        return result

    # ----------------------------------------------------------------
    # 政策传导链匹配
    # ----------------------------------------------------------------
    def _match_policy_to_sectors(self, news_content: str) -> List[str]:
        """通过政策传导链匹配新闻内容到受影响行业"""
        matched_sectors = []

        # 遍历政策传导链的所有分类
        for category in ["monetary_policy", "fiscal_policy", "industry_policy",
                         "geopolitical", "macro_indicators"]:
            chain = self.policy_chain.get(category, {})
            for policy_name, policy_info in chain.items():
                if policy_name.startswith("_"):
                    continue
                # 检查新闻中是否包含政策关键词
                if policy_name in news_content:
                    matched_sectors.extend(policy_info.get("affected_sectors", []))

        return list(set(matched_sectors))

    # ----------------------------------------------------------------
    # 时间衰减权重
    # ----------------------------------------------------------------
    @staticmethod
    def _time_weight(news: Dict, half_life_hours: float = 24.0) -> float:
        """根据新闻发布时间计算权重（政策新闻半衰期更长）"""
        t = news.get("time") or news.get("published_at") or news.get("timestamp")
        if t is None:
            return 0.5
        if isinstance(t, str):
            try:
                t = datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                return 0.5
        if not isinstance(t, datetime):
            return 0.5
        delta_hours = (datetime.now() - t).total_seconds() / 3600.0
        if delta_hours < 0:
            return 1.0
        return math.exp(-delta_hours / half_life_hours)
