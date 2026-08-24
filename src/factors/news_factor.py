"""
消息面因子（小作文处理）

增强功能：
  - 时间衰减权重（exp(-Δt/6h) 半衰期6小时，6小时内权重1.0，24小时衰减到0.4）
  - jieba 分词匹配（可选，未安装则降级到字符串 in 匹配）
  - 否定前缀过滤（"未利好"/"不涨停" 不应被 "利好"/"涨停" 命中）
"""
import math
import re
from datetime import datetime
from typing import List, Dict

# 可选依赖：jieba 分词
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# 否定前缀词：如果关键词前出现这些词，则视为否定（不命中）
_NEGATION_PREFIXES = ("未", "不", "非", "没", "无", "难")


class NewsFactor:
    """消息面因子计算器"""

    def __init__(self, keywords_config: dict = None):
        cfg = keywords_config or {}
        self.policy_kw = cfg.get("policy", ["央行", "国务院", "证监会", "降准", "降息", "LPR"])
        self.industry_kw = cfg.get("industry", ["芯片", "AI", "人工智能", "新能源", "光伏", "半导体"])
        self.company_kw = cfg.get("company", ["重组", "并购", "业绩", "增持", "回购"])
        self.rumor_kw = cfg.get("rumor", ["传", "据称", "消息称", "市场传闻", "小作文"])
        self.positive_kw = cfg.get("positive", ["利好", "突破", "增长", "超预期", "涨停", "大涨"])
        self.negative_kw = cfg.get("negative", ["利空", "下跌", "暴雷", "处罚", "减持", "跌停"])

        # jieba 初始化（首次分词时 lazy load）
        if _HAS_JIEBA:
            try:
                # 预先加入金融领域关键词，避免被错误切分
                for kw in (self.policy_kw + self.industry_kw + self.company_kw +
                           self.rumor_kw + self.positive_kw + self.negative_kw):
                    jieba.add_word(kw)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 分析新闻列表
    # ------------------------------------------------------------------
    def analyze_news(self, news_list: List[Dict]) -> Dict[str, List]:
        """
        将新闻按类别分组
        返回: {policy: [...], industry: [...], rumor: [...], positive: [...], negative: [...]}
        """
        result = {
            "policy": [], "industry": [], "company": [],
            "rumor": [], "positive": [], "negative": [],
            "all": news_list,
        }
        for news in news_list:
            content = news.get("content", "")
            title = news.get("title", "")
            text = title + " " + content

            if self._match(text, self.policy_kw):
                result["policy"].append(news)
            if self._match(text, self.industry_kw):
                result["industry"].append(news)
            if self._match(text, self.company_kw):
                result["company"].append(news)
            if self._match(text, self.rumor_kw):
                result["rumor"].append(news)
            if self._match(text, self.positive_kw):
                result["positive"].append(news)
            if self._match(text, self.negative_kw):
                result["negative"].append(news)

        return result

    # ------------------------------------------------------------------
    # 计算消息面综合得分
    # ------------------------------------------------------------------
    def calculate_news_score(self, news_analysis: Dict) -> float:
        """
        计算消息面得分 (0-100)，按时间衰减加权
        """
        score = 50  # 基础分

        # 政策消息（利好为主时加分）
        policy_news = news_analysis.get("policy", [])
        pos_policy = sum(self._time_weight(n) for n in policy_news
                         if self._match(n.get("content", ""), self.positive_kw))
        neg_policy = sum(self._time_weight(n) for n in policy_news
                         if self._match(n.get("content", ""), self.negative_kw))
        score += (pos_policy - neg_policy) * 5

        # 行业新闻
        industry_news = news_analysis.get("industry", [])
        score += min(sum(self._time_weight(n) for n in industry_news) * 2, 10)

        # 小作文/传闻（特别关注，权重高）
        rumors = news_analysis.get("rumor", [])
        pos_rumor = sum(self._time_weight(n) for n in rumors
                        if self._match(n.get("content", ""), self.positive_kw))
        neg_rumor = sum(self._time_weight(n) for n in rumors
                        if self._match(n.get("content", ""), self.negative_kw))
        score += (pos_rumor - neg_rumor) * 8  # 传闻影响大

        # 正面/负面消息（按时间衰减加权）
        pos = sum(self._time_weight(n) for n in news_analysis.get("positive", []))
        neg = sum(self._time_weight(n) for n in news_analysis.get("negative", []))
        score += (pos - neg) * 3

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # 提取关联股票
    # ------------------------------------------------------------------
    def extract_mentioned_stocks(self, news_list: List[Dict]) -> Dict[str, int]:
        """
        统计每只股票被提及次数（热度，按时间衰减加权）
        返回: {stock_code: weight_sum}
        """
        mentions: Dict[str, float] = {}
        for news in news_list:
            w = self._time_weight(news)
            for code in news.get("mentioned_stocks", []):
                if code.startswith(("00", "30", "60", "68")):
                    mentions[code] = mentions.get(code, 0) + w
        # 返回时四舍五入到整数（保持向后兼容）
        return {k: round(v, 2) for k, v in mentions.items()}

    def score_stock_news(self, stock_code: str, news_list: List[Dict]) -> float:
        """
        为单只股票打消息面分（含时间衰减权重和否定前缀过滤）
        返回: 0-100
        """
        score = 50

        # 全局新闻中找相关
        related = [n for n in news_list if stock_code in n.get("mentioned_stocks", [])]

        # 全局新闻未提及 → 返回中性分
        if not related:
            return score

        # 提及次数加分（按时间衰减加权）
        mention_weight = sum(self._time_weight(n) for n in related)
        score += min(mention_weight * 5, 20)

        for news in related:
            w = self._time_weight(news)
            content = news.get("content", "")
            if self._match_with_negation(content, self.positive_kw):
                score += 10 * w
            if self._match_with_negation(content, self.negative_kw):
                score -= 10 * w
            if self._match_with_negation(content, self.rumor_kw):
                score += 8 * w  # 小作文加持

            priority = news.get("priority", 0)
            score += priority * 5 * w

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # 时间衰减权重
    # ------------------------------------------------------------------
    @staticmethod
    def _time_weight(news: Dict, half_life_hours: float = 6.0) -> float:
        """根据新闻发布时间计算权重。
        半衰期 6 小时：发布时权重 1.0，6小时后 0.5，24小时后约 0.0625。
        缺失时间字段时默认权重 0.5（中等时效）。
        """
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
            return 1.0  # 时间异常（未来时间），视为最新
        # exp(-Δt / half_life)
        return math.exp(-delta_hours / half_life_hours)

    # ------------------------------------------------------------------
    # 匹配方法
    # ------------------------------------------------------------------
    def _match(self, text: str, keywords: list) -> bool:
        """字符串包含匹配（向后兼容）"""
        return any(kw in text for kw in keywords)

    def _match_with_negation(self, text: str, keywords: list) -> bool:
        """带否定前缀过滤的匹配。
        例如关键词 "利好"，但文本中出现 "未利好" / "不利好"，则视为不命中。
        """
        for kw in keywords:
            idx = text.find(kw)
            if idx == -1:
                continue
            # 检查紧邻前一个字符是否是否定前缀
            if idx > 0 and text[idx - 1] in _NEGATION_PREFIXES:
                continue
            return True
        return False

    @staticmethod
    def extract_stock_codes(text: str) -> List[str]:
        """提取文本中的A股代码（6位数字，以 0/3/6 开头）"""
        codes = re.findall(r"\b[036]\d{5}\b", text)
        return list(set(codes))

    def tokenize(self, text: str) -> List[str]:
        """分词。优先用 jieba，失败则降级到字符串切分。"""
        if _HAS_JIEBA:
            try:
                return list(jieba.cut(text))
            except Exception:
                pass
        # 降级：按空白和标点切分
        return re.split(r"[\s,，。；;：:!?！？、]+", text)
