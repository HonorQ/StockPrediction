"""
多源新闻获取：财联社 + 东财 + 财新 + 新浪 + 同花顺
政策新闻核心来源增强
"""
from src.utils.log import get_logger
logger = get_logger()

import requests
import hashlib
import re
from typing import List, Dict
from datetime import datetime


class NewsFetcher:
    """新闻/小作文获取器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cls.cn/telegraph",
        })

    # ------------------------------------------------------------------
    # 财联社电报（核心）
    # ------------------------------------------------------------------
    def get_cls_telegraph(self, limit: int = 50) -> List[Dict]:
        """
        获取财联社7x24电报
        注意：财联社签名算法已变更，用akshare内置API作为最后备选
        """
        url = "https://www.cls.cn/v1/roll/get_roll_list"
        params = {
            "app": "CailianpressWeb",
            "os": "web",
            "rn": str(limit),
        }
        params["sign"] = self._generate_cls_sign(params)

        try:
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            # 检查签名错误
            if data.get("errno") == "10012":
                logger.info('  ⚠️ 财联社签名已变更，尝试akshare备选...')
                return self._get_cls_via_akshare(limit)

            news_list = []
            for item in data.get("data", {}).get("roll_data", []):
                content = self._clean_html(item.get("content", ""))
                news_list.append({
                    "source": "cls",
                    "time": datetime.fromtimestamp(item.get("ctime", 0)),
                    "content": content,
                    "title": item.get("title", ""),
                    "priority": self._get_priority(content),
                    "mentioned_stocks": self._extract_stock_codes(content),
                })

            if news_list:
                logger.info(f'  ✅ 财联社电报: {len(news_list)} 条')
                return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 财联社电报获取失败: {e}')
        return []

    def _get_cls_via_akshare(self, limit: int = 50) -> List[Dict]:
        """通过akshare内置API获取财联社电报（签名已封装）"""
        try:
            import akshare as ak
            df = ak.stock_info_global_cls()
            if df is None or df.empty:
                return []
            news_list = []
            for _, row in df.head(limit).iterrows():
                content = str(row.get("content", ""))
                news_list.append({
                    "source": "cls",
                    "time": row.get("ctime", ""),
                    "content": content,
                    "title": str(row.get("title", "")),
                    "priority": self._get_priority(content),
                    "mentioned_stocks": self._extract_stock_codes(content),
                })
            logger.info(f'  ✅ 财联社电报(akshare): {len(news_list)} 条')
            return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 财联社电报(akshare)也失败: {e}')
            return []

    # ------------------------------------------------------------------
    # 东财全球资讯（备选）
    # ------------------------------------------------------------------
    def get_eastmoney_news(self, limit: int = 50) -> List[Dict]:
        """东财全局新闻（akshare）"""
        try:
            import akshare as ak
            df = ak.stock_info_global_em()
            if df is None or df.empty:
                return []

            news_list = []
            for _, row in df.head(limit).iterrows():
                content = str(row.get("摘要", "")) or str(row.get("标题", ""))
                news_list.append({
                    "source": "eastmoney",
                    "time": row.get("发布时间", ""),
                    "content": content,
                    "title": str(row.get("标题", "")),
                    "priority": 1 if "利好" in content or "突破" in content else 0,
                    "mentioned_stocks": self._extract_stock_codes(content),
                })

            logger.info(f'  ✅ 东财全局新闻: {len(news_list)} 条')
            return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 东财全局新闻获取失败: {e}')
            return []
    # ------------------------------------------------------------------
    # 个股新闻
    # ------------------------------------------------------------------
    def get_stock_news(self, stock_code: str, limit: int = 10) -> List[Dict]:
        """获取个股相关新闻"""
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=stock_code)
            if df is not None and not df.empty:
                news_list = []
                for _, row in df.head(limit).iterrows():
                    news_list.append({
                        "time": row.get("发布时间", ""),
                        "content": row.get("新闻内容", ""),
                        "title": row.get("新闻标题", ""),
                        "source": row.get("文章来源", ""),
                    })
                return news_list
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # 财新新闻（第3来源）
    # ------------------------------------------------------------------
    def get_caixin_news(self, limit: int = 50) -> List[Dict]:
        """财新资讯（akshare）"""
        try:
            import akshare as ak
            df = ak.stock_news_main_cx()
            if df is None or df.empty:
                return []
            news_list = []
            for _, row in df.head(limit).iterrows():
                content = str(row.get("summary", ""))
                tag = str(row.get("tag", ""))
                news_list.append({
                    "source": "caixin",
                    "time": "",
                    "content": content,
                    "title": f"[{tag}] {content[:40]}" if tag else content[:50],
                    "priority": 1 if any(k in content for k in ["利好", "突破", "大涨", "突发", "重磅"]) else 0,
                    "mentioned_stocks": self._extract_stock_codes(content),
                })
            logger.info(f'  ✅ 财新资讯: {len(news_list)} 条')
            return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 财新资讯获取失败: {e}')
            return []

    # ------------------------------------------------------------------
    # 新浪全球快讯（第4来源）— 政策新闻覆盖率高
    # ------------------------------------------------------------------
    def get_sina_global(self, limit: int = 50) -> List[Dict]:
        """新浪财经全球快讯（akshare）"""
        try:
            import akshare as ak
            df = ak.stock_info_global_sina()
            if df is None or df.empty:
                return []
            news_list = []
            for _, row in df.head(limit).iterrows():
                content = str(row.get("内容", ""))
                news_list.append({
                    "source": "sina",
                    "time": row.get("时间", ""),
                    "content": content,
                    "title": content[:50] if content else "",
                    "priority": self._get_priority(content),
                    "mentioned_stocks": self._extract_stock_codes(content),
                })
            logger.info(f'  ✅ 新浪全球快讯: {len(news_list)} 条')
            return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 新浪全球快讯获取失败: {e}')
            return []

    # ------------------------------------------------------------------
    # 同花顺快讯（第5来源）— 政策解读+个股公告
    # ------------------------------------------------------------------
    def get_ths_news(self, limit: int = 50) -> List[Dict]:
        """同花顺全球快讯（akshare）"""
        try:
            import akshare as ak
            df = ak.stock_info_global_ths()
            if df is None or df.empty:
                return []
            news_list = []
            for _, row in df.head(limit).iterrows():
                title = str(row.get("标题", ""))
                content = str(row.get("内容", ""))
                news_list.append({
                    "source": "ths",
                    "time": row.get("发布时间", ""),
                    "content": content,
                    "title": title,
                    "priority": self._get_priority(content),
                    "mentioned_stocks": self._extract_stock_codes(content),
                })
            logger.info(f'  ✅ 同花顺快讯: {len(news_list)} 条')
            return news_list
        except Exception as e:
            logger.info(f'  ⚠️ 同花顺快讯获取失败: {e}')
        return []

    # ------------------------------------------------------------------
    # 多源合并（主入口）
    # ------------------------------------------------------------------
    def get_all_news(self, limit_per_source: int = 50) -> List[Dict]:
        """
        从5个新闻源并行获取，合并去重
        返回时按时间排序（最新的在前）
        """
        all_news = []
        seen = set()

        # 源1: 财联社电报
        cls_news = self.get_cls_telegraph(limit_per_source)
        for n in cls_news:
            key = n.get("content", "")[:60]
            if key and key not in seen:
                seen.add(key)
                all_news.append(n)

        # 源2: 东财全局新闻
        em_news = self.get_eastmoney_news(limit_per_source)
        for n in em_news:
            key = n.get("content", "")[:60]
            if key and key not in seen:
                seen.add(key)
                all_news.append(n)

        # 源3: 财新资讯
        cx_news = self.get_caixin_news(limit_per_source)
        for n in cx_news:
            key = n.get("content", "")[:60]
            if key and key not in seen:
                seen.add(key)
                all_news.append(n)

        # 源4: 新浪全球快讯（政策新闻覆盖率高）
        sina_news = self.get_sina_global(limit_per_source)
        for n in sina_news:
            key = n.get("content", "")[:60]
            if key and key not in seen:
                seen.add(key)
                all_news.append(n)

        # 源5: 同花顺快讯（政策解读+个股公告）
        ths_news = self.get_ths_news(limit_per_source)
        for n in ths_news:
            key = n.get("content", "")[:60]
            if key and key not in seen:
                seen.add(key)
                all_news.append(n)

        # 按时间排序（最新的在前）
        def _parse_time(n):
            t = n.get("time", "")
            if isinstance(t, datetime):
                return t
            if isinstance(t, str):
                try:
                    return datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S")
                except:
                    pass
            return datetime.min
        all_news.sort(key=_parse_time, reverse=True)

        logger.info(f'  ✅ 新闻汇总: CLS({len(cls_news)}) + 东财({len(em_news)}) + 财新({len(cx_news)}) + 新浪({len(sina_news)}) + 同花顺({len(ths_news)}) = {len(all_news)} 条（去重后）')
        return all_news

    # 工具方法
    # ------------------------------------------------------------------
    def _generate_cls_sign(self, params: dict) -> str:
        """生成财联社本地签名（无需API Key）"""
        sorted_params = sorted(params.items())
        query = "&".join(f"{k}={v}" for k, v in sorted_params)
        sha1 = hashlib.sha1(query.encode()).hexdigest()
        md5 = hashlib.md5(sha1.encode()).hexdigest()
        return md5

    @staticmethod
    def _clean_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    @staticmethod
    def _get_priority(content: str) -> int:
        """2=突发重要, 1=重要, 0=普通"""
        if "【突发】" in content or "【重磅】" in content:
            return 2
        elif "【重要】" in content or "【关注】" in content:
            return 1
        return 0

    @staticmethod
    def _extract_stock_codes(content: str) -> List[str]:
        """从文本中提取A股股票代码。

        \\b 词边界在中文语境失效：中文与数字同属 \\w，"股票代码600519今天大涨"
        中 600519 两侧无词边界 → 永远提取不到。改用环视断言：
        前后都不能是数字/字母（中文、标点、行首尾均允许）。
        """
        codes = re.findall(r"(?<![0-9A-Za-z])([036]\d{5})(?![0-9A-Za-z])", content)
        return list(set(codes))
