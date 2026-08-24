"""
A股特色因子：融资融券、股东增减持、解禁风险、季节性效应
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

from src.utils.helpers import get_limit_pct

logger = logging.getLogger(__name__)


class MarginTradingFactor:
    """融资融券因子（杠杆资金）"""

    # 类级缓存：同一次 premarket 只需获取一次全市场融资融券数据
    _margin_cache: Dict[str, pd.DataFrame] = {}
    _cache_date: str = None

    def __init__(self):
        pass

    def _get_margin_data(self) -> pd.DataFrame:
        """获取最近交易日的融资融券明细（带类级缓存）。

        融资融券数据 T 日收盘后发布，盘前查当天必然为空。
        尝试过去 5 天，命中第一个有数据的日期后缓存。
        """
        import akshare as ak

        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today_str and self._margin_cache:
            # 返回最近一个有数据的日期对应的数据
            latest = max(self._margin_cache.keys())
            return self._margin_cache[latest]

        for days_ago in range(1, 6):
            d = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
            if d in self._margin_cache:
                continue
            try:
                df = ak.stock_margin_detail_sse(date=d)
                if df is not None and not df.empty:
                    self._margin_cache[d] = df
                    self._cache_date = today_str
                    return df
            except Exception as e:
                logger.debug("stock_margin_detail_sse(%s) 失败: %s", d, e)
                continue
        return pd.DataFrame()

    def calculate(self, stock_code: str) -> Dict:
        """计算融资融券相关因子"""
        result = {
            "margin_score": 50,
            "margin_trend": "neutral",
            "margin_signals": [],
        }

        try:
            margin_data = self._get_margin_data()
            if margin_data is not None and not margin_data.empty:
                # 查找个股数据
                code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")
                stock_margin = margin_data[margin_data["标的证券代码"].astype(str) == code]

                if not stock_margin.empty:
                    row = stock_margin.iloc[0]
                    margin_buy = float(row.get("融资买入额", 0)) if pd.notna(row.get("融资买入额")) else 0
                    margin_balance = float(row.get("融资余额", 0)) if pd.notna(row.get("融资余额")) else 0
                    short_sell = float(row.get("融券卖出量", 0)) if pd.notna(row.get("融券卖出量")) else 0
                    short_balance = float(row.get("融券余额", 0)) if pd.notna(row.get("融券余额")) else 0

                    # 融资买入活跃度
                    if margin_buy > 0:
                        if margin_buy > 1e8:  # 融资买入>1亿
                            result["margin_trend"] = "active"
                            result["margin_signals"].append(f"融资买入{margin_buy/1e8:.1f}亿（活跃）")
                        elif margin_buy > 5e7:
                            result["margin_trend"] = "moderate"

                    # 融资余额变化（需历史数据对比，此处简化）
                    if margin_balance > 0:
                        result["margin_balance"] = margin_balance

                    # 融券卖出（做空信号）
                    if short_sell > 1e6:
                        result["margin_signals"].append(f"融券卖出{short_sell/1e6:.0f}万股（做空）")
                        result["margin_score"] -= 10

                    # 融资融券比
                    if margin_balance > 0 and short_balance > 0:
                        ratio = margin_balance / short_balance
                        if ratio > 10:
                            result["margin_signals"].append("融资远大于融券（看多）")
                            result["margin_score"] += 10
                        elif ratio < 2:
                            result["margin_signals"].append("融券占比高（看空）")
                            result["margin_score"] -= 10

        except Exception as e:
            logger.debug("MarginTradingFactor.calculate(%s) 失败: %s", stock_code, e)

        return result


class ShareholderChangeFactor:
    """股东增减持因子（内部人交易）"""

    def __init__(self):
        pass

    def calculate(self, stock_code: str) -> Dict:
        """计算股东增减持因子"""
        result = {
            "shareholder_score": 50,
            "shareholder_trend": "neutral",
            "shareholder_signals": [],
        }

        try:
            import akshare as ak
            # 获取股东增减持数据
            code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")

            # stock_share_change_cninfo 返回的是股本变动表，列名无"变动方向"
            # 通过变动原因编码和持股结构变化判断方向
            try:
                change_data = ak.stock_share_change_cninfo(symbol=code)
                if change_data is not None and not change_data.empty:
                    # 取最近2期对比持股变化
                    recent = change_data.head(5)
                    # 变动原因编码中含"减持"关键词或持股下降→减持
                    for _, row in recent.iterrows():
                        reason_code = str(row.get("变动原因编码", ""))
                        reason_desc = str(row.get("变动原因", ""))
                        # 部分版本有变动原因描述列
                        if "减持" in reason_desc or "减持" in reason_code:
                            result["shareholder_trend"] = "decrease"
                            result["shareholder_signals"].append("股东减持（内部人卖出）")
                            result["shareholder_score"] -= 15
                            break
                        elif "增持" in reason_desc or "增持" in reason_code:
                            result["shareholder_trend"] = "increase"
                            result["shareholder_signals"].append("股东增持（内部人看好）")
                            result["shareholder_score"] += 15
                            break
            except Exception as e:
                logger.debug("stock_share_change_cninfo(%s) 失败: %s", code, e)

        except Exception as e:
            logger.debug("ShareholderChangeFactor.calculate(%s) 失败: %s", stock_code, e)

        return result


class LockUpExpiryFactor:
    """解禁风险因子（限售股解禁）"""

    def __init__(self):
        pass

    def calculate(self, stock_code: str) -> Dict:
        """计算解禁风险因子"""
        result = {
            "lockup_score": 50,
            "lockup_risk": "low",
            "lockup_signals": [],
        }

        try:
            import akshare as ak
            # 获取解禁数据
            code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")

            try:
                lockup_data = ak.stock_restricted_release_queue_sina(symbol=code)
                if lockup_data is not None and not lockup_data.empty:
                    # 近90日解禁（扩大窗口到90天，捕捉更多解禁风险）
                    today = datetime.now()
                    for _, row in lockup_data.iterrows():
                        try:
                            unlock_date = pd.to_datetime(row.get("解禁日期", ""))
                            if pd.notna(unlock_date):
                                days_to_unlock = (unlock_date - today).days
                                if 0 <= days_to_unlock <= 90:
                                    # 字段名是"解禁数量"不是"解禁股数"
                                    unlock_shares = float(row.get("解禁数量", 0)) if pd.notna(row.get("解禁数量")) else 0
                                    if unlock_shares > 0:
                                        if days_to_unlock <= 30:
                                            result["lockup_risk"] = "high"
                                            result["lockup_score"] -= 20
                                        else:
                                            result["lockup_risk"] = "medium"
                                            result["lockup_score"] -= 10
                                        result["lockup_signals"].append(f"{days_to_unlock}天后解禁{unlock_shares/1e4:.0f}万股")
                                        break
                        except:
                            continue
            except:
                pass

        except Exception:
            pass

        return result


class SeasonalityFactor:
    """季节性因子（时间效应）"""

    def __init__(self):
        # A股季节性规律
        self.seasonal_patterns = {
            1: {"trend": "up", "reason": "春季躁动"},
            2: {"trend": "up", "reason": "春季躁动"},
            3: {"trend": "neutral", "reason": "年报披露期"},
            4: {"trend": "down", "reason": "业绩地雷"},
            5: {"trend": "down", "reason": "五穷"},
            6: {"trend": "down", "reason": "六绝"},
            7: {"trend": "up", "reason": "七翻身"},
            8: {"trend": "neutral", "reason": "中报披露"},
            9: {"trend": "neutral", "reason": "节前效应"},
            10: {"trend": "up", "reason": "国庆行情"},
            11: {"trend": "up", "reason": "年末布局"},
            12: {"trend": "down", "reason": "年末资金紧张"},
        }

    def calculate(self) -> Dict:
        """计算季节性因子"""
        result = {
            "season_score": 50,
            "season_trend": "neutral",
            "season_signals": [],
        }

        current_month = datetime.now().month
        pattern = self.seasonal_patterns.get(current_month, {})

        if pattern:
            result["season_trend"] = pattern.get("trend", "neutral")
            result["season_signals"].append(pattern.get("reason", ""))

            # 根据趋势调整评分
            if pattern["trend"] == "up":
                result["season_score"] = 70
            elif pattern["trend"] == "down":
                result["season_score"] = 30
            else:
                result["season_score"] = 50

        # 周几效应
        current_weekday = datetime.now().weekday()
        if current_weekday == 0:
            result["season_signals"].append("周一效应（谨慎）")
            result["season_score"] -= 5
        elif current_weekday == 4:
            result["season_signals"].append("周五效应（观望）")
            result["season_score"] -= 3

        return result


class LimitUpQualityFactor:
    """涨停板质量因子（封单强度）"""

    def __init__(self):
        pass

    def calculate(self, df: pd.DataFrame, code: str = "", name: str = "") -> Dict:
        """计算涨停板质量因子"""
        result = {
            "limit_quality_score": 50,
            "limit_quality": "unknown",
            "limit_signals": [],
        }

        if df is None or df.empty:
            return result

        today = df.iloc[-1]
        change_pct = today.get("涨跌幅", 0)
        close = today.get("收盘", 0)
        pre_close = today.get("昨收", 0)
        volume = today.get("成交量", 0)

        if pre_close <= 0:
            return result

        # 按板块口径计算涨跌停价（原硬编码1.1/0.9对20cm/30cm/ST全部误判）
        limit_pct = get_limit_pct(code, name)
        limit_up_price = round(pre_close * (1 + limit_pct / 100), 2)

        # 涨停检测
        if abs(close - limit_up_price) / limit_up_price < 0.005:
            # 涨停了
            if len(df) >= 2:
                yesterday = df.iloc[-2]
                yesterday_change = yesterday.get("涨跌幅", 0)

                # 连板检测
                if yesterday_change > limit_pct - 1:
                    result["limit_quality"] = "连板"
                    result["limit_signals"].append("连板（强势）")
                    result["limit_quality_score"] = 80
                else:
                    result["limit_quality"] = "首板"
                    result["limit_signals"].append("首板涨停")
                    result["limit_quality_score"] = 65

                # 一字板检测（开盘=收盘=涨停价）
                open_price = today.get("开盘", 0)
                high = today.get("最高", 0)
                low = today.get("最低", 0)
                if abs(open_price - limit_up_price) / limit_up_price < 0.005:
                    if abs(high - low) / low < 0.001:
                        result["limit_quality"] = "一字板"
                        result["limit_signals"].append("一字板（极强）")
                        result["limit_quality_score"] = 95

        # 跌停检测
        limit_down_price = round(pre_close * (1 - limit_pct / 100), 2)
        if abs(close - limit_down_price) / limit_down_price < 0.005:
            result["limit_quality"] = "跌停"
            result["limit_signals"].append("跌停（极弱）")
            result["limit_quality_score"] = 10

        return result
