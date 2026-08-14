"""
三重筛选器
第一层：硬性条件排除（ST、涨跌停、市值、成交额）
第二层：信号强度筛选（必须有足够多的强信号）
第三层：综合评分筛选（必须达到最低分数线）
"""
from src.utils.log import get_logger
logger = get_logger()

import pandas as pd
from typing import List, Dict


class StockScreener:
    """股票三重筛选器"""

    def __init__(self, config: dict):
        self.cfg = config
        self.min_market_cap = config.get("min_market_cap", 50) * 1e8  # 亿→元
        self.min_turnover = config.get("min_turnover", 10000) * 1e4    # 万→元
        self.exclude_st = config.get("exclude_st", True)
        self.exclude_new = config.get("exclude_new", True)
        self.new_days = config.get("new_days", 60)

    # ------------------------------------------------------------------
    # 第一层：硬性条件排除
    # ------------------------------------------------------------------
    def filter_hard_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤不合格股票，返回可交易的候选池
        """
        if df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        # 排除ST
        if self.exclude_st:
            mask &= ~df["名称"].str.contains("ST", case=False, na=False)

        # 排除停牌（成交额为0）
        mask &= df["成交额"] > 0

        # 排除涨停（无法买入）
        if "涨跌幅" in df.columns:
            mask &= df["涨跌幅"] < 9.8

        # 排除跌停（风险过大）
        if "涨跌幅" in df.columns:
            mask &= df["涨跌幅"] > -9.8

        # 最小市值（仅当列存在且有有效数据时）
        if "总市值" in df.columns:
            valid_mcap = df["总市值"].dropna()
            if not valid_mcap.empty and valid_mcap.max() > 0:
                mask &= df["总市值"] >= self.min_market_cap
            else:
                logger.info('  ⚠️ 总市值列无有效数据，跳过市值筛选')
            if '成交额' in df.columns:
                mask &= df["成交额"] >= self.min_turnover

        filtered = df[mask].copy()
        logger.info(f'  📋 第一层硬性筛选: {len(df)} → {len(filtered)} 只')
        return filtered

    # ------------------------------------------------------------------
    # 第二层：信号强度筛选
    # ------------------------------------------------------------------
    def filter_signal_strength(
        self, stocks: List[Dict], min_strong: int = 3
    ) -> List[Dict]:
        """
        必须有足够多的"强信号"（因子值接近极端值）
        """
        qualified = []
        for stock in stocks:
            factors = stock.get("factors", {})
            # 计算强信号数量
            strong_count = 0

            # 技术面强信号
            if factors.get("macd_golden_cross", 0) == 1:
                strong_count += 1
            if factors.get("ma_bullish", 0) == 1:
                strong_count += 1
            if factors.get("breakout_20d_high", 0) == 1:
                strong_count += 1
            if factors.get("volume_price_up", 0) == 1:
                strong_count += 1
            if factors.get("boll_position", 0.5) > 0.7:
                strong_count += 1

            # 消息面强信号
            if factors.get("news_score", 50) > 70:
                strong_count += 1

            # 策略强信号
            strategy_count = factors.get("strategy_count", 0)
            if strategy_count >= 2:
                strong_count += 1

            # 筹码强信号
            chip_score = factors.get("chip_score", 50)
            if chip_score > 70:
                strong_count += 1

            stock["strong_signals"] = strong_count

            if strong_count >= min_strong:
                qualified.append(stock)

        logger.info(f'  🎯 第二层信号筛选: {len(stocks)} → {len(qualified)} 只 (强信号≥{min_strong})')
        return qualified

    # ------------------------------------------------------------------
    # 第三层：综合评分筛选
    # ------------------------------------------------------------------
    def filter_by_score(
        self, stocks: List[Dict], min_score: float = 85
    ) -> List[Dict]:
        """按综合评分筛选"""
        qualified = [s for s in stocks if s.get("score", 0) >= min_score]
        logger.info(f'  ⭐ 第三层评分筛选: {len(stocks)} → {len(qualified)} 只 (≥{min_score}分)')
        return qualified

    # ------------------------------------------------------------------
    # 第四层：行业分散约束（避免推荐股票高度集中在单一行业）
    # ------------------------------------------------------------------
    def diversify_by_industry(
        self,
        stocks: List[Dict],
        max_per_industry: int = 1,
        industry_map: Dict[str, str] = None,
        max_total: int = None,
    ) -> List[Dict]:
        """
        按行业分散推荐：同一行业最多保留 max_per_industry 只。
        输入的 stocks 应已按 score 降序排好。
        若设置 max_total，最终返回数量不超过该值。

        参数：
          stocks: 已排序的候选股票列表
          max_per_industry: 单个行业最多保留的股票数
          industry_map: {code: industry} 映射，若为 None 则跳过分散约束
          max_total: 返回的最大数量
        """
        if not industry_map:
            return stocks[:max_total] if max_total else stocks

        # stocks 已按 score 降序，逐个挑选，每选中一只就给该行业计数+1
        result = []
        industry_count: Dict[str, int] = {}
        skipped = []
        for s in stocks:
            code = s.get("code", "")
            industry = industry_map.get(code, "未知")
            cnt = industry_count.get(industry, 0)
            if cnt < max_per_industry:
                result.append(s)
                industry_count[industry] = cnt + 1
                if max_total and len(result) >= max_total:
                    break
            else:
                skipped.append((s.get("name", code), industry))

        if skipped:
            logger.info(f'  🏭 行业分散约束: 跳过 {len(skipped)} 只同行业重复')
            for s in skipped[:3]:
                name, ind = s
                logger.info(f'      • {name} ({ind})')
        return result
