"""
热点股票 + 情绪指标（完全基于新浪/腾讯公开行情，不依赖东财）
"""
from src.utils.log import get_logger
logger = get_logger()

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.utils.helpers import get_limit_pct


def _limit_thresholds(df: pd.DataFrame) -> Optional[np.ndarray]:
    """按每只股票的板块/ST口径计算涨停判定阈值数组（limit_pct - 0.5）。
    缺代码列时返回 None，调用方退回 9.5 一刀切。"""
    if "代码" not in df.columns:
        return None
    names = df["名称"].astype(str).values if "名称" in df.columns else [""] * len(df)
    codes = df["代码"].astype(str).values
    return np.array([get_limit_pct(c, n) - 0.5 for c, n in zip(codes, names)])


class HotStocksFetcher:
    """热点股票获取器（全免费公开数据源）"""

    def __init__(self, market_df: Optional[pd.DataFrame] = None):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._market_df = market_df

    def set_market_data(self, df: pd.DataFrame):
        self._market_df = df

    def _get_market_data(self) -> pd.DataFrame:
        """获取实时行情（优先用已注入的数据，否则内部获取）"""
        if self._market_df is not None and not self._market_df.empty:
            return self._market_df
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        df = fetcher.get_a_share_realtime()
        self._market_df = df
        return df

    # ------------------------------------------------------------------
    # 人气榜（按成交额排序模拟）
    # ------------------------------------------------------------------
    def get_hot_rank(self) -> pd.DataFrame:
        """获取人气榜 — 按成交额降序模拟（替代东财人气榜API）"""
        try:
            df = self._get_market_data()
            if df.empty:
                return pd.DataFrame()

            amount_col = None
            for c in ["成交额", "amount", "成交金额", "total_amount"]:
                if c in df.columns:
                    amount_col = c
                    break
            if amount_col is None:
                return pd.DataFrame()

            sorted_df = df.sort_values(amount_col, ascending=False).head(100).copy()
            sorted_df.reset_index(drop=True, inplace=True)

            result = pd.DataFrame({
                "当前排名": range(1, len(sorted_df) + 1),
                "排名": range(1, len(sorted_df) + 1),
                "代码": sorted_df.get("代码", sorted_df.get("股票代码", "")),
                "股票代码": sorted_df.get("代码", sorted_df.get("股票代码", "")),
                "股票名称": sorted_df.get("名称", sorted_df.get("股票名称", "")),
                "成交额": sorted_df[amount_col].values,
                "涨跌幅": sorted_df.get("涨跌幅", sorted_df.get("pctChg", 0)),
                "换手率": sorted_df.get("换手率", sorted_df.get("turnover", 0)),
            })
            logger.info(f'  ✅ 人气榜(成交额排序): {len(result)} 只')
            return result
        except Exception as e:
            logger.info(f'  ⚠️ 获取人气榜失败: {e}')
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 涨停/跌停板块（从实时行情涨跌幅计算）
    # ------------------------------------------------------------------
    def get_limit_up_pool(self) -> pd.DataFrame:
        """获取涨停板 — 从实时行情中筛选涨跌幅>=9.5%的股票"""
        try:
            df = self._get_market_data()
            if df.empty:
                return pd.DataFrame()
            change_col = None
            for c in ["涨跌幅", "pctChg", "change_pct"]:
                if c in df.columns:
                    change_col = c
                    break
            if change_col is None:
                return pd.DataFrame()
            # 涨停判定按板块口径(10/20/30cm,ST 5%)，原 ±9.5 一刀切漏掉 20cm/30cm、误判 ST
            thresholds = _limit_thresholds(df)
            if thresholds is not None:
                limit_up = df[df[change_col].values >= thresholds].copy()
            else:
                limit_up = df[df[change_col] >= 9.5].copy()
            logger.info(f'  ✅ 涨停板: {len(limit_up)} 只')
            return limit_up
        except Exception as e:
            logger.info(f'  ⚠️ 获取涨停池失败: {e}')
            return pd.DataFrame()

    def get_limit_down_pool(self) -> pd.DataFrame:
        """获取跌停板 — 从实时行情中筛选涨跌幅<=-9.5%的股票"""
        try:
            df = self._get_market_data()
            if df.empty:
                return pd.DataFrame()
            change_col = None
            for c in ["涨跌幅", "pctChg", "change_pct"]:
                if c in df.columns:
                    change_col = c
                    break
            if change_col is None:
                return pd.DataFrame()
            thresholds = _limit_thresholds(df)
            if thresholds is not None:
                limit_down = df[df[change_col].values <= -thresholds].copy()
            else:
                limit_down = df[df[change_col] <= -9.5].copy()
            logger.info(f'  ✅ 跌停板: {len(limit_down)} 只')
            return limit_down
        except Exception as e:
            logger.info(f'  ⚠️ 获取跌停池失败: {e}')
            return pd.DataFrame()

    def get_broken_limit_pool(self) -> pd.DataFrame:
        """炸板池 — 新浪无逐笔数据，返回空"""
        logger.info('  ℹ️ 炸板池: 新浪源无逐笔数据，跳过')
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 龙虎榜（无免费替代，降级）
    # ------------------------------------------------------------------
    def get_dragon_tiger(self) -> pd.DataFrame:
        """龙虎榜 — 东财API被封，返回空"""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 概念板块（从本地缓存读取）
    # ------------------------------------------------------------------
    def get_hot_concepts(self) -> pd.DataFrame:
        """概念板块涨幅排名 — 东财API被封，返回空（从concept_map.json获取）"""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 市场情绪综合指标
    # ------------------------------------------------------------------
    def calculate_market_sentiment(self, market_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算市场情绪综合指标（基于实时行情）
        如果传入 market_df 则复用，否则内部获取
        """
        if market_df is not None:
            self._market_df = market_df

        limit_up_df = self.get_limit_up_pool()
        limit_down_df = self.get_limit_down_pool()

        limit_up = len(limit_up_df)
        limit_down = len(limit_down_df)

        score = 50
        score += min(limit_up * 0.5, 25)
        score -= min(limit_down * 1.0, 25)
        score = max(0, min(100, score))

        if score >= 70:
            emotion_level = "🔥 偏热"
        elif score >= 50:
            emotion_level = "😐 中性"
        else:
            emotion_level = "❄️ 偏冷"

        result = {
            "emotion_level": emotion_level,
            "emotion_score": round(score, 1),
            "limit_up": limit_up,
            "limit_down": limit_down,
            "broken": 0,
            "max_board": 0,
        }

        logger.info(f'  ✅ 市场情绪: {emotion_level} (得分={score:.0f}, 涨停={limit_up}, 跌停={limit_down})')
        return result
