# -*- coding: utf-8 -*-
"""SQLite数据库 - 历史因子截面 + 推荐追踪 + 回测记录。

按表域拆分为多个 Mixin(AGENTS.md 技术债偿还), 本文件为组合入口:
对外接口 StockDatabase 与拆分前完全一致。
"""

from src.data.db.base import BaseStockDB
from src.data.db.recommendations import RecommendationsMixin
from src.data.db.portfolio import PortfolioMixin
from src.data.db.market import MarketMixin
from src.data.db.feedback import FeedbackMixin


class StockDatabase(PortfolioMixin, MarketMixin, FeedbackMixin,
                    RecommendationsMixin, BaseStockDB):
    """组合类: MRO 保证 BaseStockDB.__init__ 最后执行。"""


# 兼容 re-export(旧引用 from src.data.database import ...)
