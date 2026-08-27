# -*- coding: utf-8 -*-
"""数据库常量与序列化工具(由 database.py 拆出)。"""

import os
import json
import datetime
class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy / datetime 类型 → JSON 序列化"""
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "stock_history.db")


# 24 维子评分维度（与 StockScorer.get_sub_scores 完全一致）
# 用于 factor_snapshots 表的独立列，便于 SQL 直接过滤/排序/聚合，无需解析 JSON
SUB_SCORE_DIMS = [
    "trend", "strategy", "chip", "fundamental", "institutional",
    "sector_rotation", "oversold_rebound", "news", "relative_strength",
    "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
    "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
    "policy_event", "national_team", "capital_behavior", "moat",
    "holder_concentration", "circulate_holder",
]
