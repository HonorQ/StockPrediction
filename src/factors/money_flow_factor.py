"""
资金因子计算 - 增加占比（相对值）
"""
from typing import Dict


class MoneyFlowFactors:
    """资金因子计算器"""

    def calculate_all(self, flow_data: dict, north_data: dict = None) -> Dict[str, float]:
        """
        flow_data: 从 MoneyFlowFetcher.get_stock_fund_flow() 获取
        north_data: 北向资金数据
        """
        f = {}

        # 绝对值
        main_flow = flow_data.get("main_flow", 0) or 0
        super_large = flow_data.get("super_large", 0) or 0
        large = flow_data.get("large", 0) or 0
        medium = flow_data.get("medium", 0) or 0
        small = flow_data.get("small", 0) or 0

        f["main_flow_raw"] = main_flow
        f["main_flow"] = 1.0 if main_flow > 0 else (-1.0 if main_flow < 0 else 0)
        f["super_large_flow"] = 1.0 if super_large > 0 else (-1.0 if super_large < 0 else 0)
        f["large_flow"] = 1.0 if large > 0 else (-1.0 if large < 0 else 0)
        f["medium_flow"] = 1.0 if medium > 0 else (-1.0 if medium < 0 else 0)
        f["small_flow"] = 1.0 if small > 0 else (-1.0 if small < 0 else 0)
        f["big_money_direction"] = 1.0 if (main_flow + super_large) > 0 else -1.0

        # ============ 新增：占比（相对值）============
        total = abs(main_flow) + abs(super_large) + abs(large) + abs(medium) + abs(small)
        if total > 0:
            # 主力净流入占比（主力+超大单 / 总量）
            f["main_flow_ratio"] = round((main_flow + super_large) / total, 4)
            # 超大单占比
            f["super_large_ratio"] = round(super_large / total, 4)
            # 大单占比
            f["large_ratio"] = round(large / total, 4)
            # 散户净流入占比（反向指标）
            f["small_flow_ratio"] = round(small / total, 4)
            # 主力集中度（主力+超大单占比的绝对值）
            f["main_concentration"] = round(abs(main_flow + super_large) / total, 4)
        else:
            f["main_flow_ratio"] = 0
            f["super_large_ratio"] = 0
            f["large_ratio"] = 0
            f["small_flow_ratio"] = 0
            f["main_concentration"] = 0

        # 北向资金
        if north_data:
            net = north_data.get("net_flow", 0) or 0
            f["north_flow"] = 1.0 if net > 0 else (-1.0 if net < 0 else 0)
        else:
            f["north_flow"] = 0.0

        return f
