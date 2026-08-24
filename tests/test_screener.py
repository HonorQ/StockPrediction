"""StockScreener 单元测试（离线、纯函数、无网络）"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.screener import StockScreener


def _make_df():
    return pd.DataFrame([
        {"代码": "600001", "名称": "正常股", "成交额": 2e8, "涨跌幅": 3.0, "总市值": 100e8},
        {"代码": "600002", "名称": "ST风险", "成交额": 2e8, "涨跌幅": 1.0, "总市值": 100e8},
        {"代码": "600003", "名称": "涨停股", "成交额": 2e8, "涨跌幅": 10.0, "总市值": 100e8},
        {"代码": "600004", "名称": "跌停股", "成交额": 2e8, "涨跌幅": -9.9, "总市值": 100e8},
        {"代码": "600005", "名称": "小市值", "成交额": 0.5e8, "涨跌幅": 1.0, "总市值": 30e8},
        {"代码": "600006", "名称": "停牌股", "成交额": 0, "涨跌幅": 0.0, "总市值": 100e8},
    ])


class TestHardConditions(unittest.TestCase):
    def setUp(self):
        self.screener = StockScreener({
            "min_market_cap": 50, "min_turnover": 10000,
            "exclude_st": True, "exclude_new": True, "new_days": 60,
        })

    def test_filters_all_rules(self):
        out = self.screener.filter_hard_conditions(_make_df())
        # 仅"正常股"通过(ST/涨停/跌停/小市值/停牌均被排除)
        self.assertEqual(list(out["代码"]), ["600001"])

    def test_empty_input(self):
        out = self.screener.filter_hard_conditions(pd.DataFrame())
        self.assertTrue(out.empty)


class TestSignalStrength(unittest.TestCase):
    def setUp(self):
        self.screener = StockScreener({})

    def test_strong_signal_count_and_threshold(self):
        stocks = [
            {"code": "A", "factors": {
                "macd_golden_cross": 1, "ma_bullish": 1,
                "breakout_20d_high": 1, "news_score": 75}},
            {"code": "B", "factors": {}},
        ]
        out = self.screener.filter_signal_strength(stocks, min_strong=3)
        self.assertEqual([s["code"] for s in out], ["A"])
        self.assertEqual(stocks[0]["strong_signals"], 4)
        self.assertEqual(stocks[1]["strong_signals"], 0)


class TestDiversify(unittest.TestCase):
    def setUp(self):
        self.screener = StockScreener({})
        self.stocks = [
            {"code": "A1", "name": "甲1"}, {"code": "A2", "name": "甲2"},
            {"code": "B1", "name": "乙1"},
        ]
        self.industry_map = {"A1": "科技", "A2": "科技", "B1": "金融"}

    def test_max_per_industry(self):
        out = self.screener.diversify_by_industry(
            self.stocks, max_per_industry=1, industry_map=self.industry_map)
        self.assertEqual([s["code"] for s in out], ["A1", "B1"])

    def test_max_total(self):
        out = self.screener.diversify_by_industry(
            self.stocks, max_per_industry=2, industry_map=self.industry_map, max_total=2)
        self.assertEqual(len(out), 2)

    def test_no_map_passthrough(self):
        out = self.screener.diversify_by_industry(self.stocks, max_per_industry=1)
        self.assertEqual(len(out), len(self.stocks))


if __name__ == "__main__":
    unittest.main()
