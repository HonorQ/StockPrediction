"""MarketRegimeFilter 单元测试（离线、纯函数、无网络）"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.market_regime import MarketRegimeFilter


def _make_filter(enabled=True):
    f = MarketRegimeFilter(market_fetcher=None)
    f._regime_config = {
        "enabled": enabled,
        "empty_on_bear": True,
        "bear_score_threshold": 35,
        "weights": {
            "bear": {"trend": 0.001, "strategy": 0.03, "volatility": 0.08},
            "crash": {"trend": 0.001, "strategy": 0.01, "volatility": 0.1},
            "bull": {"trend": 0.05},
            "neutral": {"trend": 0.01},
        },
    }
    return f


class TestGetRegimeWeights(unittest.TestCase):
    def test_overrides_applied(self):
        # 2026-09-11 语义修订: 覆盖致总和漂移超差时等比缩放回基准总和
        # (防评分尺度膨胀), 断言改为相对结构: 覆盖生效 + 未覆盖维度等比保留。
        base = {"trend": 0.05, "strategy": 0.05, "moat": 0.03}
        out = _make_filter().get_regime_weights("bear", base)
        # bear 覆盖 trend=0.001, strategy=0.03 (volatility 跳过), moat 未覆盖
        self.assertAlmostEqual(
            out["trend"] / out["strategy"], 0.001 / 0.03, places=4)
        # 覆盖后总和 = 基准总和(0.13), 未覆盖维度与被覆盖维度同比例缩放
        self.assertAlmostEqual(sum(out.values()), 0.13, places=4)
        self.assertAlmostEqual(
            out["moat"] / out["strategy"], 0.03 / 0.03, places=4)

    def test_invalid_dims_skipped(self):
        # volatility 不在24维评分体系，必须被跳过而不是新增为废维度
        base = {"trend": 0.05, "strategy": 0.05}
        out = _make_filter().get_regime_weights("crash", base)
        self.assertNotIn("volatility", out)
        self.assertEqual(set(out.keys()), set(base.keys()))

    def test_disabled_returns_base(self):
        base = {"trend": 0.05}
        f = _make_filter(enabled=False)
        self.assertIs(f.get_regime_weights("bear", base), base)


class TestAssessRegime(unittest.TestCase):
    def assess(self, sh_chg, sentiment, cyb_chg=0.0, sz_chg=0.0):
        f = _make_filter()
        return f.assess_regime(
            market_overview={
                "上证指数": {"change": sh_chg},
                "深证成指": {"change": sz_chg},
                "创业板指": {"change": cyb_chg},
            },
            sentiment=sentiment,
            northbound_data=None,
            global_markets=None,
        )

    def test_crash_on_index_plunge_and_limit_down_wave(self):
        r = self.assess(-2.5, {"limit_up": 10, "limit_down": 40, "broken": 0, "max_board": 1})
        self.assertEqual(r["regime"], "crash")
        self.assertEqual(r["max_recommend"], 0)

    def test_bear_on_double_bear_signals(self):
        r = self.assess(-1.2, {"limit_up": 15, "limit_down": 20, "broken": 0, "max_board": 2})
        self.assertEqual(r["regime"], "bear")
        self.assertEqual(r["max_recommend"], 1)

    def test_bull_on_double_bull_signals(self):
        r = self.assess(1.5, {"limit_up": 60, "limit_down": 5, "broken": 0, "max_board": 5},
                        cyb_chg=2.0)
        self.assertEqual(r["regime"], "bull")
        self.assertEqual(r["max_recommend"], 5)

    def test_neutral_default(self):
        r = self.assess(0.3, {"limit_up": 30, "limit_down": 8, "broken": 2, "max_board": 3})
        self.assertEqual(r["regime"], "neutral")

    def test_northbound_outflow_triggers_crash_signal(self):
        f = _make_filter()
        r = f.assess_regime(
            market_overview={"上证指数": {"change": -2.2}},
            sentiment={"limit_up": 10, "limit_down": 35, "broken": 0, "max_board": 1},
            northbound_data={"net_inflow": -100, "consecutive_days": 4},
            global_markets=None,
        )
        self.assertIn("北向大幅净流出", " ".join(r["reasons"]))


class TestShouldEmptyPortfolio(unittest.TestCase):
    def test_crash_always_empty(self):
        self.assertTrue(_make_filter().should_empty_portfolio("crash", 80))

    def test_bear_below_threshold_empty(self):
        self.assertTrue(_make_filter().should_empty_portfolio("bear", 30))

    def test_bear_above_threshold_not_empty(self):
        self.assertFalse(_make_filter().should_empty_portfolio("bear", 50))

    def test_disabled_never_empty(self):
        self.assertFalse(_make_filter(enabled=False).should_empty_portfolio("crash", 10))


if __name__ == "__main__":
    unittest.main()
