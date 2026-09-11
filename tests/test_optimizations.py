"""2026-08 第二轮优化回归测试（离线、无网络）

覆盖：
  - scorer: quant_behavior 不再反向加分 / deep_research 透传 / penalty 深度惩罚
  - factor_attribution: classify_signal 绝对偏离兜底(优质股盲区)
  - news_factor._time_weight: asof 基准时间(软前视修复)
  - backtest: 交易成本计入 pnl
  - fundamental/institutional: 数据缺失走中性而非低分
"""
import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestQuantBehaviorScoring(unittest.TestCase):
    def test_high_risk_adds_score_per_market_ic(self):
        """全市场IC终审(2026-08): quant_behavior显著正(+0.038,t=2.2)
        → 检测到量化异动应加分(游资活跃=关注度),仅闪崩类破坏信号减分"""
        from src.engine.scorer import StockScorer
        s = StockScorer()
        risky = {"quant_risk": "high",
                 "quant_signals": ["大单对倒", "秒板"]}
        self.assertGreater(s._score_quant_behavior(risky), 50)

    def test_flash_crash_still_penalized(self):
        from src.engine.scorer import StockScorer
        s = StockScorer()
        crash = {"quant_risk": "low", "quant_signals": ["盘中闪崩"]}
        self.assertLess(s._score_quant_behavior(crash), 50)

    def test_clean_stock_neutral(self):
        from src.engine.scorer import StockScorer
        s = StockScorer()
        clean = {"quant_risk": "low", "quant_signals": []}
        self.assertEqual(s._score_quant_behavior(clean), 50)


class TestDeepResearchPassthrough(unittest.TestCase):
    def test_score_passthrough_not_inverted(self):
        from src.engine.scorer import StockScorer
        s = StockScorer()
        # raw=80(因子内部语义:高分好) → 子评分应为 80 而非 20
        self.assertEqual(s._score_deep_research({"deep_research_score": 80}), 80)
        self.assertEqual(s._score_deep_research({"deep_research_score": 30}), 30)


class TestPenaltyExpectation(unittest.TestCase):
    def test_deep_penalty_below_old_floor(self):
        from src.engine.scorer import StockScorer
        s = StockScorer()
        f = {
            "expectation_overvalued": True,
            "expectation_score": 20,
            "event_priced": True, "event_pricing_degree": 100,
            "profit_taking_detected": True, "profit_taking_intensity": 100,
        }
        p = s.penalty_expectation(f)
        self.assertEqual(p, -75)   # 全部命中 → 实质性一票否决

    def test_clean_zero_penalty(self):
        from src.engine.scorer import StockScorer
        self.assertEqual(StockScorer().penalty_expectation({}), 0)


class TestClassifySignalAbsoluteFallback(unittest.TestCase):
    def test_uniformly_high_stock_not_silent(self):
        """全因子均80的股票(median=80)：相对偏离不足但绝对强度够 → bullish"""
        from src.engine.factor_attribution import classify_signal
        self.assertEqual(classify_signal(85, factor_median=80), "bullish")

    def test_moderate_deviation_still_neutral(self):
        from src.engine.factor_attribution import classify_signal
        # score=62 > 60 但偏离中位数仅2且绝对强度不足 → neutral
        self.assertEqual(classify_signal(62, factor_median=60), "neutral")

    def test_bearish_paths(self):
        from src.engine.factor_attribution import classify_signal
        self.assertEqual(classify_signal(35, factor_median=50), "bearish")
        self.assertEqual(classify_signal(20, factor_median=25), "bearish")


class TestTimeWeightAsof(unittest.TestCase):
    def test_asof_prevents_lookahead_decay(self):
        from src.factors.news_factor import NewsFactor
        news = {"time": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")}
        # 真实now: 24小时前的新闻权重极低
        w_now = NewsFactor._time_weight(news)
        # asof 设为发布后1小时: 权重应接近1
        w_asof = NewsFactor._time_weight(news, asof=datetime.now() - timedelta(hours=23))
        self.assertLess(w_now, 0.3)
        self.assertGreater(w_asof, 0.8)


def _mk_history(dates, closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "日期": dates, "开盘": closes * 0.995, "收盘": closes,
        "最高": np.maximum(closes * 0.995, closes) * 1.01,
        "最低": np.minimum(closes * 0.995, closes) * 0.99,
        "成交量": np.full(len(closes), 2e6), "成交额": closes * 2e6,
        "涨跌幅": np.zeros(len(closes)),
    })


class TestBacktestCosts(unittest.TestCase):
    def _make_trades(self, **engine_kwargs):
        from src.engine.backtest import BacktestEngine
        n = 90
        dates = pd.date_range("2026-05-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        histories = {}
        for i in range(3):
            closes = list(np.linspace(10 + i, 14 + i, n))
            histories[f"sz00000{i}"] = _mk_history(dates, closes)
        eng = BacktestEngine(**engine_kwargs)
        signals = []
        for sc, h in histories.items():
            signals.extend(eng._generate_signals(sc, h, train_size=60))
        eng._score_cross_sectional(signals, None)
        trades = eng._simulate_holding(signals, histories, hold_days=3,
                                       max_positions=3, top_n=3, entry_score=-1e9)
        return eng, trades

    def test_costs_reduce_pnl(self):
        # guard_config={} 关闭实盘 guard 链(合成数据成交额低于流动性阈值会被全剔)
        _, trades_free = self._make_trades(
            commission_rate=0, stamp_tax=0, slippage=0, guard_config={})
        _, trades_cost = self._make_trades(guard_config={})
        pnl_free = sum(t["pnl"] for t in trades_free)
        pnl_cost = sum(t["pnl"] for t in trades_cost)
        self.assertLess(pnl_cost, pnl_free)
        # 成本占毛收益比例应 <50%（hold_days=3 的短线交易，
        # 单笔毛收益仅1-2%时 0.3% 左右的摩擦成本占比自然偏高——真实情况）
        if abs(pnl_free) > 1:
            drag = (pnl_free - pnl_cost) / abs(pnl_free)
            self.assertLess(drag, 0.5)


class TestMissingDataNeutral(unittest.TestCase):
    def test_fundamental_missing_mv_neutral(self):
        from src.factors.fundamental import FundamentalFactors
        ff = FundamentalFactors()
        out = ff.calculate_all({"总市值": 0})
        self.assertEqual(out["market_cap_score"], 50)

    def test_fundamental_real_loss_vs_missing(self):
        from src.factors.fundamental import FundamentalFactors
        ff = FundamentalFactors()
        loss = ff.calculate_all({"市盈率-动态": -10})
        missing = ff.calculate_all({})
        self.assertEqual(loss["pe_score"], 20)
        self.assertEqual(missing["pe_score"], 50)

    def test_institutional_missing_fields_neutral(self):
        from src.factors.institutional import InstitutionalFactors
        inst = InstitutionalFactors()
        # 行情源完全没有机构字段 → 各分项中性，综合分=50
        out = inst.calculate_all("sh600000", market_data={})
        self.assertAlmostEqual(out["institutional_score"], 50, places=6)

    def test_institutional_explicit_zero_still_scores_low(self):
        from src.factors.institutional import InstitutionalFactors
        inst = InstitutionalFactors()
        out = inst.calculate_all("sh600000", market_data={"机构持股家数": 0, "机构持股比例": 0})
        self.assertEqual(out["inst_count_score"], 30)


if __name__ == "__main__":
    unittest.main()
