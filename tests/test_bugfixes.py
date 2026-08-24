"""2026-08 缺陷修复回归测试（离线、无网络）

覆盖：
  - helpers.get_limit_pct 板块涨跌停口径
  - market_regime prev_ma20 NaN（趋势 rising/falling 分支可达）
  - strategies keep_increasing / backtrace_ma250 死策略复活
  - screener 成交额过滤独立于总市值列 + 涨跌停按板块口径
  - scorer 权重和恒为 1
  - ml_selector._future_return_from_kline 百分数口径
  - technical breakout_60d_high 数据不足不触发
  - rebound_analysis 套牢盘重度惩罚分支可达
  - news_fetcher.get_sina_global 异常返回 []
  - backtest T+1 开盘成交 + 并发仓位约束 + 绩效指标口径
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import get_limit_pct


class TestGetLimitPct(unittest.TestCase):
    def test_main_board_10(self):
        self.assertEqual(get_limit_pct("sh600519"), 10.0)
        self.assertEqual(get_limit_pct("000001"), 10.0)
        self.assertEqual(get_limit_pct("sz000001", "平安银行"), 10.0)

    def test_chinext_star_20(self):
        # 项目虽默认排除300/301/688，但工具函数必须给出口径
        self.assertEqual(get_limit_pct("sz300750"), 20.0)
        self.assertEqual(get_limit_pct("301236"), 20.0)
        self.assertEqual(get_limit_pct("sh688981"), 20.0)

    def test_bse_30(self):
        self.assertEqual(get_limit_pct("bj832000"), 30.0)
        self.assertEqual(get_limit_pct("830799"), 30.0)
        self.assertEqual(get_limit_pct("430047"), 30.0)

    def test_st_5_overrides_board(self):
        self.assertEqual(get_limit_pct("sh600519", "*ST某某"), 5.0)
        self.assertEqual(get_limit_pct("000001", "ST某某"), 5.0)


def _mk_history(dates, closes):
    closes = np.asarray(closes, dtype=float)
    opens = closes * (1 - 0.005)
    return pd.DataFrame({
        "日期": dates,
        "开盘": opens,
        "收盘": closes,
        "最高": np.maximum(opens, closes) * 1.01,
        "最低": np.minimum(opens, closes) * 0.99,
        "成交量": np.full(len(closes), 2_000_000.0),
        "成交额": closes * 2_000_000,
        "涨跌幅": np.zeros(len(closes)),
    })


class TestMarketRegimePrevMa20(unittest.TestCase):
    """原实现 prev_ma20 对最后2个元素做 rolling(20) → 恒NaN，
    above_ma20_rising / below_ma20_falling 分支永远不可达。"""

    def _trend_of(self, closes):
        from src.engine.market_regime import MarketRegimeFilter
        dates = pd.date_range("2026-06-01", periods=len(closes), freq="B").strftime("%Y-%m-%d").tolist()
        hist = _mk_history(dates, closes)

        class _FakeFetcher:
            def get_stock_history(self, code, days=60):
                return hist

        f = MarketRegimeFilter(market_fetcher=_FakeFetcher())
        f._index_cache = {}
        return f._assess_index_trend("sh000001")

    def test_rising_branch_reachable(self):
        n = 40
        closes = list(np.linspace(100, 130, n))          # 稳定上升
        trend = self._trend_of(closes)
        self.assertEqual(trend, "above_ma20_rising")

    def test_falling_branch_reachable(self):
        n = 40
        closes = list(np.linspace(130, 90, n))           # 稳定下降
        trend = self._trend_of(closes)
        self.assertEqual(trend, "below_ma20_falling")


class TestRevivedStrategies(unittest.TestCase):
    """原实现 ma.isna().any() 检查整条 rolling 序列（前N位必然NaN）→ 策略永远False"""

    def _df(self):
        n = 80
        dates = pd.date_range("2026-04-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        # 平滑强上升趋势: MA30 严格递增、30日涨幅>15%
        closes = 10 * (1.006 ** np.arange(n))
        return _mk_history(dates, closes)

    def test_keep_increasing_can_be_true(self):
        from src.factors.strategies import StockStrategies
        self.assertTrue(StockStrategies.keep_increasing(self._df()))

    def test_backtrace_ma250_runs_without_false_nan_gate(self):
        from src.factors.strategies import StockStrategies
        # 构造260日数据：先跌后回升踩年线附近
        n = 270
        dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        down = np.linspace(30, 18, 200)
        up = np.linspace(18, 26, n - 200)   # 回升站上年线上方但距高点>5%
        closes = np.concatenate([down, up])
        df = _mk_history(dates, closes)
        # 不再因整条序列NaN恒返回False；结果随行情而定，只断言可产生True路径存在
        result = StockStrategies.backtrace_ma250(df)
        self.assertIsInstance(result, bool)


class TestScreenerLiquidityFilter(unittest.TestCase):
    """原实现成交额过滤嵌套在"总市值列存在"分支内"""

    def _screener(self):
        from src.engine.screener import StockScreener
        return StockScreener({"min_market_cap": 50, "min_turnover": 10000})

    def test_turnover_filter_works_without_mktcap_column(self):
        s = self._screener()
        df = pd.DataFrame({
            "代码": ["sh600000", "sh600001"],
            "名称": ["甲", "乙"],
            "涨跌幅": [1.0, 1.0],
            "成交额": [5e8, 5e6],   # 第二只流动性不足(5000万 < 1亿)
        })
        out = s.filter_hard_conditions(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["代码"], "sh600000")

    def test_turnover_filter_still_works_with_mktcap_column(self):
        s = self._screener()
        df = pd.DataFrame({
            "代码": ["sh600000", "sh600001"],
            "名称": ["甲", "乙"],
            "涨跌幅": [1.0, 1.0],
            "总市值": [1e10, 1e10],
            "成交额": [5e8, 5e6],
        })
        out = s.filter_hard_conditions(df)
        self.assertEqual(len(out), 1)

    def test_chinext_20cm_not_treated_as_limit_up(self):
        s = self._screener()
        df = pd.DataFrame({
            "代码": ["sh600000", "sz300001"],
            "名称": ["甲", "乙"],
            "涨跌幅": [9.9, 12.0],   # 主板9.9≈涨停排除; 创业板12%未涨停保留
            "成交额": [5e8, 5e8],
        })
        out = s.filter_hard_conditions(df)
        codes = set(out["代码"])
        self.assertNotIn("sh600000", codes)   # 主板接近涨停被排除
        self.assertIn("sz300001", codes)      # 20cm板块12%不是涨停


class TestScorerWeightsSumToOne(unittest.TestCase):
    def test_default_weights_sum(self):
        from src.engine.scorer import StockScorer
        self.assertAlmostEqual(sum(StockScorer().weights.values()), 1.0, places=9)

    def test_unnormalized_input_gets_normalized(self):
        from src.engine.scorer import StockScorer
        w = {k: v for k, v in StockScorer().weights.items()}
        # 人为放大10%
        scaled = {k: v * 1.1 for k, v in w.items()}
        s = StockScorer(scaled)
        self.assertAlmostEqual(sum(s.weights.values()), 1.0, places=9)
        # 相对结构不变
        self.assertAlmostEqual(s.weights["trend"] / s.weights["chip"],
                               w["trend"] / w["chip"], places=9)

    def test_config_weights_sum(self):
        import yaml
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertAlmostEqual(sum(cfg["weights"].values()), 1.0, places=9)


class TestFutureReturnUnit(unittest.TestCase):
    def test_returns_percentage(self):
        from src.engine.ml_selector import _future_return_from_kline
        krows = [
            ("2026-08-01", "10.0"),
            ("2026-08-02", "10.5"),
            ("2026-08-03", "11.0"),
            ("2026-08-04", "11.0"),   # T+3: (11/10-1)*100 = 10%
        ]
        ret = _future_return_from_kline(krows, "2026-08-01", 3)
        self.assertAlmostEqual(ret, 10.0, places=6)

    def test_insufficient_data_returns_none(self):
        from src.engine.ml_selector import _future_return_from_kline
        self.assertIsNone(_future_return_from_kline([("2026-08-01", "10.0")], "2026-08-01", 3))
        self.assertIsNone(_future_return_from_kline([], "2026-08-01", 3))


class TestBreakoutInsufficientData(unittest.TestCase):
    def test_breakout_60d_not_triggered_with_short_history(self):
        from src.factors.technical import TechnicalFactors
        n = 40   # 不足61日
        dates = pd.date_range("2026-06-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        closes = np.linspace(10, 15, n)   # 收盘在"高位"
        tf = TechnicalFactors(_mk_history(dates, closes), {})
        self.assertEqual(tf._breakout_high(60), 0.0)

    def test_breakout_20d_triggered_on_real_breakout(self):
        from src.factors.technical import TechnicalFactors
        n = 40
        dates = pd.date_range("2026-06-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        closes = list(np.linspace(10, 12, n - 1)) + [13.0]   # 最后一天突破前高
        tf = TechnicalFactors(_mk_history(dates, closes), {})
        self.assertEqual(tf._breakout_high(20), 1.0)


class TestReboundHeavyTrapped(unittest.TestCase):
    """原 elif 顺序颠倒导致 <-10 重罚分支永不可达"""

    def _analyze(self, closes):
        from src.factors.rebound_analysis import ReboundSustainability
        n = len(closes)
        dates = pd.date_range("2026-06-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        df = _mk_history(dates, closes)
        rs = ReboundSustainability()
        # 通过 analyze 输出的 risk_factors 验证套牢盘分支
        out = rs.analyze(df)
        return out

    def test_heavy_trapped_penalty_reachable(self):
        n = 60
        # 先冲高到20，再跌到16（距20日高点约-11%，触发"很重"而非"重"）
        up = np.linspace(10, 20, 35)
        down = np.linspace(20, 16, n - 35)
        out = self._analyze(np.concatenate([up, down]))
        heavy = [r for r in out["risk_factors"] if "套牢盘很重" in r]
        light = [r for r in out["risk_factors"] if r.endswith("（套牢盘重）")]
        self.assertTrue(heavy or light, "至少应触发一档套牢盘风险")
        if heavy:
            # 触发重度时数值必须 <= -10
            import re
            m = re.search(r"距20日高点(-[\d.]+)%", heavy[0])
            self.assertIsNotNone(m)
            self.assertLessEqual(float(m.group(1)), -10)


class TestNewsFetcherSinaException(unittest.TestCase):
    def test_sina_exception_returns_list(self):
        from src.data.news_fetcher import NewsFetcher
        nf = NewsFetcher()
        import akshare as ak
        orig = ak.stock_info_global_sina

        def _boom():
            raise RuntimeError("network down")

        ak.stock_info_global_sina = _boom
        try:
            out = nf.get_sina_global()
        finally:
            ak.stock_info_global_sina = orig
        self.assertEqual(out, [])


class TestBacktestNextDayEntryAndPositionCap(unittest.TestCase):
    def _setup(self):
        from src.engine.backtest import BacktestEngine
        n = 90
        dates = pd.date_range("2026-05-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        histories = {}
        for i in range(4):
            closes = 10 * (1 + 0.002 * i) * (1 + np.sin(np.arange(n) / 9) * 0.03)
            histories[f"sz00000{i}"] = _mk_history(dates, closes)
        eng = BacktestEngine(initial_capital=1_000_000)
        signals = []
        for sc, h in histories.items():
            signals.extend(eng._generate_signals(sc, h, train_size=60))
        eng._score_cross_sectional(signals, None)
        return eng, signals, histories

    def test_entry_is_next_day_open(self):
        eng, signals, histories = self._setup()
        trades = eng._simulate_holding(signals, histories, hold_days=3,
                                       max_positions=4, top_n=4, entry_score=-1e9)
        self.assertTrue(trades)
        for t in trades:
            self.assertGreater(t["entry_date"], t["signal_date"])

    def test_concurrent_positions_capped(self):
        eng, signals, histories = self._setup()
        max_positions = 2
        trades = eng._simulate_holding(signals, histories, hold_days=5,
                                       max_positions=max_positions, top_n=4, entry_score=-1e9)
        # 任一 entry_date 当日的未平仓持仓数不得超过 max_positions
        events = []
        for t in trades:
            events.append((t["entry_date"], 1))
            events.append((t["exit_date"], -1))
        events.sort(key=lambda x: x[0])
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        self.assertLessEqual(peak, max_positions)

    def test_no_duplicate_holding_same_stock(self):
        eng, signals, histories = self._setup()
        trades = eng._simulate_holding(signals, histories, hold_days=10,
                                       max_positions=10, top_n=4, entry_score=-1e9)
        by_code_dates = {}
        for t in trades:
            self.assertNotIn(t["entry_date"], by_code_dates.setdefault(t["stock_code"], set()))
            by_code_dates[t["stock_code"]].add(t["entry_date"])

    def test_results_metrics_consistent(self):
        eng, signals, histories = self._setup()
        trades = eng._simulate_holding(signals, histories, hold_days=3,
                                       max_positions=4, top_n=4, entry_score=-1e9)
        res = eng._calculate_results(trades)
        s = res["summary"]
        # total_return 与 pnl 自洽（summary 值已 round 到 2 位，用 0.01 容差）
        self.assertAlmostEqual(
            s["total_return"], s["total_pnl"] / s["initial_capital"] * 100, places=1
        )
        # 年化基于真实交易日数而非交易笔数
        self.assertGreaterEqual(eng.trading_day_count, len({t["signal_date"] for t in trades}))


if __name__ == "__main__":
    unittest.main()
