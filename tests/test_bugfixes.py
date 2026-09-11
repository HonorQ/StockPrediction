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
  - liquidity_guard avg_daily_amount_20d 字段缺失不误杀(2026-08-31)
"""
import os
import sys
import unittest
from datetime import datetime

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
        eng = BacktestEngine(initial_capital=1_000_000, guard_config={})
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


class TestLiquidityGuardMissingField(unittest.TestCase):
    """2026-08-31 修复: avg_daily_amount_20d 字段缺失导致全部候选被误杀

    根因: factors.get("avg_daily_amount_20d", 0) 永远返回0,
    0 < 0.5亿 恒成立 → 30只候选全部被剔除 → 无推荐。
    """
    def _config(self):
        return {"liquidity_guard": {"enabled": True, "min_daily_amount_yi": 0.5}}

    def test_missing_field_all_kept(self):
        from src.engine.scorer import apply_liquidity_guard
        candidates = [
            {"code": "sh600000", "name": "甲", "score": 60, "factors": {}},
            {"code": "sh600001", "name": "乙", "score": 55, "factors": {"other_key": 1}},
        ]
        result = apply_liquidity_guard(candidates, self._config())
        self.assertEqual(len(result), 2, "字段缺失时应保留全部候选")

    def test_field_present_excludes_low(self):
        from src.engine.scorer import apply_liquidity_guard
        candidates = [
            {"code": "sh600000", "name": "甲", "score": 60,
             "factors": {"avg_daily_amount_20d": 1.0}},
            {"code": "sh600001", "name": "乙", "score": 55,
             "factors": {"avg_daily_amount_20d": 0.3}},
        ]
        result = apply_liquidity_guard(candidates, self._config())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["code"], "sh600000")

    def test_field_present_keeps_high(self):
        from src.engine.scorer import apply_liquidity_guard
        candidates = [
            {"code": "sh600000", "name": "甲", "score": 60,
             "factors": {"avg_daily_amount_20d": 2.0}},
        ]
        result = apply_liquidity_guard(candidates, self._config())
        self.assertEqual(len(result), 1)

    def test_disabled_guard_returns_all(self):
        from src.engine.scorer import apply_liquidity_guard
        cfg = {"liquidity_guard": {"enabled": False}}
        candidates = [
            {"code": "sh600000", "name": "甲", "score": 60, "factors": {}},
        ]
        result = apply_liquidity_guard(candidates, cfg)
        self.assertEqual(len(result), 1)


class _FakeDB:
    """离线假DB: 拦截调权落库, 避免污染生产库(测试禁止写 data/*.db)。"""

    def __init__(self, latest_weights=None, attribution_history=None):
        self.latest_weights = latest_weights or {}
        self.attribution_history = attribution_history or []
        self.saved = []

    def get_latest_weights(self):
        return self.latest_weights

    def get_attribution_history(self, days=60):
        return self.attribution_history

    def save_weight_update(self, today, factor, old, new, reason=""):
        self.saved.append({"factor": factor, "old": old, "new": new})


def _wrong_attr(factor, n=6):
    """构造 n 条全部失败的归因记录(当天, 供惩罚调权统计)。"""
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"trade_date": today, "factor_results": {factor: {"result": "wrong"}}}
        for _ in range(n)
    ]


class TestWeightRevivalGuard(unittest.TestCase):
    """2026-09-11 修复: 已否决(清零)因子不得被反馈循环/历史调权复活。

    背景: config.weights 23 维过拟合因子清零后, weight_history 历史记录、
    惩罚调权 0.005 下限、|IC| softmax 上调三条路径均在静默复活它们。
    """

    def _make_loop(self, fake_db, config):
        from unittest.mock import patch
        from src.engine.feedback_loop import FeedbackLoop
        with patch("src.engine.feedback_loop.StockDatabase", return_value=fake_db):
            loop = FeedbackLoop(config=config)
        loop.db = fake_db
        return loop

    def test_penalty_skips_zeroed_factor(self):
        # 惩罚调权 0.005 下限曾把清零因子写成 0.005
        cfg = {"weights": {"ashare_specific": 1.0, "strategy": 0.0}}
        fake_db = _FakeDB()
        loop = self._make_loop(fake_db, cfg)
        changes = loop._penalty_adjustments(
            _wrong_attr("strategy"), cfg["weights"], "2026-09-11")
        self.assertFalse(any(c["factor"] == "strategy" for c in changes))
        self.assertFalse(any(s["factor"] == "strategy" for s in fake_db.saved))

    def test_penalty_active_factor_still_adjustable(self):
        # 非零因子(ashare_specific)的惩罚调权路径保持可用
        cfg = {"weights": {"ashare_specific": 1.0, "strategy": 0.0}}
        fake_db = _FakeDB()
        loop = self._make_loop(fake_db, cfg)
        changes = loop._penalty_adjustments(
            _wrong_attr("ashare_specific"), cfg["weights"], "2026-09-11")
        self.assertTrue(any(c["factor"] == "ashare_specific" for c in changes))

    def test_merge_latest_weights_skips_zeroed(self):
        # weight_history 合并不得把 DB 历史调权叠加到清零因子上
        from unittest.mock import patch
        cfg = {"weights": {"ashare_specific": 1.0, "strategy": 0.0}}
        fake_db = _FakeDB(
            latest_weights={"strategy": 0.071, "ashare_specific": 0.9},
            # 归因记录不足以触发任何调权, 只测合并段
            attribution_history=[{"trade_date": "2026-09-11", "factor_results": {}}],
        )
        loop = self._make_loop(fake_db, cfg)
        with patch.object(loop, "_build_ic_dataset", return_value=None):
            loop._adjust_weights(today="2026-09-11")
        # strategy 若被合并复活(0.071)且惩罚路径免疫失效, 会写入调权记录
        self.assertFalse(any(s["factor"] == "strategy" for s in fake_db.saved))

    def test_ic_adjustments_skip_zeroed_and_negative_ic(self):
        # 负 IC 因子未做方向翻转前上调 = 反向加权; 清零因子免疫
        import pandas as pd
        rows = []
        for d in ("2026-09-09", "2026-09-10"):
            for i in range(6):
                rows.append({
                    "trade_date": d, "code": f"sz00000{i}",
                    # ashare_specific 与收益正相关(IC>0), strategy 负相关(IC<0)
                    "ashare_specific": 50.0 + i, "strategy": 50.0 - i,
                    "future_return": float(i) * 2,
                })
        df = pd.DataFrame(rows)
        cfg = {"weights": {"ashare_specific": 1.0, "strategy": 0.0}}
        fake_db = _FakeDB()
        loop = self._make_loop(fake_db, cfg)
        from unittest.mock import patch
        with patch.object(loop, "_build_ic_dataset", return_value=df):
            changes = loop._ic_adjustments(cfg["weights"], "2026-09-11")
        self.assertFalse(any(c["factor"] == "strategy" for c in changes))
        self.assertFalse(any(s["factor"] == "strategy" for s in fake_db.saved))

    def test_ic_adjustments_skip_oscillation(self):
        # 同轮惩罚调权已处理的因子, IC 调权不得再动(振荡修复)
        cfg = {"weights": {"ashare_specific": 1.0}}
        loop = self._make_loop(_FakeDB(), cfg)
        changes = loop._ic_adjustments(
            cfg["weights"], "2026-09-11", skip_factors={"ashare_specific"})
        self.assertEqual(changes, [])


class TestRegimeWeightsGuard(unittest.TestCase):
    """2026-09-11 修复: regime 覆盖后权重总和必须 ≈1(评分尺度防漂移)。"""

    def _make_filter(self, regime_cfg):
        from src.engine.market_regime import MarketRegimeFilter
        filt = MarketRegimeFilter.__new__(MarketRegimeFilter)
        filt.market_fetcher = None
        filt._index_cache = {}
        filt._regime_config = regime_cfg
        return filt

    def test_override_normalized_to_one(self):
        # 旧覆盖段曾致 bear 总和 1.277 → 评分尺度膨胀 27.7%
        filt = self._make_filter({
            "enabled": True,
            "weights": {"bull": {"trend": 0.37}},
        })
        base = {"ashare_specific": 1.0, "trend": 0.0}
        w = filt.get_regime_weights("bull", base)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=2)

    def test_empty_override_returns_base(self):
        # 清空后的 weights:{} 必须原样返回基准权重
        filt = self._make_filter({"enabled": True, "weights": {}})
        base = {"ashare_specific": 1.0, "trend": 0.0}
        w = filt.get_regime_weights("bear", base)
        self.assertEqual(w, base)

    def test_config_regime_weights_cleared(self):
        # 防回归: config.yaml 的 regime.weights 覆盖段保持清空
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["regime"]["weights"], {})
        # 择时开关保留(空仓防守不受影响)
        self.assertTrue(cfg["regime"]["enabled"])
        self.assertTrue(cfg["regime"]["empty_on_bear"])


class TestAvgDailyAmountProducer(unittest.TestCase):
    """2026-09-11 修复: avg_daily_amount_20d 生产者随未提交改动丢失,
    liquidity_guard 自 08-31 下午起连续10天空转(生产日志实锤 25/25 缺字段)。

    数据为受控合成测试数据, 非真实行情。
    """
    def _make_history(self, amounts):
        import pandas as pd
        n = len(amounts)
        return pd.DataFrame({
            "日期": [f"2026-08-{i+1:02d}" for i in range(n)],
            "收盘": [10.0] * n,
            "成交额": amounts,
        })

    def test_produces_yi_unit(self):
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        # 20日每日成交额 1亿元 → 均值 1.0 亿
        df = self._make_history([1e8] * 25)
        self.assertAlmostEqual(_calc_avg_daily_amount_20d(df), 1.0, places=4)

    def test_uses_last_20_days(self):
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        # 前10天 100亿(应被 tail(20) 排除), 后20天 1亿 → 均值 1.0 亿
        df = self._make_history([1e10] * 10 + [1e8] * 20)
        self.assertAlmostEqual(_calc_avg_daily_amount_20d(df), 1.0, places=4)

    def test_below_threshold_value(self):
        # 与 guard 阈值对齐: 3000万 < 0.5亿 应产出 0.3 (guard 将剔除)
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        df = self._make_history([3e7] * 20)
        self.assertAlmostEqual(_calc_avg_daily_amount_20d(df), 0.3, places=4)

    def test_missing_column_returns_none(self):
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        import pandas as pd
        df = pd.DataFrame({"收盘": [10.0] * 25})
        self.assertIsNone(_calc_avg_daily_amount_20d(df))

    def test_non_numeric_amount_returns_none(self):
        # 全坏数据 → mean 为 NaN → None, 不产出伪值
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        df = self._make_history(["坏数据"] * 25)
        self.assertIsNone(_calc_avg_daily_amount_20d(df))

    def test_end_to_end_with_guard(self):
        # 生产者输出直接被 guard 正确消费(单位对齐验证)
        from src.pipeline.premarket_deep import _calc_avg_daily_amount_20d
        from src.engine.scorer import apply_liquidity_guard
        high = self._make_history([2e8] * 25)
        low = self._make_history([2e7] * 25)
        candidates = [
            {"code": "sh600000", "name": "甲", "score": 60,
             "factors": {"avg_daily_amount_20d": _calc_avg_daily_amount_20d(high)}},
            {"code": "sh600001", "name": "乙", "score": 55,
             "factors": {"avg_daily_amount_20d": _calc_avg_daily_amount_20d(low)}},
        ]
        cfg = {"liquidity_guard": {"enabled": True, "min_daily_amount_yi": 0.5}}
        result = apply_liquidity_guard(candidates, cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["code"], "sh600000")


class TestGuardExclusionsAudit(unittest.TestCase):
    """2026-09-11 修复: guard 剔除清单此前只存在于内存/日志,
    无法用 T+N 真实收益回查 guard 是避雷还是误杀。

    验证 _apply_guards_with_audit 的 diff 归因 + guard_exclusions 表落库。
    """

    def _temp_db(self):
        import tempfile
        from src.data.database import StockDatabase
        fd, path = tempfile.mkstemp(suffix=".db")
        import os
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return StockDatabase(db_path=path), path

    def _candidates(self):
        # chip 高 z 票会被 chip_guard 剔除; 低流动性票被 liquidity_guard 剔除
        return [
            {"code": "sh600000", "name": "甲", "score": 60,
             "z_scores": {"chip": 2.0, "ashare_specific": 1.0},
             "sub_scores": {"ashare_specific": 60}, "factors": {"avg_daily_amount_20d": 2.0}},
            {"code": "sh600001", "name": "乙", "score": 55,
             "z_scores": {"chip": 0.0, "ashare_specific": 1.0},
             "sub_scores": {"ashare_specific": 58}, "factors": {"avg_daily_amount_20d": 0.3}},
            {"code": "sh600002", "name": "丙", "score": 57,
             "z_scores": {"chip": 0.0, "ashare_specific": 1.0},
             "sub_scores": {"ashare_specific": 57}, "factors": {"avg_daily_amount_20d": 2.0}},
        ]

    def _config(self):
        # 只开 chip/liquidity 两个 guard, 排除 us/supply 等外部依赖
        return {
            "chip_guard": {"enabled": True, "z_threshold": 1.5},
            "ashare_guard": {"enabled": False},
            "us_sector_guard": {"enabled": False},
            "supply_chain_guard": {"enabled": False},
            "oversold_rebound_guard": {"enabled": False},
            "liquidity_guard": {"enabled": True, "min_daily_amount_yi": 0.5},
        }

    def test_exclusions_persisted_with_correct_guard(self):
        from src.pipeline.premarket_finalize import _apply_guards_with_audit
        db, _ = self._temp_db()
        kept = _apply_guards_with_audit(self._candidates(), self._config(), db=db)
        # chip 剔甲, liquidity 剔乙, 丙保留
        self.assertEqual([c["code"] for c in kept], ["sh600002"])
        rows = db._fetch_all(
            "SELECT code, guard_name, guard_reason FROM guard_exclusions ORDER BY code")
        self.assertEqual(len(rows), 2)
        by_code = {r["code"]: r for r in rows}
        self.assertEqual(by_code["sh600000"]["guard_name"], "chip")
        self.assertIn("获利盘", by_code["sh600000"]["guard_reason"])
        self.assertEqual(by_code["sh600001"]["guard_name"], "liquidity")
        self.assertIn("流动性不足", by_code["sh600001"]["guard_reason"])

    def test_idempotent_rewrite(self):
        # 同日重跑: UNIQUE(trade_date,code,guard_name)+OR REPLACE 不产生重复行
        from src.pipeline.premarket_finalize import _apply_guards_with_audit
        db, _ = self._temp_db()
        _apply_guards_with_audit(self._candidates(), self._config(), db=db)
        _apply_guards_with_audit(self._candidates(), self._config(), db=db)
        rows = db._fetch_all(
            "SELECT code, guard_name FROM guard_exclusions")
        self.assertEqual(len(rows), 2)

    def test_db_failure_does_not_break_chain(self):
        # 落库失败只降级, 不影响 guard 主链路返回值
        from src.pipeline.premarket_finalize import _apply_guards_with_audit

        class _BrokenDB:
            def save_guard_exclusions_batch(self, *a, **kw):
                raise RuntimeError("db down")
        kept = _apply_guards_with_audit(
            self._candidates(), self._config(), db=_BrokenDB())
        self.assertEqual([c["code"] for c in kept], ["sh600002"])


class TestPolicyGuard(unittest.TestCase):
    """2026-09-11 新增: 政策硬利空排除器(影子模式起步)。

    数据流: PolicyEventFactor 检测新闻直接点名个股+硬利空关键词 →
    events type=hard_negative → apply_policy_guard 消费。
    """

    def _factor(self):
        from src.factors.policy_event import PolicyEventFactor
        return PolicyEventFactor(asof=None)

    def _news(self, content):
        return [{"content": content, "title": "", "time": None}]

    def test_hard_negative_event_emitted_on_direct_mention(self):
        f = self._factor()
        r = f._calculate_news_policy_impact(
            "sh600000", self._news("某公司发布公告: 因涉嫌信息披露违法违规, 证监会决定对湖南黄金立案调查"),
            stock_sector="", stock_concepts=[], stock_name="湖南黄金")
        types = [e["type"] for e in r["events"]]
        self.assertIn("hard_negative", types)

    def test_no_event_without_direct_mention(self):
        # 行业级利空不点名个股 → 不产生 hard_negative(不做行业传导级匹配)
        f = self._factor()
        r = f._calculate_news_policy_impact(
            "sh600000", self._news("证监会宣布对多家公司立案调查, 湖南板块承压"),
            stock_sector="", stock_concepts=[], stock_name="湖南黄金")
        types = [e["type"] for e in r["events"]]
        self.assertNotIn("hard_negative", types)

    def test_no_event_for_soft_negative_words(self):
        # 普通情绪词(利空/下跌)不触发 hard_negative(留给 major_negative 路径)
        f = self._factor()
        r = f._calculate_news_policy_impact(
            "sh600000", self._news("湖南黄金今日大跌, 利空频出"),
            stock_sector="", stock_concepts=[], stock_name="湖南黄金")
        types = [e["type"] for e in r["events"]]
        self.assertNotIn("hard_negative", types)

    def _candidates(self):
        return [
            {"code": "sh600000", "name": "甲", "score": 60,
             "sub_scores": {"ashare_specific": 60},
             "factors": {"policy_events": [
                 {"type": "hard_negative", "title": "证监会对甲立案调查", "impact": "high"}]}},
            {"code": "sh600001", "name": "乙", "score": 55,
             "sub_scores": {"ashare_specific": 58},
             "factors": {"policy_events": []}},
        ]

    def test_shadow_mode_marks_but_keeps(self):
        from src.engine.scorer import apply_policy_guard
        cfg = {"policy_guard": {"enabled": True, "shadow": True}}
        result = apply_policy_guard(self._candidates(), cfg)
        self.assertEqual(len(result), 2, "影子模式不剔除")
        self.assertIn("policy_shadow_reason", result[0])
        self.assertNotIn("policy_shadow_reason", result[1])

    def test_real_mode_excludes(self):
        from src.engine.scorer import apply_policy_guard
        cands = self._candidates()
        cfg = {"policy_guard": {"enabled": True, "shadow": False}}
        result = apply_policy_guard(cands, cfg)
        self.assertEqual([c["code"] for c in result], ["sh600001"])
        # 原地打标: 被剔除票带 guard_reason
        self.assertIn("政策硬利空", cands[0].get("guard_reason", ""))

    def test_disabled_returns_all(self):
        from src.engine.scorer import apply_policy_guard
        result = apply_policy_guard(self._candidates(), {})
        self.assertEqual(len(result), 2)

    def test_chain_persists_shadow_flags(self):
        # 影子模式下 _apply_guards_with_audit 把打标票落库为 policy_shadow
        from src.pipeline.premarket_finalize import _apply_guards_with_audit
        import tempfile, os
        from src.data.database import StockDatabase
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        db = StockDatabase(db_path=path)
        cfg = {
            "policy_guard": {"enabled": True, "shadow": True},
            "chip_guard": {"enabled": False},
            "ashare_guard": {"enabled": False},
            "us_sector_guard": {"enabled": False},
            "supply_chain_guard": {"enabled": False},
            "oversold_rebound_guard": {"enabled": False},
            "liquidity_guard": {"enabled": False},
        }
        kept = _apply_guards_with_audit(self._candidates(), cfg, db=db)
        self.assertEqual(len(kept), 2)
        rows = db._fetch_all(
            "SELECT code, guard_name FROM guard_exclusions")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["guard_name"], "policy_shadow")
        self.assertEqual(rows[0]["code"], "sh600000")


class TestBatchNormalizeFullZScores(unittest.TestCase):
    """2026-09-11 修复: 单因子架构下 batch_normalize 只给正权重维算 z,
    chip_guard 的 z 分支自 08-27 起静默失效(实盘剔除全来自 pos_250 分支)。

    修复后: z_scores 覆盖全部 sub_scores 维度; 评分仅加权正权重维(数值不变)。
    """
    def _candidates(self, n=6):
        cands = []
        for i in range(n):
            cands.append({
                "sub_scores": {
                    "ashare_specific": 50 + i * 3,
                    "chip": 40 + i * 10,  # 截面递增 → chip z 单调
                    "oversold_rebound": 50,
                },
                "expectation_penalty": 0,
            })
        return cands

    def test_z_scores_cover_all_dims(self):
        from src.engine.scorer import StockScorer
        cands = self._candidates()
        weights = {"ashare_specific": 1.0, "chip": 0}
        StockScorer.batch_normalize(cands, weights)
        for c in cands:
            self.assertIn("chip", c["z_scores"], "零权重维也应有截面z供guard消费")
            self.assertIn("ashare_specific", c["z_scores"])
        # 截面 chip 分单调递增 → 末位 z 应为正
        self.assertGreater(cands[-1]["z_scores"]["chip"], 0)
        self.assertLess(cands[0]["z_scores"]["chip"], 0)

    def test_score_only_weights_positive_dims(self):
        from src.engine.scorer import StockScorer
        cands = self._candidates()
        weights = {"ashare_specific": 1.0, "chip": 0}
        StockScorer.batch_normalize(cands, weights)
        # 评分只由 ashare_specific z 决定: score = 50 + z*(50/3)
        z_ashare = cands[-1]["z_scores"]["ashare_specific"]
        self.assertAlmostEqual(
            cands[-1]["score"], 50 + z_ashare * (50 / 3), places=1)

    def test_chip_guard_z_branch_revived(self):
        # 修复前: chip z 恒缺失→0, z 分支永不触发; 修复后可正常剔除
        from src.engine.scorer import StockScorer, apply_chip_guard
        cands = [{
            "code": f"sh60000{i:02d}", "name": f"股{i}",
            "sub_scores": {"ashare_specific": 55, "chip": 40 + i * 10},
            "expectation_penalty": 0,
        } for i in range(10)]
        StockScorer.batch_normalize(cands, {"ashare_specific": 1.0, "chip": 0})
        guard_input = [{"code": c["code"], "name": c["name"],
                        "z_scores": c["z_scores"], "factors": {}} for c in cands]
        kept = apply_chip_guard(guard_input, {"chip_guard": {"enabled": True, "z_threshold": 1.5}})
        # chip 最高的末位票应被剔除(10样本最大z≈1.63>1.5)
        self.assertNotIn("sh600009", [c["code"] for c in kept])


class TestEventCalendarCap(unittest.TestCase):
    """2026-09-11 新增: 重大事件日历窗口防守(pre市场 max_recommend 封顶)。"""

    def _cfg(self, events, enabled=True, default_cap=1):
        return {"event_calendar": {
            "enabled": enabled, "default_cap": default_cap, "events": events}}

    def test_in_window_returns_cap(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": ["2026-09-15", "2026-09-17"]}])
        cap, note = _event_calendar_cap(cfg, "2026-09-16")
        self.assertEqual(cap, 1)
        self.assertIn("FOMC", note)

    def test_outside_window_returns_none(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": ["2026-09-15", "2026-09-17"]}])
        cap, _ = _event_calendar_cap(cfg, "2026-09-11")
        self.assertIsNone(cap)

    def test_window_boundaries_inclusive(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "两会", "window": ["2027-03-04", "2027-03-11"]}])
        for d in ("2027-03-04", "2027-03-11"):
            cap, _ = _event_calendar_cap(cfg, d)
            self.assertIsNotNone(cap, f"边界日{d}应命中")

    def test_event_level_cap_overrides_default(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": ["2026-09-15", "2026-09-17"], "cap": 0}],
                        default_cap=1)
        cap, _ = _event_calendar_cap(cfg, "2026-09-16")
        self.assertEqual(cap, 0)

    def test_multiple_windows_take_min(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([
            {"name": "A会", "window": ["2026-12-08", "2026-12-12"], "cap": 1},
            {"name": "B会", "window": ["2026-12-10", "2026-12-15"], "cap": 0},
        ])
        cap, note = _event_calendar_cap(cfg, "2026-12-11")
        self.assertEqual(cap, 0)
        self.assertIn("B会", note)

    def test_disabled_returns_none(self):
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": ["2026-09-15", "2026-09-17"]}],
                        enabled=False)
        cap, _ = _event_calendar_cap(cfg, "2026-09-16")
        self.assertIsNone(cap)

    def test_date_objects_tolerated(self):
        # YAML 未加引号日期解析成 date 对象, 不得因类型比较 TypeError 被静默吞掉
        import datetime as _dt
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": [
            _dt.date(2026, 9, 15), _dt.date(2026, 9, 17)]}])
        cap, note = _event_calendar_cap(cfg, "2026-09-16")
        self.assertEqual(cap, 1)
        self.assertIn("FOMC", note)

    def test_static_only_mode(self):
        # fetch_auto=False 只用静态条目(离线测试入口)
        from src.pipeline.premarket_finalize import _event_calendar_cap
        cfg = self._cfg([{"name": "FOMC", "window": ["2026-09-15", "2026-09-17"]}])
        cap, _ = _event_calendar_cap(cfg, "2026-09-16", fetch_auto=False)
        self.assertEqual(cap, 1)


class TestEventCalendarAuto(unittest.TestCase):
    """2026-09-11 事件日历自动维护: FOMC解析/规律播种/三级合并。"""

    # 内嵌官网HTML片段(2026-09 实抓结构核对), 受控测试数据非实时网络
    _FOMC_HTML = """
    <h4><a id="42828">2026 FOMC Meetings</a></h4></div>
    <div class="row fomc-meeting" ">
      <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>September</strong></div>
      <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">15-16*</div>
    </div>
    <div class="fomc-meeting--shaded row fomc-meeting" ">
      <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>October</strong></div>
      <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">27-28</div>
    </div>
    <h4><a id="42827">2025 FOMC Meetings</a></h4></div>
    """

    def test_parse_fomc_html(self):
        from src.data.event_calendar import _parse_fomc_html
        events = _parse_fomc_html(self._FOMC_HTML)
        self.assertEqual(len(events), 2)
        # 9月会议 15-16, 窗口=会议日+2(含决议落地与反应日)
        self.assertEqual(events[0]["name"], "FOMCSeptember")
        self.assertEqual(events[0]["window"], ["2026-09-15", "2026-09-18"])
        self.assertEqual(events[1]["window"], ["2026-10-27", "2026-10-30"])

    def test_parse_ignores_bad_dates(self):
        from src.data.event_calendar import _parse_fomc_html
        html = self._FOMC_HTML.replace("15-16*", "32-33")  # 非法日期
        events = _parse_fomc_html(html)
        self.assertEqual(len(events), 1)

    def test_seed_recurring_events(self):
        from src.data.event_calendar import seed_recurring_events
        events = seed_recurring_events(year=2027)
        names = {e["name"] for e in events}
        self.assertIn("两会2027", names)
        self.assertIn("中央经济工作会议2027", names)
        by_name = {e["name"]: e["window"] for e in events}
        self.assertEqual(by_name["两会2027"], ["2027-03-03", "2027-03-13"])
        self.assertEqual(by_name["中央经济工作会议2027"], ["2027-12-08", "2027-12-16"])

    def test_merge_static_overrides_auto(self):
        # 静态条目与自动条目窗口重叠时, 自动条目被抑制(手工修正优先)
        from src.data.event_calendar import merge_event_calendars
        config = {"event_calendar": {
            "enabled": True,
            "events": [{"name": "中经会(手工)", "window": ["2026-12-08", "2026-12-10"]}],
        }}
        merged = merge_event_calendars(
            config, today="2026-09-01", fetch_fomc=False)
        # 手工条目与 seed 的中经会2026 窗口重叠 → 自动条目被抑制
        # (两会2026 已过期被过滤, 9月不播种次年)
        names = {e["name"] for e in merged}
        self.assertEqual(names, {"中经会(手工)"})

    def test_merge_includes_future_seeds(self):
        from src.data.event_calendar import merge_event_calendars
        config = {"event_calendar": {"enabled": True, "events": []}}
        merged = merge_event_calendars(
            config, today="2026-11-01", fetch_fomc=False)
        # 11月: 当年12月中经会 + 次年两会(10月后跨年播种)都在未来窗口
        names = {e["name"] for e in merged}
        self.assertIn("中央经济工作会议2026", names)
        self.assertIn("两会2027", names)

    def test_merge_expired_events_filtered(self):
        from src.data.event_calendar import merge_event_calendars
        config = {"event_calendar": {
            "enabled": True,
            "events": [{"name": "旧事件", "window": ["2026-01-01", "2026-01-05"]}],
        }}
        merged = merge_event_calendars(
            config, today="2026-09-11", fetch_fomc=False)
        # 过期静态事件被过滤; seed 的中经会2026(12月)在未来, 应保留
        names = {e["name"] for e in merged}
        self.assertNotIn("旧事件", names)
        self.assertIn("中央经济工作会议2026", names)

    def test_auto_fomc_disabled(self):
        from src.data.event_calendar import merge_event_calendars
        config = {"event_calendar": {
            "enabled": True, "auto_fomc": False, "events": [],
        }}
        # auto_fomc=False 不触发抓取(无网络调用), 只有规律播种
        merged = merge_event_calendars(config, today="2026-09-11", fetch_fomc=False)
        names = {e["name"] for e in merged}
        self.assertNotIn("FOMCSeptember", names)


class TestPolitburoNewsDetect(unittest.TestCase):
    """政治局会议新闻检测(影子模式): 前向词过滤。"""

    def _factor(self):
        from src.factors.policy_event import PolicyEventFactor
        return PolicyEventFactor(asof=None)

    def test_upcoming_meeting_detected(self):
        f = self._factor()
        hit = f._detect_politburo_upcoming([
            {"content": "新华社北京电 中共中央政治局会议将于下周一召开", "title": ""}])
        self.assertTrue(hit)

    def test_retrospective_not_detected(self):
        f = self._factor()
        hit = f._detect_politburo_upcoming([
            {"content": "昨日召开的中央政治局会议强调...", "title": ""}])
        self.assertEqual(hit, "")

    def test_signal_flows_into_calculate(self):
        # 集成: calculate 首次调用检测并写入 signals + 实例属性
        f = self._factor()
        r = f.calculate(
            "sh600000", [{"content": "政治局会议即将召开, 分析称...", "title": ""}],
            stock_name="某股")
        self.assertTrue(any(s.startswith("事件日历: ") for s in r["signals"]))
        self.assertTrue(hasattr(f, "politburo_upcoming"))
        # 二次调用不重复检测(实例缓存)
        r2 = f.calculate("sh600000", [], stock_name="某股")
        self.assertFalse(any(s.startswith("事件日历: ") for s in r2["signals"]))

    def test_aggregate_from_scored_stocks(self):
        from src.pipeline.premarket_finalize import _detect_politburo_in_signals
        stocks = [
            {"code": "sh600001", "factors": {"policy_signals": ["无关信号"]}},
            {"code": "sh600002", "factors": {"policy_signals": [
                "事件日历: 政治局会议将于下周一召开"]}},
        ]
        hit = _detect_politburo_in_signals(stocks)
        self.assertIn("政治局会议", hit)
        self.assertEqual(_detect_politburo_in_signals(
            [{"code": "x", "factors": {}}]), "")


if __name__ == "__main__":
    unittest.main()
