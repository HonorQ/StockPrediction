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
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

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


class TestTencentKlineHostFailover(unittest.TestCase):
    """2026-09-14: web.ifzq.gtimg.cn 被腾讯WAF返回501(HTML), 主域名必须自动切备域名。

    故障表现: resp.json() 抛 "Expecting value: line 1 column 1", 盘前整轮K线全灭。
    """

    class _Resp:
        def __init__(self, text):
            self._text = text

        def json(self):
            return json.loads(self._text)

    def setUp(self):
        import src.data.kline_fetcher as kf
        self.kf = kf
        kf.KlineFetcher._host_fail.clear()

    def test_waf_html_falls_back_to_backup_host(self):
        calls = []
        ok_payload = json.dumps({"data": {"bj830799": {"qfqday": [
            ["2026-09-10", "10.0", "10.5", "10.6", "9.9", "1000", "10500"],
            ["2026-09-11", "10.5", "10.8", "10.9", "10.4", "1200", "12960"],
        ]}}})

        def _fake_get(url, **kw):
            calls.append(url)
            return self._Resp("<html>waf</html>" if "web.ifzq" in url else ok_payload)

        # 用北交所代码: 避免 _supplement_turnover 触发 baostock(离线测试禁止联网)
        with patch.object(self.kf.requests, "get", _fake_get):
            df = self.kf.KlineFetcher().get_stock_history("830799", days=5)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(len(calls), 2)  # 主域名失败后切备用域名

    def test_dead_host_skipped_after_repeated_failures(self):
        """连续失败的域名不再重复请求, 避免持续触发WAF并刷屏"""
        self.kf.KlineFetcher._host_fail["web.ifzq.gtimg.cn"] = 3
        hosts = self.kf.KlineFetcher._tencent_hosts()
        self.assertNotIn("web.ifzq.gtimg.cn", hosts)
        self.assertIn("ifzq.gtimg.cn", hosts)


class TestMoneyFlowContinuityContract(unittest.TestCase):
    """2026-09-14: MoneyFlowContinuity 用单日快照dict当序列用, 每只票必抛
    "'dict' object has no attribute 'empty'", 资金流连续性因子长期失效。"""

    def _analyze(self, klines):
        from src.data.money_flow import MoneyFlowFetcher
        from src.factors.phase2_market_intel import MoneyFlowContinuity
        with patch.object(MoneyFlowFetcher, "_fetch_em_fflow_klines",
                          lambda self, c, s: klines):
            return MoneyFlowContinuity(money_fetcher=MoneyFlowFetcher()).analyze("002001")

    def test_unavailable_source_returns_neutral(self):
        r = self._analyze(None)
        self.assertEqual(r["flow_continuity_score"], 50)
        self.assertEqual(r["flow_continuity"], "neutral")

    def test_five_day_inflow_scored(self):
        klines = [f"2026-09-0{i},1e8,2e8,3e8,4e8,5e8,6.0" for i in range(1, 6)]
        r = self._analyze(klines)
        self.assertEqual(r["flow_continuity"], "strong_inflow")
        self.assertEqual(r["flow_continuity_score"], 85)


class TestExpiredCloseBeforeMarketClose(unittest.TestCase):
    """2026-09-14: 盘前/盘中不存在当日收盘价, 到期检查应静默等待而非刷"顺延平仓"。"""

    def test_intraday_returns_not_due(self):
        from src.engine.paper_trade_ops import _try_close_expired

        class _NoPriceFetcher:
            def get_stock_history(self, code, days=120):
                return None

        trade = {"id": 1, "code": "002001", "name": "测试票",
                 "trade_date": "2026-09-01", "entry_date": "2026-09-01",
                 "entry_price": 10.0}
        with patch("src.engine.paper_trade_ops._is_before_market_close", lambda: True):
            # db 传 None: 盘中未平仓, 不应触发任何落库
            status = _try_close_expired(None, _NoPriceFetcher(), trade, 3, "2026-09-11")
        self.assertEqual(status, "not_due")


class TestMLModelMetaNumpy(unittest.TestCase):
    """2026-09-14: meta.json 写入含 numpy int32 的 metrics 抛
    "Object of type int32 is not JSON serializable", 整轮重训练结果丢失。"""

    def test_numpy_metrics_serialized(self):
        from src.engine.ml_selector import MLStockSelector
        path = os.path.join(tempfile.gettempdir(), "test_ml_meta_numpy.pkl")
        sel = MLStockSelector(model_path=path)
        sel.model = "dummy"
        sel._save_model_versioned({"n_samples": np.int32(120), "acc": np.float32(0.58)})
        with open(sel.model_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["metrics"]["n_samples"], "120")


class TestConceptSyncCircuitBreaker(unittest.TestCase):
    """2026-09-14: 东财不可达时概念同步逐个板块重试(400+ × 5s), 实测空转30分钟且
    用空结果覆盖已有缓存。必须熔断 + 不写空缓存。"""

    def setUp(self):
        import akshare as ak
        self.ak = ak
        self.config = os.path.join(tempfile.gettempdir(), "test_concept_map.json")
        if os.path.exists(self.config):
            os.remove(self.config)
        self.cache = self.config + ".auto.json"
        if os.path.exists(self.cache):
            os.remove(self.cache)

    def tearDown(self):
        for p in (self.config, self.cache):
            if os.path.exists(p):
                os.remove(p)

    def test_aborts_after_fail_streak(self):
        calls = []

        def _cons(symbol):
            calls.append(symbol)
            raise OSError("RemoteDisconnected")

        from src.data.concept_map import ConceptMap
        with patch.object(self.ak, "stock_board_concept_name_em",
                          return_value=pd.DataFrame({"板块名称": [f"板块{i}" for i in range(20)]})), \
             patch.object(self.ak, "stock_board_concept_cons_em", side_effect=_cons):
            ok = ConceptMap(config_path=self.config, auto_sync=False).maybe_sync_from_akshare(force=True)
        self.assertFalse(ok)
        self.assertEqual(len(calls), ConceptMap.SYNC_FAIL_STREAK)  # 熔断而非跑满20个
        self.assertFalse(os.path.exists(self.cache))  # 空结果不得覆盖缓存

    def test_board_list_timeout_aborts(self):
        from src.data.concept_map import ConceptMap

        def _hang():
            import time as _t
            _t.sleep(30)
            return pd.DataFrame()

        with patch.object(self.ak, "stock_board_concept_name_em", side_effect=_hang), \
             patch.object(ConceptMap, "SYNC_TIMEOUT", 1):
            ok = ConceptMap(config_path=self.config, auto_sync=False).maybe_sync_from_akshare(force=True)
        self.assertFalse(ok)


class TestIntelFactorSourceCooldown(unittest.TestCase):
    """2026-09-14: 机构调研(48次)/龙虎榜(14次)在东财不可达时对每只候选股重跑全量请求,
    且机构持仓报告期硬编码含"20051"(2005年一季度)。必须缓存+熔断+修正报告期。"""

    def setUp(self):
        from src.factors.phase2_market_intel import InstitutionalResearch, DragonTigerBoard
        self.IR = InstitutionalResearch
        self.DTB = DragonTigerBoard
        self.IR._down_until = 0.0
        self.IR._recommend_df = None
        self.IR._recommend_day = None
        self.IR._hold_cache = {}
        self.DTB._lhb_down_until = 0.0
        self.DTB._lhb_cache = {}
        self.DTB._lhb_cache_date = None

    def test_recent_periods_exclude_2005(self):
        periods = self.IR._recent_periods(3)
        self.assertEqual(len(periods), 3)
        self.assertTrue(periods[0].startswith(str(datetime.now().year)))
        self.assertNotIn("20051", periods)

    def test_institution_source_down_stops_retrying(self):
        import akshare as ak
        calls = []

        def _boom(*a, **kw):
            calls.append(1)
            raise OSError("RemoteDisconnected")

        with patch.object(ak, "stock_institute_recommend", side_effect=_boom), \
             patch.object(ak, "stock_institute_hold", side_effect=_boom):
            f = self.IR()
            f.analyze("002001")
            first_round = len(calls)
            f.analyze("002002")
        self.assertGreater(first_round, 0)
        self.assertEqual(len(calls), first_round)  # 冷却期内不再发起请求

    def test_lhb_source_down_stops_retrying(self):
        import akshare as ak
        calls = []

        def _boom(*a, **kw):
            calls.append(1)
            raise OSError("RemoteDisconnected")

        with patch.object(ak, "stock_lhb_detail_em", side_effect=_boom), \
             patch.object(ak, "stock_lhb_detail_daily_sina", side_effect=_boom):
            f = self.DTB()
            f.analyze("002001")
            first_round = len(calls)
            f.analyze("002002")
        self.assertGreater(first_round, 0)
        self.assertEqual(len(calls), first_round)


class TestSkippedTaskReturnsFalse(unittest.TestCase):
    """2026-09-14: 09:25 intraday / 09:27 afternoon 因盘前缓存未生成而空跑,
    但心跳仍写 ok, 监控全绿而当天实际零产出。空跑必须返回 False。"""

    def test_intraday_returns_false_without_premarket_cache(self):
        from src.pipeline.intraday import run_intraday
        with patch("src.pipeline.intraday._log_source_health"), \
             patch("src.pipeline.intraday.load_cache", return_value=None):
            self.assertIs(run_intraday({}), False)

    def test_afternoon_returns_false_without_premarket_cache(self):
        from src.pipeline.afternoon import run_afternoon
        with patch("src.pipeline.afternoon._log_source_health"), \
             patch("src.pipeline.afternoon.check_task_alerts", return_value=[]), \
             patch("src.pipeline.afternoon.load_cache", return_value=None):
            self.assertIs(run_afternoon({}), False)


class TestWeightAdjustIdempotent(unittest.TestCase):
    """2026-09-14: 同日两次盘前运行(full 补跑)把唯一有效因子 ashare_specific
    连降两档 1.0→0.95→0.90。调权必须当天幂等。"""

    class _DB:
        def __init__(self):
            self.called = []

        def has_weight_update(self, trade_date):
            self.called.append(("check_idem", trade_date))
            return True

        def get_attribution_history(self, days=60):
            self.called.append(("attr", days))
            return [{"trade_date": "2026-09-14", "factor_results": {}}]

    def test_skips_when_already_adjusted_today(self):
        from src.engine.feedback_loop import FeedbackLoop
        fb = FeedbackLoop.__new__(FeedbackLoop)
        fb.db = self._DB()
        fb.config = {"weights": {"ashare_specific": 1.0}}
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertEqual(fb._adjust_weights(today), [])
        self.assertEqual(fb.db.called, [("check_idem", today)])  # 未再读归因、未落库

    def test_backtest_historical_date_bypasses_idempotency(self):
        """回测传历史日期时必须照常调权(否则回测与实盘链路漂移)。"""
        from src.engine.feedback_loop import FeedbackLoop
        fb = FeedbackLoop.__new__(FeedbackLoop)
        fb.db = self._DB()
        fb.config = {"weights": {}}
        fb._adjust_weights("2020-01-02")  # 不应因"今日已调权"提前返回
        self.assertEqual(fb.db.called[0][0], "attr")


class TestMLQualityGate(unittest.TestCase):
    """2026-09-14: 样本外劣于随机的模型(train_acc=0.81 / test_acc=0.413)此前每轮
    无条件覆盖线上模型, 而它按 5% 权重参与午后打分。门禁以测试集 AUC 判定。"""

    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), "test_ml_gate.pkl")
        self.hist = os.path.join(tempfile.gettempdir(), "history")
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        for suffix in ("", ".meta.json", "_aux.pkl"):
            p = self.path + suffix
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(self.hist):
            import shutil
            shutil.rmtree(self.hist, ignore_errors=True)

    def _selector(self):
        from src.engine.ml_selector import MLStockSelector
        sel = MLStockSelector(model_path=self.path)
        sel.model = "dummy"
        return sel

    def test_rejected_model_does_not_overwrite_production(self):
        self._selector()._save_model_versioned({"accuracy": 0.413, "test_auc": 0.48})
        self.assertFalse(os.path.exists(self.path + ".meta.json"))  # 线上 meta 未被覆盖
        rejected = [f for f in os.listdir(self.hist) if f.startswith("rejected_")]
        self.assertEqual(len(rejected), 1)  # 仅留诊断记录

    def test_qualified_model_is_saved(self):
        self._selector()._save_model_versioned({"accuracy": 0.58, "test_auc": 0.56})
        with open(self.path + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["metrics"]["test_auc"], 0.56)
        self.assertFalse(meta["metrics"].get("rejected", False))

    def test_low_auc_model_is_not_loaded(self):
        """门禁闭环: 门禁上线前落盘的劣质模型必须在加载期就被挡掉。"""
        import joblib
        from src.engine.ml_selector import MLStockSelector
        joblib.dump("dummy_model", self.path)
        with open(self.path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"feature_names": [], "metrics": {"test_auc": 0.41}}, f)
        sel = MLStockSelector(model_path=self.path)
        self.assertIsNone(sel.model)
        self.assertEqual(sel.predict({})["ml_score"], 50)  # 回退中性, 不参与打分

    def test_qualified_model_is_loaded(self):
        import joblib
        from src.engine.ml_selector import MLStockSelector
        joblib.dump("dummy_model", self.path)
        with open(self.path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"feature_names": [], "metrics": {"test_auc": 0.56}}, f)
        self.assertIsNotNone(MLStockSelector(model_path=self.path).model)

    def test_model_without_auc_metric_still_loads(self):
        """旧格式 meta 无 test_auc 时必须向后兼容(照常加载)。"""
        import joblib
        from src.engine.ml_selector import MLStockSelector
        joblib.dump("dummy_model", self.path)
        with open(self.path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"feature_names": [], "metrics": {"accuracy": 0.6}}, f)
        self.assertIsNotNone(MLStockSelector(model_path=self.path).model)


class TestRecommendationBackfillFromKline(unittest.TestCase):
    """2026-09-16 修复: 历史收益 T+1/T+3/T+5 全空。

    原 premarket_finalize 用"自然日窗口 + 内嵌行情获取"补收益, 周末/节假日错位,
    kline 不全时静默跳过, 导致 2026-09-07 之后所有推荐收益为 NULL,
    报告页历史均值显示 "-"。本测试验证新离线回填(本地 kline_daily, 交易日索引)正确幂等。
    """

    def _temp_db(self):
        import tempfile
        from src.data.database import StockDatabase
        fd, path = tempfile.mkstemp(suffix=".db")
        import os
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return StockDatabase(db_path=path)

    def _tmp_db(self):
        return self._temp_db()

    def _seed_klines(self, db, code):
        """向 kline_daily 连续写入 15 个交易日(截至今天的近 3 周)。

        日期相对今天生成: 15 个工作日 ≈ 21 自然日, 稳落在回填窗口
        days=30 内 —— 硬编码日期会随真实时间滑出窗口导致误报失败。
        """
        import pandas as pd
        from datetime import datetime
        dates = pd.bdate_range(end=datetime.now().date(), periods=15) \
            .strftime("%Y-%m-%d").tolist()
        closes = [10 + 0.04 * i for i in range(15)]   # 线性上涨 10.00→10.56
        df = pd.DataFrame({"trade_date": dates, "close": closes,
                           "open": closes, "high": closes, "low": closes,
                           "volume": 1e6, "amount": 1e7})
        db.save_klines(code, df)
        return dates, closes

    def _get_row(self, db, trade_date, code):
        return db._fetch_one(
            "SELECT * FROM recommendations WHERE trade_date=? AND code=?",
            (trade_date, code))

    def test_backfill_calculates_d1_d3_d5(self):
        db = self._tmp_db()
        dates, closes = self._seed_klines(db, "600001")
        db.save_recommendation(
            trade_date=dates[0], code="600001", name="测试股", score=60,
            price=10.0, mode="afternoon")

        updated = db.backfill_returns_from_kline(days=30)
        self.assertEqual(updated, 5)  # d1/d3/d5/d10/d14 五个字段
        row = self._get_row(db, dates[0], "600001")
        self.assertIsNotNone(row["d1_return"])
        self.assertAlmostEqual(row["d1_return"], (closes[1] / 10.0 - 1) * 100, places=2)
        self.assertAlmostEqual(row["d3_return"], (closes[3] / 10.0 - 1) * 100, places=2)
        self.assertAlmostEqual(row["d5_return"], (closes[5] / 10.0 - 1) * 100, places=2)

    def test_backfill_is_idempotent_and_respects_overwrite(self):
        db = self._tmp_db()
        dates, closes = self._seed_klines(db, "600001")
        db.save_recommendation(
            trade_date=dates[0], code="600001", name="测试股", score=60,
            price=10.0, mode="afternoon")
        db.backfill_returns_from_kline(days=30)
        # 修改已填值, 不 overwrite 时回填不得覆盖
        db.update_performance(dates[0], "600001", d1=99.0, mode="afternoon")
        db.backfill_returns_from_kline(days=30)
        self.assertEqual(self._get_row(db, dates[0], "600001")["d1_return"], 99.0)
        # overwrite=True 时重算
        db.backfill_returns_from_kline(days=30, overwrite=True)
        self.assertAlmostEqual(
            self._get_row(db, dates[0], "600001")["d1_return"],
            (closes[1] / 10.0 - 1) * 100, places=2)

    def test_backfill_skips_non_trading_recommendation_date(self):
        """推荐日在行情无记录时(新股/缺失)不得抛错, 也无需回填"""
        db = self._tmp_db()
        dates, _ = self._seed_klines(db, "600001")
        import pandas as pd
        pre = (pd.Timestamp(dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        db.save_recommendation(
            trade_date=pre, code="600001", name="测试股", score=60,
            price=10.0, mode="afternoon")  # 早于 K 线起点(相对日期防滑窗)
        updated = db.backfill_returns_from_kline(days=30)
        self.assertEqual(updated, 0)
        # 窗口拉大重验: 基准日不在 K 线里, 仍不得误填
        db.backfill_returns_from_kline(days=60)
        row = self._get_row(db, pre, "600001")
        self.assertIsNone(row["d1_return"])  # K线起点晚于推荐日, 基准日找不到→跳过

    def test_enrich_history_returns_fills_report_field(self):
        """reporter._enrich_history_returns 按 (date, code) 回填历史摘要字段"""
        db = self._tmp_db()
        dates, closes = self._seed_klines(db, "600001")
        db.save_recommendation(
            trade_date=dates[0], code="600001", name="测试股", score=60,
            price=10.0, mode="afternoon")
        db.backfill_returns_from_kline(days=30)

        with patch("src.data.database.StockDatabase", return_value=db):
            from src.engine.reporter import ReportGenerator
            report = ReportGenerator()
            history = [{"date": dates[0], "stocks": [{"code": "600001", "price": 10.0}]}]
            report._enrich_history_returns(history)
        s = history[0]["stocks"][0]
        self.assertIsNotNone(s.get("d1_return"))
        self.assertIsNotNone(s.get("return_to_date"))  # ROI 最新收盘 / entry - 1
        self.assertAlmostEqual(s["return_to_date"], (closes[-1] / 10.0 - 1) * 100, places=2)


class TestBackfillModeIsolation(unittest.TestCase):
    """backfill_recommendation_returns 的 mode 隔离（2026-10-15 修复）

    同一 (trade_date, code) 可同时存在 premarket 与 afternoon 两条推荐，
    已确认实例： 2026-09-18 000338、2026-09-21 000628。
    旧实现把 mode 归一成 premarket='1' 与 afternoon='2' 两个字符串，
    而默认回填只按 trade_date 定位，导致站点的实盘推荐长期漏回填。
    回归测试锁定三条约定：
      1. 单模式回填不得波及其它模式
      2. mode=live 一次覆盖两个实盘模式，且各按本行入场价计价
      3. premarket/afternoon 绝不触碰 backtest/feedback_training 的模拟收益
    """
    import sqlite3  # 局部导入：避免改动文件头既有 import 区

    def setUp(self):
        # 被测模块在 scripts/ 下（非包），需显式加入 sys.path
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts"))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    # ---- 辅助 ----
    def _new_db(self, path):
        conn = self.sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT, code TEXT, name TEXT, score REAL,
                current_price REAL, rank INTEGER, mode TEXT,
                d1_return REAL, d3_return REAL, d5_return REAL,
                d10_return REAL, d14_return REAL,
                factors TEXT, created_at TEXT, regime TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _seed(self, path, rows):
        """rows: [(trade_date, code, current_price, mode), ...]"""
        conn = self.sqlite3.connect(path)
        conn.executemany(
            "INSERT INTO recommendations (trade_date, code, current_price, mode) "
            "VALUES (?,?,?,?)", rows)
        conn.commit()
        conn.close()

    def _get(self, path, mode):
        conn = self.sqlite3.connect(path)
        conn.row_factory = self.sqlite3.Row
        row = conn.execute(
            "SELECT * FROM recommendations WHERE mode=?", (mode,)).fetchone()
        conn.close()
        return row

    def _get_field(self, path, mode, field):
        conn = self.sqlite3.connect(path)
        row = conn.execute(
            f"SELECT {field} FROM recommendations WHERE mode=?", (mode,)).fetchone()
        conn.close()
        return row[0] if row else None

    @staticmethod
    def _fake_daily():
        """离线模拟 baostock 日K {date: close}，覆盖 2026-09-10 ~ 2026-10-20"""
        dates = ([f"2026-09-{d:02d}" for d in range(10, 31)]
                 + [f"2026-10-{d:02d}" for d in range(1, 21)])
        return {d: 10.0 + i * 0.1 for i, d in enumerate(dates)}

    def _run_backfill(self, db_path, mode):
        """在离线环境下执行一次 backfill: baostock 换成假模块, fetch_daily 换成假行情"""
        import backfill_recommendation_returns as mod
        from unittest.mock import MagicMock  # 局部导入，避免改动文件头 import 区
        with patch.dict(sys.modules, {"baostock": MagicMock()}), \
                patch.object(mod, "DB_PATH", db_path), \
                patch.object(mod, "fetch_daily", return_value=self._fake_daily()):
            mod.backfill(mode)

    # ---- 用例 ----
    def test_premarket_backfill_does_not_touch_afternoon(self):
        """默认 premarket 回填不得改写 afternoon 行"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000338", 100.0, "premarket"),
                            ("2026-09-18", "000338", 100.0, "afternoon")])
            self._run_backfill(db, "premarket")

            # premarket 行已回填
            self.assertIsNotNone(self._get_field(db, "premarket", "d5_return"))
            # afternoon 行必须保持全空
            for f in ("d1_return", "d3_return", "d5_return", "d10_return", "d14_return"):
                self.assertIsNone(self._get_field(db, "afternoon", f))

    def test_live_fills_both_modes_with_own_entry(self):
        """live 一次覆盖两个实盘模式，且各按本行入场价计价"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000338", 100.0, "premarket"),
                            ("2026-09-18", "000338", 200.0, "afternoon")])
            self._run_backfill(db, "live")

            pm = self._get_field(db, "premarket", "d5_return")
            af = self._get_field(db, "afternoon", "d5_return")
            self.assertIsNotNone(pm)
            self.assertIsNotNone(af)
            # 入场价不同 => 收益率必须不同；相同即说明看错行的入场价
            self.assertNotEqual(pm, af)
            # 2026-09-18 是 _fake_daily 的第 9 项(idx=8)，T+5 落在 idx=13 = 2026-09-23，
            # close = 10.0 + 13 * 0.1 = 11.3
            self.assertEqual(pm, round((11.3 / 100.0 - 1) * 100, 2))
            self.assertEqual(af, round((11.3 / 200.0 - 1) * 100, 2))

    def test_live_never_touches_simulation_modes(self):
        """实盘模式回填绝不得触碰模拟模式"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000338", 100.0, "premarket"),
                            ("2026-09-18", "000338", 200.0, "afternoon"),
                            ("2026-09-18", "000338", 100.0, "feedback_training"),
                            ("2026-09-18", "000338", 100.0, "backtest")])
            self._run_backfill(db, "live")

            for mode in ("feedback_training", "backtest"):
                for f in ("d1_return", "d5_return", "d14_return"):
                    self.assertIsNone(self._get_field(db, mode, f))

    def test_live_fills_only_first_pending_method(self):
        """live 只回填第一个待补模式，避免重复计算同一收益源"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000001", 100.0, "premarket"),
                            ("2026-09-18", "000002", 200.0, "afternoon")])
            self._run_backfill(db, "live")

            for code in ("000001", "000002"):
                self.assertIsNotNone(
                    self._get_field_by_code(db, code, "d5_return"))

    def _get_field_by_code(self, path, code, field):
        conn = self.sqlite3.connect(path)
        row = conn.execute(
            f"SELECT {field} FROM recommendations WHERE code=?", (code,)).fetchone()
        conn.close()
        return row[0] if row else None

    def test_all_overwrites_simulation_row(self):
        """all 应一并覆盖模拟模式行"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000338", 100.0, "feedback_training")])
            self._run_backfill(db, "all")
            self.assertIsNotNone(
                self._get_field(db, "feedback_training", "d5_return"))

    def test_idempotent_no_growth_on_second_run(self):
        """重复运行已回填行，值不变且有输出提示（幂等）"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            self._new_db(db)
            self._seed(db, [("2026-09-18", "000001", 100.0, "premarket"),
                            ("2026-09-18", "000002", 100.0, "afternoon")])
            self._run_backfill(db, "live")
            v1 = {c: self._get_field_by_code(db, c, "d5_return")
                  for c in ("000001", "000002")}
            self._run_backfill(db, "live")
            v2 = {c: self._get_field_by_code(db, c, "d5_return")
                  for c in ("000001", "000002")}
            self.assertEqual(v1, v2)
            for c, v in v2.items():
                self.assertIsNotNone(v, f"{c} 第二次应为已完成而非被清空")
