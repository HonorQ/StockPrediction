"""
回测系统 - 向量化Walk-Forward回测 + 17维因子评分 + 完整绩效指标

评分统一说明：
    生产环境（run_daily.py）通过 StockScorer.batch_normalize 做截面 Z-Score 归一化，
    保证不同交易日的得分可比。本回测引擎亦使用同一套 batch_normalize 流程，
    在每个交易日对所有候选股票做截面评分，避免回测分数与生产分数口径不一致。
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BacktestEngine:
    """向量化回测引擎"""

    # 入场阈值：与 config.yaml 中 filter.min_score 默认值一致
    DEFAULT_ENTRY_SCORE = 55
    # 截面评分的最小样本数：少于该值时降级为单股票 calculate_score
    MIN_CROSS_SECTIONAL_SAMPLES = 5

    def __init__(self, initial_capital: float = 100000,
                 commission_rate: float = 0.0003, stamp_tax: float = 0.0005,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        # A股交易成本：佣金万2.5(双边,最低5元)、印花税卖出0.05%、滑点单边0.1%。
        # 零成本假设会把短线策略的收益系统性高估
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        # 回测覆盖的真实交易日数，由 _simulate_holding 统计（年化/Sharpe 用）
        self.trading_day_count = 0

    def _trade_costs(self, buy_amount: float, sell_amount: float) -> float:
        """一笔完整交易的总摩擦成本（佣金双边+印花税卖出+滑点已计入价格）"""
        commission = max(buy_amount * self.commission_rate, 5.0) + \
                     max(sell_amount * self.commission_rate, 5.0)
        stamp = sell_amount * self.stamp_tax
        return commission + stamp

    # ------------------------------------------------------------------
    # 训练窗口权重拟合（补 ML_TRAINING_PLAN 缺口#6：真 walk-forward）
    # ------------------------------------------------------------------
    def _fit_window_weights(self, window_histories: Dict[str, pd.DataFrame],
                            train_end: int, hold_days: int = 3,
                            min_samples: int = 30,
                            min_ic: float = 0.01) -> Optional[Dict[str, float]]:
        """在训练窗口 [60, train_end) 内拟合因子权重：
        逐日计算 24 维子评分与 T+hold_days 收益的 Rank IC → 均值 → softmax 加权。

        仅取 IC>min_ic 的正贡献维度；其余维度保留 0.001 底权并归一化(24维齐全)。
        样本不足或无正 IC 因子时返回 None，调用方沿用基准权重。
        min_ic 暴露为参数便于测试(随机数据下真实因子IC均值难以稳定超过0.01)。

        注意：逐日因子计算成本高(每只股票约 train_window 天)，仅建议
        run_walk_forward(fit_weights=True) 时开启。
        """
        from src.engine.scorer import StockScorer

        scorer = StockScorer()
        samples = []  # (date_key, sub_scores, fwd_ret)
        for sc, h in window_histories.items():
            close_col = "收盘" if "收盘" in h.columns else "close"
            if close_col not in h.columns:
                continue
            closes = h[close_col].astype(float).values
            for i in range(60, train_end - hold_days):
                if pd.isna(closes[i]) or closes[i] <= 0:
                    continue
                fwd = closes[i + hold_days] / closes[i] - 1
                if pd.isna(fwd):
                    continue
                window = h.iloc[max(0, i - 120):i + 1]
                factors = self._calc_factors(window, sc)
                samples.append((self._date_of(h, i), scorer.get_sub_scores(factors), float(fwd)))

        if len(samples) < min_samples:
            return None

        by_date: Dict[str, list] = defaultdict(list)
        for d, ss, fr in samples:
            by_date[d].append((ss, fr))

        dims = list(scorer.weights.keys())
        ic_by_dim: Dict[str, list] = {d: [] for d in dims}
        for group in by_date.values():
            if len(group) < 5:
                continue
            rets = pd.Series([g[1] for g in group]).rank()
            for dim in dims:
                vals = pd.Series([g[0].get(dim, 50) for g in group])
                if vals.nunique() < 2:
                    continue
                ic = vals.rank().corr(rets)
                if not pd.isna(ic):
                    ic_by_dim[dim].append(ic)

        positive = {
            d: float(np.mean(v)) for d, v in ic_by_dim.items()
            if v and np.mean(v) > min_ic
        }
        if not positive:
            return None

        exps = np.exp(np.array(list(positive.values())) * 5)
        fitted = {d: e / exps.sum() for d, e in zip(positive.keys(), exps)}
        out = {d: 0.001 for d in dims}
        out.update(fitted)
        total = sum(out.values()) or 1.0
        return {d: round(v / total, 4) for d, v in out.items()}

    def run_walk_forward(self, stocks: List[str], weights: Dict[str, float],
                         train_window: int = 120, test_window: int = 20,
                         hold_days: int = 3, max_positions: int = 5,
                         top_n: int = 5,
                         entry_score: float = DEFAULT_ENTRY_SCORE,
                         regime_aware: bool = False,
                         fit_weights: bool = False) -> Dict:
        """
        Walk-Forward回测（截面评分统一版）
        train_window: 训练窗口天数（用于因子计算时的lookback）
        test_window: 测试窗口天数
        流程: [train 120天 | test 20天] → 滚动 → [train 120天 | test 20天]
        每个测试窗口内：逐日收集所有股票的因子，按日做 batch_normalize 截面评分。
        regime_aware: True 时按指数历史逐日判定简化 regime 并切换因子权重，
            与实盘 run_daily 的 regime 权重切换口径对齐。
        fit_weights: True 时每个测试窗口前在训练窗口内拟合 IC 权重(真 walk-forward)，
            拟合失败沿用传入的基准权重。开启后计算耗时显著增加。
        """
        from src.data.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher()

        # 1. 拉取所有股票的历史K线
        stock_histories: Dict[str, pd.DataFrame] = {}
        for sc in stocks:
            try:
                h = fetcher.get_stock_history(sc, days=250)
            except Exception:
                h = None
            if h is not None and not h.empty and len(h) >= train_window + test_window:
                stock_histories[sc] = h

        if not stock_histories:
            return {"error": "无可用历史数据"}

        regime_by_date = self._build_regime_by_date(fetcher=fetcher) if regime_aware else None

        all_trades: List[Dict] = []

        # 2. 对齐所有股票的日期索引，找出共同的窗口起止位置
        # Walk-forward按test_window步长滚动，每轮在所有股票上同时取相同日期段
        min_len = min(len(h) for h in stock_histories.values())
        windows = 0
        while windows * test_window + train_window + test_window <= min_len:
            train_end = windows * test_window + train_window
            test_start = train_end
            test_end = min(test_start + test_window, min_len)

            if test_end - test_start < 3:
                windows += 1
                continue

            # 对每只股票取 train+test 段
            window_histories = {
                sc: h.iloc[:test_end] for sc, h in stock_histories.items()
            }

            # 真 walk-forward：训练窗口内拟合权重，测试窗口用拟合结果评分；
            # 拟合失败(None)沿用基准 weights，行为与旧版一致
            win_weights = weights
            if fit_weights:
                try:
                    fitted = self._fit_window_weights(
                        window_histories, train_end, hold_days=hold_days)
                    if fitted:
                        win_weights = fitted
                except Exception:
                    win_weights = weights

            trades = self._backtest_window_cross_sectional(
                window_histories, win_weights,
                train_size=train_window,
                test_start=test_start,
                test_end=test_end,
                hold_days=hold_days,
                max_positions=max_positions,
                top_n=top_n,
                entry_score=entry_score,
                regime_by_date=regime_by_date,
            )
            all_trades.extend(trades)
            windows += 1

        if not all_trades:
            return {"error": "无交易信号"}

        return self._calculate_results(all_trades)

    def run(self, stocks: List[str], weights: Dict[str, float],
            hold_days: int = 3, max_positions: int = 5,
            top_n: int = 5, benchmark_code: str = "sh000300",
            entry_score: float = DEFAULT_ENTRY_SCORE,
            regime_aware: bool = False) -> Dict:
        """
        标准回测（全量数据，截面评分统一版）
        与生产环境 StockScorer.batch_normalize 完全一致：
            1. 为每只股票逐日生成原始因子 + sub_scores
            2. 按交易日分组，调用 batch_normalize 做截面 Z-Score 归一化
            3. 每日选 top_n 入场，持仓 hold_days 后平仓
        regime_aware: True 时按指数历史逐日判定简化 regime 并切换因子权重，
            与实盘 run_daily 的 regime 权重切换口径对齐。
        """
        # 1. 拉取所有股票的历史K线
        stock_histories: Dict[str, pd.DataFrame] = {}
        for sc in stocks:
            h = self._get_history(sc)
            if h is not None and len(h) >= 60:
                stock_histories[sc] = h

        if not stock_histories:
            return {"error": "无可用历史数据"}

        regime_by_date = self._build_regime_by_date() if regime_aware else None

        # 2. 全量逐日生成信号
        all_signals: List[Dict] = []
        for sc, h in stock_histories.items():
            sigs = self._generate_signals(sc, h, train_size=60)
            all_signals.extend(sigs)

        if not all_signals:
            return {"error": "无交易信号"}

        # 3. 截面评分（与生产 batch_normalize 一致）
        self._score_cross_sectional(all_signals, weights, regime_by_date=regime_by_date)

        # 4. 每日选 top_n 模拟持仓
        trades = self._simulate_holding(
            all_signals, stock_histories,
            hold_days=hold_days,
            max_positions=max_positions,
            top_n=top_n,
            entry_score=entry_score,
        )

        if not trades:
            return {"error": "无交易记录"}

        return self._calculate_results(trades)

    def _get_history(self, stock_code: str) -> Optional[pd.DataFrame]:
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        try:
            return fetcher.get_stock_history(stock_code, days=250)
        except Exception:
            return None

    @staticmethod
    def _date_of(history: pd.DataFrame, i: int) -> str:
        """取第 i 行的真实交易日字符串。
        get_stock_history 返回 RangeIndex(已 reset_index)，此前直接用
        history.index[i] 导致日期是整数位置(如 "120")而非日历日期，
        按日截面分组虽能工作，但报告中的日期与 regime 对齐全部失真。
        """
        for col in ("日期", "date"):
            if col in history.columns:
                return str(history.iloc[i][col])[:10]
        idx_val = history.index[i]
        return idx_val.strftime("%Y-%m-%d") if hasattr(idx_val, "strftime") else str(idx_val)

    # ------------------------------------------------------------------
    # 简化历史 regime 判定（回测与实盘的权重口径对齐用）
    # ------------------------------------------------------------------
    def _build_regime_by_date(self, fetcher=None, index_code: str = "sh000001",
                              days: int = 250) -> Dict[str, str]:
        """用指数K线逐日构建简化版 regime 序列 {date_str: bull/neutral/bear/crash}。

        仅复现 assess_regime 中可从历史K线回溯的两个维度：主指数涨跌、MA20趋势；
        涨跌停结构/北向/外盘无逐日历史数据故缺席，整体判定偏中性——
        但保证 bear/crash 的防守权重切换在回测中生效，消除与实盘的口径差。
        阈值与 market_regime.assess_regime 保持一致。
        """
        try:
            if fetcher is None:
                from src.data.market_data import MarketDataFetcher
                fetcher = MarketDataFetcher()
            hist = fetcher.get_stock_history(index_code, days=days)
            if hist is None or hist.empty or len(hist) < 25:
                return {}
            close_col = "收盘" if "收盘" in hist.columns else "close"
            closes = hist[close_col].astype(float).reset_index(drop=True)
            ma20 = closes.rolling(20).mean()
            out: Dict[str, str] = {}
            for i in range(21, len(closes)):
                chg = (closes.iloc[i] / closes.iloc[i - 1] - 1) * 100
                cur, _prev = closes.iloc[i], closes.iloc[i - 1]
                ma_now, ma_prev = ma20.iloc[i], ma20.iloc[i - 1]
                if cur > ma_now and ma_now >= ma_prev:
                    trend_pts = 5      # above_ma20_rising
                elif cur > ma_now:
                    trend_pts = 0      # above_ma20
                elif cur < ma_now and ma_now <= ma_prev:
                    trend_pts = -10    # below_ma20_falling
                else:
                    trend_pts = -5     # below_ma20
                score = 50.0
                if chg <= -2.0:
                    score -= 25
                elif chg <= -1.0:
                    score -= 12
                elif chg >= 1.0:
                    score += 12
                score += trend_pts
                score = max(0.0, min(100.0, score))
                if score <= 20:
                    regime = "crash"
                elif score <= 35:
                    regime = "bear"
                elif score >= 65:
                    regime = "bull"
                else:
                    regime = "neutral"
                out[self._date_of(hist, i)] = regime
            return out
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # 信号生成：单股票逐日生成原始因子 + sub_scores（未评分）
    # ------------------------------------------------------------------
    def _generate_signals(self, stock_code: str, history: pd.DataFrame,
                          train_size: int = 60) -> List[Dict]:
        """对单只股票逐日生成交易信号（含原始因子和子评分，未做截面归一化）"""
        from src.engine.scorer import StockScorer
        scorer = StockScorer()  # 默认权重，仅用于 get_sub_scores

        signals: List[Dict] = []
        close_col = "收盘" if "收盘" in history.columns else "close"

        for i in range(train_size, len(history)):
            window = history.iloc[i - train_size:i + 1]
            row = history.iloc[i]
            close = row[close_col] if close_col in row.index else row.get("close", 0)
            if pd.isna(close) or close <= 0:
                continue

            factors = self._calc_factors(window, stock_code)
            sub_scores = scorer.get_sub_scores(factors)
            expectation_penalty = scorer.penalty_expectation(factors)

            date_str = self._date_of(history, i)

            signals.append({
                "stock_code": stock_code,
                "date": date_str,
                "idx": i,
                "price": float(close),
                "factors": factors,
                "sub_scores": sub_scores,
                "expectation_penalty": expectation_penalty,
            })

        return signals

    # ------------------------------------------------------------------
    # 截面评分：按日分组，调用 StockScorer.batch_normalize
    # ------------------------------------------------------------------
    @staticmethod
    def _score_cross_sectional(signals: List[Dict], weights: Dict[str, float],
                               regime_by_date: Dict[str, str] = None) -> None:
        """按交易日分组，对每组信号调用 batch_normalize 做截面 Z-Score 评分。
        样本数 < MIN_CROSS_SECTIONAL_SAMPLES 时降级为单股票 calculate_score。
        直接修改 signals 中每一项的 'score' 字段。

        regime_by_date: {date_str: bull/neutral/bear/crash}，提供时按日切换
        regime 因子权重(与实盘 run_daily 5.2 的口径一致)；config.yaml 中
        regime.enabled=False 时 get_regime_weights 返回原权重，行为不变。
        """
        from src.engine.scorer import StockScorer
        from src.engine.market_regime import MarketRegimeFilter

        scorer = StockScorer(weights)
        regime_filter = None
        if regime_by_date:
            try:
                regime_filter = MarketRegimeFilter()
            except Exception:
                regime_filter = None

        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for sig in signals:
            by_date[sig["date"]].append(sig)

        for date, group in by_date.items():
            day_weights = weights
            if regime_filter is not None:
                rg = regime_by_date.get(date)
                if rg:
                    day_weights = regime_filter.get_regime_weights(rg, weights)
            if len(group) >= BacktestEngine.MIN_CROSS_SECTIONAL_SAMPLES:
                # 与生产完全一致：截面 Z-Score + 截尾 + 正交化
                # batch_normalize 需要字段：sub_scores, expectation_penalty
                candidates = [
                    {
                        "sub_scores": s["sub_scores"],
                        "expectation_penalty": s["expectation_penalty"],
                    }
                    for s in group
                ]
                StockScorer.batch_normalize(candidates, day_weights)
                for s, c in zip(group, candidates):
                    s["score"] = c["score"]
            else:
                # 降级：单股票评分（无截面归一化）
                for s in group:
                    s["score"] = scorer.calculate_score(s["factors"])

    # ------------------------------------------------------------------
    # 持仓模拟：每日选 top_n，持仓 hold_days
    # ------------------------------------------------------------------
    def _simulate_holding(self, signals: List[Dict],
                          histories: Dict[str, pd.DataFrame],
                          hold_days: int, max_positions: int,
                          top_n: int, entry_score: float) -> List[Dict]:
        """对每个交易日的截面评分结果选 top_n，模拟持仓 hold_days 后平仓。

        成交与仓位规则（消除前视偏差与敞口失控）：
          1. T 日信号以 T+1 日开盘价成交（实盘只能收盘后决策、次日开盘买入；
             原实现用 T 日收盘价入场，因子窗口含 T 日收盘 → 前视偏差）
          2. 平仓为 T+1+hold_days 日收盘价（持有 hold_days 个完整交易日）
          3. 并发持仓上限 max_positions：持仓未平时不重复开仓、同票不加仓
             （原实现每日独立开仓，重叠期内总敞口可达本金数倍）
        """
        from collections import defaultdict

        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for sig in signals:
            by_date[sig["date"]].append(sig)

        capital_per_stock = self.initial_capital / max_positions if max_positions > 0 else self.initial_capital
        trades: List[Dict] = []

        # 并发持仓表: [{code, exit_date}]，ISO日期字符串可直接比较
        open_positions: List[Dict] = []

        for date in sorted(by_date.keys()):
            # 先释放已到期持仓（exit_date <= 当日视为已平仓腾出仓位）
            open_positions = [p for p in open_positions if p["exit_date"] > date]

            group = by_date[date]
            qualified = [s for s in group if s["score"] >= entry_score]
            qualified.sort(key=lambda x: -x["score"])
            picks = qualified[:top_n]

            held_codes = {p["code"] for p in open_positions}

            for sig in picks:
                # 并发仓位已满则不再开新仓
                if max_positions > 0 and len(open_positions) >= max_positions:
                    break
                sc = sig["stock_code"]
                # 同一股票持仓期间不重复开仓（避免同票叠加敞口）
                if sc in held_codes:
                    continue
                history = histories.get(sc)
                if history is None:
                    continue
                sig_idx = sig["idx"]

                open_col = "开盘" if "开盘" in history.columns else "open"
                close_col = "收盘" if "收盘" in history.columns else "close"

                # T+1 日开盘成交；越界(数据末端)则放弃该笔
                entry_idx = sig_idx + 1
                if entry_idx >= len(history):
                    continue
                entry_row = history.iloc[entry_idx]
                entry_price = entry_row[open_col] if open_col in entry_row.index else entry_row.get("open", 0)

                exit_idx = min(entry_idx + hold_days, len(history) - 1)
                exit_row = history.iloc[exit_idx]
                exit_price = exit_row[close_col] if close_col in exit_row.index else exit_row.get("close", 0)

                if pd.isna(exit_price) or pd.isna(entry_price) or entry_price <= 0 or exit_price <= 0:
                    continue

                # 数据末端被截断导致持有期不足 → 放弃（避免把半程收益当全程）
                if exit_idx < entry_idx + hold_days and entry_idx + hold_days < len(history):
                    continue

                # 滑点：买入按开盘上浮、卖出按下浮
                buy_price = entry_price * (1 + self.slippage)
                sell_price = exit_price * (1 - self.slippage)
                ret = (sell_price / buy_price - 1) * 100

                shares = int(capital_per_stock / entry_price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = shares * buy_price
                sell_amount = shares * sell_price
                costs = self._trade_costs(buy_amount, sell_amount)
                pnl = sell_amount - buy_amount - costs

                entry_date_str = self._date_of(history, entry_idx)
                exit_date_str = self._date_of(history, exit_idx)

                trades.append({
                    "stock_code": sc,
                    "entry_date": entry_date_str,
                    "exit_date": exit_date_str,
                    "signal_date": date,
                    # entry/exit 记录未含滑点的成交参考价；return_pct/pnl 已含全部摩擦成本
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(float(exit_price), 2),
                    "return_pct": round(ret, 2),
                    "pnl": round(pnl, 2),
                    "score": round(sig["score"], 1),
                    "shares": shares,
                })
                open_positions.append({"code": sc, "exit_date": exit_date_str})
                held_codes.add(sc)

        # 记录回测覆盖的真实交易日数（供年化/Sharpe 使用，替代原来的"交易笔数冒充天数"）
        all_dates = set()
        for s in signals:
            all_dates.add(s["date"])
        for t in trades:
            all_dates.add(t["entry_date"])
            all_dates.add(t["exit_date"])
        for h in histories.values():
            col = "日期" if "日期" in h.columns else ("date" if "date" in h.columns else None)
            if col:
                all_dates.update(h[col].astype(str).str[:10].tolist())
        if all_dates:
            dmin = min(t["signal_date"] for t in trades) if trades else min(all_dates)
            dmax = max(all_dates)
            self.trading_day_count = len([d for d in all_dates if dmin <= d <= dmax])
        else:
            self.trading_day_count = 0

        return trades

    # ------------------------------------------------------------------
    # Walk-forward窗口内的截面回测
    # ------------------------------------------------------------------
    def _backtest_window_cross_sectional(self, window_histories: Dict[str, pd.DataFrame],
                                          weights: Dict[str, float],
                                          train_size: int, test_start: int, test_end: int,
                                          hold_days: int, max_positions: int,
                                          top_n: int, entry_score: float,
                                          regime_by_date: Dict[str, str] = None) -> List[Dict]:
        """Walk-forward 的单个测试窗口：对所有股票同一窗口内做截面回测"""
        all_signals: List[Dict] = []
        for sc, h in window_histories.items():
            # 在该窗口内逐日生成信号（test_start ~ test_end）
            for i in range(test_start, test_end):
                if i >= len(h) or i < train_size:
                    continue
                window = h.iloc[i - train_size:i + 1]
                row = h.iloc[i]
                close_col = "收盘" if "收盘" in h.columns else "close"
                close = row[close_col] if close_col in row.index else row.get("close", 0)
                if pd.isna(close) or close <= 0:
                    continue

                factors = self._calc_factors(window, sc)
                from src.engine.scorer import StockScorer
                scorer = StockScorer()
                sub_scores = scorer.get_sub_scores(factors)
                expectation_penalty = scorer.penalty_expectation(factors)

                date_str = self._date_of(h, i)

                all_signals.append({
                    "stock_code": sc,
                    "date": date_str,
                    "idx": i,
                    "price": float(close),
                    "factors": factors,
                    "sub_scores": sub_scores,
                    "expectation_penalty": expectation_penalty,
                })

        if not all_signals:
            return []

        self._score_cross_sectional(all_signals, weights, regime_by_date=regime_by_date)

        # 持仓模拟（exit_idx 相对各自 history 全长）
        return self._simulate_holding(
            all_signals, window_histories,
            hold_days=hold_days,
            max_positions=max_positions,
            top_n=top_n,
            entry_score=entry_score,
        )

    def _calc_factors(self, window: pd.DataFrame, stock_code: str) -> Dict:
        """计算单窗口因子"""
        from src.factors.technical import TechnicalFactors
        from src.factors.strategies import StockStrategies

        tech = TechnicalFactors(window, {})
        factors = tech.calculate_all()

        strategies = StockStrategies.get_triggered_strategies(window)
        factors["strategies_triggered"] = strategies
        factors["strategy_count"] = len(strategies)

        chip_col = "收盘" if "收盘" in window.columns else "close"
        factors["chip_avg_cost"] = float(window[chip_col].mean())
        return factors

    def _calculate_results(self, trades: List[Dict]) -> Dict:
        """计算完整绩效指标。

        指标口径说明（原实现把"逐笔交易收益序列"当"日收益率序列"使用，
        年化/Sharpe/Sortino/Calmar 四个指标全部失真）：
          - 资金曲线按 exit_date 聚合每日已实现盈亏逐日推进
          - 日收益率 = 当日盈亏 / 前日资金
          - 年化用回测覆盖的真实交易日数(self.trading_day_count，由
            _simulate_holding 统计)，而非交易笔数
        """
        if not trades:
            return {"error": "无交易记录"}

        total_trades = len(trades)
        win_trades = [t for t in trades if t["return_pct"] > 0]
        lose_trades = [t for t in trades if t["return_pct"] <= 0]
        win_count = len(win_trades)
        lose_count = len(lose_trades)

        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        total_return = (total_pnl / self.initial_capital) * 100

        avg_win = np.mean([t["pnl"] for t in win_trades]) if win_count > 0 else 0
        avg_lose = np.mean([t["pnl"] for t in lose_trades]) if lose_count > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else float('inf')

        # 日度资金曲线：按平仓日聚合盈亏（同一日多笔平仓合并为当日损益）
        from collections import defaultdict as _dd
        pnl_by_exit: Dict[str, float] = _dd(float)
        for t in trades:
            pnl_by_exit[t["exit_date"]] += t["pnl"]

        capital = self.initial_capital
        daily_returns: List[float] = []
        capital_curve = [capital]
        for d in sorted(pnl_by_exit.keys()):
            day_pnl = pnl_by_exit[d]
            if capital > 0:
                daily_returns.append(day_pnl / capital)
            capital += day_pnl
            capital_curve.append(capital)

        final_capital = capital
        # 与实际盈亏一致（浮点兜底）
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        capital_series = np.array(capital_curve)
        peak = np.maximum.accumulate(capital_series)
        drawdowns = (peak - capital_series) / peak * 100
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 1 else 0.0

        returns_arr = np.array(daily_returns)
        n_days = getattr(self, "trading_day_count", 0) or max(len(returns_arr), 1)
        if len(returns_arr) >= 2 and np.std(returns_arr) > 0:
            annual_return = ((1 + total_return / 100) ** (252 / max(1, n_days)) - 1) * 100
            sharpe = np.sqrt(252) * np.mean(returns_arr) / np.std(returns_arr)
            downside = [r for r in returns_arr if r < 0]
            sortino_ratio = np.sqrt(252) * np.mean(returns_arr) / (
                np.std(downside) if downside else 0.001
            )
        else:
            annual_return = 0
            sharpe = 0
            sortino_ratio = 0

        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        result = {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_capital": round(final_capital, 2),
                "total_return": round(total_return, 2),
                "annual_return": round(annual_return, 2),
                "total_pnl": round(total_pnl, 2),
                "total_trades": total_trades,
                "win_trades": win_count,
                "lose_trades": lose_count,
                "win_rate": round(win_rate, 2),
                "profit_loss_ratio": round(profit_loss_ratio, 2) if profit_loss_ratio != float('inf') else "N/A",
                "max_drawdown": round(max_drawdown, 2),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino_ratio, 3),
                "calmar_ratio": round(calmar, 3),
            },
            "trades": trades[-20:],
            "total_trades": total_trades,
        }
        return result


class ATRStopLoss:
    """ATR动态止损"""

    def __init__(self, atr_period: int = 14, multiplier: float = 2.0):
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["最高"] if "最高" in df.columns else df["high"]
        low = df["最低"] if "最低" in df.columns else df["low"]
        close = df["收盘"] if "收盘" in df.columns else df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        return atr

    def get_stop_loss(self, entry_price: float, atr: float, direction: str = "long") -> float:
        if direction == "long":
            stop_loss = entry_price - atr * self.multiplier
        else:
            stop_loss = entry_price + atr * self.multiplier
        return round(stop_loss, 2)

    def get_position_size(self, capital: float, risk_per_trade: float,
                         entry_price: float, stop_loss: float) -> int:
        risk_amount = capital * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        shares = int(risk_amount / risk_per_share / 100) * 100
        return max(100, shares)


class PositionManager:
    """仓位管理器"""

    def __init__(self, max_position_pct: float = 0.25, min_position_pct: float = 0.05,
                 max_stop_loss_pct: float = -0.08, volatility_adaptive: bool = True,
                 volatility_factor: float = 1.5):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_stop_loss_pct = max_stop_loss_pct  # ATR止损空间上限(小数)
        self.volatility_adaptive = volatility_adaptive  # 是否启用波动率自适应止损
        self.volatility_factor = volatility_factor  # 波动率调整系数

    def calculate_position(self, score: float, capital: float, price: float) -> Dict:
        if score >= 85:
            position_pct = self.max_position_pct
            risk_level = "高"
        elif score >= 75:
            position_pct = 0.20
            risk_level = "中高"
        elif score >= 65:
            position_pct = 0.15
            risk_level = "中"
        elif score >= 55:
            position_pct = 0.10
            risk_level = "中低"
        else:
            position_pct = self.min_position_pct
            risk_level = "低"

        position_amount = capital * position_pct
        shares = int(position_amount / price / 100) * 100

        return {
            "position_pct": position_pct,
            "position_amount": round(position_amount, 2),
            "shares": max(100, shares),
            "risk_level": risk_level,
        }

    def calculate_stop_loss(self, entry_price: float, atr: float, score: float,
                           volatility_ratio: float = 1.0) -> Dict:
        """
        计算动态止损
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            score: 综合评分
            volatility_ratio: 波动率比率 (当前波动率/历史平均波动率)
        """
        if score >= 85:
            multiplier = 2.5
        elif score >= 75:
            multiplier = 2.0
        elif score >= 65:
            multiplier = 1.5
        else:
            multiplier = 1.0

        # 波动率自适应调整：高波动率时放宽止损，低波动率时收紧止损
        if self.volatility_adaptive and volatility_ratio > 0:
            # 波动率翻倍，止损放宽 factor 倍
            volatility_adjustment = 1 + (volatility_ratio - 1) * (self.volatility_factor - 1)
            multiplier *= volatility_adjustment

        stop_loss = entry_price - atr * multiplier
        stop_loss_pct = (stop_loss / entry_price - 1) * 100

        # 止损空间上限:防止高价股/高ATR票止损被设得过宽(如协创数据-11.4%被穿透到-22.8%)
        max_sl_pct = self.max_stop_loss_pct * 100  # 转百分数,如 -8.0
        if stop_loss_pct < max_sl_pct:
            stop_loss_pct = max_sl_pct
            stop_loss = round(entry_price * (1 + self.max_stop_loss_pct), 2)

        return {
            "stop_loss": round(stop_loss, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "atr_multiplier": round(multiplier, 2),
            "volatility_ratio": round(volatility_ratio, 2),
        }

    def calculate_take_profit(self, entry_price: float, atr: float, score: float) -> Dict:
        if score >= 85:
            multiplier = 4.0
        elif score >= 75:
            multiplier = 3.0
        elif score >= 65:
            multiplier = 2.5
        else:
            multiplier = 2.0

        take_profit = entry_price + atr * multiplier
        take_profit_pct = (take_profit / entry_price - 1) * 100

        return {
            "take_profit": round(take_profit, 2),
            "take_profit_pct": round(take_profit_pct, 2),
            "atr_multiplier": multiplier,
        }
