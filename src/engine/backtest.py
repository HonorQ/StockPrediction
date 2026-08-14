"""
回测系统 - 向量化Walk-Forward回测 + 17维因子评分 + 完整绩效指标

评分统一说明：
    生产环境（run_daily.py）通过 StockScorer.batch_normalize 做截面 Z-Score 归一化，
    保证不同交易日的得分可比。本回测引擎亦使用同一套 batch_normalize 流程，
    在每个交易日对所有候选股票做截面评分，避免回测分数与生产分数口径不一致。
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable, Optional
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

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run_walk_forward(self, stocks: List[str], weights: Dict[str, float],
                         train_window: int = 120, test_window: int = 20,
                         hold_days: int = 3, max_positions: int = 5,
                         top_n: int = 5,
                         entry_score: float = DEFAULT_ENTRY_SCORE) -> Dict:
        """
        Walk-Forward回测（截面评分统一版）
        train_window: 训练窗口天数（用于因子计算时的lookback）
        test_window: 测试窗口天数
        流程: [train 120天 | test 20天] → 滚动 → [train 120天 | test 20天]
        每个测试窗口内：逐日收集所有股票的因子，按日做 batch_normalize 截面评分。
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

            trades = self._backtest_window_cross_sectional(
                window_histories, weights,
                train_size=train_window,
                test_start=test_start,
                test_end=test_end,
                hold_days=hold_days,
                max_positions=max_positions,
                top_n=top_n,
                entry_score=entry_score,
            )
            all_trades.extend(trades)
            windows += 1

        if not all_trades:
            return {"error": "无交易信号"}

        return self._calculate_results(all_trades)

    def run(self, stocks: List[str], weights: Dict[str, float],
            hold_days: int = 3, max_positions: int = 5,
            top_n: int = 5, benchmark_code: str = "sh000300",
            entry_score: float = DEFAULT_ENTRY_SCORE) -> Dict:
        """
        标准回测（全量数据，截面评分统一版）
        与生产环境 StockScorer.batch_normalize 完全一致：
            1. 为每只股票逐日生成原始因子 + sub_scores
            2. 按交易日分组，调用 batch_normalize 做截面 Z-Score 归一化
            3. 每日选 top_n 入场，持仓 hold_days 后平仓
        """
        # 1. 拉取所有股票的历史K线
        stock_histories: Dict[str, pd.DataFrame] = {}
        for sc in stocks:
            h = self._get_history(sc)
            if h is not None and len(h) >= 60:
                stock_histories[sc] = h

        if not stock_histories:
            return {"error": "无可用历史数据"}

        # 2. 全量逐日生成信号
        all_signals: List[Dict] = []
        for sc, h in stock_histories.items():
            sigs = self._generate_signals(sc, h, train_size=60)
            all_signals.extend(sigs)

        if not all_signals:
            return {"error": "无交易信号"}

        # 3. 截面评分（与生产 batch_normalize 一致）
        self._score_cross_sectional(all_signals, weights)

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

            date_val = history.index[i]
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)

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
    def _score_cross_sectional(signals: List[Dict], weights: Dict[str, float]) -> None:
        """按交易日分组，对每组信号调用 batch_normalize 做截面 Z-Score 评分。
        样本数 < MIN_CROSS_SECTIONAL_SAMPLES 时降级为单股票 calculate_score。
        直接修改 signals 中每一项的 'score' 字段。
        """
        from src.engine.scorer import StockScorer
        scorer = StockScorer(weights)

        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for sig in signals:
            by_date[sig["date"]].append(sig)

        for date, group in by_date.items():
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
                StockScorer.batch_normalize(candidates, weights)
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
        """对每个交易日的截面评分结果选 top_n，模拟持仓 hold_days 后平仓"""
        from collections import defaultdict

        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for sig in signals:
            by_date[sig["date"]].append(sig)

        capital_per_stock = self.initial_capital / max_positions if max_positions > 0 else self.initial_capital
        trades: List[Dict] = []

        for date in sorted(by_date.keys()):
            group = by_date[date]
            # 过滤入场阈值，按分排序取 top_n
            qualified = [s for s in group if s["score"] >= entry_score]
            qualified.sort(key=lambda x: -x["score"])
            picks = qualified[:top_n]

            for sig in picks:
                sc = sig["stock_code"]
                history = histories.get(sc)
                if history is None:
                    continue
                entry_idx = sig["idx"]
                exit_idx = min(entry_idx + hold_days, len(history) - 1)
                close_col = "收盘" if "收盘" in history.columns else "close"

                entry_price = sig["price"]
                exit_row = history.iloc[exit_idx]
                exit_price = exit_row[close_col] if close_col in exit_row.index else exit_row.get("close", 0)
                if pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
                    continue

                ret = (exit_price / entry_price - 1) * 100
                shares = int(capital_per_stock / entry_price / 100) * 100
                if shares <= 0:
                    continue
                pnl = shares * entry_price * ret / 100

                exit_date_val = history.index[exit_idx]
                exit_date_str = exit_date_val.strftime("%Y-%m-%d") if hasattr(exit_date_val, "strftime") else str(exit_date_val)

                trades.append({
                    "stock_code": sc,
                    "entry_date": date,
                    "exit_date": exit_date_str,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(float(exit_price), 2),
                    "return_pct": round(ret, 2),
                    "pnl": round(pnl, 2),
                    "score": round(sig["score"], 1),
                    "shares": shares,
                })

        return trades

    # ------------------------------------------------------------------
    # Walk-forward窗口内的截面回测
    # ------------------------------------------------------------------
    def _backtest_window_cross_sectional(self, window_histories: Dict[str, pd.DataFrame],
                                          weights: Dict[str, float],
                                          train_size: int, test_start: int, test_end: int,
                                          hold_days: int, max_positions: int,
                                          top_n: int, entry_score: float) -> List[Dict]:
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

                date_val = h.index[i]
                date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)

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

        self._score_cross_sectional(all_signals, weights)

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
        """计算完整绩效指标"""
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
        final_capital = self.initial_capital + total_pnl

        avg_win = np.mean([t["pnl"] for t in win_trades]) if win_count > 0 else 0
        avg_lose = np.mean([t["pnl"] for t in lose_trades]) if lose_count > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else float('inf')

        trades_sorted = sorted(trades, key=lambda x: x["entry_date"])
        cumulative = 0
        capital_curve = [self.initial_capital]
        for t in trades_sorted:
            cumulative += t["pnl"]
            capital_curve.append(self.initial_capital + cumulative)

        capital_series = np.array(capital_curve)
        peak = np.maximum.accumulate(capital_series)
        drawdowns = (peak - capital_series) / peak * 100
        max_drawdown = float(np.max(drawdowns))

        daily_returns = []
        for t in trades_sorted:
            daily_returns.append(t["pnl"] / self.initial_capital)
        returns_arr = np.array(daily_returns)
        if len(returns_arr) >= 2 and np.std(returns_arr) > 0:
            annual_return = ((1 + total_return / 100) ** (252 / max(1, len(returns_arr))) - 1) * 100
            sharpe = np.sqrt(252) * np.mean(returns_arr) / np.std(returns_arr)
            sortino_ratio = np.sqrt(252) * np.mean(returns_arr) / (
                np.std([r for r in returns_arr if r < 0]) if any(r < 0 for r in returns_arr) else 0.001
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
                 max_stop_loss_pct: float = -0.08):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_stop_loss_pct = max_stop_loss_pct  # ATR止损空间上限(小数)

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

    def calculate_stop_loss(self, entry_price: float, atr: float, score: float) -> Dict:
        if score >= 85:
            multiplier = 2.5
        elif score >= 75:
            multiplier = 2.0
        elif score >= 65:
            multiplier = 1.5
        else:
            multiplier = 1.0

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
            "atr_multiplier": multiplier,
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
