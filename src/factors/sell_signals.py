"""
卖出信号检测 - 技术面/时间面/风控面
"""
import pandas as pd
from typing import Dict, List, Optional


class SellSignalDetector:
    """卖出信号检测器"""

    @staticmethod
    def detect_all(df: pd.DataFrame, current_price: float, entry_price: float,
                   hold_days: int = 0, entry_date: Optional[str] = None) -> List[Dict]:
        """
        检测所有卖出信号
        df: 历史K线数据
        current_price: 当前价
        entry_price: 买入价
        hold_days: 已持仓天数
        """
        if df is None or df.empty or len(df) < 20:
            return []

        signals = []

        signals.extend(SellSignalDetector._technical_signals(df, current_price))
        signals.extend(SellSignalDetector._time_signals(hold_days, entry_date))
        signals.extend(SellSignalDetector._risk_signals(current_price, entry_price))

        return signals

    @staticmethod
    def _technical_signals(df: pd.DataFrame, current_price: float) -> List[Dict]:
        """技术面卖出信号"""
        signals = []

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        df["最高"] if "最高" in df.columns else df["high"]
        df["最低"] if "最低" in df.columns else df["low"]

        if len(close) < 20:
            return signals

        # === MA死叉：MA5跌破MA10 ===
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        prev_ma5 = ma5.iloc[-2] if len(ma5) > 1 else 0
        prev_ma10 = ma10.iloc[-2] if len(ma10) > 1 else 0
        curr_ma5 = ma5.iloc[-1]
        curr_ma10 = ma10.iloc[-1]
        if curr_ma5 < curr_ma10 and prev_ma5 >= prev_ma10:
            signals.append({
                "type": "technical", "reason": "MA5死叉MA10",
                "strength": "strong", "action": "卖出"
            })

        # === MACD死叉：DIF跌破DEA ===
        # adjust=False 与 technical.py 的 MACD 口径对齐（默认 adjust=True
        # 会用全样本权重重算历史，两处信号可能不同步）
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        if len(dif) > 1 and len(dea) > 1:
            prev_dif = dif.iloc[-2]
            prev_dea = dea.iloc[-2]
            curr_dif = dif.iloc[-1]
            curr_dea = dea.iloc[-1]
            if curr_dif < curr_dea and prev_dif >= prev_dea:
                signals.append({
                    "type": "technical", "reason": "MACD死叉",
                    "strength": "strong", "action": "卖出"
                })

        # === 跌破20日均线 ===
        ma20 = close.rolling(20).mean()
        if len(ma20) > 0 and len(close) > 1:
            if close.iloc[-1] < ma20.iloc[-1] and close.iloc[-2] >= ma20.iloc[-2]:
                signals.append({
                    "type": "technical", "reason": "跌破20日均线",
                    "strength": "medium", "action": "减仓"
                })

        # === 量价背离：价格上涨但成交量萎缩 ===
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        if vol_col in df.columns and len(df) > 5:
            vol = df[vol_col].iloc[-5:]
            price_change = close.iloc[-1] / close.iloc[-5] - 1
            vol_change = vol.iloc[-1] / vol.iloc[0] - 1
            if price_change > 0.03 and vol_change < -0.3:
                signals.append({
                    "type": "technical", "reason": "量价背离",
                    "strength": "medium", "action": "减仓观察"
                })

        # === 高开低走（出货信号）===
        open_col = "开盘" if "开盘" in df.columns else "open"
        if open_col in df.columns:
            last = df.iloc[-1]
            last_open = last[open_col]
            last_close = last[close.name] if isinstance(close, pd.Series) else last.get("收盘", last.get("close", 0))
            if last_open > last_close * 1.02 and (last_open - last_close) / last_open > 0.03:
                signals.append({
                    "type": "technical", "reason": "高开低走",
                    "strength": "medium", "action": "减仓"
                })

        return signals

    @staticmethod
    def _time_signals(hold_days: int, entry_date: Optional[str] = None) -> List[Dict]:
        """时间面卖出信号"""
        signals = []

        if hold_days >= 7:
            signals.append({
                "type": "time", "reason": f"持仓{hold_days}天，超时卖出",
                "strength": "strong", "action": "卖出"
            })
        elif hold_days >= 5:
            signals.append({
                "type": "time", "reason": f"持仓{hold_days}天，接近超时",
                "strength": "medium", "action": "减仓准备"
            })

        return signals

    @staticmethod
    def _risk_signals(current_price: float, entry_price: float) -> List[Dict]:
        """风控面卖出信号"""
        signals = []

        if entry_price <= 0:
            return signals

        return_pct = (current_price / entry_price - 1) * 100

        if return_pct <= -7:
            signals.append({
                "type": "risk", "reason": f"止损-7%（实际{return_pct:.1f}%）",
                "strength": "critical", "action": "强制卖出"
            })
        elif return_pct <= -5:
            signals.append({
                "type": "risk", "reason": f"触发止损线（{return_pct:.1f}%）",
                "strength": "strong", "action": "卖出"
            })

        if return_pct >= 10:
            signals.append({
                "type": "risk", "reason": f"触发止盈线+10%（实际{return_pct:.1f}%）",
                "strength": "strong", "action": "卖出"
            })

        return signals

    @staticmethod
    def get_sell_advice(signals: List[Dict]) -> str:
        """根据信号综合判断卖出建议"""
        if not signals:
            return "持有观望"

        has_critical = any(s["strength"] == "critical" for s in signals)
        has_strong_sell = any(s["strength"] == "strong" and s["action"] == "卖出" for s in signals)

        if has_critical:
            return "🔴 强制卖出"
        if has_strong_sell:
            return "🟡 建议卖出"
        if any(s["action"] == "减仓" for s in signals):
            return "🟢 建议减仓"

        return "持有观望"
