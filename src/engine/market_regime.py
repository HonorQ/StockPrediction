"""
大盘择时层（Market Regime Filter）

根据大盘走势、市场情绪、北向资金、外盘环境判断当前市场状态，
动态调整最大推荐数量和风险系数。避免在系统性下跌期间仍然推荐股票。

状态：
  bull      - 强势上行（满仓推荐）
  neutral   - 震荡中性（正常推荐）
  bear      - 弱势下行（缩减推荐，提高门槛）
  crash     - 系统性风险（仅保留最强 1 只或全部观望）
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd


class MarketRegimeFilter:
    """大盘择时过滤器"""

    # regime 到最大推荐数的映射（最终值 = min(config_max, regime_max)）
    REGIME_MAX_RECOMMEND = {
        "bull": 5,       # 强势满仓
        "neutral": 3,    # 正常
        "bear": 1,       # 弱势仅推 1 只
        "crash": 0,      # 系统性风险，全部观望
    }

    # regime 到风险调整系数的映射（影响仓位）
    REGIME_RISK_ADJUST = {
        "bull": 1.2,
        "neutral": 1.0,
        "bear": 0.5,
        "crash": 0.0,
    }

    def __init__(self, market_fetcher=None):
        self.market_fetcher = market_fetcher
        self._index_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    def assess_regime(
        self,
        market_overview: Dict,
        sentiment: Dict,
        northbound_data: Optional[Dict] = None,
        global_markets: Optional[Dict] = None,
    ) -> Dict:
        """
        评估当前市场状态。

        参数：
          market_overview: 主指数数据 {"上证指数": {"change": ...}, ...}
          sentiment: 情绪数据 {"limit_up":..., "limit_down":..., "broken":..., "max_board":...}
          northbound_data: 北向资金 {"net_inflow": ..., "consecutive_days": ...}
          global_markets: 外盘数据

        返回：
          {
            "regime": "bull|neutral|bear|crash",
            "max_recommend": int,
            "risk_adjustment": float,
            "score": float,         # 大盘综合得分 0-100
            "reasons": List[str],   # 触发原因
          }
        """
        reasons: List[str] = []
        score = 50.0  # 中性起点
        bear_signals = 0
        crash_signals = 0
        bull_signals = 0

        # === 1. 主指数涨跌 ===
        sh_data = market_overview.get("上证指数", {})
        sh_chg = sh_data.get("change", 0) if isinstance(sh_data, dict) else 0
        sz_data = market_overview.get("深证成指", {})
        sz_chg = sz_data.get("change", 0) if isinstance(sz_data, dict) else 0
        cyb_data = market_overview.get("创业板指", {})
        cyb_chg = cyb_data.get("change", 0) if isinstance(cyb_data, dict) else 0

        if sh_chg <= -2.0 or sz_chg <= -2.5 or cyb_chg <= -3.0:
            crash_signals += 1
            score -= 25
            reasons.append(f"主指数大跌 上证{sh_chg:+.2f}% 创业板{cyb_chg:+.2f}%")
        elif sh_chg <= -1.0 or cyb_chg <= -1.5:
            bear_signals += 1
            score -= 12
            reasons.append(f"主指数下跌 上证{sh_chg:+.2f}% 创业板{cyb_chg:+.2f}%")
        elif sh_chg >= 1.0 and cyb_chg >= 1.5:
            bull_signals += 1
            score += 12
            reasons.append(f"主指数上涨 上证{sh_chg:+.2f}% 创业板{cyb_chg:+.2f}%")

        # === 2. 涨跌停结构 ===
        limit_up = sentiment.get("limit_up", 0) or 0
        limit_down = sentiment.get("limit_down", 0) or 0
        broken = sentiment.get("broken", 0) or 0
        max_board = sentiment.get("max_board", 0) or 0

        if limit_down >= 30:
            crash_signals += 1
            score -= 15
            reasons.append(f"跌停潮 {limit_down}家")
        elif limit_down >= 15:
            bear_signals += 1
            score -= 8
            reasons.append(f"跌停较多 {limit_down}家")
        elif limit_up >= 50 and max_board >= 4:
            bull_signals += 1
            score += 8
            reasons.append(f"涨停 {limit_up}家 最高{max_board}板")

        # 炸板率高 = 情绪退潮
        total_attacks = limit_up + broken
        if total_attacks > 0 and broken / total_attacks > 0.4 and broken >= 8:
            bear_signals += 1
            score -= 6
            reasons.append(f"炸板率高 {broken}/{total_attacks}")

        # === 3. 北向资金 ===
        if northbound_data:
            nb_net = northbound_data.get("net_inflow", 0) or 0
            nb_days = northbound_data.get("consecutive_days", 0) or 0
            # net_inflow 单位：亿元
            if nb_net <= -80:
                crash_signals += 1
                score -= 12
                reasons.append(f"北向大幅净流出 {nb_net:.0f}亿")
            elif nb_net <= -30 and nb_days >= 3:
                bear_signals += 1
                score -= 8
                reasons.append(f"北向连续{nb_days}日净流出 {nb_net:.0f}亿")
            elif nb_net >= 50:
                bull_signals += 1
                score += 6
                reasons.append(f"北向净流入 {nb_net:.0f}亿")

        # === 4. 大盘趋势（用上证指数 MA20 判断）===
        if self.market_fetcher is not None:
            sh_trend = self._assess_index_trend("sh000001")
            if sh_trend == "below_ma20_falling":
                bear_signals += 1
                score -= 10
                reasons.append("上证跌破MA20且下行")
            elif sh_trend == "below_ma20":
                bear_signals += 1
                score -= 5
                reasons.append("上证运行于MA20之下")
            elif sh_trend == "above_ma20_rising":
                bull_signals += 1
                score += 5
                reasons.append("上证站稳MA20且上行")

        # === 5. 外盘联动 ===
        if global_markets:
            ext_crash_count = 0
            for name, data in global_markets.items():
                if not isinstance(data, dict):
                    continue
                chg = data.get("change", 0) or 0
                if chg <= -2.0:
                    ext_crash_count += 1
            if ext_crash_count >= 2:
                bear_signals += 1
                score -= 5
                reasons.append(f"外盘普跌 {ext_crash_count}个市场大跌")

        # === 决策 ===
        score = max(0, min(100, score))
        if crash_signals >= 2 or score <= 20:
            regime = "crash"
        elif bear_signals >= 2 or score <= 35:
            regime = "bear"
        elif bull_signals >= 2 or score >= 65:
            regime = "bull"
        else:
            regime = "neutral"

        return {
            "regime": regime,
            "max_recommend": self.REGIME_MAX_RECOMMEND[regime],
            "risk_adjustment": self.REGIME_RISK_ADJUST[regime],
            "score": round(score, 1),
            "reasons": reasons,
            "bear_signals": bear_signals,
            "bull_signals": bull_signals,
            "crash_signals": crash_signals,
        }

    # ------------------------------------------------------------------
    def _assess_index_trend(self, code: str) -> str:
        """判断指数相对MA20的趋势。
        返回值：above_ma20_rising / above_ma20 / below_ma20 / below_ma20_falling / unknown
        """
        try:
            if code not in self._index_cache:
                if self.market_fetcher is None:
                    return "unknown"
                history = self.market_fetcher.get_stock_history(code, days=60)
                if history is None or history.empty or len(history) < 25:
                    return "unknown"
                self._index_cache[code] = history

            history = self._index_cache[code]
            close_col = "收盘" if "收盘" in history.columns else "close"
            closes = history[close_col].astype(float)
            if len(closes) < 25:
                return "unknown"

            ma20 = closes.rolling(window=20).mean().iloc[-1]
            current = closes.iloc[-1]
            prev = closes.iloc[-2]
            prev_ma20 = closes.iloc[-2:].rolling(window=20).mean().iloc[-1] if len(closes) >= 21 else ma20

            if current > ma20 and ma20 > prev_ma20:
                return "above_ma20_rising"
            elif current > ma20:
                return "above_ma20"
            elif current < ma20 and ma20 < prev_ma20:
                return "below_ma20_falling"
            else:
                return "below_ma20"
        except Exception:
            return "unknown"
