"""
大盘择时层（Market Regime Filter）

根据大盘走势、市场情绪、北向资金、外盘环境判断当前市场状态，
动态调整最大推荐数量、风险系数和因子权重。避免在系统性下跌期间仍然推荐股票。

状态：
  bull      - 强势上行（满仓推荐，偏进攻因子）
  neutral   - 震荡中性（正常推荐）
  bear      - 弱势下行（缩减推荐，偏防守因子）
  crash     - 系统性风险（仅保留最强 1 只或全部观望）
"""
from typing import Dict, List, Optional
import pandas as pd
import os
import yaml
import logging

logger = logging.getLogger(__name__)


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
        self._regime_config = self._load_regime_config()

    def _load_regime_config(self) -> Dict:
        """加载regime权重配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("regime", {})
        except Exception as e:
            logger.warning(f"加载regime配置失败: {e}")
            return {}

    def get_regime_weights(self, regime: str, base_weights: Dict[str, float]) -> Dict[str, float]:
        """根据regime返回调整后的因子权重（部分覆盖语义：仅覆盖列出的因子）

        Args:
            regime: 市场状态 (bull/neutral/bear/crash)
            base_weights: 基准权重

        Returns:
            调整后的权重字典
        """
        if not self._regime_config.get("enabled", False):
            return base_weights

        regime_overrides = self._regime_config.get("weights", {}).get(regime, {})
        if not regime_overrides:
            return base_weights

        adjusted = base_weights.copy()
        # 只覆盖 base_weights 中已存在的评分维度。
        # config.yaml 的 regime.weights 中含 24 维评分体系之外的 key(如 volatility)，
        # 若新增进去会成为"恒为50分的废维度"(sub_scores 无此维，z 恒为 0)，
        # 且违反权重-评分维度一致性原则(_validate_weights_consistency)，故跳过并告警。
        skipped = []
        for factor, weight in regime_overrides.items():
            if factor in adjusted:
                adjusted[factor] = weight
            else:
                skipped.append(factor)
        if skipped:
            logger.warning(
                "get_regime_weights(%s): 忽略不在评分体系中的权重维度: %s",
                regime, sorted(skipped),
            )
        # 总和校验(2026-09-11): 覆盖发生在 scorer 归一化之后、batch_normalize 之前,
        # 无人把关的总和漂移会直接放大评分尺度(旧覆盖段曾致 bear 总和 1.277)。
        # 超差时等比缩放回基准总和(生产中基准恒为 1.0), 保持覆盖的相对结构。
        _base_total = sum(float(v) for v in base_weights.values())
        _total = sum(float(v) for v in adjusted.values())
        if _total > 0 and abs(_total - _base_total) > 0.02:
            _scale = _base_total / _total
            logger.warning(
                "get_regime_weights(%s): 覆盖致权重总和 %.4f→%.4f，已等比缩放回基准",
                regime, _base_total, _total,
            )
            adjusted = {k: v * _scale for k, v in adjusted.items()}
        return adjusted

    def should_empty_portfolio(self, regime: str, score: float) -> bool:
        """判断是否应该空仓"""
        if not self._regime_config.get("enabled", False):
            return False
        if not self._regime_config.get("empty_on_bear", True):
            return False
        if regime in ("crash",):
            return True
        if regime == "bear" and score <= self._regime_config.get("bear_score_threshold", 35):
            return True
        return False

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
            # 昨日的MA20：对全序列 shift(1) 后取20窗口均值。
            # 原写法 closes.iloc[-2:].rolling(20) 对最后2个元素做20窗口 → 恒NaN，
            # 趋势上升/下降分支永远不可达
            prev_ma20 = closes.shift(1).rolling(window=20).mean().iloc[-1] if len(closes) >= 21 else ma20

            if current > ma20 and ma20 > prev_ma20:
                return "above_ma20_rising"
            elif current > ma20:
                return "above_ma20"
            elif current < ma20 and ma20 < prev_ma20:
                return "below_ma20_falling"
            else:
                return "below_ma20"
        except Exception as e:
            logger.warning(f"判断MA20趋势失败: {e}")
            return "unknown"
