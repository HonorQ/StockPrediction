"""
综合评分器 - 24维度评分（含护城河moat+深度研究+板块联动+资金流向+机构调研+龙虎榜+龙头检测）

维度清单见 get_sub_scores；config.yaml weights 必须与此处24维一一对应，
启动时由 run_daily._validate_weights_consistency 强制校验。
"""
from typing import Dict, List
import logging
import os
import pandas as pd

logger = logging.getLogger("stock_prediction")


class StockScorer:
    """股票综合评分器"""

    # 预定义高相关因子对：(target, regressor) — 对 target 做残差化以剥离与 regressor 的共线部分
    # 基于代码分析：trend/strategy 共享 MACD/MA 信号，ashare_specific/relative_strength 共享 RSI 等
    ORTHOGONALIZE_PAIRS = [
        ("strategy", "trend"),               # K线形态与趋势共享均线信号
        ("ashare_specific", "relative_strength"),  # A股特色与相对强弱共享RSI/超卖
        ("oversold_rebound", "ashare_specific"),   # 超跌反弹与A股特色共享超卖信号
        ("deep_research", "chip"),            # 深度研究与筹码共享60日高低点位置
    ]

    # 新架构(2026-08-27): 仅ashare_specific参与评分, 其余因子权重=0
    # NEGATIVE_IC_DIMS 仅在 invert_signals=True 时生效, 新架构下无需反转
    NEGATIVE_IC_DIMS = ()

    def __init__(self, weights: Dict[str, float] = None,
                 invert_signals: bool = False):
        self.invert_signals = invert_signals
        self.weights = weights or {
            # K线类: 11% (从26%大幅降低，因为K线可被主力操纵)
            "trend": 0.03,
            "strategy": 0.02,
            "chip": 0.03,
            "ashare_specific": 0.02,
            "relative_strength": 0.01,

            # 资金行为: 15% (核心跟踪主力)
            "quant_behavior": 0.06,
            "capital_behavior": 0.09,

            # 基本面+事件: 28% (真正驱动股价)
            "fundamental": 0.09,
            "policy_event": 0.07,
            "news": 0.07,
            "deep_research": 0.05,

            # 其他: 46%
            "sector_rotation": 0.05,
            "institutional": 0.05,
            "dragon_tiger": 0.04,
            "national_team": 0.04,
            "oversold_rebound": 0.02,
            "leader": 0.03,
            "margin_trading": 0.02,
            "shareholder": 0.04,
            "lockup_risk": 0.04,
            "research": 0.01,
            "moat": 0.04,
            "holder_concentration": 0.05,
            "circulate_holder": 0.03,
        }
        # 权重和保护：原默认表总和=1.10、config.yaml=0.9821，
        # 尺度漂移会让 entry_score/min_score 阈值的实际含义失真。
        # 相对结构不变，等比缩放回总和=1。
        _total = sum(self.weights.values())
        if abs(_total - 1.0) > 1e-9:
            logger.warning(
                "权重总和=%.4f ≠ 1.0，已按比例归一化(×%.4f)",
                _total, 1.0 / _total,
            )
            self.weights = {k: v / _total for k, v in self.weights.items()}

    def get_sub_scores(self, factors: Dict) -> Dict[str, float]:
        sub = {
            "trend": self._score_trend(factors),
            "strategy": self._score_strategy(factors),
            "chip": self._score_chip(factors),
            "fundamental": self._score_fundamental(factors),
            "institutional": self._score_institutional(factors),
            "sector_rotation": self._score_sector_rotation(factors),
            "oversold_rebound": self._score_oversold_rebound(factors),
            "news": self._score_news(factors),
            "relative_strength": self._score_relative_strength(factors),
            "ashare_specific": self._score_ashare_specific(factors),
            "quant_behavior": self._score_quant_behavior(factors),
            "margin_trading": self._score_margin_trading(factors),
            "shareholder": self._score_shareholder(factors),
            "lockup_risk": self._score_lockup_risk(factors),
            "research": self._score_research(factors),
            "dragon_tiger": self._score_dragon_tiger(factors),
            "leader": self._score_leader(factors),
            "deep_research": self._score_deep_research(factors),
            "policy_event": self._score_policy_event(factors),
            "national_team": self._score_national_team(factors),
            "capital_behavior": self._score_capital_behavior(factors),
            "moat": self._score_moat(factors),
            "holder_concentration": self._score_holder_concentration(factors),
            "circulate_holder": self._score_circulate_holder(factors),
        }
        if self.invert_signals:
            for dim in self.NEGATIVE_IC_DIMS:
                sub[dim] = round(100.0 - sub[dim], 2)
        return sub

    def _score_moat(self, factors: Dict) -> float:
        """护城河/不可替代性子评分，直接取 MoatFactors 合成分（已在 0-100）。"""
        return float(factors.get("moat_score", 50))

    def calculate_score(self, factors: Dict[str, float]) -> float:
        """计算综合评分 (0-100) 单只股票版本（兼容旧调用）"""
        sub_scores = self.get_sub_scores(factors)

        total = 0
        for key, weight in self.weights.items():
            total += sub_scores.get(key, 50) * weight

        # 预期透支惩罚（一票否决）
        expectation_penalty = self.penalty_expectation(factors)
        total += expectation_penalty

        return round(max(0, min(100, total)), 1)

    @staticmethod
    def batch_normalize(candidates: List[Dict], weights: Dict[str, float],
                        extra_ortho_pairs=None):
        """截面Z-Score归一化：对一批候选的24维子评分做标准化，计算最终得分
        candidates: list of dicts, 每项需包含 'sub_scores' 和 'expectation_penalty'
        修改每个candidate，添加 'score' 和 'z_scores'
        extra_ortho_pairs: 动态正交化对 [(target, regressor)]，
            来自 config/dynamic_ortho_pairs.json(feedback_loop IC 分析产出)。
            不再允许运行时修改类属性——那是进程级全局副作用，重启即丢、
            回测与生产行为漂移。

        评分口径说明: score = 50 + z_total*(50/3)，z 截断 ±3。
        min_score 门槛在此口径下含义是"z_total ≥ (min_score-50)*3/50"，
        如 min_score=53 仅相当于 z_total≈0.18(极宽松)——真正的数量约束
        是 max_recommend；调 min_score 时应按此换算预期通过率。

        新增：在 Z-Score 之前对预定义的高相关因子对做残差化正交处理，
        防止共线因子在加权时被重复计算（双重 beta 暴露）。
        """
        dims = [k for k in weights.keys() if weights.get(k, 0) > 0]
        if not dims or not candidates:
            return

        ortho_pairs = list(StockScorer.ORTHOGONALIZE_PAIRS) + list(extra_ortho_pairs or [])

        raw = {dim: [] for dim in dims}
        for c in candidates:
            ss = c.get("sub_scores", {})
            for dim in dims:
                raw[dim].append(ss.get(dim, 50))

        # 截尾处理：去除极端值，防止1-2只异常股扭曲整个截面评分
        for dim in dims:
            series = pd.Series(raw[dim])
            raw[dim] = series.clip(
                series.quantile(0.05), series.quantile(0.95)
            ).tolist()

        # === 因子正交化：对高相关因子对做残差化 ===
        # 仅在样本数 >= 10 时执行，避免小样本下回归不稳定
        if len(candidates) >= 10:
            for target, regressor in ortho_pairs:
                if target not in raw or regressor not in raw:
                    continue
                t_vals = raw[target]
                r_vals = raw[regressor]
                # 计算 OLS beta = cov(t,r) / var(r)
                n = len(t_vals)
                mu_t = sum(t_vals) / n
                mu_r = sum(r_vals) / n
                cov_tr = sum((t - mu_t) * (r - mu_r) for t, r in zip(t_vals, r_vals)) / n
                var_r = sum((r - mu_r) ** 2 for r in r_vals) / n
                if var_r < 1e-6:
                    continue
                beta = cov_tr / var_r
                # 残差化: t_new = t - beta * r (保持均值不变)
                raw[target] = [t - beta * (r - mu_r) for t, r in zip(t_vals, r_vals)]

        for i, c in enumerate(candidates):
            z_total = 0.0
            z_scores = {}
            for dim in dims:
                vals = raw[dim]
                mu = sum(vals) / len(vals)
                sigma = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                if sigma < 1e-6:
                    z = 0.0
                else:
                    z = max(-3.0, min(3.0, (vals[i] - mu) / sigma))
                z_scores[dim] = round(z, 3)
                z_total += z * weights[dim]

            score = 50 + z_total * (50 / 3)
            penalty = c.get("expectation_penalty", 0)
            score += penalty
            c["score"] = round(max(0, min(100, score)), 1)
            c["z_scores"] = z_scores

    def penalty_expectation(self, f: Dict) -> float:
        """预期透支/风险信号深度惩罚。

        各分项自带封顶(单项最多-20)，叠加最坏约-75，构成实质性一票否决；
        旧实现人为截断到-15使 docstring 与行为不符。
        """
        penalty = 0

        # 预期透支
        if f.get("expectation_overvalued"):
            penalty -= 20
        expectation_score = f.get("expectation_score", 50)
        if expectation_score < 30:
            penalty -= 15

        # 利好兑现
        if f.get("event_priced"):
            pricing_degree = f.get("event_pricing_degree", 0)
            penalty -= min(pricing_degree / 5, 20)

        # 获利盘出逃
        if f.get("profit_taking_detected"):
            intensity = f.get("profit_taking_intensity", 0)
            penalty -= min(intensity / 5, 20)

        return max(penalty, -75)

    def _score_trend(self, f: Dict) -> float:
        score = 50
        if f.get("macd_golden_cross") == 1:
            score += 15
        elif f.get("macd_golden_cross") == -1:
            score -= 15
        if f.get("ma_bullish") == 1:
            score += 12
        if f.get("macd_above_zero") == 1:
            score += 5
        if f.get("breakout_20d_high") == 1:
            score += 10
        if f.get("volume_price_up") == 1:
            score += 8
        rsi = f.get("rsi_6", 50)
        if 40 <= rsi <= 60:
            score += 0
        elif rsi > 75:
            score -= 8
        elif rsi < 25:
            score += 8
        md = f.get("momentum_decay", 0)
        score -= md * 15
        return max(0, min(100, score))

    def _score_strategy(self, f: Dict) -> float:
        score = 50
        strategies = f.get("strategies_triggered", [])
        strategy_scores = {
            "放量上涨": 10, "均线多头": 8, "停机坪": 6, "回踩年线": 6,
            "突破平台": 8, "无大幅回撤": 5, "海龟交易": 6, "高窄旗形": 5,
            "旭日东升": 8, "多方炮": 7, "拨云见日": 7, "早晨之星": 7,
            "曙光初现": 6, "穿头破脚": 5, "三连阳": 5, "五连阳": 6,
            "连涨放量": 5, "天量突破": 7, "窄幅整理": 0,
            "锤头线": 7, "十字星": 0, "蜻蜓十字": 6, "三白兵": 9,
            "南方三星": 6, "上升三法": 8, "弃婴": 8, "孕线十字": 0,
            "三只乌鸦": -8, "下降三法": -6, "墓碑十字": -5, "上吊线": -5,
            "大敌当前": -4,
        }
        for s in strategies:
            score += strategy_scores.get(s, 3)
        if len(strategies) >= 3:
            score += 8
        return max(0, min(100, score))

    def _score_chip(self, f: Dict) -> float:
        score = 50
        # 同时尝试带 chip_ 前缀和不带前缀的字段（all_factors.update(chip) 写入的是原始名）
        pr = f.get("chip_profit_ratio", f.get("profit_ratio", 0.5))
        if 0.5 <= pr <= 0.8:
            score += 10
        elif pr > 0.9:
            score -= 10
        conc = f.get("chip_concentration_70", f.get("concentration_70", 0.5))
        if conc < 0.1:
            score += 15
        elif conc < 0.2:
            score += 8
        above_peak = f.get("chip_above_peak", f.get("price_above_peak"))
        if above_peak == 1:
            score += 10
        return max(0, min(100, score))

    def _score_fundamental(self, f: Dict) -> float:
        score = 50
        score += (f.get("pe_score", 50) - 50) * 0.3
        score += (f.get("pb_score", 50) - 50) * 0.25
        score += (f.get("market_cap_score", 50) - 50) * 0.25
        score += (f.get("turnover_score", 50) - 50) * 0.2
        return max(0, min(100, score))

    def _score_institutional(self, f: Dict) -> float:
        score = 50
        score += (f.get("inst_count_score", 50) - 50) * 0.4
        score += (f.get("inst_ratio_score", 50) - 50) * 0.4
        score += (f.get("inst_rating_score", 50) - 50) * 0.2
        return max(0, min(100, score))

    def _score_sector_rotation(self, f: Dict) -> float:
        score = 50
        # 板块轮动评分
        rotation_score = f.get("rotation_score", 50)
        score += (rotation_score - 50) * 0.5
        # 科技板块强度
        tech_strength = f.get("tech_sector_strength", 50)
        score += (tech_strength - 50) * 0.3
        # 轮动信号
        signal = f.get("rotation_signal", "neutral")
        if signal == "tech_strong":
            score += 10
        elif signal == "oversold_rebound":
            score += 15
        return max(0, min(100, score))

    def _score_oversold_rebound(self, f: Dict) -> float:
        score = 50
        reboot_detected = f.get("oversold_rebound_detected", False)
        reboot_score = f.get("oversold_rebound_score", 0)
        if reboot_detected:
            score += min(reboot_score * 0.5, 30)
        reboot_signals = f.get("oversold_rebound_signals", [])
        score += len(reboot_signals) * 5
        return max(0, min(100, score))

    def _score_news(self, f: Dict) -> float:
        return f.get("news_score", 50)

    def _score_relative_strength(self, f: Dict) -> float:
        # 优先用相对大盘的超额收益（需要 market_df 传入）
        rs_5d = f.get("relative_strength_5d")
        rs_20d = f.get("relative_strength_20d")
        if rs_5d is not None or rs_20d is not None:
            score = 50
            if rs_5d is not None:
                if rs_5d > 3:
                    score += 15
                elif rs_5d > 1:
                    score += 10
                elif rs_5d < -3:
                    score -= 15
            if rs_20d is not None:
                if rs_20d > 5:
                    score += 15
                elif rs_20d > 2:
                    score += 10
                elif rs_20d < -5:
                    score -= 15
            return max(0, min(100, score))
        # 降级：大盘数据缺失时用个股自身 strength_score（已基于动量/价格位置计算）
        return max(0, min(100, float(f.get("strength_score", 50))))

    def _score_research(self, f: Dict) -> float:
        """机构调研评分"""
        score = 50
        research_score = f.get("research_score", 50)
        score += (research_score - 50) * 0.6
        research_count = f.get("research_count", 0)
        if research_count >= 10:
            score += 10
        return max(0, min(100, score))

    def _score_dragon_tiger(self, f: Dict) -> float:
        """龙虎榜评分"""
        score = 50
        lh_score = f.get("lh_score", 50)
        score += (lh_score - 50) * 0.5
        hot_money_count = f.get("hot_money_count", 0)
        if hot_money_count >= 2:
            score += 15
        return max(0, min(100, score))

    def _score_leader(self, f: Dict) -> float:
        """龙头检测评分"""
        score = 50
        leader_score = f.get("leader_score", 50)
        score += (leader_score - 50) * 0.5
        if f.get("is_leader"):
            score += 15
        return max(0, min(100, score))
    def _score_ashare_specific(self, f: Dict) -> float:
        """ashare_specific子评分。

        方向依据(2026-08-27 胜率校准):
        ashare_specific Top10胜率58.9%, 样本外稳定(训练58.8%/测试58.9%).
        均值回归信号: 超卖(偏离MA20负)→后续反弹概率高→给高分.
        注意: 这与全市场IC方向相反(全市场IC为负), 但Top10选股有效.
        """
        score = 50

        # 均值回归: 超卖→高分, 超买→低分
        reversion_signal = f.get("reversion_signal", "neutral")
        if reversion_signal == "strong_buy":
            score += 30  # 严重超卖, 反弹概率高
        elif reversion_signal == "buy":
            score += 15  # 超卖
        elif reversion_signal == "strong_sell":
            score -= 25  # 严重超买, 回调概率高
        elif reversion_signal == "sell":
            score -= 12  # 超买

        volatility_score = f.get("volatility_score", 50)
        score -= (volatility_score - 50) * 0.2

        turnover_anomaly = f.get("turnover_anomaly", "normal")
        if turnover_anomaly == "extreme_high":
            score -= 10
        elif turnover_anomaly == "low":
            score += 5

        position_score = f.get("position_score", 50)
        score -= (position_score - 50) * 0.3

        sentiment_cycle = f.get("sentiment_cycle", {})
        cycle_score = sentiment_cycle.get("cycle_score", 50)
        score -= (cycle_score - 50) * 0.2

        hot_money_score = f.get("hot_money_score", 50)
        score -= (hot_money_score - 50) * 0.15

        limit_up_20d = f.get("limit_up_20d", 0)
        if limit_up_20d > 2:
            score -= 10

        return max(0, min(100, score))

    def _score_quant_behavior(self, f: Dict) -> float:
        """量化行为评分。

        方向依据全市场大样本 IC 终审(2026-08, 5273只×37截面):
        quant_behavior IC=+0.032/+0.038(h5/h10), t≈+2.1 显著正——
        检测到量化异动的股票整体后续表现更好(游资/量化活跃=关注度)。
        曾以"常识+报告矛盾"改为防御扣分，被大样本证据推翻，现恢复正向，
        但对明确破坏性信号(闪崩/天地板)保留适度减分。
        报告层 🚫 标记保留(提示博弈风险)，与评分层职责分离。
        """
        score = 50

        quant_risk = f.get("quant_risk", "low")
        quant_signals = f.get("quant_signals", [])

        if quant_risk == "high":
            score += 20
        elif quant_risk == "medium":
            score += 10

        for signal in quant_signals:
            if "闪崩" in signal or "天地板" in signal:
                score -= 12          # 唯一保留的防御项:极端破坏性信号
            elif "秒板" in signal:
                score += 10
            elif "尾盘" in signal:
                score += 8
            elif "对倒" in signal or "诱多" in signal:
                score += 8
            elif "动量追逐" in signal:
                score += 8
            elif "涨停未封住" in signal:
                score -= 3

        return max(0, min(100, score))

    def _score_margin_trading(self, f: Dict) -> float:
        """融资融券评分"""
        score = 50

        margin_trend = f.get("margin_trend", "neutral")
        margin_signals = f.get("margin_signals", [])

        # 融资趋势
        if margin_trend == "active":
            score += 15
        elif margin_trend == "moderate":
            score += 5

        # 融资信号
        for signal in margin_signals:
            if "融资买入" in signal and "活跃" in signal:
                score += 10
            elif "融券卖出" in signal:
                score -= 10
            elif "看多" in signal:
                score += 10
            elif "看空" in signal:
                score -= 10

        return max(0, min(100, score))

    def _score_shareholder(self, f: Dict) -> float:
        """股东增减持评分"""
        score = 50

        shareholder_trend = f.get("shareholder_trend", "neutral")
        shareholder_signals = f.get("shareholder_signals", [])

        # 增减持趋势
        if shareholder_trend == "increase":
            score += 20
        elif shareholder_trend == "decrease":
            score -= 20

        # 信号
        for signal in shareholder_signals:
            if "增持" in signal:
                score += 15
            elif "减持" in signal:
                score -= 15

        return max(0, min(100, score))

    def _score_lockup_risk(self, f: Dict) -> float:
        """解禁风险评分"""
        score = 50

        lockup_risk = f.get("lockup_risk", "low")
        lockup_signals = f.get("lockup_signals", [])

        # 解禁风险
        if lockup_risk == "high":
            score -= 25
        elif lockup_risk == "medium":
            score -= 10

        # 信号
        for signal in lockup_signals:
            if "解禁" in signal:
                score -= 15

        return max(0, min(100, score))

    def _score_deep_research(self, f: Dict) -> float:
        """深度研究因子评分：直接透传。

        deep_research_score 内部以 50 为中性、催化/信心加分为正向构造，
        分越高越好。旧实现 100-raw 整体取反基于样本内"低分正确率89%"，
        与因子自身语义相反（data snooping），已移除。
        """
        raw = f.get("deep_research_score", 50)
        return max(0, min(100, float(raw)))

    def _score_policy_event(self, f: Dict) -> float:
        """政策事件因子评分"""
        score = 50
        policy_score = f.get("policy_score", 50)
        score += (policy_score - 50) * 0.6

        # 政策方向
        policy_direction = f.get("policy_direction", "neutral")
        if policy_direction == "positive":
            score += 10
        elif policy_direction == "negative":
            score -= 10

        # 宏观事件影响
        macro_impact = f.get("macro_impact", {})
        if macro_impact:
            impact_score = macro_impact.get("impact_score", 50)
            score += (impact_score - 50) * 0.3

        return max(0, min(100, score))

    def _score_national_team(self, f: Dict) -> float:
        """国家队因子评分。

        NationalTeamFactor 返回:
          national_team_holding: bool (True=检测到国家队特征)
          national_team_change: float (-1 到 1, 正=增持倾向, 负=减持倾向)
        """
        score = 50
        holding = f.get("national_team_holding", False)
        change = f.get("national_team_change", 0.0)

        # 国家队特征检测（bool → 加分）
        if holding:
            score += 15

        # 增减持倾向（float → 方向判断）
        if isinstance(change, (int, float)):
            if change > 0.3:
                score += 20
            elif change > 0.1:
                score += 10
            elif change < -0.3:
                score -= 25
            elif change < -0.1:
                score -= 12
        elif isinstance(change, str):
            # 兼容旧版字符串格式
            if change == "increase":
                score += 20
            elif change == "decrease":
                score -= 25

        return max(0, min(100, score))

    def _score_capital_behavior(self, f: Dict) -> float:
        """资金行为模式因子评分"""
        score = 50
        behavior = f.get("capital_behavior", "normal")
        confidence = f.get("capital_behavior_confidence", 0)

        # 行为模式
        if behavior == "accumulation":
            score += 20 * confidence
        elif behavior == "markup":
            score += 15 * confidence
        elif behavior == "distribution":
            score -= 25 * confidence
        elif behavior == "shakeout":
            score -= 10 * confidence

        # 资金连续性（CapitalBehaviorFactor 返回 float -1~1，正=流入，负=流出）
        flow_continuity = f.get("flow_continuity", 0.0)
        if isinstance(flow_continuity, (int, float)):
            if flow_continuity > 0.3:
                score += 15
            elif flow_continuity > 0.1:
                score += 8
            elif flow_continuity < -0.3:
                score -= 15
            elif flow_continuity < -0.1:
                score -= 8
        elif isinstance(flow_continuity, str):
            # 兼容旧版字符串格式
            if flow_continuity == "strong_inflow":
                score += 15
            elif flow_continuity == "inflow":
                score += 8
            elif flow_continuity == "strong_outflow":
                score -= 15
            elif flow_continuity == "outflow":
                score -= 8

        return max(0, min(100, score))

    def _score_holder_concentration(self, f: Dict) -> float:
        holder_score = f.get("holder_score", 50)
        return max(0, min(100, float(holder_score)))

    def _score_circulate_holder(self, f: Dict) -> float:
        circulate_score = f.get("circulate_score", 50)
        return max(0, min(100, float(circulate_score)))

def apply_alpha_overlay(candidates: List[Dict], config: Dict) -> None:
    """positive2 合成信号 overlay：对最终得分施加 alpha 调整。

    boost = clip(weight * (z_quant + z_oversold)/2, ±cap)
    直接修改 candidates 各项的 score，并记录 alpha_boost 字段供报告展示。
    """
    cfg = config.get("alpha_overlay") or {}
    if not cfg.get("enabled", True):
        return candidates
    weight = float(cfg.get("weight", 3.0))
    cap = float(cfg.get("cap", 6.0))

    applied = 0
    for c in candidates:
        zs = c.get("z_scores") or {}
        zq = zs.get("quant_behavior", 0) or 0
        zo = zs.get("oversold_rebound", 0) or 0
        boost = max(-cap, min(cap, weight * (zq + zo) / 2.0))
        if abs(boost) < 0.05:
            continue
        c["score"] = round(max(0, min(100, float(c["score"]) + boost)), 1)
        c["alpha_boost"] = round(boost, 2)
        applied += 1
    if applied:
        logging.getLogger("stock_prediction").info(
            "Alpha overlay 已应用 %d 只 (weight=%.1f cap=%.1f)", applied, weight, cap)


def apply_chip_guard(candidates: List[Dict], config: Dict) -> List[Dict]:
    """chip 风控排除器：剔除截面 chip z-score 过高的票(获利盘沉重→跑输)。

    返回过滤后的列表；被剔除项带 guard_reason 字段供日志/报告说明。
    """
    cfg = config.get("chip_guard") or {}
    if not cfg.get("enabled", True):
        return candidates
    threshold = float(cfg.get("z_threshold", 1.5))
    # pos_250 一年位置排除(全市场IC h10=-0.112, t=-6.9 最强负因子):
    # 实盘复用 ashare_specific.price_position_250d(0~1), 高位阈值默认0.9
    pos_high = float(cfg.get("pos250_high", 0.9))

    # bottom 排除(组合模拟实证2026-08): positive_v2 截面后20%平均跑输全池
    # 30bp+/日——信号的真实预测力在空头端。候选池内按
    # (quant_behavior+oversold_rebound)/2 排序剔除后20%。
    # 注意: 候选池是全市场筛过的高分票, 其"后20%"不等于全市场底部,
    # 但相对排序仍具参考意义
    bottom_frac = float(cfg.get("bottom_exclude_frac", 0.0))  # 默认关闭,需显式启用

    scored_for_bottom = []
    for idx, c in enumerate(candidates):
        zs = c.get("z_scores") or {}
        composite = ((zs.get("quant_behavior", 0) or 0)
                     + (zs.get("oversold_rebound", 0) or 0)) / 2.0
        c["_bottom_sort"] = composite
        candidates[idx] = c
    if bottom_frac > 0 and len(candidates) >= 10:
        cutoff_rank = int(len(candidates) * bottom_frac)
        bottom_codes = {c["code"] for c in
                        sorted(candidates, key=lambda x: x["_bottom_sort"])[:cutoff_rank]}
    else:
        bottom_codes = set()

    kept, excluded = [], []
    for c in candidates:
        zc = (c.get("z_scores") or {}).get("chip", 0) or 0
        pp = (c.get("factors") or {}).get("price_position_250d")
        if c.get("code") in bottom_codes:
            c["guard_reason"] = "positive_v2 截面后列(风格失效保护)"
            excluded.append(c)
            continue
        if zc > threshold:
            c["guard_reason"] = f"获利盘压力(chip_z={zc:.2f}>{threshold})"
            excluded.append(c)
            continue
        if pp is not None and float(pp) > pos_high:
            c["guard_reason"] = f"一年高位(pos={float(pp):.2f}>{pos_high})"
            excluded.append(c)
            continue
        kept.append(c)
    for c in candidates:
        c.pop("_bottom_sort", None)
    if excluded:
        logging.getLogger("stock_prediction").warning(
            "Chip guard 剔除 %d 只获利盘压力票(z>%.1f): %s",
            len(excluded), threshold,
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept


def apply_ashare_guard(candidates: List[Dict], config: Dict) -> List[Dict]:
    """ashare_specific 风控排除器：剔除截面 ashare_specific 最低 N% 的票。

    依据(2026-08 组合模拟A/B):
      - hold=5 多头年化-22%, 空头年化+8.8%, 胜率60.9%, Sharpe+0.44
      - alpha集中在空头端, A股无做空机制下转为多头保护
      - 排除截面 ashare_specific z-score 最低 bottom_exclude_frac 比例

    返回过滤后的列表；被剔除项带 guard_reason 字段供日志/报告说明。
    """
    cfg = config.get("ashare_guard") or {}
    if not cfg.get("enabled", False):
        return candidates
    exclude_frac = float(cfg.get("bottom_exclude_frac", 0.10))
    min_candidates = int(cfg.get("min_candidates", 10))

    if exclude_frac <= 0 or len(candidates) < min_candidates:
        return candidates

    cutoff = int(len(candidates) * exclude_frac)
    if cutoff <= 0:
        return candidates

    sorted_cands = sorted(candidates,
                          key=lambda x: (x.get("z_scores") or {}).get("ashare_specific", 0) or 0)
    bottom_codes = {c["code"] for c in sorted_cands[:cutoff]}

    kept, excluded = [], []
    for c in candidates:
        if c.get("code") in bottom_codes:
            zs = c.get("z_scores") or {}
            z_ashare = zs.get("ashare_specific", 0) or 0
            c["guard_reason"] = f"ashare_specific弱信号(z={z_ashare:.2f},截面后{exclude_frac:.0%})"
            excluded.append(c)
        else:
            kept.append(c)

    if excluded:
        logging.getLogger("stock_prediction").warning(
            "Ashare guard 剔除 %d 只弱信号票(ashare_specific后%.0f%%): %s",
            len(excluded), exclude_frac * 100,
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept


# 美股板块→A股行业映射(风控用)
_US_SECTOR_MAP = {
    "半导体": ["电子信息", "电子器件"],
    "液冷/服务器": ["电子信息", "机械行业"],
    "光通信/CPO": ["电子信息"],
    "AI/云计算": ["电子信息", "机械行业"],
    "新能源/电车": ["电力行业", "发电设备", "汽车制造"],
    "金融": ["金融行业"],
}


def apply_us_sector_guard(candidates: List[Dict], config: Dict,
                          industry_map: Dict[str, str] = None) -> List[Dict]:
    """美股板块风控排除器: 美股相关板块跌>2%时, 排除A股对应行业。

    依据(2026-08-27 全年数据分析 reports/us_a_sector_correlation.json):
      - AI/云计算跌>2%→A股电子信息次日+0.39%/胜率55%(vs涨时+1.93%/73%)
      - 新能源/电车跌>2%→A股发电设备次日-1.97%/胜率26%
      - 金融跌>2%→A股金融行业次日-1.34%/胜率0%
      极端行情下美股→A股板块联动显著, 常规行情无相关性。

    Args:
        candidates: 候选股列表(需含 code/name 字段)
        config: 配置(读取 us_sector_guard 段)
        industry_map: {code: industry} 映射, None 时从 market_data 获取
    """
    cfg = config.get("us_sector_guard") or {}
    if not cfg.get("enabled", False):
        return candidates

    threshold = float(cfg.get("drop_threshold", -2.0))
    lookback_hours = int(cfg.get("lookback_hours", 12))

    # 读取美股板块最新数据
    import os
    us_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "cache", "us_daily")
    sector_path = os.path.join(us_dir, "us_sector_returns.csv")
    if not os.path.exists(sector_path):
        return candidates

    try:
        import pandas as pd
        us_df = pd.read_csv(sector_path, index_col=0)
        latest = us_df.iloc[-1]  # 最新一天
    except Exception:
        return candidates

    # 找出跌幅超阈值的板块
    dropped_sectors = [s for s, v in latest.items() if v < threshold]
    if not dropped_sectors:
        return candidates

    # 映射到A股行业
    affected_industries = set()
    for sector in dropped_sectors:
        if sector in _US_SECTOR_MAP:
            affected_industries.update(_US_SECTOR_MAP[sector])

    if not affected_industries:
        return candidates

    # 获取行业映射
    if industry_map is None:
        try:
            industry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "data", "industry_map.json")
            with open(industry_path, encoding="utf-8") as f:
                industry_map = json.load(f)
        except Exception:
            return candidates

    # 排除受影响行业的股票
    kept, excluded = [], []
    for c in candidates:
        code = c.get("code", "")
        # 标准化代码格式
        norm_code = code.replace(".", "")
        industry = industry_map.get(norm_code, "")
        if industry in affected_industries:
            c["guard_reason"] = f"美股板块风险({','.join(dropped_sectors)}跌>{abs(threshold):.0f}%)"
            excluded.append(c)
        else:
            kept.append(c)

    if excluded:
        logging.getLogger("stock_prediction").warning(
            "US sector guard 剔除 %d 只(%s行业, 美股跌>%.0f%%): %s",
            len(excluded), ",".join(affected_industries), abs(threshold),
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept


def load_supply_chain():
    """加载美股→A股供应链映射"""
    sc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "us_a_supply_chain.json")
    if not os.path.exists(sc_path):
        return {}
    try:
        with open(sc_path, encoding="utf-8") as f:
            data = json.load(f)
        # 构建反向索引: A股code → [(us_name, link)]
        index = {}
        for us_code, info in data.get("supply_chain", {}).items():
            for s in info.get("a_share_suppliers", []):
                code = s.get("code", "")
                if code not in index:
                    index[code] = []
                index[code].append({"us_name": info["name"], "link": s["link"], "degree": s["degree"]})
        return index
    except Exception:
        return {}


def apply_supply_chain_guard(candidates: List[Dict], config: Dict,
                             supply_chain_index: Dict = None) -> List[Dict]:
    """美股供应链风控: 美股跌>2%时排除其A股供应商。

    依据(2026-08-27 全年数据分析):
      - 特斯拉跌>2%→A股供应链次日跌(9家供应商平均lift=-8.3%)
      - 英伟达跌>2%→A股PCB/光模块次日跌(lift=-3.9%)
      - 亚马逊跌>2%→A股服务器次日跌(lift=-11.3%)

    Args:
        candidates: 候选股列表(需含 code 字段)
        config: 配置(读取 supply_chain_guard 段)
        supply_chain_index: {code: [{us_name, link, degree}]} 反向索引
    """
    cfg = config.get("supply_chain_guard") or {}
    if not cfg.get("enabled", False):
        return candidates

    threshold = float(cfg.get("drop_threshold", -2.0))

    # 读取美股最新数据
    us_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "cache", "us_daily")
    sector_path = os.path.join(us_dir, "us_sector_returns.csv")
    if not os.path.exists(sector_path):
        return candidates

    try:
        import pandas as pd
        us_df = pd.read_csv(sector_path, index_col=0)
        latest = us_df.iloc[-1]
    except Exception:
        return candidates

    # 加载供应链索引
    if supply_chain_index is None:
        supply_chain_index = load_supply_chain()
    if not supply_chain_index:
        return candidates

    # 找出跌>阈值的美股板块
    dropped_sectors = [s for s, v in latest.items() if v < threshold]
    if not dropped_sectors:
        return candidates

    # 映射到A股供应商 (板块→行业→股票, 通过 us_sector_guard 已覆盖)
    # 这里做更精准的: 直接匹配供应链公司
    # 用美股个股数据(如果有的话)
    us_stock_path = os.path.join(us_dir, "us_sector_returns.csv")
    # 简化: 用板块收益近似个股收益
    # 跌的板块 → 该板块内的美股公司 → A股供应商

    kept, excluded = [], []
    for c in candidates:
        code = c.get("code", "")
        norm_code = code.replace(".", "")
        if norm_code in supply_chain_index:
            suppliers = supply_chain_index[norm_code]
            # 检查该供应商的美股关联公司是否在跌的板块
            for s in suppliers:
                us_name = s["us_name"]
                # 简单映射: 英伟达→半导体, 特斯拉→新能源, etc
                sector_map = {
                    "英伟达": "半导体", "AMD": "半导体",
                    "特斯拉": "新能源/电车",
                    "亚马逊": "AI/云计算", "微软": "AI/云计算",
                    "Meta": "AI/云计算", "谷歌": "AI/云计算",
                }
                related_sector = sector_map.get(us_name, "")
                if related_sector in dropped_sectors:
                    c["guard_reason"] = f"美股供应链风险({us_name}相关板块跌>{abs(threshold):.0f}%)"
                    excluded.append(c)
                    break
            else:
                kept.append(c)
        else:
            kept.append(c)

    if excluded:
        logging.getLogger("stock_prediction").warning(
            "Supply chain guard 剔除 %d 只(美股供应链跌>%.0f%%): %s",
            len(excluded), abs(threshold),
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept


def apply_oversold_rebound_guard(candidates: List[Dict], config: Dict) -> List[Dict]:
    """oversold_rebound 反向排除器：剔除截面 oversold_rebound 最高 N% 的票。

    依据(2026-08 分层分析):
      - Q10-Q1=-42.1bp (t=-1.98), 高分位反而跌
      - 单调性=-0.36, 方向完全反向
      - 排除截面 oversold_rebound z-score 最高 top_exclude_frac 比例

    返回过滤后的列表；被剔除项带 guard_reason 字段供日志/报告说明。
    """
    cfg = config.get("oversold_rebound_guard") or {}
    if not cfg.get("enabled", False):
        return candidates
    exclude_frac = float(cfg.get("top_exclude_frac", 0.10))
    min_candidates = int(cfg.get("min_candidates", 10))

    if exclude_frac <= 0 or len(candidates) < min_candidates:
        return candidates

    cutoff = int(len(candidates) * exclude_frac)
    if cutoff <= 0:
        return candidates

    sorted_cands = sorted(candidates,
                          key=lambda x: (x.get("z_scores") or {}).get("oversold_rebound", 0) or 0,
                          reverse=True)
    top_codes = {c["code"] for c in sorted_cands[:cutoff]}

    kept, excluded = [], []
    for c in candidates:
        if c.get("code") in top_codes:
            zs = c.get("z_scores") or {}
            z_rebound = zs.get("oversold_rebound", 0) or 0
            c["guard_reason"] = f"oversold_rebound反向信号(z={z_rebound:.2f},截面前{exclude_frac:.0%})"
            excluded.append(c)
        else:
            kept.append(c)

    if excluded:
        logging.getLogger("stock_prediction").warning(
            "Oversold rebound guard 剔除 %d 只反向信号票(oversold_rebound前%.0f%%): %s",
            len(excluded), exclude_frac * 100,
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept


def apply_liquidity_guard(candidates: List[Dict], config: Dict) -> List[Dict]:
    """流动性风控：排除日均成交额过低的票。

    依据：A股无做空机制，低流动性=无法及时止损=硬约束。
    默认阈值 0.5 亿元（5000万），低于该值的票在极端行情下可能无法卖出。

    返回过滤后的列表；被剔除项带 guard_reason 字段供日志/报告说明。
    """
    cfg = config.get("liquidity_guard") or {}
    if not cfg.get("enabled", False):
        return candidates
    min_amount = float(cfg.get("min_daily_amount_yi", 0.5))  # 亿元

    kept, excluded = [], []
    for c in candidates:
        factors = c.get("factors") or {}
        avg_amount = factors.get("avg_daily_amount_20d", 0) or 0
        if avg_amount < min_amount:
            c["guard_reason"] = f"流动性不足({avg_amount:.2f}亿<{min_amount}亿)"
            excluded.append(c)
        else:
            kept.append(c)

    if excluded:
        logging.getLogger("stock_prediction").warning(
            "Liquidity guard 剔除 %d 只低流动性票(<%.2f亿): %s",
            len(excluded), min_amount,
            ", ".join(f"{c.get('name', c.get('code', '?'))}" for c in excluded[:5]))
    return kept
