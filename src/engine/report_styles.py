# -*- coding: utf-8 -*-
"""每日报告的 CSS 样式(由 reporter.py 外移, 纯静态资源)。"""
import html as _html
from typing import List, Dict

# ruff: noqa: E501
CSS = """\
:root {{
  --paper:#faf8f4; --card:#ffffff; --ink:#1a1a1a; --ink-2:#555;
  --muted:#8a8578; --line:#e7e1d3; --accent:#9a3b2a; --gold:#b8912e;
  --up:#c0392b; --down:#1e7d4f;
  --serif: Georgia, 'Songti SC', 'SimSun', serif;
  --sans: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); line-height: 1.7; -webkit-font-smoothing: antialiased; }}
.container {{ max-width: 820px; margin: 0 auto; padding: 48px 28px 64px; }}
.up {{ color: var(--up); font-weight: 600; }}
.down {{ color: var(--down); font-weight: 600; }}
.flat {{ color: var(--muted); }}

/* 报头 */
.masthead {{ border-bottom: 3px double var(--ink); padding-bottom: 18px; margin-bottom: 8px; }}
.masthead .kicker {{ font-size: 12px; letter-spacing: .28em; text-transform: uppercase; color: var(--muted); }}
.masthead h1 {{ font-family: var(--serif); font-size: 34px; font-weight: 700; margin-top: 6px; line-height: 1.2; }}
.masthead .sub {{ font-size: 13px; color: var(--muted); margin-top: 6px; }}
.dateline {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); padding: 8px 0 12px; }}
.dateline strong {{ color: var(--ink); font-weight: 600; }}

/* 段落标题 */
.sec {{ font-family: var(--serif); font-size: 19px; font-weight: 700; margin: 26px 0 16px; display: flex; align-items: baseline; gap: 12px; }}
.sec::after {{ content: ""; flex: 1; height: 1px; background: var(--line); }}

/* 卡片 */
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 2px; padding: 16px 18px; margin-bottom: 18px; }}
.card h2 {{ font-size: 13px; letter-spacing: .1em; color: var(--muted); font-weight: 600; margin-bottom: 12px; text-transform: uppercase; }}
.card h3 {{ font-size: 12px; letter-spacing: .1em; color: var(--muted); font-weight: 600; margin-bottom: 12px; text-transform: uppercase; }}

/* 市场 */
.market-grid {{ display: grid; grid-template-columns: 1fr 1.4fr; gap: 18px; margin-bottom: 8px; }}
.i-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.i-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 2px; padding: 14px 16px; }}
.i-card .nm {{ font-size: 12px; color: var(--muted); }}
.i-card .v {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
.i-card .d {{ font-size: 12px; font-weight: 600; margin-top: 2px; }}
.e-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.e-item .val {{ font-size: 17px; font-weight: 600; }}
.e-item .lab {{ font-size: 11px; color: var(--muted); }}
.sent-extra {{ font-size: 12.5px; color: var(--ink-2); margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--line); }}
.badge {{ display: inline-block; padding: 1px 8px; border-radius: 2px; font-size: 11px; font-weight: 600; }}
.badge-pos {{ background: #f6e5e2; color: var(--up); }}
.badge-neg {{ background: #e2efe9; color: var(--down); }}
.badge-neu {{ background: #f0ece2; color: var(--muted); }}
.badge-warn {{ background: #f6efe0; color: #856404; }}

/* 表格 */
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th {{ text-align: left; color: var(--muted); font-weight: 600; padding: 9px 8px; border-bottom: 1px solid var(--line); font-size: 11px; letter-spacing: .05em; }}
td {{ padding: 8px; border-bottom: 1px solid #f3efe4; }}

/* 股票卡 */
.stock {{ border: 1px solid var(--line); border-top: 3px solid var(--accent); background: var(--card); border-radius: 2px; margin-bottom: 22px; overflow: hidden; }}
.stk-head {{ padding: 18px 22px; border-bottom: 1px solid var(--line); display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 6px; }}
.stk-name {{ font-family: var(--serif); font-size: 22px; font-weight: 700; }}
.stk-code {{ font-size: 13px; color: var(--muted); margin-left: 8px; }}
.stk-score {{ font-size: 13px; color: var(--ink-2); }}
.score-num {{ font-family: var(--serif); font-size: 26px; font-weight: 700; color: var(--accent); margin-left: 6px; }}
.stars {{ color: var(--gold); letter-spacing: 2px; font-size: 15px; }}
.stk-body {{ padding: 6px 20px 4px; }}

.reason {{ padding: 14px 0 4px; }}
.reason .rt {{ font-size: 12px; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }}
.reason ul {{ list-style: none; }}
.reason li {{ position: relative; padding: 5px 0 5px 22px; font-size: 14px; color: var(--ink-2); border-bottom: 1px dashed var(--line); }}
.reason li:last-child {{ border-bottom: none; }}
.reason li::before {{ content: "◆"; position: absolute; left: 2px; top: 6px; font-size: 9px; color: var(--accent); }}
.reason li b {{ color: var(--ink); font-weight: 600; }}

.op-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 8px 0 6px; }}
.op {{ background: #f4f0e6; border-radius: 2px; padding: 10px 12px; }}
.op .l {{ font-size: 11px; color: var(--muted); }}
.op .v {{ font-size: 15px; font-weight: 600; margin-top: 2px; }}

.factor {{ margin: 8px 0 0; border-top: 1px dashed var(--line); padding: 12px 0 4px; }}
.factor .ft {{ font-size: 12px; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }}
.f-row {{ display: flex; justify-content: space-between; padding: 5px 0; font-size: 13px; border-bottom: 1px solid #f3efe4; }}
.f-row:last-child {{ border-bottom: none; }}
.f-row .fl {{ color: var(--ink-2); }}
.f-row .fv {{ font-weight: 600; color: var(--ink); }}
.f-row .fv bar {{ display: inline-block; width: 60px; height: 5px; background: #e7e1d3; margin-left: 8px; vertical-align: middle; border-radius: 2px; }}
.f-row .fv bar i {{ display: block; height: 100%; background: var(--accent); border-radius: 2px; }}
.warn-row {{ font-size: 12.5px; color: #856404; padding: 4px 0; }}

/* 新闻 */
.news-card {{ border: 1px solid var(--line); background: var(--card); border-radius: 2px; padding: 6px 22px 18px; }}
.news-section {{ margin-bottom: 6px; }}
.news-section h3 {{ font-size: 12px; font-weight: 700; color: var(--accent); letter-spacing: .06em; margin: 14px 0 4px; }}
.news-item {{ padding: 9px 0; border-bottom: 1px solid var(--line); }}
.news-item:last-child {{ border-bottom: none; }}
.news-txt {{ font-size: 13.5px; color: var(--ink-2); }}
.news-txt.full {{ display: block; }}
details {{ font-size: 13.5px; color: var(--ink-2); }}
details summary {{ cursor: pointer; color: var(--ink); font-weight: 600; list-style: none; position: relative; padding-right: 60px; }}
details summary::-webkit-details-marker {{ display: none; }}
details summary::after {{ content: "展开全文 ▾"; position: absolute; right: 0; top: 0; font-size: 12px; color: var(--accent); font-weight: 600; white-space: nowrap; }}
details[open] summary::after {{ content: "收起 ▴"; }}
details .news-full {{ font-size: 13px; color: var(--ink); margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--line); }}

.biz-box {{ background: #f4f0e6; border-left: 3px solid var(--accent); border-radius: 2px; padding: 12px 14px; margin: 10px 0 0; font-size: 12.5px; }}
.biz-box .op-title {{ font-weight: 600; color: var(--ink); margin-bottom: 6px; }}
.biz-box .biz-line {{ color: var(--ink-2); margin: 3px 0; }}
.biz-box .moat-val {{ font-size: 16px; font-weight: 700; }}
.tag {{ display: inline-block; background: #efe8d9; color: #6b5d3e; border-radius: 2px; padding: 1px 7px; margin: 2px 4px 2px 0; font-size: 11px; }}
.tag-moat {{ background: #f6e5e2; color: var(--up); }}
.tag-moat-neu {{ background: #f0ece2; color: var(--muted); }}

.footer {{ text-align: center; color: var(--muted); font-size: 11.5px; margin-top: 34px; line-height: 1.8; }}
.footer::before {{ content: ""; display: block; width: 60px; height: 3px; background: var(--accent); margin: 0 auto 12px; }}
@media (max-width: 640px) {{
  .container {{ padding: 28px 16px; }}
  .market-grid {{ grid-template-columns: 1fr; }}
  .i-grid, .e-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .op-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .masthead h1 {{ font-size: 26px; }}
}}
"""


class FactorRowBuilder:
    """因子行构建器"""

    @staticmethod
    def _stars(n):
        """★/☆ 强度条文本(最多5星)。"""
        return "★" * min(n, 5) + "☆" * max(0, 5 - n)

    def _build_factor_rows(self, factors: Dict) -> List[str]:
        rows = []

        def add(category, label, val_str, strength_text, force=False):
            # 根据强度文本映射 bar 宽度：★越多越强
            stars = strength_text.count("★")
            emoji_strong = ("🚫" in strength_text) or ("⚠️" in strength_text)
            if emoji_strong:
                bar_w = "100"
            elif stars >= 4:
                bar_w = "90"
            elif stars == 3:
                bar_w = "75"
            elif stars == 2:
                bar_w = "55"
            elif stars == 1:
                bar_w = "40"
            else:
                bar_w = "30"
            cat_short = category.replace("⚠️ ", "").replace("🏆 ", "")[:2]
            rows.append(
                f'<div class="f-row"><span class="fl">{_html.escape(str(label))} {_html.escape(str(val_str))}</span>'
                f'<span class="fv">{cat_short}<bar><i style="width:{bar_w}%"></i></bar></span></div>'
            )

        self._factor_rows_tech(factors, add)
        self._factor_rows_sentiment(factors, add)
        self._factor_rows_fundamental(factors, add)
        self._factor_rows_event(factors, add)
        return rows

    def _factor_rows_tech(self, factors: Dict, add) -> None:
        """技术面行: 基础信号/策略/筹码/强弱/超跌反弹/持续性/均值回归/波动率/价格位置/资金持续。"""
        star = self._stars

        signal_map = {
            "macd_golden_cross": ("MACD金叉", "趋势"),
            "ma_bullish": ("均线多头排列", "趋势"),
            "breakout_20d_high": ("突破20日新高", "趋势"),
            "volume_price_up": ("量价齐升", "量价"),
            "news_score": ("消息面利好", "消息"),
            "in_hot_list": ("同花顺热股榜", "情绪"),
        }
        for factor_key, (label, category) in signal_map.items():
            val = factors.get(factor_key, 0)
            if factor_key == "news_score":
                if val > 75:
                    add(category, label, f"({val:.0f}分)", star(4))
                elif val > 60:
                    add(category, label, f"({val:.0f}分)", star(3))
            elif val == 1:
                add(category, label, "", star(5))

        strategies = factors.get("strategies_triggered", [])
        if strategies:
            add("策略", "、".join(strategies), "", star(5))

        chip_score = factors.get("chip_score", 50)
        if chip_score > 65:
            chip_info = []
            if factors.get("chip_above_peak") == 1:
                chip_info.append("突破筹码峰")
            if factors.get("chip_profit_ratio", 0) > 0.6:
                chip_info.append(f"获利盘{factors.get('chip_profit_ratio', 0)*100:.0f}%")
            conc = factors.get("chip_concentration_70", 0)
            if conc < 0.15:
                chip_info.append("筹码高度集中")
            if chip_info:
                add("筹码", ", ".join(chip_info), "", star(4))

        rs_5d = factors.get("relative_strength_5d", 0)
        rs_20d = factors.get("relative_strength_20d", 0)
        if rs_5d > 2:
            add("强弱", f"5日超额收益+{rs_5d:.1f}%", "", star(4))
        elif rs_5d < -2:
            add("弱势", f"5日超额收益{rs_5d:.1f}%", "", "⚠️")
        if rs_20d > 5:
            add("强弱", f"20日超额收益+{rs_20d:.1f}%", "", star(5))

        if factors.get("oversold_rebound_detected"):
            rebound_signals = factors.get("oversold_rebound_signals", [])
            if rebound_signals:
                add("超跌反弹", ", ".join(rebound_signals[:3]), "", star(5))

        sustainability_score = factors.get("sustainability_score", 50)
        rebound_type = factors.get("rebound_type", "unknown")
        type_map = {"capital_driven": "资金驱动型", "emotion_repair": "情绪修复型", "technical": "技术性反弹"}
        if rebound_type in type_map:
            if sustainability_score >= 65:
                add("持续性", f"{type_map[rebound_type]}（{sustainability_score}分）", "", star(4))
            elif sustainability_score <= 40:
                add("持续性", f"{type_map[rebound_type]}（{sustainability_score}分）", "", "⚠️")

        reversion_signal = factors.get("reversion_signal", "neutral")
        if reversion_signal in ["strong_buy", "buy"]:
            # 均值回归信号: 超卖→高分(评分层加30分), 标为机会
            add("均值回归", f"偏离20日均线{factors.get('deviation_20d', 0):.1f}%（超卖→机会）", "", star(4))
        elif reversion_signal in ["strong_sell", "sell"]:
            add("均值回归", f"偏离20日均线{factors.get('deviation_20d', 0):.1f}%（超买→风险）", "", "⚠️")

        volatility_level = factors.get("volatility_level", "")
        if volatility_level == "low":
            add("波动率", "低波动（安全）", "", star(3))
        elif volatility_level in ["extreme_high", "high"]:
            add("波动率", "高波动（风险大）", "", "⚠️")

        position_signal = factors.get("position_signal", "")
        if position_signal:
            if "底部" in position_signal:
                add("价格位置", position_signal, "", star(5))
            elif "顶部" in position_signal:
                add("价格位置", position_signal, "", "⚠️")
            else:
                add("价格位置", position_signal, "", star(3))

        flow_signals = factors.get("flow_continuity_signals", [])
        flow_continuity = factors.get("flow_continuity", "neutral")
        if flow_signals:
            if flow_continuity in ["strong_inflow", "inflow"]:
                add("资金持续", flow_signals[0], "", star(5))
            elif flow_continuity in ["strong_outflow", "outflow"]:
                add("⚠️ 资金流出", flow_signals[0], "", "🚫")
            elif flow_continuity == "reversal":
                add("资金逆转", flow_signals[0], "", star(4))

    def _factor_rows_sentiment(self, factors: Dict, add) -> None:
        """情绪面行: 人气/游资/情绪周期/涨停质量/板块联动/龙头/龙虎榜。"""
        star = self._stars

        hot_rank = factors.get("hot_rank", 0)
        if 0 < hot_rank <= 30:
            add("人气", f"热股榜第{hot_rank}名", "", star(5))

        hot_money_signals = factors.get("hot_money_signals", [])
        if hot_money_signals:
            add("游资", hot_money_signals[0], "", star(4))

        sentiment_cycle = factors.get("sentiment_cycle", {})
        cycle_phase = sentiment_cycle.get("cycle_phase", "")
        if cycle_phase in ["冰点", "低迷"]:
            add("情绪周期", f"{cycle_phase}（可布局）", "", star(4))
        elif cycle_phase in ["亢奋"]:
            add("情绪周期", f"{cycle_phase}（注意风险）", "", "⚠️")

        limit_signals = factors.get("limit_signals", [])
        limit_quality = factors.get("limit_quality", "unknown")
        if limit_signals:
            if limit_quality in ["连板", "一字板"]:
                add("涨停质量", limit_signals[0], "", star(5))
            elif limit_quality == "首板":
                add("涨停质量", limit_signals[0], "", star(4))
            elif limit_quality == "跌停":
                add("⚠️ 涨停质量", limit_signals[0], "", "🚫")

        sector_signals = factors.get("sector_signals", [])
        if sector_signals:
            for s in sector_signals[:2]:
                if "涨" in s:
                    add("板块联动", s, "", star(4))
                else:
                    add("板块联动", s, "", "⚠️")

        leader_signals = factors.get("leader_signals", [])
        is_leader = factors.get("is_leader", False)
        if leader_signals:
            if is_leader:
                add("🏆 龙头", leader_signals[0], "", star(5))
            else:
                add("龙头", leader_signals[0], "", star(3))

        lh_signals = factors.get("lh_signals", [])
        if lh_signals:
            if any("出货" in s for s in lh_signals):
                add("⚠️ 龙虎榜", lh_signals[0], "", "🚫")
            else:
                add("龙虎榜", lh_signals[0], "", star(4))

    def _factor_rows_fundamental(self, factors: Dict, add) -> None:
        """基本面行: PE/PB/市值/机构持股/两融/机构调研。"""
        star = self._stars

        pe = factors.get("pe", 0)
        pb = factors.get("pb", 0)
        mv = factors.get("total_mv_yi", 0)
        if pe > 0 and pe < 25:
            add("基本面", f"PE={pe:.1f}（低估）", "", star(4))
        if pb > 0 and pb < 2:
            add("基本面", f"PB={pb:.1f}（低估）", "", star(3))
        if 50 < mv < 500:
            add("基本面", f"市值{mv:.0f}亿（中盘）", "", star(3))

        inst_count = factors.get("inst_holding_count", 0)
        inst_ratio = factors.get("inst_holding_ratio", 0)
        if inst_count > 10:
            add("机构", f"{inst_count}家机构持股", "", star(4))
        elif inst_count > 0:
            add("机构", f"{inst_count}家机构持股", "", star(3))
        if inst_ratio > 20:
            add("机构", f"持股比例{inst_ratio:.1f}%", "", star(4))

        margin_signals = factors.get("margin_signals", [])
        margin_trend = factors.get("margin_trend", "neutral")
        if margin_signals:
            if margin_trend == "active":
                add("融资", margin_signals[0], "", star(4))
            elif "融券" in margin_signals[0]:
                add("⚠️ 融券", margin_signals[0], "", "⚠️")

        research_signals = factors.get("research_signals", [])
        if research_signals:
            add("机构", research_signals[0], "", star(4))

    def _factor_rows_event(self, factors: Dict, add) -> None:
        """事件面行: 次日预判/深度研究/预期透支/利好兑现/获利出逃/量化/股东/解禁。"""
        star = self._stars

        next_day = factors.get("next_day_prediction", {})
        if not isinstance(next_day, dict):
            next_day = {}
        if next_day:
            bias = next_day.get("bias", "neutral")
            suggestion = next_day.get("trading_suggestion", "")
            bias_map = {"bullish": "📈 看多", "slightly_bullish": "📈 偏多", "bearish": "📉 看空",
                        "slightly_bearish": "📉 偏空", "neutral": "➡️ 中性"}
            if bias in bias_map:
                s = star(3) if "bullish" in bias else ("⚠️" if "bearish" in bias else star(2))
                add("次日预判", bias_map[bias], "", s)
            if suggestion:
                add("操作建议", suggestion, "", "-")

        deep_score = factors.get("deep_research_score", 0)
        if deep_score > 0:
            ref = factors.get("historical_reference", "")
            factors.get("prediction_confidence", 0)
            factors.get("volatility_regime", "")
            factors.get("position_at", "")
            if deep_score > 65:
                add("深度研究", f"历史匹配({deep_score}分)", ref, star(4))
            elif deep_score > 50:
                add("深度研究", f"历史匹配({deep_score}分)", ref, star(3))
            else:
                add("深度研究", f"历史匹配({deep_score}分)", ref, "⚠️")
            dr_signals = factors.get("deep_research_signals", [])
            for s in dr_signals[:2]:
                if "⚠" in s:
                    add("深度研究", s, "", "⚠️")
                else:
                    add("深度研究", s, "", star(3))

        if factors.get("expectation_overvalued"):
            reasons = factors.get("expectation_reasons", [])
            if reasons:
                add("⚠️ 预期透支", reasons[0], "", "🚫")

        if factors.get("event_priced"):
            signals = factors.get("event_pricing_signals", [])
            if signals:
                add("⚠️ 利好兑现", signals[0], "", "🚫")

        if factors.get("profit_taking_detected"):
            signals = factors.get("profit_taking_signals", [])
            if signals:
                add("⚠️ 获利出逃", signals[0], "", "🚫")

        quant_detected = factors.get("quant_detected", False)
        quant_risk = factors.get("quant_risk", "low")
        quant_signals = factors.get("quant_signals", [])
        if quant_detected:
            qs = quant_signals[0] if quant_signals else ""
            if quant_risk == "high":
                add("⚠️ 量化风险", f"高（{qs}）", "", "🚫")
            elif quant_risk == "medium":
                add("⚠️ 量化风险", f"中（{qs}）", "", "⚠️")
            else:
                add("量化行为", qs, "", star(3))

        shareholder_signals = factors.get("shareholder_signals", [])
        shareholder_trend = factors.get("shareholder_trend", "neutral")
        if shareholder_signals:
            if shareholder_trend == "increase":
                add("股东", shareholder_signals[0], "", star(5))
            elif shareholder_trend == "decrease":
                add("⚠️ 股东", shareholder_signals[0], "", "🚫")

        lockup_signals = factors.get("lockup_signals", [])
        lockup_risk = factors.get("lockup_risk", "low")
        if lockup_signals and lockup_risk == "high":
            add("⚠️ 解禁", lockup_signals[0], "", "🚫")
