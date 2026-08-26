"""
报告生成器（HTML格式）
"""
from src.utils.log import get_logger
logger = get_logger()
from src.engine.report_styles import CSS

from datetime import datetime
from typing import List, Dict
import os
import html as _html


class ReportGenerator:
    """推荐报告生成器"""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    # 报告身份标注。
    # 决策流程已收敛为"每日一次"：14:40 午后版是当日唯一的正式推荐与买入依据；
    # 盘前(research)/盘中(warning)仅为过程产物，不再构成独立的推荐记录
    # （历史回顾表也按"每天一条最终推荐"展示，见 _build_history）
    REPORT_ROLES = {
        "research": {
            "title_suffix": "· 盘前研究池",
            "banner_title": "研究池说明",
            "banner": ("本报告为 07:00 盘前全量因子计算产出的观察候选池，非买入依据、不构成当日推荐。"
                       "当日唯一正式推荐以 14:40 午后《每日推荐》为准。"),
            "bg": "#eef2f6", "border": "#7a8ba3", "color": "#3f4c63",
        },
        "warning": {
            "title_suffix": "· 盘中预警(非买入依据)",
            "banner_title": "盘中预警",
            "banner": ("09:25 盘中仅负责：崩盘清仓检测、不可交易标记(涨跌停)、高风险票预警。"
                       "本报告不含推荐列表，当日唯一正式推荐以 14:40 午后《每日推荐》为准。"),
            "bg": "#fdf3e0", "border": "#e0a800", "color": "#8a6d1a",
        },
        "final": {
            "title_suffix": "· 每日推荐(14:40)",
            "banner_title": "今日推荐",
            "banner": ("本报告为每日 14:40 生成的唯一正式推荐：基于全天真实走势/主力资金流验证，"
                       "按 T+1~T+3 持有口径选股。下单窗口 14:45-14:55(收盘竞价前)。"),
            "bg": "#e9f5ec", "border": "#3d9153", "color": "#25603a",
        },
    }

    def generate_report(
        self,
        recommendations: List[Dict],
        market_overview: Dict,
        global_markets: Dict,
        news_analysis: Dict,
        sentiment: Dict,
        trade_config: dict = None,
        rotation_data: Dict = None,
        history_records: List[Dict] = None,
        intraday_note: str = None,
        report_role: str = None,
    ) -> str:
        """生成完整HTML推荐报告

        Args:
            intraday_note: 盘中修正说明标记。设置后会在标题追加"（xxx修正版）"
                并在头部插入黄色提示横幅。
            report_role: 报告身份(research=盘前研究池/warning=盘中预警/final=尾盘最终决策)，
                提供时覆盖标题后缀与横幅样式，明确各时点报告的决策地位。
        """
        role = self.REPORT_ROLES.get(report_role) if report_role else None
        trade_cfg = trade_config or {}
        stop_loss_pct = trade_cfg.get("stop_loss_pct", -0.05)
        take_profit_pct = trade_cfg.get("take_profit_pct", 0.08)
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M")

        sections = []
        sections.append(self._build_head(date_str, intraday_note, role))
        sections.append(self._build_header(date_str, time_str, intraday_note, role))
        sections.append(self._build_intraday_notice(intraday_note, role))
        sections.append(self._build_market(market_overview))
        sections.append(self._build_global(global_markets))
        sections.append(self._build_sentiment(sentiment, rotation_data))
        sections.append(self._build_recommendations(recommendations, stop_loss_pct, take_profit_pct))
        sections.append(self._build_news(news_analysis))
        sections.append(self._build_history(history_records))
        sections.append(self._build_footer())
        sections.append("</body></html>")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # 各段落构建
    # ------------------------------------------------------------------
    def _build_head(self, date_str: str, intraday_note: str = None,
                    role: dict = None) -> str:
        if role:
            title_suffix = f"（{role['title_suffix']}）"
        else:
            title_suffix = f"（{intraday_note}修正版）" if intraday_note else ""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>每日精选{title_suffix} - {date_str}</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="container">"""

    def _build_header(self, date_str: str, time_str: str, intraday_note: str = None,
                      role: dict = None) -> str:
        if role:
            title_suffix = f"（{role['title_suffix']}）"
        else:
            title_suffix = f"（{intraday_note}修正版）" if intraday_note else ""
        return f"""
<div class="masthead">
  <div class="kicker">量化选股 · 每日晨报</div>
  <h1>今日精选{title_suffix}</h1>
  <div class="masthead-sub"></div>
</div>
<div class="dateline">
  <span>{date_str} <strong>星期{'一二三四五六日'[datetime.now().weekday()]}</strong></span>
  <span>{time_str} · 仅供参考</span>
</div>"""

    def _build_intraday_notice(self, intraday_note: str = None,
                               role: dict = None) -> str:
        """报告身份横幅：report_role 提供时输出对应身份横幅，否则退回盘中修正说明"""
        if role:
            return f"""
<div style="background:{role['bg']};border-left:3px solid {role['border']};padding:10px 16px;margin:14px 0;border-radius:2px;font-size:13px;color:{role['color']};">🧭 <b>{role['banner_title']}</b>: {role['banner']}</div>"""
        if not intraday_note:
            return ""
        return """
<div style="background:#f6efe0;border-left:3px solid #e0a800;padding:10px 16px;margin:14px 0;border-radius:2px;font-size:13px;color:#6b5d3e;">⏱ <b>盘中修正说明</b>: 基于盘前深度分析(150+因子) + 集合竞价实时行情修正</div>"""

    def _build_market(self, market_overview: Dict) -> str:
        cards = []
        for name, data in market_overview.items():
            chg = data.get("change", 0)
            cls = "up" if chg > 0 else ("down" if chg < 0 else "flat")
            sign = "+" if chg > 0 else ""
            cards.append(f"""
    <div class="i-card">
      <div class="nm">{_html.escape(str(name))}</div>
      <div class="v">{data.get("price", "-")}</div>
      <div class="d {cls}">{sign}{chg:.2f}%</div>
    </div>""")
        if not cards:
            return ""
        return f"""
<h2 class="sec">市场概览</h2>
<div class="market-grid">
  <div class="i-grid">{"".join(cards)}</div>
</div>"""

    def _build_global(self, global_markets: Dict) -> str:
        rows = []
        for name, data in global_markets.items():
            if not data.get("available", True):
                rows.append(f'<tr><td>{_html.escape(str(name))}</td><td>—</td><td class="flat">—</td><td><span class="badge badge-neu">无数据</span></td></tr>')
                continue
            chg = data.get("change", 0)
            price = data.get("price", "-")
            sign = "+" if chg > 0 else ""
            cls = "up" if chg > 0 else ("down" if chg < 0 else "flat")
            impact = ("利好", "badge-pos") if chg > 0 else (("利空", "badge-neg") if chg < 0 else ("中性", "badge-neu"))
            rows.append(f'<tr><td>{_html.escape(str(name))}</td><td>{price}</td><td class="{cls}">{sign}{chg:.2f}%</td><td><span class="badge {impact[1]}">{impact[0]}</span></td></tr>')
        if not rows:
            return ""
        return f"""
<section class="sec">外盘环境</section>
<div class="card">
  <table>
    <thead><tr><th>市场</th><th>最新</th><th>涨跌</th><th>影响</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""

    def _build_sentiment(self, sentiment: Dict, rotation_data: Dict = None) -> str:
        emo_level = sentiment.get("emotion_level", "-")
        emo_score = sentiment.get("emotion_score", 0)
        limit_up = sentiment.get("limit_up", 0)
        limit_down = sentiment.get("limit_down", 0)
        broken = sentiment.get("broken", 0)
        max_board = sentiment.get("max_board", 0)

        grid_items = [
            (emo_level, "情绪评级"),
            (f"{emo_score:.0f}", "情绪得分"),
            (str(limit_up), "涨停"),
            (str(limit_down), "跌停"),
            (str(broken), "炸板"),
            (str(max_board) + "板", "最高连板"),
        ]

        grid_html = "".join(f'<div class="e-item"><div class="val">{v}</div><div class="lab">{l}</div></div>' for v, l in grid_items)

        extra = ""
        if rotation_data:
            signal = rotation_data.get("rotation_signal", "neutral")
            signal_text = {
                "tech_strong": "🔥 科技板块强势",
                "cyclical_strong": "📈 周期板块强势",
                "balanced": "⚖️ 板块均衡",
                "neutral": "📊 待观察",
            }.get(signal, signal)
            tech_str = rotation_data.get("tech_strength", 0)
            extra = f'<div style="font-size:13px;color:#555;margin-top:8px;">板块轮动: <b>{signal_text}</b> | 科技强度: {tech_str}</div>'
            if rotation_data.get("oversold_rebound"):
                rebound_sectors = rotation_data.get("rebound_sectors", [])
                extra += f'<div style="font-size:13px;color:#9a3b2a;margin-top:4px;">⚡ 超跌反弹: {", ".join(rebound_sectors[:5])}</div>'

        return f"""
<section class="sec">市场情绪</section>
<div class="card">
  <div class="e-grid">{grid_html}</div>
  {extra}
</div>"""

    def _build_news(self, news_analysis: Dict) -> str:
        category_labels = {
            "policy": "政策消息",
            "industry": "行业新闻",
            "rumor": "市场传闻（小作文）",
            "company": "公司新闻",
            "positive": "正面消息",
            "negative": "负面消息",
        }
        parts = []
        for cat, label in category_labels.items():
            items = news_analysis.get(cat, [])
            if not items:
                continue
            item_htmls = []
            for n in items[:3]:
                content = str(n.get("content", "")).strip()
                title = str(n.get("title", "")).strip()
                if not content:
                    continue
                snippet = content[:50] + "…" if len(content) > 50 else content
                item_htmls.append(f"""
    <div class="news-item">
      <div class="news-txt">{_html.escape(title + " " + snippet) if title else _html.escape(snippet)}</div>
    </div>""")
            if item_htmls:
                parts.append(f'<div class="news-section"><h3>{label}</h3>{"".join(item_htmls)}</div>')
        if not parts:
            return ""
        return f"""
<section class="sec">实时消息速览</section>
<div class="news-card">
  {"".join(parts)}
</div>"""

    # ------------------------------------------------------------------
    # 推荐段落
    # ------------------------------------------------------------------
    def _build_recommendations(self, recommendations: List[Dict], stop_loss_pct: float, take_profit_pct: float,
                               max_display: int = None) -> str:
        if not recommendations:
            return """
<section class="sec">今日精选</section>
<div class="card">
  <p style="color:#8a8578;text-align:center;padding:20px;">今日无达标股票，建议观望</p>
</div>"""

        # 展示数量与上游推荐数解耦：原硬编码 [:2] 使 regime bull 放宽到
        # 5 只的推荐永远只展示 2 只，展示层与推荐控制脱节且不可配置
        limit = max_display or len(recommendations)
        cards = []
        for i, rec in enumerate(recommendations[:limit], 1):
            cards.append(self._format_one_stock_html(rec, i, stop_loss_pct, take_profit_pct))
        return f"""
<section class="sec">今日精选</section>
{"".join(cards)}"""

    # ------------------------------------------------------------------
    # 单只股票HTML卡片
    # ------------------------------------------------------------------
    def _format_one_stock_html(self, stock: Dict, index: int, stop_loss_pct: float, take_profit_pct: float) -> str:
        name = _html.escape(str(stock.get("name", "")))
        code = _html.escape(str(stock.get("code", "")))
        score = stock.get("score", 0)
        price = stock.get("current_price", 0)
        factors = stock.get("factors", {})
        strong = stock.get("strong_signals", 0)

        star_count = min(int(score / 20) + 1, 5)
        stars_html = "★" * star_count + "☆" * (5 - star_count)


        # 因子表格行
        factor_rows = self._build_factor_rows(factors)
        factor_html = ""
        if factor_rows:
            factor_html = '<div class="factor"><div class="ft">关键因子</div>' + "".join(factor_rows) + "</div>"

        # 推荐理由
        reason_html = self._build_reasons(stock)
        # Alpha overlay 加成透明化: positive2 信号对最终分的调整量
        ab = stock.get("alpha_boost")
        if ab is not None and abs(ab) >= 0.5:
            icon = "↑" if ab > 0 else "↓"
            color = "#2e7d32" if ab > 0 else "#b3541e"
            reason_html += (
                f'<div style="margin-top:6px;font-size:12px;color:{color};">'
                f'{icon} Alpha信号加成 {ab:+.1f} 分(量化异动+超跌反弹合成)</div>')

        # 操作建议
        dynamic_stop = factors.get("dynamic_stop_loss", 0)
        dynamic_take = factors.get("dynamic_take_profit", 0)
        dynamic_stop_pct = factors.get("dynamic_stop_loss_pct", None)
        dynamic_take_pct = factors.get("dynamic_take_profit_pct", None)

        if dynamic_stop > 0 and dynamic_take > 0:
            target = dynamic_take
            stop = dynamic_stop
            # pct 字段上游可能只写了价格没写百分比，缺失时按价格反推而非 TypeError 崩溃
            if dynamic_take_pct is None:
                dynamic_take_pct = (dynamic_take / price - 1) * 100 if price else 0
            if dynamic_stop_pct is None:
                dynamic_stop_pct = (dynamic_stop / price - 1) * 100 if price else 0
            profit_pct_str = f"+{dynamic_take_pct:.1f}%"
            loss_pct_str = f"{dynamic_stop_pct:.1f}%"
        else:
            target = round(price * (1 + take_profit_pct), 2) if price else 0
            stop = round(price * (1 + stop_loss_pct), 2) if price else 0
            profit_pct_str = f"+{take_profit_pct * 100:.1f}%"
            loss_pct_str = f"{stop_loss_pct * 100:.1f}%"

        position_pct = factors.get("position_pct", None)
        position_shares = factors.get("position_shares", None)
        pos_html = ""
        if position_pct is not None and position_shares is not None:
            pos_html = f'<div class="op"><div class="l">建议仓位</div><div class="v">{position_pct*100:.0f}%（约{position_shares}股）</div></div>'

        # 卖出信号（杂志警示行）
        sell_html = ""
        sell_signals = factors.get("sell_signals", None)
        if sell_signals:
            sell_items = []
            for sig in sell_signals:
                action = sig.get("action", "")
                reason = sig.get("reason", "")
                sell_items.append(f"{action}: {reason}")
            sell_html = f'<div class="warn-row">⚠️ {" · ".join(sell_items)}</div>'

        # 业务画像 / 不可替代性（B层）
        biz_html = self._build_business_profile_html(stock.get("business_profile"))

        return f"""
<div class="stock">
  <div class="stk-head">
    <div>
      <span class="stk-name">{index}. {name}</span><span class="stk-code">({code})</span>
      <span class="stars">{stars_html}</span>
    </div>
    <div class="stk-score">综合评分<span class="score-num">{score}</span><span style="font-size:12px;color:#8a8578;">分 · 强信号 {strong}</span></div>
  </div>
  <div class="stk-body">
    {reason_html}
    <div class="op-grid">
      <div class="op"><div class="l">入手点</div><div class="v">{price:.2f} 元</div></div>
      <div class="op"><div class="l">目标价</div><div class="v up">{target:.2f} 元 {profit_pct_str}</div></div>
      <div class="op"><div class="l">止损价</div><div class="v down">{stop:.2f} 元 {loss_pct_str}</div></div>
      <div class="op"><div class="l">持仓周期</div><div class="v">1-3天</div></div>
      {pos_html}
    </div>
    {factor_html}
    {sell_html}
    {biz_html}
  </div>
</div>"""

    # ------------------------------------------------------------------
    # 推荐理由（为什么选它）—— 从真实 factors 提炼，不依赖 LLM
    # ------------------------------------------------------------------
    def _build_reasons(self, stock: Dict) -> str:
        factors = stock.get("factors", {}) or {}
        bp = stock.get("business_profile") or {}
        reasons = []

        # 护城河 / 业务地位
        moat_score = 0
        try:
            moat_score = float(bp.get("moat_score", 0) or 0)
        except (ValueError, TypeError):
            moat_score = 0
        moat_level = str(bp.get("moat_level", "") or "")
        if moat_score >= 70:
                reasons.append(("公司底子好", f"{moat_level}，护城河{moat_score:.0f}分"))
        elif moat_score >= 55:
            reasons.append(("公司底子不错", f"{moat_level}，护城河{moat_score:.0f}分"))

        is_leader = bp.get("is_leader") or factors.get("is_leader")
        leader_score = 0
        try:
            leader_score = float(bp.get("moat_leader", 0) or 0)
        except (ValueError, TypeError):
            leader_score = 0
        if is_leader and leader_score >= 75:
            reasons.append(("行业老大", "这个领域它说了算"))
        elif leader_score >= 60:
            reasons.append(("行业前几名", "在同行里排名靠前"))

        if factors.get("macd_golden_cross"):
            reasons.append(("技术指标转好", "MACD金叉，短期看涨"))
        if factors.get("ma_bullish"):
            reasons.append(("均线向上", "股价站上均线，趋势向好"))
        if factors.get("breakout_20d_high"):
            reasons.append(("突破近期高点", "放量突破20日新高"))
        if factors.get("volume_price_up"):
            reasons.append(("放量上涨", "量价齐升，有资金在买入"))

        news_score = factors.get("news_score", 0)
        if news_score > 75:
            reasons.append(("有利好消息", "市场消息面偏正面"))

        hot_rank = factors.get("hot_rank", 0)
        if 0 < hot_rank <= 30:
            reasons.append(("市场关注度高", f"热股榜第{hot_rank}名，很多人在看"))

        inst_count = factors.get("inst_holding_count", 0)
        if inst_count > 10:
            reasons.append(("机构看好", f"{inst_count}家机构持有"))

        rs_5d = factors.get("relative_strength_5d", 0)
        if rs_5d > 2:
            reasons.append(("走势比大盘强", "近5天涨得比大盘好"))

        if factors.get("reversion_signal") in ("strong_buy", "buy"):
            reasons.append(("跌多了会反弹", "之前跌太多，有反弹的可能"))

        sector_signals = factors.get("sector_signals", []) or []
        for s in sector_signals[:1]:
            if "涨" in s:
                reasons.append(("板块在涨", s))
                break

        if factors.get("limit_quality") in ("连板", "一字板"):
            reasons.append(("涨停质量高", "连续涨停，资金很看好"))

        if not reasons:
            return ""

        top = reasons[:3]
        lis = "".join(f"<li><b>{t}</b>：{d}</li>" for t, d in top)
        return f"""
    <div class="reason">
      <div class="rt">为什么选它</div>
      <ul>
      {lis}
      </ul>
    </div>"""

    # ------------------------------------------------------------------
    # 业务画像 / 不可替代性区块（B层：规则抽取，不依赖 LLM）
    # ------------------------------------------------------------------
    def _build_business_profile_html(self, bp: Dict) -> str:
        if not bp:
            return ""
        industry = _html.escape(str(bp.get("industry", "未知")))
        themes = bp.get("themes", []) or []
        concepts = bp.get("concepts", []) or []
        moat_labels = bp.get("moat_labels", []) or []
        moat_score = float(bp.get("moat_score", 50) or 50)
        moat_level = _html.escape(str(bp.get("moat_level", "")))

        theme_html = " · ".join(_html.escape(t) for t in themes) if themes else "—"
        concept_tags = "".join(
            f'<span class="tag">{_html.escape(str(c))}</span>' for c in concepts[:8]
        )
        strong_set = {"行业龙头", "机构重仓", "业务聚焦", "大市值权重"}
        moat_tags = "".join(
            f'<span class="tag tag-moat">{_html.escape(str(m))}</span>'
            if m in strong_set else f'<span class="tag tag-moat-neu">{_html.escape(str(m))}</span>'
            for m in moat_labels
        )

        level_color = "#c0392b" if moat_score >= 70 else ("#e67e22" if moat_score >= 55 else "#888")
        sub = []
        for key, label in [("moat_concept_rarity", "概念稀缺"), ("moat_leader", "龙头"),
                           ("moat_institution", "机构"), ("moat_market_position", "市值")]:
            v = bp.get(key)
            if v is not None:
                try:
                    sub.append(f"{label} {float(v):.0f}")
                except (ValueError, TypeError):
                    pass
        sub_html = f' <span style="color:#999;">({" / ".join(sub)})</span>' if sub else ""

        return f"""
<div class="biz-box">
  <div class="op-title">板块信息</div>
  <div class="biz-line"><b>行业:</b> {industry} &nbsp; <b>赛道:</b> {theme_html}</div>
  <div class="biz-line">{concept_tags}</div>
  <div class="biz-line"><b>竞争力:</b> <span class="moat-val" style="color:{level_color};">{moat_score:.1f}</span>/100 &nbsp; <b style="color:{level_color};">{moat_level}</b>{sub_html}</div>
  <div class="biz-line">{moat_tags}</div>
</div>"""

    # ------------------------------------------------------------------
    # 因子行构建
    # ------------------------------------------------------------------
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

        def star(n):
            return "★" * min(n, 5) + "☆" * max(0, 5 - n)

        # 基础因子
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

        # 策略信号
        strategies = factors.get("strategies_triggered", [])
        if strategies:
            add("策略", "、".join(strategies), "", star(5))

        # 筹码分布
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

        # 基本面
        pe = factors.get("pe", 0)
        pb = factors.get("pb", 0)
        mv = factors.get("total_mv_yi", 0)
        if pe > 0 and pe < 25:
            add("基本面", f"PE={pe:.1f}（低估）", "", star(4))
        if pb > 0 and pb < 2:
            add("基本面", f"PB={pb:.1f}（低估）", "", star(3))
        if 50 < mv < 500:
            add("基本面", f"市值{mv:.0f}亿（中盘）", "", star(3))

        # 人气
        hot_rank = factors.get("hot_rank", 0)
        if 0 < hot_rank <= 30:
            add("人气", f"热股榜第{hot_rank}名", "", star(5))

        # 机构持股
        inst_count = factors.get("inst_holding_count", 0)
        inst_ratio = factors.get("inst_holding_ratio", 0)
        if inst_count > 10:
            add("机构", f"{inst_count}家机构持股", "", star(4))
        elif inst_count > 0:
            add("机构", f"{inst_count}家机构持股", "", star(3))
        if inst_ratio > 20:
            add("机构", f"持股比例{inst_ratio:.1f}%", "", star(4))

        # 相对强弱
        rs_5d = factors.get("relative_strength_5d", 0)
        rs_20d = factors.get("relative_strength_20d", 0)
        if rs_5d > 2:
            add("强弱", f"5日超额收益+{rs_5d:.1f}%", "", star(4))
        elif rs_5d < -2:
            add("弱势", f"5日超额收益{rs_5d:.1f}%", "", "⚠️")
        if rs_20d > 5:
            add("强弱", f"20日超额收益+{rs_20d:.1f}%", "", star(5))

        # 超跌反弹
        if factors.get("oversold_rebound_detected"):
            rebound_signals = factors.get("oversold_rebound_signals", [])
            if rebound_signals:
                add("超跌反弹", ", ".join(rebound_signals[:3]), "", star(5))

        # 反弹持续性
        sustainability_score = factors.get("sustainability_score", 50)
        rebound_type = factors.get("rebound_type", "unknown")
        type_map = {"capital_driven": "资金驱动型", "emotion_repair": "情绪修复型", "technical": "技术性反弹"}
        if rebound_type in type_map:
            if sustainability_score >= 65:
                add("持续性", f"{type_map[rebound_type]}（{sustainability_score}分）", "", star(4))
            elif sustainability_score <= 40:
                add("持续性", f"{type_map[rebound_type]}（{sustainability_score}分）", "", "⚠️")

        # 次日预判
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

        # 深度研究
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
            # 深度研究信号
            dr_signals = factors.get("deep_research_signals", [])
            for s in dr_signals[:2]:
                if "⚠" in s:
                    add("深度研究", s, "", "⚠️")
                else:
                    add("深度研究", s, "", star(3))

        # 预期透支
        if factors.get("expectation_overvalued"):
            reasons = factors.get("expectation_reasons", [])
            if reasons:
                add("⚠️ 预期透支", reasons[0], "", "🚫")

        # 利好兑现
        if factors.get("event_priced"):
            signals = factors.get("event_pricing_signals", [])
            if signals:
                add("⚠️ 利好兑现", signals[0], "", "🚫")

        # 获利盘出逃
        if factors.get("profit_taking_detected"):
            signals = factors.get("profit_taking_signals", [])
            if signals:
                add("⚠️ 获利出逃", signals[0], "", "🚫")

        # A股特色因子
        reversion_signal = factors.get("reversion_signal", "neutral")
        if reversion_signal in ["strong_buy", "buy"]:
            add("均值回归", f"偏离20日均线{factors.get('deviation_20d', 0):.1f}%（超卖）", "", star(4))
        elif reversion_signal in ["strong_sell", "sell"]:
            add("均值回归", f"偏离20日均线{factors.get('deviation_20d', 0):.1f}%（超买）", "", "⚠️")

        # 波动率
        volatility_level = factors.get("volatility_level", "")
        if volatility_level == "low":
            add("波动率", "低波动（安全）", "", star(3))
        elif volatility_level in ["extreme_high", "high"]:
            add("波动率", "高波动（风险大）", "", "⚠️")

        # 游资行为
        hot_money_signals = factors.get("hot_money_signals", [])
        if hot_money_signals:
            add("游资", hot_money_signals[0], "", star(4))

        # 价格位置
        position_signal = factors.get("position_signal", "")
        if position_signal:
            if "底部" in position_signal:
                add("价格位置", position_signal, "", star(5))
            elif "顶部" in position_signal:
                add("价格位置", position_signal, "", "⚠️")
            else:
                add("价格位置", position_signal, "", star(3))

        # 情绪周期
        sentiment_cycle = factors.get("sentiment_cycle", {})
        cycle_phase = sentiment_cycle.get("cycle_phase", "")
        if cycle_phase in ["冰点", "低迷"]:
            add("情绪周期", f"{cycle_phase}（可布局）", "", star(4))
        elif cycle_phase in ["亢奋"]:
            add("情绪周期", f"{cycle_phase}（注意风险）", "", "⚠️")

        # 量化行为
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

        # 融资融券
        margin_signals = factors.get("margin_signals", [])
        margin_trend = factors.get("margin_trend", "neutral")
        if margin_signals:
            if margin_trend == "active":
                add("融资", margin_signals[0], "", star(4))
            elif "融券" in margin_signals[0]:
                add("⚠️ 融券", margin_signals[0], "", "⚠️")

        # 股东增减持
        shareholder_signals = factors.get("shareholder_signals", [])
        shareholder_trend = factors.get("shareholder_trend", "neutral")
        if shareholder_signals:
            if shareholder_trend == "increase":
                add("股东", shareholder_signals[0], "", star(5))
            elif shareholder_trend == "decrease":
                add("⚠️ 股东", shareholder_signals[0], "", "🚫")

        # 解禁风险
        lockup_signals = factors.get("lockup_signals", [])
        lockup_risk = factors.get("lockup_risk", "low")
        if lockup_signals and lockup_risk == "high":
            add("⚠️ 解禁", lockup_signals[0], "", "🚫")

        # 涨停板质量
        limit_signals = factors.get("limit_signals", [])
        limit_quality = factors.get("limit_quality", "unknown")
        if limit_signals:
            if limit_quality in ["连板", "一字板"]:
                add("涨停质量", limit_signals[0], "", star(5))
            elif limit_quality == "首板":
                add("涨停质量", limit_signals[0], "", star(4))
            elif limit_quality == "跌停":
                add("⚠️ 涨停质量", limit_signals[0], "", "🚫")

        # 板块联动
        sector_signals = factors.get("sector_signals", [])
        if sector_signals:
            for s in sector_signals[:2]:
                if "涨" in s:
                    add("板块联动", s, "", star(4))
                else:
                    add("板块联动", s, "", "⚠️")

        # 龙头信号
        leader_signals = factors.get("leader_signals", [])
        is_leader = factors.get("is_leader", False)
        if leader_signals:
            if is_leader:
                add("🏆 龙头", leader_signals[0], "", star(5))
            else:
                add("龙头", leader_signals[0], "", star(3))

        # 资金流向持续性
        flow_signals = factors.get("flow_continuity_signals", [])
        flow_continuity = factors.get("flow_continuity", "neutral")
        if flow_signals:
            if flow_continuity in ["strong_inflow", "inflow"]:
                add("资金持续", flow_signals[0], "", star(5))
            elif flow_continuity in ["strong_outflow", "outflow"]:
                add("⚠️ 资金流出", flow_signals[0], "", "🚫")
            elif flow_continuity == "reversal":
                add("资金逆转", flow_signals[0], "", star(4))

        # 机构调研
        research_signals = factors.get("research_signals", [])
        if research_signals:
            add("机构", research_signals[0], "", star(4))

        # 龙虎榜
        lh_signals = factors.get("lh_signals", [])
        if lh_signals:
            if any("出货" in s for s in lh_signals):
                add("⚠️ 龙虎榜", lh_signals[0], "", "🚫")
            else:
                add("龙虎榜", lh_signals[0], "", star(4))

        return rows

    # ------------------------------------------------------------------
    # 历史推荐
    # ------------------------------------------------------------------
    def _build_history(self, history_records: List[Dict]) -> str:
        if not history_records:
            return ""
        # 每日一次推荐口径：每天只展示一条最终推荐。
        # 过渡期同一天可能存在 afternoon + 旧盘前行，优先展示 afternoon 模式行；
        # （8/14 案例：旧"每日两次"设计下单日 9 条推荐挤占回顾表，
        #   先改为每日2条；决策流程收敛为每日一次后进一步收敛为每日1条）
        from collections import defaultdict
        by_date = defaultdict(list)
        for h in history_records:
            # .get 防御缺列(如旧库行无 score)；展示字段统一 HTML 转义，
            # 名称来自行情源，含特殊字符会破坏表格结构
            by_date[str(h.get("trade_date", ""))].append(h)
        days = sorted(by_date.keys(), reverse=True)[:7]
        rows = []
        for d in days:
            recs = sorted(
                by_date[d],
                key=lambda x: (x.get("mode") != "afternoon", -x.get("score", 0)),
            )[:1]
            for h in recs:
                d1 = f'{h.get("d1_return", 0):+.1f}%' if h.get("d1_return") is not None else "-"
                d3 = f'{h.get("d3_return", 0):+.1f}%' if h.get("d3_return") is not None else "-"
                d5 = f'{h.get("d5_return", 0):+.1f}%' if h.get("d5_return") is not None else "-"
                d10 = f'{h.get("d10_return", 0):+.1f}%' if h.get("d10_return") is not None else "-"
                d1_cls = "up" if h.get("d1_return", 0) and h["d1_return"] > 0 else ("down" if h.get("d1_return", 0) and h["d1_return"] < 0 else "flat")
                d3_cls = "up" if h.get("d3_return", 0) and h["d3_return"] > 0 else ("down" if h.get("d3_return", 0) and h["d3_return"] < 0 else "flat")
                rows.append(
                    f'<tr><td>{_html.escape(str(h.get("trade_date", "")))}</td>'
                    f'<td>{_html.escape(str(h.get("code", "")))}</td>'
                    f'<td>{_html.escape(str(h.get("name", "")))}</td>'
                    f'<td>{_html.escape(str(h.get("score", "")))}</td>'
                    f'<td class="{d1_cls}">{d1}</td>'
                    f'<td class="{d3_cls}">{d3}</td>'
                    f'<td>{d5}</td><td>{d10}</td></tr>'
                )
        return f"""
<section class="sec">历史推荐回顾</section>
<div class="card">
  <table>
    <thead><tr><th>日期</th><th>代码</th><th>名称</th><th>评分</th><th>1日</th><th>3日</th><th>5日</th><th>10日</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""

    # ------------------------------------------------------------------
    def _build_footer(self) -> str:
        return """
<div class="footer">
  本报告基于技术分析和量化因子自动生成，仅供参考，不构成投资建议。<br>
  股市有风险，投资需谨慎。
</div>"""

    # ------------------------------------------------------------------
    def save_report(self, content: str, filename: str = None) -> str:
        """保存报告到HTML文件。
        相对路径基于项目根目录解析，不再依赖进程 CWD
        （计划任务/CWD 不同的场景会把报告写到别处）。"""
        if filename is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join("reports", "daily", f"{date_str}.html")
        if not os.path.isabs(filename):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            filename = os.path.join(project_root, filename)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f'  ✅ 报告已保存: {filename}')
        return filename
