"""
报告生成器（HTML格式）
"""
from src.utils.log import get_logger
logger = get_logger()
from src.engine.report_styles import CSS, FactorRowBuilder

from datetime import datetime
from typing import List, Dict
import os
import html as _html


class ReportGenerator:
    """推荐报告生成器"""

    def __init__(self):
        self._row_builder = FactorRowBuilder()

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
        yesterday_tracking: List[Dict] = None,
    ) -> str:
        """生成完整HTML推荐报告

        Args:
            intraday_note: 盘中修正说明标记。设置后会在标题追加"（xxx修正版）"
                并在头部插入黄色提示横幅。
            report_role: 报告身份(research=盘前研究池/warning=盘中预警/final=尾盘最终决策)，
                提供时覆盖标题后缀与横幅样式，明确各时点报告的决策地位。
            yesterday_tracking: 昨日推荐跟踪列表，每项包含:
                - code: 股票代码
                - name: 股票名称
                - yesterday_score: 昨日评分
                - yesterday_price: 昨日推荐价格
                - current_price: 当前价格
                - change_pct: 今日涨跌幅(%)
                - action: 操作建议(持有/关注/淘汰)
                - reason: 操作理由
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
        sections.append(self._build_yesterday_tracking(yesterday_tracking))
        sections.append(self._build_recommendations(recommendations, stop_loss_pct, take_profit_pct))
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
        except (ValueError, TypeError) as e:
            logger.warning("捕获异常已忽略(src/engine/reporter.py:395): %s", e)
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
        except (ValueError, TypeError) as e:
            logger.warning("捕获异常已忽略(src/engine/reporter.py:407): %s", e)
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
                except (ValueError, TypeError) as e:
                    logger.warning("捕获异常已忽略(src/engine/reporter.py:496): %s", e)
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
    # 因子行构建（代理到 FactorRowBuilder）
    # ------------------------------------------------------------------
    @staticmethod
    def _stars(n):
        """★/☆ 强度条文本(最多5星)。"""
        return FactorRowBuilder._stars(n)

    def _build_factor_rows(self, factors: Dict) -> List[str]:
        return self._row_builder._build_factor_rows(factors)

    def _factor_rows_tech(self, factors: Dict, add) -> None:
        self._row_builder._factor_rows_tech(factors, add)

    def _factor_rows_sentiment(self, factors: Dict, add) -> None:
        self._row_builder._factor_rows_sentiment(factors, add)

    def _factor_rows_fundamental(self, factors: Dict, add) -> None:
        self._row_builder._factor_rows_fundamental(factors, add)

    def _factor_rows_event(self, factors: Dict, add) -> None:
        self._row_builder._factor_rows_event(factors, add)

    # ------------------------------------------------------------------
    # 昨日推荐跟踪
    # ------------------------------------------------------------------
    def _build_yesterday_tracking(self, tracking_data: List[Dict]) -> str:
        """构建昨日推荐跟踪段落，展示每只股票的今日表现和操作建议。"""
        if not tracking_data:
            return ""

        rows = []
        for item in tracking_data:
            code = _html.escape(str(item.get("code", "")))
            name = _html.escape(str(item.get("name", "")))
            yesterday_score = item.get("yesterday_score", "")
            yesterday_price = item.get("yesterday_price", 0)
            current_price = item.get("current_price", 0)
            change_pct = item.get("change_pct", 0)
            action = item.get("action", "持有")
            reason = _html.escape(str(item.get("reason", "")))

            # 涨跌幅样式
            if change_pct > 0:
                change_cls = "up"
                change_str = f"+{change_pct:.2f}%"
            elif change_pct < 0:
                change_cls = "down"
                change_str = f"{change_pct:.2f}%"
            else:
                change_cls = "flat"
                change_str = "0.00%"

            # 操作建议样式
            action_cls = {
                "持有": "action-hold",
                "关注": "action-watch",
                "淘汰": "action-discard"
            }.get(action, "action-hold")

            rows.append(
                f'<tr>'
                f'<td>{code}</td>'
                f'<td>{name}</td>'
                f'<td>{yesterday_score}</td>'
                f'<td>{yesterday_price:.2f}</td>'
                f'<td>{current_price:.2f}</td>'
                f'<td class="{change_cls}">{change_str}</td>'
                f'<td><span class="{action_cls}">{action}</span></td>'
                f'<td>{reason}</td>'
                f'</tr>'
            )

        return f"""
<section class="sec">昨日推荐跟踪</section>
<div class="card">
  <div class="tracking-note">以下为昨日推荐股票的今日表现和操作建议，仅供参考。</div>
  <table>
    <thead>
      <tr>
        <th>代码</th>
        <th>名称</th>
        <th>昨日评分</th>
        <th>昨日价格</th>
        <th>当前价格</th>
        <th>今日涨跌</th>
        <th>操作建议</th>
        <th>理由</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""

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
    def save_report(self, content: str, filename: str = None, report_role: str = None,
                    recommendations: List[Dict] = None, **kwargs) -> str:
        """保存报告到HTML文件，并更新汇总报告。

        方案B实现：单文件模式
        - 最新推荐完整展示在 index.html
        - 历史推荐以摘要表格形式保存

        Args:
            content: 完整的HTML报告内容
            filename: 自定义文件名（单文件模式下忽略）
            report_role: 报告类型(research/warning/final)
            recommendations: 推荐列表，用于生成摘要
            **kwargs: 其他参数（市场数据等，用于历史记录）

        Returns:
            报告文件路径（index.html）
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        index_path = os.path.join(project_root, "reports", "daily", "index.html")
        history_path = os.path.join(project_root, "reports", "daily", "history.json")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # 1. 加载历史数据
        history = self._load_history(history_path)

        # 2. 生成当前报告的摘要
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        weekday = "一二三四五六日"[now.weekday()]

        role_names = {
            "research": "盘前研究池",
            "warning": "盘中预警",
            "final": "每日推荐"
        }
        role_name = role_names.get(report_role, "报告")

        # 推荐股票摘要
        stock_summary = []
        if recommendations:
            for r in recommendations[:5]:
                stock_summary.append({
                    "code": r.get("code", ""),
                    "name": r.get("name", ""),
                    "score": r.get("score", 0),
                    "price": r.get("current_price", 0),
                })

        new_entry = {
            "date": date_str,
            "time": time_str,
            "weekday": weekday,
            "role": role_name,
            "role_key": report_role or "unknown",
            "stocks": stock_summary,
            "content": content,  # 保存完整HTML内容
        }

        # 3. 插入到历史最前面（最新的在上面）
        history.insert(0, new_entry)

        # 4. 限制历史记录数量（保留最近30天）
        max_history = 30
        if len(history) > max_history:
            history = history[:max_history]

        # 5. 保存历史数据
        self._save_history(history_path, history)

        # 6. 生成单文件HTML
        index_html = self._build_single_file_html(history)

        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_html)

        logger.info(f'  ✅ 报告已保存: {index_path}')
        return index_path

    def _load_history(self, history_path: str) -> List[Dict]:
        """加载历史数据"""
        import json
        if not os.path.exists(history_path):
            return []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载历史数据失败: {e}")
            return []

    def _save_history(self, history_path: str, history: List[Dict]) -> None:
        """保存历史数据"""
        import json
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _fmt_return_cell(value) -> str:
        """格式化收益率为 HTML 单元格（平均值）"""
        if value is None:
            return '<span class="return-flat">-</span>'
        if value > 0:
            return f'<span class="return-up">+{value:.2f}%</span>'
        elif value < 0:
            return f'<span class="return-down">{value:.2f}%</span>'
        return f'<span class="return-flat">{value:.2f}%</span>'

    @staticmethod
    def _fmt_return_inline(value) -> str:
        """格式化收益率为内联文本"""
        if value is None:
            return '<span class="return-flat">-</span>'
        if value > 0:
            return f'<span class="return-up">+{value:.1f}%</span>'
        elif value < 0:
            return f'<span class="return-down">{value:.1f}%</span>'
        return f'<span class="return-flat">{value:.1f}%</span>'

    def _build_single_file_html(self, history: List[Dict]) -> str:
        """生成单文件HTML：最新推荐完整展示 + 历史摘要表格（含收益率）"""
        role_names = {
            "research": "盘前研究池",
            "warning": "盘中预警",
            "final": "每日推荐"
        }

        # 最新推荐的完整内容
        latest_content = ""
        if history:
            latest_content = history[0].get("content", "")

        # 历史摘要表格（从第2条开始）
        history_rows = ""
        for idx, entry in enumerate(history[1:]):
            role_key = entry.get("role_key", "unknown")
            role_name = entry.get("role", role_names.get(role_key, "报告"))
            date = entry.get("date", "")
            time = entry.get("time", "")
            weekday = entry.get("weekday", "")
            stocks = entry.get("stocks", [])

            # 计算平均收益率
            d1_returns = [s["d1_return"] for s in stocks if s.get("d1_return") is not None]
            d3_returns = [s["d3_return"] for s in stocks if s.get("d3_return") is not None]
            d5_returns = [s["d5_return"] for s in stocks if s.get("d5_return") is not None]
            rtd_returns = [s["return_to_date"] for s in stocks if s.get("return_to_date") is not None]

            avg_d1 = sum(d1_returns) / len(d1_returns) if d1_returns else None
            avg_d3 = sum(d3_returns) / len(d3_returns) if d3_returns else None
            avg_d5 = sum(d5_returns) / len(d5_returns) if d5_returns else None
            avg_rtd = sum(rtd_returns) / len(rtd_returns) if rtd_returns else None

            # 生成收益率单元格
            d1_html = self._fmt_return_cell(avg_d1)
            d3_html = self._fmt_return_cell(avg_d3)
            d5_html = self._fmt_return_cell(avg_d5)
            rtd_html = self._fmt_return_cell(avg_rtd)

            role_class = {
                "research": "role-research",
                "warning": "role-warning",
                "final": "role-final"
            }.get(role_key, "")

            # 构建展开内容（每只股票的详细信息）
            detail_rows = ""
            for s in stocks[:10]:  # 最多展示10只
                s_d1 = self._fmt_return_inline(s.get("d1_return"))
                s_d3 = self._fmt_return_inline(s.get("d3_return"))
                s_d5 = self._fmt_return_inline(s.get("d5_return"))
                s_rtd = self._fmt_return_inline(s.get("return_to_date"))
                price = s.get('price', 0)
                price_str = f"{price:.2f}" if price else "-"
                detail_rows += f"""
                <tr class="detail-row">
                  <td>{s.get('code', '')}</td>
                  <td>{s.get('name', '')}</td>
                  <td>{s.get('score', 0):.1f}</td>
                  <td>{price_str}</td>
                  <td>{s_d1}</td>
                  <td>{s_d3}</td>
                  <td>{s_d5}</td>
                  <td>{s_rtd}</td>
                </tr>"""
            if len(stocks) > 10:
                detail_rows += f"""
                <tr class="detail-row"><td colspan="8" style="text-align:center;color:#8a8578;">... 共 {len(stocks)} 只股票</td></tr>"""

            history_rows += f"""
        <tr class="summary-row" onclick="toggleDetail('detail-{idx}')" style="cursor:pointer">
          <td>{date}</td>
          <td>{weekday}</td>
          <td><span class="role-badge {role_class}">{role_name}</span></td>
          <td>{len(stocks)}只</td>
          <td>{d1_html}</td>
          <td>{d3_html}</td>
          <td>{d5_html}</td>
          <td>{rtd_html}</td>
        </tr>
        <tr id="detail-{idx}" class="detail-container" style="display:none">
          <td colspan="8" style="padding:0">
            <table class="detail-table">
              <thead><tr><th>代码</th><th>名称</th><th>评分</th><th>入手价</th><th>T+1</th><th>T+3</th><th>T+5</th><th>至今</th></tr></thead>
              <tbody>{detail_rows}</tbody>
            </table>
          </td>
        </tr>"""

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>每日推荐报告</title>
    <style>
        :root {{
            --paper: #faf8f4;
            --card: #ffffff;
            --ink: #1a1a1a;
            --ink-2: #555;
            --muted: #8a8578;
            --line: #e7e1d3;
            --accent: #9a3b2a;
            --up: #c0392b;
            --down: #1e7d4f;
            --research-bg: #eef2f6;
            --research-border: #7a8ba3;
            --warning-bg: #fdf3e0;
            --warning-border: #e0a800;
            --final-bg: #e9f5ec;
            --final-border: #3d9153;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--paper);
            color: var(--ink);
            font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
            line-height: 1.7;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 48px 28px 64px;
        }}
        .header {{
            border-bottom: 3px double var(--ink);
            padding-bottom: 18px;
            margin-bottom: 32px;
        }}
        .header h1 {{
            font-family: Georgia, 'Songti SC', 'SimSun', serif;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .header .subtitle {{
            font-size: 14px;
            color: var(--muted);
        }}
        .section-title {{
            font-family: Georgia, 'Songti SC', 'SimSun', serif;
            font-size: 20px;
            font-weight: 700;
            margin: 32px 0 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--accent);
            display: inline-block;
        }}
        .latest-report {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 24px;
            margin-bottom: 32px;
        }}
        .history-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 4px;
            overflow: hidden;
        }}
        .history-table th {{
            background: #f5f3ee;
            text-align: left;
            padding: 14px 16px;
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            border-bottom: 1px solid var(--line);
        }}
        .history-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid #f3efe4;
            font-size: 14px;
        }}
        .history-table tr:last-child td {{
            border-bottom: none;
        }}
        .summary-row:hover {{
            background: #faf8f4;
        }}
        .summary-row td:first-child::before {{
            content: "▶";
            display: inline-block;
            width: 16px;
            font-size: 10px;
            color: var(--muted);
            margin-right: 4px;
        }}
        .role-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
        }}
        .role-research {{ background: var(--research-bg); color: var(--research-border); border: 1px solid var(--research-border); }}
        .role-warning {{ background: var(--warning-bg); color: var(--warning-border); border: 1px solid var(--warning-border); }}
        .role-final {{ background: var(--final-bg); color: var(--final-border); border: 1px solid var(--final-border); }}
        .return-up {{ color: var(--up); font-weight: 600; }}
        .return-down {{ color: var(--down); font-weight: 600; }}
        .return-flat {{ color: var(--muted); }}
        .detail-container td {{ padding: 0 !important; border-bottom: 2px solid var(--line) !important; }}
        .detail-table {{ width: 100%; border-collapse: collapse; background: #faf8f4; }}
        .detail-table th {{ background: #f0ece2; font-size: 11px; padding: 8px 12px; }}
        .detail-table td {{ font-size: 13px; padding: 6px 12px; border-bottom: 1px solid #f3efe4; }}
        .detail-table tr:last-child td {{ border-bottom: none; }}
        .empty-state {{
            text-align: center;
            padding: 40px 20px;
            color: var(--muted);
        }}
        .empty-state .icon {{
            font-size: 48px;
            margin-bottom: 16px;
        }}
        .footer {{
            text-align: center;
            color: var(--muted);
            font-size: 12px;
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--line);
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 24px 16px; }}
            .history-table {{ font-size: 12px; }}
            .history-table th, .history-table td {{ padding: 10px 8px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>每日推荐报告</h1>
            <div class="subtitle">量化选股系统</div>
        </div>

        <h2 class="section-title">最新推荐</h2>
        <div class="latest-report">
            {latest_content if latest_content else '<div class="empty-state"><div class="icon">&#x1F4CA;</div><p>暂无推荐数据</p></div>'}
        </div>

        <h2 class="section-title">历史记录</h2>
        <table class="history-table">
            <thead>
                <tr>
                    <th>日期</th>
                    <th>星期</th>
                    <th>类型</th>
                    <th>股票数</th>
                    <th>T+1均值</th>
                    <th>T+3均值</th>
                    <th>T+5均值</th>
                    <th>至今均值</th>
                </tr>
            </thead>
            <tbody>
                {history_rows if history_rows else '<tr><td colspan="8"><div class="empty-state"><p>暂无历史记录</p></div></td></tr>'}
            </tbody>
        </table>

        <div class="footer">
            本报告基于技术分析和量化因子自动生成，仅供参考，不构成投资建议。<br>
            股市有风险，投资需谨慎。
        </div>
    </div>
    <script>
    function toggleDetail(id) {{
        var el = document.getElementById(id);
        if (!el) return;
        var visible = el.style.display !== 'none';
        el.style.display = visible ? 'none' : '';
        var row = el.previousElementSibling;
        if (row) {{
            var arrow = row.querySelector('td:first-child');
            if (arrow) arrow.style.setProperty('--arrow', visible ? '&#x25B6;' : '&#x25BC;');
        }}
    }}
    </script>
</body>
</html>"""
