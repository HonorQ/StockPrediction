"""
报告生成器（HTML格式）
"""
from src.utils.log import get_logger
logger = get_logger()

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
    ) -> str:
        """生成完整HTML推荐报告

        Args:
            intraday_note: 盘中修正说明标记。设置后会在标题追加"（xxx修正版）"
                并在头部插入黄色提示横幅。
        """
        trade_cfg = trade_config or {}
        stop_loss_pct = trade_cfg.get("stop_loss_pct", -0.05)
        take_profit_pct = trade_cfg.get("take_profit_pct", 0.08)
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M")

        sections = []
        sections.append(self._build_head(date_str, intraday_note))
        sections.append(self._build_header(date_str, time_str, intraday_note))
        sections.append(self._build_intraday_notice(intraday_note))
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
    def _build_head(self, date_str: str, intraday_note: str = None) -> str:
        title_suffix = f"（{intraday_note}修正版）" if intraday_note else ""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>每日精选{title_suffix} - {date_str}</title>
<style>
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
</style>
</head>
<body>
<div class="container">"""

    def _build_header(self, date_str: str, time_str: str, intraday_note: str = None) -> str:
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

    def _build_intraday_notice(self, intraday_note: str = None) -> str:
        """盘中修正说明横幅（仅当 intraday_note 非空时返回内容）"""
        if not intraday_note:
            return ""
        return f"""
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
            for n in items[:5]:
                content = str(n.get("content", "")).strip()
                title = str(n.get("title", "")).strip()
                if not content:
                    continue
                if len(content) > 80:
                    # 完整全文始终保留在页面里，前 80 字做摘要 + details 展开
                    snippet = content[:80] + "…"
                    full = content
                    item_htmls.append(f"""
    <div class="news-item">
      <div class="news-txt"><details><summary>{_html.escape(title) if title else ""}{_html.escape(snippet)}</summary><div class="news-full">{_html.escape(full)}</div></details></div>
    </div>""")
                else:
                    item_htmls.append(f"""
    <div class="news-item">
      <div class="news-txt">{_html.escape(title) + " " if title else ""}{_html.escape(content)}</div>
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
    def _build_recommendations(self, recommendations: List[Dict], stop_loss_pct: float, take_profit_pct: float) -> str:
        if not recommendations:
            return """
<section class="sec">今日精选</section>
<div class="card">
  <p style="color:#8a8578;text-align:center;padding:20px;">今日无达标股票，建议观望</p>
</div>"""

        cards = []
        for i, rec in enumerate(recommendations[:2], 1):
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

        score_cls = "score-high" if score >= 70 else ("score-mid" if score >= 50 else "score-low")

        # 因子表格行
        factor_rows = self._build_factor_rows(factors)
        factor_html = ""
        if factor_rows:
            factor_html = '<div class="factor"><div class="ft">关键因子</div>' + "".join(factor_rows) + "</div>"

        # 推荐理由
        reason_html = self._build_reasons(stock)

        # 操作建议
        dynamic_stop = factors.get("dynamic_stop_loss", 0)
        dynamic_take = factors.get("dynamic_take_profit", 0)
        dynamic_stop_pct = factors.get("dynamic_stop_loss_pct", None)
        dynamic_take_pct = factors.get("dynamic_take_profit_pct", None)

        if dynamic_stop > 0 and dynamic_take > 0:
            target = dynamic_take
            stop = dynamic_stop
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
            reasons.append(("护城河深", f"{moat_level}（{moat_score:.0f}分）"))
        elif moat_score >= 55:
            reasons.append(("护城河稳健", f"{moat_level}（{moat_score:.0f}分）"))
        elif moat_score >= 40:
            reasons.append(("基本盘稳定", f"护城河 {moat_score:.0f} 分，行业地位中上"))

        if bp.get("is_leader") or factors.get("is_leader"):
            reasons.append(("行业龙头", "细分领域具备规模与话语权"))

        # 技术面
        if factors.get("macd_golden_cross"):
            reasons.append(("MACD 金叉", "短期趋势转强信号确认"))
        if factors.get("ma_bullish"):
            reasons.append(("均线多头", "短期均线多头排列，趋势向上"))
        if factors.get("breakout_20d_high"):
            reasons.append(("突破新高", "放量突破 20 日高点，打开上方空间"))
        if factors.get("volume_price_up"):
            reasons.append(("量价齐升", "成交量与价格同步放大，资金主动介入"))

        # 消息面
        news_score = factors.get("news_score", 0)
        if news_score > 75:
            reasons.append(("消息面偏暖", f"新闻情绪 {news_score:.0f} 分，偏正面"))

        # 情绪 / 人气
        hot_rank = factors.get("hot_rank", 0)
        if 0 < hot_rank <= 30:
            reasons.append(("人气靠前", f"热股榜第 {hot_rank} 名"))

        # 机构
        inst_count = factors.get("inst_holding_count", 0)
        if inst_count > 10:
            reasons.append(("机构关注", f"{inst_count} 家机构持股"))

        # 相对强弱
        rs_5d = factors.get("relative_strength_5d", 0)
        if rs_5d > 2:
            reasons.append(("相对强势", "5 日相对强弱显著跑赢市场"))

        # 超跌反弹 / 均值回归
        if factors.get("reversion_signal") in ("strong_buy", "buy"):
            reasons.append(("超卖修复", "偏离均线超卖，存在均值回归动能"))

        # 板块联动
        sector_signals = factors.get("sector_signals", []) or []
        for s in sector_signals[:1]:
            if "涨" in s:
                reasons.append(("板块共振", s))
                break

        # 游资 / 涨停质量
        if factors.get("limit_quality") in ("连板", "一字板"):
            reasons.append(("涨停质量高", "连板/一字板，资金一致性极强"))

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
  <div class="op-title">业务画像 / 不可替代性</div>
  <div class="biz-line"><b>行业:</b> {industry} &nbsp; <b>赛道:</b> {theme_html}</div>
  <div class="biz-line">{concept_tags}</div>
  <div class="biz-line"><span class="moat-val" style="color:{level_color};">{moat_score:.1f}</span>/100 &nbsp; <b style="color:{level_color};">{moat_level}</b>{sub_html}</div>
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
            conf = factors.get("prediction_confidence", 0)
            vol_regime = factors.get("volatility_regime", "")
            pos_at = factors.get("position_at", "")
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
        # 按日期分组：覆盖最近 N 天、每天最多2条，避免单日多条推荐占满回顾表
        # （8/14 案例：同一天盘前+午后产生 9 条推荐，旧的 [:10] 直接挤掉其他天）
        from collections import defaultdict
        by_date = defaultdict(list)
        for h in history_records:
            by_date[h["trade_date"]].append(h)
        days = sorted(by_date.keys(), reverse=True)[:7]
        rows = []
        for d in days:
            recs = sorted(by_date[d], key=lambda x: x.get("score", 0), reverse=True)[:2]
            for h in recs:
                d1 = f'{h.get("d1_return", 0):+.1f}%' if h.get("d1_return") is not None else "-"
                d3 = f'{h.get("d3_return", 0):+.1f}%' if h.get("d3_return") is not None else "-"
                d5 = f'{h.get("d5_return", 0):+.1f}%' if h.get("d5_return") is not None else "-"
                d10 = f'{h.get("d10_return", 0):+.1f}%' if h.get("d10_return") is not None else "-"
                d1_cls = "up" if h.get("d1_return", 0) and h["d1_return"] > 0 else ("down" if h.get("d1_return", 0) and h["d1_return"] < 0 else "flat")
                d3_cls = "up" if h.get("d3_return", 0) and h["d3_return"] > 0 else ("down" if h.get("d3_return", 0) and h["d3_return"] < 0 else "flat")
                rows.append(f'<tr><td>{h["trade_date"]}</td><td>{h["code"]}</td><td>{h["name"]}</td><td>{h["score"]}</td><td class="{d1_cls}">{d1}</td><td class="{d3_cls}">{d3}</td><td>{d5}</td><td>{d10}</td></tr>')
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
        """保存报告到HTML文件"""
        if filename is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join("reports", "daily", f"{date_str}.html")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f'  ✅ 报告已保存: {filename}')
        return filename
