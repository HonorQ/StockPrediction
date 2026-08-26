# -*- coding: utf-8 -*-
"""每日报告的 CSS 样式(由 reporter.py 外移, 纯静态资源)。"""

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
