"""
一次性工具：把 reports/daily/ 下的旧版 HTML 报告迁移到新杂志样式。
复用 reporter.py 的新 CSS 模板 + 老报告里已有的数据。

局限（源头已丢失，无法恢复）：
  - 历史新闻在源头已被 [:120] 截断，迁移保留已截断内容
  - 历史股票卡的 factors / business_profile 未落库，迁移仅还原操作建议与评分，
    推荐理由与关键因子块在历史报告中不渲染（数据不存在）
"""
import logging
import os
import re
import sys
from bs4 import BeautifulSoup

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from src.engine.reporter import ReportGenerator


# ------------------------------------------------------------------
# 解析辅助
# ------------------------------------------------------------------
def _to_float(v):
    try:
        txt = str(v).replace("%", "").replace("+", "").replace("元", "").strip()
        return float(txt) if txt else 0.0
    except (ValueError, TypeError) as e:
        logging.warning("捕获异常已忽略(scripts/migrate_daily_reports.py:28): %s", e)
        return 0.0


def _to_float_or_none(v):
    txt = str(v).replace("%", "").replace("+", "").replace("元", "").strip()
    if txt in ("-", "", "--"):
        return None
    try:
        return float(txt)
    except (ValueError, TypeError) as e:
        logging.warning("捕获异常已忽略(scripts/migrate_daily_reports.py:38): %s", e)
        return None


def _to_int(v):
    txt = str(v).replace("板", "").strip()
    try:
        return int(float(txt))
    except (ValueError, TypeError) as e:
        logging.warning("捕获异常已忽略(scripts/migrate_daily_reports.py:46): %s", e)
        return 0


_CAT_MAP = {
    "政策消息": "policy", "行业新闻": "industry",
    "市场传闻（小作文）": "rumor", "公司新闻": "company",
    "正面消息": "positive", "负面消息": "negative",
}


def _extract_factor_rows(sc):
    """旧股票卡 factor-table 的 <tr> -> list[dict]"""
    rows = []
    table = sc.select_one(".factor-table")
    if not table:
        return rows
    for r in table.select("tbody tr"):
        tds = r.find_all("td")
        if len(tds) >= 3:
            rows.append({"cat": tds[0].get_text(strip=True),
                         "sig": tds[1].get_text(strip=True),
                         "str": tds[2].get_text(strip=True)})
    return rows


def _parse_rotation(extra_txt):
    """从旧情绪卡文本解析板块轮动信息"""
    rotation = {}
    m = re.search(r"板块轮动[:：]\s*(\S+)", extra_txt)
    tech_m = re.search(r"科技强度[:：]?\s*([\d.]+)", extra_txt)
    signal_map = {
        "🔥科技板块强势": "tech_strong", "📈周期板块强势": "cyclical_strong",
        "⚖️板块均衡": "balanced", "📊待观察": "neutral",
    }
    if m:
        key = m.group(1)
        rotation["rotation_signal"] = signal_map.get(key, "neutral")
    if tech_m:
        try:
            rotation["tech_strength"] = float(tech_m.group(1))
        except ValueError as e:
            logging.warning("捕获异常已忽略(scripts/migrate_daily_reports.py:87): %s", e)
            pass
    return rotation or None


# ------------------------------------------------------------------
# 主解析
# ------------------------------------------------------------------
def parse_old_html(path):
    soup = BeautifulSoup(open(path, encoding="utf-8"), "html.parser")
    data = {"market_overview": {}, "global_markets": {}, "sentiment": {},
            "rotation_data": None, "news_analysis": {}, "recommendations": [],
            "history_records": []}

    for card in soup.select(".card"):
        h2 = card.find("h2")
        h2txt = h2.get_text(strip=True) if h2 else ""

        if h2txt == "市场环境":
            for tr in card.select("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 3:
                    name = tds[0].get_text(strip=True)
                    data["market_overview"][name] = {
                        "price": tds[1].get_text(strip=True),
                        "change": _to_float(tds[2].get_text(strip=True)),
                    }

        elif h2txt == "外盘环境":
            for tr in card.select("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 4:
                    name = tds[0].get_text(strip=True)
                    impact = tds[3].get_text(strip=True)
                    data["global_markets"][name] = {
                        "price": tds[1].get_text(strip=True),
                        "change": _to_float(tds[2].get_text(strip=True)),
                        "available": impact != "无数据",
                    }

        elif h2txt == "市场情绪":
            items = {}
            for it in card.select(".sentiment-item"):
                val = it.select_one(".val")
                lab = it.select_one(".label")
                if val and lab:
                    items[lab.get_text(strip=True)] = val.get_text(strip=True)
            data["sentiment"] = {
                "emotion_level": items.get("情绪评级", "-"),
                "emotion_score": _to_float(items.get("情绪得分", 0)),
                "limit_up": _to_int(items.get("涨停", 0)),
                "limit_down": _to_int(items.get("跌停", 0)),
                "broken": _to_int(items.get("炸板", 0)),
                "max_board": _to_int(items.get("最高连板", "0")),
            }
            data["rotation_data"] = _parse_rotation(card.get_text(" ", strip=True))

        elif h2txt == "实时消息":
            for sec in card.select(".news-section"):
                h3 = sec.find("h3")
                cat = "industry"
                if h3:
                    cat = _CAT_MAP.get(h3.get_text(strip=True), "industry")
                for li in sec.select("li"):
                    content = li.get_text(strip=True)
                    if content:
                        data["news_analysis"].setdefault(cat, []).append({"content": content})

        elif h2txt == "历史推荐回顾":
            for tr in card.select(".history-table tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) >= 8:
                    data["history_records"].append({
                        "trade_date": tds[0].get_text(strip=True),
                        "code": tds[1].get_text(strip=True),
                        "name": tds[2].get_text(strip=True),
                        "score": _to_float(tds[3].get_text(strip=True)),
                        "d1_return": _to_float_or_none(tds[4].get_text(strip=True)),
                        "d3_return": _to_float_or_none(tds[5].get_text(strip=True)),
                        "d5_return": _to_float_or_none(tds[6].get_text(strip=True)),
                        "d10_return": _to_float_or_none(tds[7].get_text(strip=True)),
                    })

    # 股票卡（独立于 .card）
    for sc in soup.select(".stock-card"):
        title_el = sc.select_one(".stock-title")
        title = title_el.get_text(strip=True) if title_el else ""
        m = re.match(r"\d+\.\s*(.+?)\s*\(([^)]+)\)\s*(★*)", title)
        name = m.group(1).strip() if m else title
        code = m.group(2).strip() if m else ""

        score = _to_float(sc.select_one(".score-val").get_text(strip=True)) if sc.select_one(".score-val") else 0.0

        price = target = stop = 0.0
        for item in sc.select(".op-item"):
            txt = item.get_text(" ", strip=True)
            if "入手点" in txt:
                price = _to_float(txt.split("入手点", 1)[1])
            elif "目标价" in txt:
                target = _to_float(txt.split("目标价", 1)[1].split("元")[0])
            elif "止损价" in txt:
                stop = _to_float(txt.split("止损价", 1)[1].split("元")[0])

        factor_rows = _extract_factor_rows(sc)
        # 从旧因子表反向映射成 factors，尽量还原关键因子块
        factors = {}
        for fr in factor_rows:
            sig = fr["sig"]
            if "MACD" in sig:
                factors["macd_golden_cross"] = 1
            if "均线多头" in sig:
                factors["ma_bullish"] = 1
            if "突破" in sig and "新高" in sig:
                factors["breakout_20d_high"] = 1
            if "量价" in sig:
                factors["volume_price_up"] = 1
            if fr["cat"] == "策略":
                factors.setdefault("strategies_triggered", []).append(sig)
            if fr["cat"] == "🏆 龙头":
                factors["is_leader"] = True
        if target > 0 and stop > 0 and price > 0:
            factors["dynamic_take_profit"] = target
            factors["dynamic_stop_loss"] = stop
            factors["dynamic_take_profit_pct"] = (target / price - 1) * 100 if price else 0
            factors["dynamic_stop_loss_pct"] = (stop / price - 1) * 100 if price else 0

        data["recommendations"].append({
            "name": name, "code": code, "score": score, "current_price": price,
            "strong_signals": 0, "business_profile": None, "factors": factors,
        })

    return data


# ------------------------------------------------------------------
# 迁移入口
# ------------------------------------------------------------------
def migrate(path, out_dir):
    data = parse_old_html(path)
    rg = ReportGenerator()
    html = rg.generate_report(
        recommendations=data["recommendations"],
        market_overview=data["market_overview"],
        global_markets=data["global_markets"],
        news_analysis=data["news_analysis"],
        sentiment=data["sentiment"],
        rotation_data=data["rotation_data"],
        trade_config={"stop_loss_pct": -0.05, "take_profit_pct": 0.08},
        history_records=data["history_records"],
    )
    out_path = os.path.join(out_dir, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ {os.path.basename(path)} -> 票:{len(data['recommendations'])} 新闻块:{len(data['news_analysis'])}")


def main():
    daily_dir = os.path.join("reports", "daily")
    files = sorted(f for f in os.listdir(daily_dir) if f.endswith(".html"))
    print(f"发现 {len(files)} 个历史报告")
    for f in files:
        migrate(os.path.join(daily_dir, f), daily_dir)


if __name__ == "__main__":
    main()
