# -*- coding: utf-8 -*-
"""事件日历自动维护 — FOMC 官网抓取 + 国内规律事件播种 + 合并。

事件日历窗口防守(_event_calendar_cap)的日期来源分三级:
  1. config.yaml event_calendar.events 静态条目(最高优先, 手工修正永不覆盖)
  2. FOMC 官网抓取(federalreserve.gov, 官方提前一年公布全年日程, 格式稳定)
  3. 规律事件播种(两会/中央经济工作会议, 时间规律强, 宽窗口纯本地计算)

降级链(规则§4): 抓取失败 → 7天TTL过期缓存 → 仅静态/规律事件。
所有日期统一为 ISO 字符串(YYYY-MM-DD), 合并时按 (name, window) 去重,
冲突时 config 静态优先。

政治局会议日期无官方日历(仅提前几天由新闻放出), 由新闻关键词检测
(PolicyEventFactor 层)处理, 不在本模块。
"""
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("stock_prediction")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_ROOT, "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "fomc_calendar.json")
_CACHE_TTL = 7 * 86400  # 官方日程一年才变一次, 7天足够

_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
_TIMEOUT = 8


def fetch_fomc_events(force: bool = False) -> List[Dict]:
    """抓取 FOMC 未来会议日程, 返回 events 列表 [{name, window: [起, 止]}]。

    窗口语义: 会议两日 + 决议落地日(北京时间会议次日凌晨2:00公布,
    A股在第三日反应)。只保留未结束的会议(历史会议无防守意义)。
    降级链: 网络失败 → 过期缓存 → 返回 [](调用方继续用静态/规律事件)。
    """
    today = date.today().isoformat()

    # 缓存命中
    if not force:
        cached = _load_cache()
        if cached is not None:
            return [e for e in cached if e["window"][1] >= today]

    # 抓取
    html = _fetch_fomc_html()
    if html is None:
        cached = _load_cache(allow_expired=True)
        if cached is not None:
            logger.warning("FOMC日历抓取失败, 使用过期缓存(%d条)", len(cached))
            return [e for e in cached if e["window"][1] >= today]
        logger.warning("FOMC日历抓取失败且无缓存, 本轮仅用静态/规律事件")
        return []

    events = _parse_fomc_html(html)
    if not events:
        cached = _load_cache(allow_expired=True)
        if cached:
            logger.warning("FOMC日历解析为空, 使用过期缓存(%d条)", len(cached))
            return [e for e in cached if e["window"][1] >= today]
        return []

    _save_cache(events)
    return [e for e in events if e["window"][1] >= today]


def _fetch_fomc_html() -> Optional[str]:
    try:
        resp = requests.get(_FOMC_URL, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("FOMC日历HTTP %d", resp.status_code)
            return None
        return resp.text
    except Exception as e:
        logger.warning("FOMC日历抓取异常: %s", e)
        return None


def _parse_fomc_html(html: str) -> List[Dict]:
    """解析官网日历页: 定位 '<h4>YYYY FOMC Meetings' 年度段,
    逐段提取 fomc-meeting__month/__date 块。页面为静态 HTML, 结构多年稳定。

    真实结构(2026-09 核对):
      <h4><a id="42828">2026 FOMC Meetings</a></h4>
      <div class="fomc-meeting__month ..."><strong>January</strong></div>
      <div class="fomc-meeting__date ...">27-28</div>   ← 日期可带 * 后缀

    受控说明: 单元测试用内嵌 HTML 片段验证, 不依赖网络。
    """
    events = []
    year_blocks = re.split(r"<h4[^>]*>\s*(?:<a[^>]*>)?\s*(\d{4})\s*FOMC\s*Meetings", html)
    # split 结果: [前文, 年份1, 段1, 年份2, 段2, ...]
    for i in range(1, len(year_blocks) - 1, 2):
        year = year_blocks[i]
        block = year_blocks[i + 1]
        for m in re.finditer(
                r"fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Z][a-z]+)\s*"
                r"(?:</strong>)?\s*</div>\s*"
                r"<div[^>]*fomc-meeting__date[^>]*>\s*(\d{1,2})-(\d{1,2})\*?\s*<",
                block, re.DOTALL):
            month_name, d1, d2 = m.group(1), int(m.group(2)), int(m.group(3))
            month = _MONTH_NUM.get(month_name)
            if month is None or d2 <= d1:
                continue
            try:
                start = date(int(year), month, d1)
                end = date(int(year), month, d2)
            except ValueError:
                continue
            # 窗口 = 会议两日 + 次日(决议落地, 北京时间凌晨) + 反应日, 共4天
            window_end = (end + timedelta(days=2)).isoformat()
            events.append({
                "name": f"FOMC{month_name}",
                "window": [start.isoformat(), window_end],
            })
    return events


_MONTH_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12,
}


def _load_cache(allow_expired: bool = False) -> Optional[List[Dict]]:
    try:
        if not os.path.exists(_CACHE_FILE):
            return None
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            ts, events = json.load(f)
        if not allow_expired and datetime.now().timestamp() - ts > _CACHE_TTL:
            return None
        return events
    except Exception as e:
        logger.warning("FOMC日历缓存读取失败: %s", e)
        return None


def _save_cache(events: List[Dict]):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump([datetime.now().timestamp(), events], f, ensure_ascii=False)
    except Exception as e:
        logger.warning("FOMC日历缓存写入失败: %s", e)


def seed_recurring_events(year: int = None, today: str = None) -> List[Dict]:
    """规律事件播种: 两会/中央经济工作会议。纯本地计算零网络。

    窗口取得比实际会议宽(代价只是多防守两天):
      - 两会: 3月3日~3月13日(全国政协3/4 + 人大3/5 开幕, 闭幕约3/11)
      - 中央经济工作会议: 12月8日~12月16日(惯例12月中旬, 为期约3天)
    today: 评估基准日(ISO字符串), 10月及以后同时播种次年两会(跨年窗口);
           None=用真实今天(回测/测试可传历史日避免软前视)。
    """
    ref = today or date.today().isoformat()
    y = year if year is not None else int(ref[:4])
    # 10月及以后播种次年两会(跨年窗口)
    years = {y}
    if int(ref[5:7]) >= 10:
        years.add(y + 1)
    events = []
    for yy in sorted(years):
        events.append({
            "name": f"两会{yy}",
            "window": [f"{yy}-03-03", f"{yy}-03-13"],
            "source": "seed",
        })
        events.append({
            "name": f"中央经济工作会议{yy}",
            "window": [f"{yy}-12-08", f"{yy}-12-16"],
            "source": "seed",
        })
    return events


def merge_event_calendars(config: Dict, today: str = None,
                          fetch_fomc: bool = True) -> List[Dict]:
    """合并三级日历: config静态(优先) + FOMC抓取 + 规律播种。

    去重键 (name, window): 若静态与自动源窗口重叠且名字相近
    (同为 FOMC 前缀或同日), 保留静态条目——手工修正永不覆盖。
    只返回窗口未结束的事件。
    """
    today = today or date.today().isoformat()
    cal_cfg = config.get("event_calendar") or {}
    static_events = []
    for ev in cal_cfg.get("events", []) or []:
        win = [str(w)[:10] for w in (ev.get("window") or [])]
        if len(win) == 2:
            static_events.append({
                "name": ev.get("name", "?"), "window": win,
                "cap": ev.get("cap"), "source": "config",
            })

    merged = [e for e in static_events if e["window"][1] >= today]

    # FOMC 抓取(config 可关)
    auto_enabled = cal_cfg.get("auto_fomc", True)
    fomc_events = []
    if auto_enabled and fetch_fomc:
        try:
            fomc_events = fetch_fomc_events()
        except Exception as e:
            logger.warning("FOMC日历获取失败(降级为静态+规律事件): %s", e)
            fomc_events = []

    # 规律播种(config 可关)
    seed_enabled = cal_cfg.get("auto_seed", True)
    seed_events = seed_recurring_events(today=today) if seed_enabled else []

    # 静态已覆盖的日期段不再重复添加(同窗口重叠即视为已覆盖)
    def _overlap(a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        return a[0] <= b[1] and b[0] <= a[1]

    static_windows = [tuple(e["window"]) for e in merged]
    for ev in fomc_events + seed_events:
        if ev["window"][1] < today:
            continue
        if any(_overlap(w, tuple(ev["window"])) for w in static_windows):
            continue  # 静态条目已覆盖此日期段
        merged.append({**ev, "source": ev.get("source", "auto")})

    return merged


def check_fomc_source() -> bool:
    """source_health 探针: FOMC 日历页可达性。"""
    html = _fetch_fomc_html()
    return html is not None and "FOMC" in html and "Meetings" in html
