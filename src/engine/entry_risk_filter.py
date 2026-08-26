# -*- coding: utf-8 -*-
"""入场风险过滤：追涨/套牢盘/量价背离/低开概率拦截 + 盘中信号刷新。"""
import io as _io_unused  # placeholder 保持 import 区整洁
import re

from src.utils.log import get_logger

logger = get_logger()


def _filter_high_risk_entries(stocks: list, filter_cfg: dict) -> list:
    """过滤追涨型高风险入场点（基于 rebound_analysis 已算出的风险信号）：
    - 今日大涨(entry前一日涨幅>5%, gap_open_probability>=70) → 追涨风险
    - 套牢盘很重(距20日高点<-10%) → 严重套牢盘压力,反弹易被砸
    - 量价背离(价涨量缩/放量滞涨) → 假反弹风险
    这些信号系统已算出但之前未用作过滤,导致7-23/7-24/7-28的9笔交易平均亏损-7.47%。
    """
    import re
    risk_cfg = filter_cfg.get("entry_risk", {})
    max_gap_prob = risk_cfg.get("max_gap_open_prob", 70)
    block_high_chip = risk_cfg.get("block_high_chip", True)
    block_volume_divergence = risk_cfg.get("block_volume_divergence", True)

    filtered = []
    rejected = []
    for s in stocks:
        f = s.get("factors", {}) or {}
        nd = f.get("next_day_prediction", {})
        if not isinstance(nd, dict):
            nd = {}
        gap_prob = nd.get("gap_open_probability", 0) or 0
        reasons = nd.get("reasons", []) or []
        risks = f.get("sustainability_risks", []) or []

        reject_reason = None
        all_reasons = [str(r) for r in list(reasons) + list(risks)]
        # 1. 今日大涨(entry前一日>5%) → 追涨风险
        if gap_prob >= max_gap_prob and any("今日大涨" in r for r in all_reasons):
            reject_reason = "今日大涨追涨"
        # 2. 套牢盘很重(距20日高点<-10%, 从risk字符串提取数值判断)
        elif block_high_chip:
            for r in all_reasons:
                if "套牢盘" in r:
                    m = re.search(r'距20日高点([-\d.]+)%', r)
                    if m and float(m.group(1)) <= -10:
                        reject_reason = "套牢盘很重(%s)" % r
                        break
        # 3. 量价背离(价涨量缩/放量滞涨)
        if not reject_reason and block_volume_divergence:
            if any("价涨量缩" in r or "放量滞涨" in r for r in all_reasons):
                reject_reason = "量价背离"
        # 4. 次日低开概率高(系统预判次日低开, e.g. "今日下跌，次日低开概率70%")
        if not reject_reason:
            m = re.search(r'次日低开概率(\d+)%', " ".join(all_reasons))
            if m and int(m.group(1)) >= 60:
                reject_reason = "次日低开概率高(%s%%)" % m.group(1)
        # 5. 历史类似反弹后易下跌(历史反弹后多日走弱, 反弹延续性差)
        if not reject_reason and any("历史类似反弹" in r for r in all_reasons):
            reject_reason = "历史类似反弹后易下跌"

        if reject_reason:
            rejected.append((s.get("code", ""), s.get("name", ""), reject_reason))
        else:
            filtered.append(s)

    if rejected:
        print(f"  🚫 入场风险过滤: 剔除 {len(rejected)} 只追涨/高位票")
        for code, name, reason in rejected:
            print(f"    • {name}({code}): {reason}")

    return filtered


def _refresh_intraday_gap_signal(change_pct, base_sig):
    """用当日实时涨跌幅重算次日高开/低开信号，替换盘前缓存的过期信号。

    盘前缓存的 gap_open_probability / gap_reasons 由 NextDayPredictor 基于"最近交易日"K线
    计算（8/17 案例：周五涨>5%的票，周一午后仍被当"今日大涨追涨"误拦，导致午后 0 推荐）。
    午后/盘中运行时"今日"已变化，须用当日实时涨跌幅重算：
      - 当日涨 >+5%  → 真追高，判"今日大涨"(gap=70) 拦截
      - 当日跌 <-3%  → 真弱势，判"次日低开概率70%"(gap=30) 拦截
      - 其余(横盘/温和波动) → gap=50 中性，不触发追涨/低开拦截
    保留与"今日涨跌"无关的原始理由（持续性/套牢盘等风险信息）。
    """
    base = base_sig or {}
    reasons = [
        r for r in (base.get("gap_reasons", []) or [])
        if not any(k in str(r) for k in ("今日大涨", "今日上涨", "今日下跌", "次日高开概率", "次日低开概率"))
    ]
    chg = float(change_pct or 0)
    if chg > 5:
        gap = 70
        reasons.append("今日大涨，次日高开概率70%")
    elif chg < -3:
        gap = 30
        reasons.append("今日下跌，次日低开概率70%")
    else:
        gap = 50
    return {"gap_open_probability": gap, "reasons": reasons}
