# -*- coding: utf-8 -*-
"""盘中模式(09:25): 崩盘清仓检测/竞价快照落库/预警

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger
from src.utils.helpers import get_limit_pct

logger = get_logger()

from src.engine.entry_risk_filter import (
    _refresh_intraday_gap_signal, _filter_high_risk_entries)

from src.engine.paper_trade_ops import _check_crash_and_close_positions
from src.data.auction_snapshot import _save_auction_snapshot

from src.data.market_data import MarketDataFetcher
from src.engine.reporter import ReportGenerator
from src.pipeline.common import (  # noqa: F401
    _DualOutput, setup_logging, write_summary, _prefetch_history_and_flow,
    save_cache, load_cache, _log_source_health, _cleanup_cache,
)


def _id_snapshot_and_crash(ctx):
    """[盘中·阶段1] 实时行情获取 → 大盘崩盘清仓检测 → 竞价快照落库。

    行情为空时返回 None(调用方终止)；成功则把 price_map 写回 ctx。
    """
    config = ctx["config"]
    today = ctx["today"]

    # 快速拉取实时行情
    print("\n[1/1] 获取实时开盘行情...")
    market_fetcher = MarketDataFetcher()
    fresh = market_fetcher.get_a_share_realtime()
    if fresh.empty:
        print("  ❌ 获取行情失败")
        return None

    # 建立 代码→行 的快速索引
    fresh["代码_str"] = fresh["代码"].astype(str)
    price_map = {}
    for _, r in fresh.iterrows():
        price_map[r["代码_str"]] = r

    # 大盘崩盘检测:对 open 仓位止损清仓(解决7-24盘前开仓后当天崩盘问题)
    _check_crash_and_close_positions(fresh, config)

    # === 集合竞价快照落库：为竞价强度因子积累 IC 样本 ===
    # 9:25 时刻 涨跌幅=开盘涨幅、最新价=开盘撮合价、量比=竞价强度核心字段。
    # 失败不影响盘中预警主流程
    try:
        from src.data.database import StockDatabase
        _save_auction_snapshot(StockDatabase(), today, fresh)
        print('  💾 竞价快照已落库')
    except Exception as e:
        logger.warning('竞价快照落库失败(不影响主流程): %s', e)

    ctx["price_map"] = price_map
    return fresh


def _id_filter_risky(updated, filter_cfg):
    """入场风险过滤 → 返回 (kept, risky)。

    盘中模式复用盘前缓存，须与盘前一致的入场风险过滤，
    避免盘前判死的高风险票被盘中捞回。失败时原样返回(不影响主流程)。
    """
    risky = []
    try:
        risk_inputs = []
        for c in updated:
            sig = c.get("signals", {}) or {}
            gap = _refresh_intraday_gap_signal(c.get("intraday_change", 0), sig)
            risk_inputs.append({
                "code": c["code"],
                "name": c["name"],
                "factors": {
                    "next_day_prediction": {
                        "gap_open_probability": gap["gap_open_probability"],
                        "reasons": gap["reasons"],
                    },
                    "sustainability_risks": c.get("sustainability_risks", []) or [],
                },
            })
        kept = _filter_high_risk_entries(risk_inputs, filter_cfg)
        kept_codes = {x["code"] for x in kept}
        for x in risk_inputs:
            if x["code"] not in kept_codes:
                risky.append(x)
        updated = [c for c in updated if c["code"] in kept_codes]
    except Exception as e:
        logger.warning("盘中入场风险过滤失败(不影响主流程): %s", e)
    return updated, risky


def _id_alerts_and_rescore(ctx):
    """[盘中·阶段2] 候选池实时刷新评分 + 入场风险过滤 + 预警输出。

    仅作观察池快照与预警依据，不产出买入推荐(T+1 设计约定)；
    结果 updated/excluded/risky 写回 ctx。
    """
    config = ctx["config"]
    cache = ctx["cache"]
    price_map = ctx["price_map"]

    # 对候选池刷新评分（仅作观察池快照与预警依据，不产出买入推荐）
    print("\n刷新候选池评分...")
    candidates = cache["candidates"]
    updated = []
    excluded = []  # 涨跌停/停牌等不可交易票 → 预警输出
    for c in candidates:
        code = c["code"]
        name = c["name"]
        old_score = c["score"]
        row = price_map.get(code)

        if row is None:
            excluded.append({"code": code, "name": name, "reason": "停牌或无行情"})
            continue

        new_price = float(row.get("最新价", 0))
        change_pct = float(row.get("涨跌幅", 0) or 0)
        volume_ratio = float(row.get("量比", 0) or 0)
        turnover = float(row.get("换手率", 0) or 0)

        if new_price <= 0:
            excluded.append({"code": code, "name": name, "reason": "价格无效"})
            continue

        # 基础分 = 盘前评分（因子分），叠加价格偏离/量比/换手修正
        intraday_score = old_score

        # 价格偏离修正：高开加分，低开减分
        old_price = c.get("premarket_price", 0)
        if old_price > 0:
            price_change_pct = (new_price - old_price) / old_price * 100
            if price_change_pct > 3:
                intraday_score += 3
            elif price_change_pct > 1:
                intraday_score += 1
            elif price_change_pct < -3:
                intraday_score -= 3
            elif price_change_pct < -1:
                intraday_score -= 1

            # 涨停/跌停过滤 → 转为不可交易预警(按板块口径, 禁止硬编码9.5)
            limit_pct = get_limit_pct(code, name)
            if change_pct >= limit_pct - 0.5:
                excluded.append({"code": code, "name": name,
                                 "reason": f"涨停 {change_pct:+.1f}%，买不进"})
                continue
            if change_pct <= -limit_pct + 0.5:
                excluded.append({"code": code, "name": name,
                                 "reason": f"跌停 {change_pct:+.1f}%，回避"})
                continue

        # 量比修正
        if volume_ratio > 2:
            intraday_score += 2
        elif volume_ratio < 0.5:
            intraday_score -= 1

        # 换手修正
        if turnover > 2:
            intraday_score += 1

        c["intraday_price"] = new_price
        c["intraday_change"] = round(change_pct, 2)
        c["intraday_score"] = round(intraday_score, 1)
        c["volume_ratio"] = volume_ratio
        updated.append(c)

    # 排序（仅作观察池快照）
    updated.sort(key=lambda x: x["intraday_score"], reverse=True)

    # === 入场风险过滤 → 转为高风险预警标记 ===
    updated, risky = _id_filter_risky(updated, config.get("filter", {}))

    # === 预警输出 ===
    # 设计约定(T+1)：尾盘持有期从收盘起算，14:40 午后模式的报告才是买入决策依据；
    # 09:25 盘中只负责三件事：崩盘清仓 / 不可交易预警 / 高风险票预警。
    # 此前盘中产出独立推荐排序，修正幅度小(±3)却制造了"早盘 vs 午盘推荐不一致"的困惑。
    print("\n" + "-" * 40)
    print("  ⚠️ 盘中预警(非买入依据):")
    for w in excluded[:8]:
        print(f"    ⛔ {w['name']}({w['code']}): {w['reason']}")
    for x in risky[:5]:
        sig = (x.get("factors") or {}).get("next_day_prediction") or {}
        reasons = "；".join((sig.get("reasons") or [])[:1])
        print(f"    🚫 {x['name']}({x['code']}): 高风险入场 {reasons}")
    if not excluded and not risky:
        print("    无异常")

    top5 = updated[:5]
    print(f"\n  👀 观察池 Top{len(top5)} (按实时行情刷新，最终决策以14:40午后报告为准):")
    for i, c in enumerate(top5, 1):
        drift = c["intraday_score"] - c["score"]
        arrow = "↑" if drift >= 1 else ("↓" if drift <= -1 else "→")
        print(f"  {i}. {c['name']}({c['code']}) "
              f"盘前{c['score']}分 → 实时{c['intraday_score']}分 {arrow} "
              f"现价{c.get('intraday_price', 0):.2f} "
              f"{c.get('intraday_change', 0):+.1f}% "
              f"量比{c.get('volume_ratio', 0):.1f}")

    ctx["updated"] = updated
    ctx["excluded"] = excluded
    ctx["risky"] = risky


def run_intraday(config):
    print("=" * 60)
    print("  盘中模式 — 快速修正")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    _log_source_health()

    today = datetime.now().strftime("%Y-%m-%d")
    cache = load_cache(f"premarket_{today}.json")
    if cache is None:
        print(f"  ❌ 未找到盘前缓存 cache/premarket_{today}.json")
        print("  → 请先运行盘前模式: python run_daily.py --premarket")
        # 返回 False: 空跑不得被记为"成功", 否则计划任务静默失效无人察觉
        return False

    print(f"  ✅ 载入盘前候选池: {len(cache['candidates'])} 只 (计算于 {cache['run_at']})")

    ctx = {"config": config, "today": today, "cache": cache}
    if _id_snapshot_and_crash(ctx) is None:
        return False
    _id_alerts_and_rescore(ctx)
    updated = ctx["updated"]
    excluded = ctx["excluded"]
    risky = ctx["risky"]

    # 生成预警版报告（不含推荐列表）
    print("\n生成盘中预警报告...")
    reporter = ReportGenerator()
    now_str = datetime.now().strftime("%H:%M")

    index_data = cache.get("index_data", {})
    global_markets_list = cache.get("global_markets", [])
    news_analysis = cache.get("news_analysis", {})
    sentiment_dict = cache.get("sentiment", {})
    rotation_dict = cache.get("rotation_data", {})
    trade_cfg = cache.get("trade_config", config.get("trade", {}))

    report_content = reporter.generate_report(
        recommendations=[],
        market_overview=index_data,
        global_markets=global_markets_list,
        news_analysis=news_analysis,
        sentiment=sentiment_dict,
        trade_config=trade_cfg,
        rotation_data=rotation_dict,
        history_records=None,
        report_role="warning",
        intraday_note=f"盘中预警于 {now_str}",
    )
    report_path = reporter.save_report(report_content, report_role="warning")

    write_summary("intraday_watch", len(cache["candidates"]), [],
                   excluded=len(excluded), risky=len(risky))

    print("\n" + "=" * 60)
    print("  盘中预警完成! (本档不产出买入推荐)")
    print(f"  观察池: {len(updated)} 只 | 不可交易: {len(excluded)} | 高风险: {len(risky)}")
    print(f"  报告文件: {report_path}")
    print("  💡 买入决策请以 14:40 午后《尾盘最终决策版》为准")
    print("=" * 60)
    return True


# ======================================================================
# 午后模式：早盘真实成交数据深度修正（建议14:40运行——
# 尾盘最后半小时是主力拉升/出货的揭晓时段，越接近收盘信息越完整；
# 14:40启动约3-5分钟出报告，留足14:45-14:55下单窗口，避免被迫进收盘集合竞价）
# ======================================================================
