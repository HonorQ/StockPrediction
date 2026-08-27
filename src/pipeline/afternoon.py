# -*- coding: utf-8 -*-
"""午后模式(14:40): 全天数据修正评分, 每日唯一正式推荐

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

from src.utils.heartbeat import check_task_alerts
from src.engine.paper_trade_ops import _open_paper_trades_for_recs

from src.data.market_data import MarketDataFetcher
from src.data.money_flow import MoneyFlowFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.hot_stocks import HotStocksFetcher
from src.factors.news_factor import NewsFactor
from src.engine.reporter import ReportGenerator
from src.data.database import StockDatabase
from src.pipeline.common import (  # noqa: F401
    _DualOutput, setup_logging, write_summary, _prefetch_history_and_flow,
    save_cache, load_cache, _log_source_health, _cleanup_cache,
)


def _am_init(ctx):
    """[午后·阶段0] 组件实例化 + 盘前缓存载入 + 运维告警检查 + 数据拉取。

    初始化失败返回 False(调用方终止)；中间状态全部写入 ctx。
    """
    config = ctx["config"]

    print("=" * 60)
    print("  午后模式 — 早盘数据深度修正")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    _log_source_health()

    # 失败告警：盘前/盘中异常时午后报告显著提示
    afternoon_alerts = check_task_alerts()
    if afternoon_alerts:
        for a in afternoon_alerts:
            print(f"  🚨 运维告警: {a}")

    today = datetime.now().strftime("%Y-%m-%d")
    cache = load_cache(f"premarket_{today}.json")
    if cache is None:
        print(f"  ❌ 未找到盘前缓存 cache/premarket_{today}.json")
        print("  → 请先运行盘前模式: python run_daily.py --premarket")
        return False

    print(f"  ✅ 载入盘前候选池: {len(cache['candidates'])} 只 (计算于 {cache['run_at']})")

    filter_cfg = config.get("filter", {})
    trade_cfg = config.get("trade", {})
    keywords_cfg = config.get("keywords", {})

    # 1. 实时行情
    print("\n[1/5] 获取早盘实时行情...")
    market_fetcher = MarketDataFetcher()
    fresh = market_fetcher.get_a_share_realtime()
    if fresh.empty:
        print("  ❌ 获取行情失败")
        return False

    fresh["代码_str"] = fresh["代码"].astype(str)
    price_map = {r["代码_str"]: r for _, r in fresh.iterrows()}

    # 2. 资金流向
    print("\n[2/5] 获取早盘资金流向...")
    money_fetcher = MoneyFlowFetcher()

    # 3. 午间新闻
    print("\n[3/5] 获取午间新闻...")
    news_fetcher = NewsFetcher()
    news_list = news_fetcher.get_all_news(limit_per_source=30)

    # 4. 最新热点+情绪
    print("\n[4/5] 获取最新热点...")
    hot_fetcher = HotStocksFetcher(market_df=fresh)
    hot_df = hot_fetcher.get_hot_rank()
    sentiment = hot_fetcher.calculate_market_sentiment()
    hot_codes = set()
    if not hot_df.empty:
        hot_codes.update(hot_df["股票代码"].astype(str).tolist())

    ctx.update(
        today=today, cache=cache, afternoon_alerts=afternoon_alerts,
        filter_cfg=filter_cfg, trade_cfg=trade_cfg,
        market_fetcher=market_fetcher, price_map=price_map,
        money_fetcher=money_fetcher, news_list=news_list,
        sentiment=sentiment, hot_codes=hot_codes,
    )
    return True


def _am_rescore_one(c, idx, price_map, money_fetcher,
                    news_factor, news_list, hot_codes, total):
    """[午后·阶段5] 单票深度修正评分(5a~5g)；不可交易返回 None 跳过。

    修正项: 价格行为/量比/换手/资金流向/午间新闻/人气热度/盘前信号验证。
    """
    code = c["code"]
    name = c["name"]
    row = price_map.get(code)
    if row is None:
        return None

    new_price = float(row.get("最新价", 0))
    change_pct = float(row.get("涨跌幅", 0) or 0)
    volume_ratio = float(row.get("量比", 0) or 0)
    turnover = float(row.get("换手率", 0) or 0)

    if new_price <= 0:
        return None

    # 涨停/跌停过滤(按板块口径, 禁止硬编码9.5)
    limit_pct = get_limit_pct(code, name)
    if change_pct >= limit_pct - 0.5 or change_pct <= -limit_pct + 0.5:
        print(f"    ⛔ {name}({code}) {'涨停' if change_pct >= limit_pct - 0.5 else '跌停'}，跳过")
        return None

    afternoon_score = c["score"]
    signals = c.get("signals", {})

    # 5a. 早盘价格行为：真金白银的趋势确认
    old_price = c.get("premarket_price", 0)
    if old_price > 0:
        price_change_pct = (new_price - old_price) / old_price * 100
        if 0.5 <= price_change_pct < 3:
            afternoon_score += 3
        elif 3 <= price_change_pct < 7:
            afternoon_score += 2
        elif -2 <= price_change_pct < -0.5:
            afternoon_score -= 2
        elif price_change_pct < -2:
            afternoon_score -= 4

    # 5b. 量比：真实成交意愿
    if volume_ratio > 2:
        afternoon_score += 3
    elif volume_ratio > 1.5:
        afternoon_score += 2
    elif volume_ratio > 1.2:
        afternoon_score += 1
    elif volume_ratio < 0.5:
        afternoon_score -= 3
    elif volume_ratio < 0.8:
        afternoon_score -= 1

    # 5c. 换手率活跃度
    if turnover > 5:
        afternoon_score += 2
    elif turnover > 3:
        afternoon_score += 1
    elif turnover < 0.5:
        afternoon_score -= 1

    # 5d. 资金流向：主力态度
    flow_data = money_fetcher.get_stock_fund_flow(code)
    net_inflow = 0
    if flow_data:
        net_inflow = flow_data.get("main_flow", 0) or 0
        if net_inflow > 2000:
            afternoon_score += 5
        elif net_inflow > 500:
            afternoon_score += 3
        elif net_inflow > 0:
            afternoon_score += 2
        elif net_inflow < -2000:
            afternoon_score -= 5
        elif net_inflow < -500:
            afternoon_score -= 3
        elif net_inflow < 0:
            afternoon_score -= 2

    # 5e. 午间新闻新鲜度
    ns = news_factor.score_stock_news(code, news_list)
    if ns > 60:
        afternoon_score += min((ns - 50) / 10, 3)
    elif ns < 40:
        afternoon_score -= min((50 - ns) / 10, 3)

    # 5f. 人气热度
    if code in hot_codes:
        afternoon_score += 3

    # 5g. 信号验证：盘前预测 vs 实际走势
    # 修复：signals.next_day_prediction 缓存的是 str(dict)，原 == "up" 恒不成立，
    #      改为从 next_day dict 读取 bias 判断盘前看多/看空倾向
    nd = signals.get("next_day")
    if not isinstance(nd, dict):
        raw = signals.get("next_day_prediction")
        nd = raw if isinstance(raw, dict) else {}
    bias = nd.get("bias", "neutral")
    if bias in ("bullish", "slightly_bullish") and change_pct > 0:
        afternoon_score += 2
    if bias in ("bullish", "slightly_bullish") and change_pct < -2:
        afternoon_score -= 3

    row_out = {
        "code": code, "name": name, "current_price": new_price,
        "change_pct": round(change_pct, 2),
        "volume_ratio": volume_ratio, "turnover": turnover,
        "premarket_score": c["score"],
        "afternoon_score": round(afternoon_score, 1),
        "net_inflow": net_inflow, "news_score": ns,
        "strong_signals": signals.get("strategy_count", 0),
    }
    print(f"    [{idx+1}/{total}] "
          f"{name}({code}) {c['score']}→{afternoon_score:.1f}分 "
          f"{change_pct:+.1f}% 量比{volume_ratio:.1f} 净流入{net_inflow:.0f}万")
    return row_out


def _am_ml_assist(ctx, updated):
    """ML模型辅助信号（可选，不影响主流程）。

    特征来源必须是盘前 factor_snapshots 的完整原始因子(与训练数据同源)。
    此前传 updated 里不存在的 factors 键 → 永远 predict({}) → 特征全默认值50，
    属于分布外推理的伪修正，已修复。
    """
    today = ctx["today"]
    try:
        from src.engine.ml_selector import MLStockSelector
        ml = MLStockSelector()
        if os.path.exists(ml.model_path):
            print("\n  🤖 ML模型预测(基于盘前因子快照)...")
            snap_db = StockDatabase()
            n_ml = 0
            for c in updated:
                raw_factors = snap_db.get_factor_snapshot_raw_factors(c["code"], today)
                if not raw_factors:
                    continue  # 无快照则跳过该股，不用默认值硬预测
                pred = ml.predict(raw_factors)
                ml_score = pred.get("ml_score", 50)
                c["ml_score"] = ml_score
                c["afternoon_score"] = round(c["afternoon_score"] + (ml_score - 50) * 0.05, 1)
                n_ml += 1
            print(f"    ML辅助修正完成: {n_ml}/{len(updated)} 只有权重5%的微调")
    except Exception as e:
        logger.warning("ML模型预测失败(不影响主流程): %s", e)


def _am_rescore(ctx):
    """[午后·阶段5] 逐票深度修正评分 + ML辅助 + 排序 + 入场风险过滤。

    结果 updated(风险过滤后) 与 cand_by_code 写回 ctx。
    """
    keywords_cfg = ctx["config"].get("keywords", {})
    cache = ctx["cache"]
    price_map = ctx["price_map"]
    money_fetcher = ctx["money_fetcher"]
    news_list = ctx["news_list"]
    hot_codes = ctx["hot_codes"]
    candidates = cache["candidates"]
    bp_by_code = {c["code"]: c.get("business_profile") for c in candidates}

    print("\n[5/5] 基于早盘数据深度修正评分...")
    news_factor = NewsFactor(keywords_cfg)
    updated = []
    for idx, c in enumerate(candidates):
        row_out = _am_rescore_one(c, idx, price_map, money_fetcher,
                                  news_factor, news_list, hot_codes,
                                  total=len(candidates))
        if row_out is not None:
            updated.append(row_out)

    # === ML模型辅助信号 ===
    _am_ml_assist(ctx, updated)

    updated.sort(key=lambda x: x["afternoon_score"], reverse=True)

    print("\n午后修正 Top5:")
    for i, c in enumerate(updated[:5], 1):
        print(f"  {i}. {c['name']}({c['code']}) "
              f"盘前{c['premarket_score']}分 → 午后{c['afternoon_score']}分 "
              f"现价{c['current_price']:.2f} {c['change_pct']:+.1f}% "
              f"量比{c['volume_ratio']:.1f} "
              f"净流入{c['net_inflow']:.0f}万")

    # === 入场风险过滤：与盘前模式同口径（剔除追涨/套牢盘重/量价背离） ===
    # 盘前模式已剔除的追涨/套牢盘票，午后不能又捞回来
    # （2026-08-14 案例：紫光股份"今日大涨追涨"、智度股份"套牢盘很重"被盘前剔除，
    #  但午后模式此前不做该过滤，导致这两只高风险票被重新推荐）
    filter_cfg = ctx["filter_cfg"]
    try:
        cand_by_code = {cc["code"]: cc for cc in candidates}
        risk_inputs = []
        for c in updated:
            src = cand_by_code.get(c["code"], {})
            sig = src.get("signals", {}) or {}
            gap = _refresh_intraday_gap_signal(c.get("change_pct", 0), sig)
            risk_inputs.append({
                "code": c["code"],
                "name": c["name"],
                "factors": {
                    "next_day_prediction": {
                        "gap_open_probability": gap["gap_open_probability"],
                        "reasons": gap["reasons"],
                    },
                    "sustainability_risks": src.get("sustainability_risks", []) or [],
                },
            })
        kept = _filter_high_risk_entries(risk_inputs, filter_cfg)
        kept_codes = {x["code"] for x in kept}
        updated = [c for c in updated if c["code"] in kept_codes]
    except Exception as e:
        logger.warning("午后入场风险过滤失败(不影响主流程): %s", e)

    ctx["updated"] = updated
    ctx["candidates"] = candidates
    ctx["bp_by_code"] = bp_by_code


def _am_filter_persist(ctx):
    """[午后·阶段6] 达标筛选 + 策略健康度降级 + DB保存 + 同步开纸面仓。

    结果 final / afternoon_health_note / db / cand_by_code 写回 ctx。
    """
    config = ctx["config"]
    today = ctx["today"]
    filter_cfg = ctx["filter_cfg"]
    trade_cfg = ctx["trade_cfg"]
    market_fetcher = ctx["market_fetcher"]
    afternoon_alerts = ctx["afternoon_alerts"]
    updated = ctx["updated"]
    candidates = ctx["candidates"]
    cand_by_code = {cc["code"]: cc for cc in candidates}

    # 生成推荐（达标线与盘前/盘中统一取 config，避免三档口径漂移）
    min_score = filter_cfg.get("min_score", 53)
    max_rec = filter_cfg.get("max_recommend", 2)
    qualified = [c for c in updated if c["afternoon_score"] >= min_score]
    final = qualified[:max_rec]

    # === 策略健康度层(与盘前同口径)：风格失效时收紧午后推荐上限 ===
    afternoon_health_note = None
    if afternoon_alerts:
        afternoon_health_note = "🚨 " + "; ".join(afternoon_alerts)
    try:
        from src.engine.strategy_health import assess_strategy_health, apply_action
        db_h = StockDatabase()
        health = assess_strategy_health(db_h, today=today,
                                        config=filter_cfg.get("strategy_health"))
        if health.get("status") == "degraded":
            new_max = apply_action(max_rec, health["action"])
            if new_max != max_rec or new_max <= 1:
                afternoon_health_note = (
                    f"⚠️ 策略健康度降级: 近{health['samples']}条推荐 "
                    f"胜率{health['win_rate']}% / 平均d5 {health['avg_d5']:+.2f}% "
                    f"→ 推荐上限 {max_rec} → {new_max}")
                print(f"  🩺 {afternoon_health_note}")
            max_rec = new_max
            final = final[:max_rec]
        else:
            print(f"  🩺 策略健康度: {health.get('status')} "
                  f"(样本{health.get('samples', 0)})")
    except Exception as e:
        logger.warning("策略健康度监控失败(不影响主流程): %s", e)

    # 保存到数据库
    db = StockDatabase()
    for i, rec in enumerate(final, 1):
        db.save_recommendation(
            trade_date=today,
            code=rec["code"],
            name=rec["name"],
            score=rec["afternoon_score"],
            price=rec.get("current_price", 0),
            rank=i,
            mode="afternoon",
            factors=rec,
        )

    # 同步开纸面仓位：以午后实际价格为入场基准(尾盘决策的真实成本)。
    # 与盘前推荐分开归因，积累数据后即可对比"盘前 vs 午后"推荐的历史准确率
    _open_paper_trades_for_recs(db, market_fetcher, config.get("trade", {}), final,
                                 today, score_field="afternoon_score",
                                 label="午后")

    ctx["final"] = final
    ctx["afternoon_health_note"] = afternoon_health_note
    ctx["db"] = db
    ctx["cand_by_code"] = cand_by_code


def _am_report(ctx):
    """[午后·阶段7] 报告生成(合并盘前信号字段) + 总结输出。"""
    today = ctx["today"]
    cache = ctx["cache"]
    db = ctx["db"]
    final = ctx["final"]
    updated = ctx["updated"]
    candidates = ctx["candidates"]
    sentiment = ctx["sentiment"]
    trade_cfg = ctx["trade_cfg"]
    bp_by_code = ctx["bp_by_code"]
    cand_by_code = ctx["cand_by_code"]
    afternoon_health_note = ctx["afternoon_health_note"]

    # 生成报告
    print("\n生成午后报告...")
    reporter = ReportGenerator()
    index_data = cache.get("index_data", {})
    global_markets_list = cache.get("global_markets", [])
    news_analysis = cache.get("news_analysis", {})
    rotation_dict = cache.get("rotation_data", {})

    recommendations = []
    for c in final:
        # 合并盘前缓存中的信号字段，保证报告能提炼"为什么选它"和关键因子
        # （此前只传 6 个盘中字段，导致报告丢失 MACD/均线/量价/次日预判等推荐原因）
        src = cand_by_code.get(c["code"], {})
        sig = src.get("signals", {}) or {}
        recommendations.append({
            "code": c["code"],
            "name": c["name"],
            "current_price": c["current_price"],
            "score": c["afternoon_score"],
            "business_profile": bp_by_code.get(c["code"]),
            "factors": {
                "strategies_triggered": sig.get("strategies", []),
                "strategy_count": sig.get("strategy_count", 0),
                "macd_golden_cross": sig.get("macd_golden_cross", 0),
                "ma_bullish": sig.get("ma_bullish", 0),
                "volume_price_up": sig.get("volume_price_up", 0),
                "breakout_20d_high": sig.get("breakout_20d_high", 0),
                "next_day_prediction": sig.get("next_day", sig.get("next_day_prediction", "neutral")),
                "oversold_rebound_detected": sig.get("oversold_rebound_detected", False),
                "quant_risk": sig.get("quant_risk", "low"),
                "quant_detected": sig.get("quant_detected", False),
                "sustainability_score": sig.get("sustainability_score", 50),
                "profit_taking_detected": sig.get("profit_taking_detected", False),
                "hot_concepts": sig.get("hot_concepts", ""),
                "premarket_score": c["premarket_score"],
                "afternoon_change": c["change_pct"],
                "volume_ratio": c["volume_ratio"],
                "turnover": c["turnover"],
                "net_inflow": c["net_inflow"],
                "news_score": c["news_score"],
            },
        })

    # 报告历史回顾：每日一次推荐口径。
    # 只含盘前类模式(afternoon/premarket/历史feedback_training)，排除回测(backtest)行——
    # 否则跨窗口基线的回测推荐会污染回顾表；同日多行由 reporter 按 afternoon 优先去重为1条
    history = db.get_recent_recommendations(
        days=14, modes=("afternoon", "premarket", "feedback_training"))
    report_content = reporter.generate_report(
        recommendations=recommendations,
        market_overview=index_data,
        global_markets=global_markets_list,
        news_analysis=news_analysis,
        sentiment=sentiment,
        trade_config=trade_cfg,
        rotation_data=rotation_dict,
        history_records=history,
        report_role="final",
        intraday_note=afternoon_health_note,
    )
    report_path = reporter.save_report(report_content)

    write_summary("afternoon", len(candidates), final,
                   updated=len(updated))

    print("\n" + "=" * 60)
    print("  午后模式完成!")
    print(f"  修正候选: {len(updated)} 只")
    print(f"  达标推荐: {len(final)} 只")
    print(f"  报告文件: {report_path}")
    print("=" * 60)
    if final:
        print("\n午后推荐:")
        for i, c in enumerate(final, 1):
            print(f"  {i}. {c['name']}({c['code']}) - {c['afternoon_score']}分 {c['change_pct']:+.1f}% 净流入{c['net_inflow']:.0f}万")
    else:
        print("\n今日无达标推荐，建议观望")


def run_afternoon(config):
    """午后模式主入口：初始化 → 修正评分 → 筛选落库 → 报告。"""
    ctx = {"config": config}
    if not _am_init(ctx):
        return
    _am_rescore(ctx)
    _am_filter_persist(ctx)
    _am_report(ctx)


# ======================================================================
# 因子IC评估
# ======================================================================
