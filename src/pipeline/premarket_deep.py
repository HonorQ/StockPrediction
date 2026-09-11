# -*- coding: utf-8 -*-
"""盘前模式(07:00): 全量因子计算与候选池产出

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import pandas as pd
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger
from src.pipeline.common import _prefetch_history_and_flow

logger = get_logger()
from src.factors.moat import MoatFactors
from src.factors.strategies import StockStrategies
from src.factors.technical import TechnicalFactors
from src.pipeline.common import _pack


def _pm_screen(ctx):
    """硬性条件筛选: ST/停牌/板块排除。

    显式依赖注入(AGENTS.md 二节): 只 unpack 本阶段实际需要的键,
    其余上下文经 ctx 对后续阶段可见, 不做全量往返。
    """
    screener, a_stock_df = ctx.unpack('screener', 'a_stock_df')
    config = ctx["config"]

    print("\n[6/6] 筛选股票并计算因子...")
    print("  --- 第一层: 硬性条件筛选 ---")
    candidates_df = screener.filter_hard_conditions(a_stock_df)
    before = len(candidates_df)

    # 排除指定板块（从配置读取）
    stock_cfg = config.get("stock", {})
    exclude_boards = stock_cfg.get("exclude_boards", ["300", "301", "688"])
    if exclude_boards:
        candidates_df = candidates_df[~candidates_df["代码"].astype(str).str.startswith(tuple(exclude_boards))]
        print(f"  📋 排除板块{exclude_boards}: {before} → {len(candidates_df)} 只")
    else:
        print(f"  📋 未排除任何板块: {len(candidates_df)} 只")

    # 阶段1: 快速预评分
    # 权重参数从 config.yaml 的 filter.quick_score 读取，默认值与历史硬编码一致；
    # 候选数量按 regime 动态调整：bull 放宽进入深度分析的票数，bear/crash 收紧省时
    _pack(ctx, dict(locals()))
    ctx['config'] = config

def _pm_quick_topn(ctx):
    """实时快评 → TopN候选(regime动态) + 并行K线预取。

    显式依赖注入: 本阶段只需快评输入(candidates_df/hot_codes/news)与 regime 配置。
    """
    candidates_df, filter_cfg, hot_codes, news_factor, news_list, regime = ctx.unpack(
        'candidates_df', 'filter_cfg', 'hot_codes', 'news_factor', 'news_list', 'regime',
    )
    config = ctx["config"]

    print("  --- 阶段1: 实时行情快速预评分 ---")
    qs_cfg = filter_cfg.get("quick_score", {}) or {}
    # QS 快评参数: 从 config 读取(默认值与历史硬编码一致)
    _QSD = {k: float(qs_cfg.get(k, d)) for k, d in [
        ("change_weight", 3), ("change_cap", 20),
        ("turnover2_bonus", 10), ("turnover5_bonus", 5),
        ("volume_ratio_bonus", 10), ("hot_bonus", 15),
        ("mv_sweet_bonus", 5), ("news_bonus_cap", 10)]}
    QS_CHANGE_W = _QSD["change_weight"]
    QS_CHANGE_CAP = _QSD["change_cap"]
    QS_TURNOVER2 = _QSD["turnover2_bonus"]
    QS_TURNOVER5 = _QSD["turnover5_bonus"]
    QS_VOLUME_RATIO = _QSD["volume_ratio_bonus"]
    QS_HOT = _QSD["hot_bonus"]
    QS_MV_SWEET = _QSD["mv_sweet_bonus"]
    QS_NEWS_CAP = _QSD["news_bonus_cap"]

    if regime == "bull":
        top_n = int(qs_cfg.get("top_n_bull", 50))
    elif regime in ("bear", "crash"):
        top_n = int(qs_cfg.get("top_n_defensive", 20))
    else:
        top_n = int(qs_cfg.get("top_n_default", 30))

    quick_scored = []
    for _, row in candidates_df.iterrows():
        stock_code = str(row["代码"])
        stock_name = row["名称"]
        current_price = row.get("最新价", 0)
        change_pct = row.get("涨跌幅", 0) or 0
        turnover = row.get("换手率", 0) or 0
        volume_ratio = row.get("量比", 0) or 0
        total_mv = row.get("总市值", 0) or 0
        if current_price == 0 or current_price is None:
            continue
        qs = 50
        if change_pct > 0:
            qs += min(change_pct * QS_CHANGE_W, QS_CHANGE_CAP)
        if turnover > 2:
            qs += QS_TURNOVER2
        if turnover > 5:
            qs += QS_TURNOVER5
        if volume_ratio > 1.5:
            qs += QS_VOLUME_RATIO
        if stock_code in hot_codes:
            qs += QS_HOT
        mv_yi = total_mv / 1e8 if total_mv else 0
        if 50 < mv_yi < 500:
            qs += QS_MV_SWEET
        ns = news_factor.score_stock_news(stock_code, news_list)
        if ns > 60:
            qs += min((ns - 50) / 5, QS_NEWS_CAP)
        quick_scored.append({
            "code": stock_code, "name": stock_name,
            "current_price": float(current_price),
            "quick_score": qs, "row": row,
        })
    quick_scored.sort(key=lambda x: x["quick_score"], reverse=True)
    top_candidates = quick_scored[:top_n]
    # 把北交所(bj)股排后面（baostock不支持，先处理sz/sh）
    top_candidates.sort(key=lambda x: (x["code"].startswith(('bj', 'BJ', '8', '9')), -x["quick_score"]))
    print(f"  实时预评分完成: {len(candidates_df)} → Top{len(top_candidates)} 进入深度分析 (regime={regime})")

    # 阶段2: 深度因子计算
    _pack(ctx, dict(locals()))
    ctx['config'] = config

def _calc_one_stock(ctx, idx, cand):
    """单票深度因子计算(自 _pm_deep_loop 大循环机械提取)。

    组件与上下文经 ctx 注入; 因子计算拆为五个块(见下方 _calc_*_block);
    结果 append 到 scored_stocks。
    """
    (history_cache, market_fetcher, phase2_start, scored_stocks,
     top_candidates) = ctx.unpack(
        'history_cache', 'market_fetcher', 'phase2_start', 'scored_stocks',
        'top_candidates',
    )

    if (datetime.now() - phase2_start).total_seconds() > 480:
        print(f"  ⏰ 阶段2已运行超过8分钟，停止计算（已处理{idx}/{len(top_candidates)}只）")
        return "timeout"

    stock_code = cand["code"]
    stock_name = cand["name"]
    current_price = cand["current_price"]
    row = cand["row"]
    print(f"    [{idx+1}/{len(top_candidates)}] {stock_name}({stock_code})...")
    history = history_cache.get(stock_code)
    if history is None:
        history = market_fetcher.get_stock_history(stock_code, days=120)
    if history is None or history.empty or len(history) < 20:
        print("      ⚠️ 历史数据不足，跳过")
        return "skip"

    all_factors = {}
    # 20日日均成交额(亿元): liquidity_guard 消费字段
    # 2026-09-11 修复: 原生产者随未提交改动丢失, guard 连续10天空转(见 logs/2026-09-*)
    amt_yi = _calc_avg_daily_amount_20d(history)
    if amt_yi is not None:
        all_factors["avg_daily_amount_20d"] = amt_yi
    _calc_tech_block(ctx, cand, history, all_factors)
    _calc_sustain_block(ctx, cand, history, all_factors)
    _calc_sentiment_event_block(ctx, cand, history, all_factors)
    _calc_policy_deep_block(ctx, cand, history, all_factors)
    _calc_composite_block(ctx, cand, history, all_factors)
    return None


def _calc_tech_block(ctx, cand, history, all_factors):
    """技术面块: 技术因子/资金流/新闻/情绪/策略/筹码/基本面/人气/机构/强弱/超跌/轮动。"""
    (chip_calc, flow_cache, fundamental_factor, hot_codes, hot_df,
     institutional_factor, money_factor, money_fetcher, news_factor, news_list,
     popularity_factor, relative_strength, market_history, oversold_strategy,
     rotation_data, sector_rotation, sentiment, sentiment_factor, tech_cfg) = ctx.unpack(
        'chip_calc', 'flow_cache', 'fundamental_factor', 'hot_codes', 'hot_df',
        'institutional_factor', 'money_factor', 'money_fetcher', 'news_factor', 'news_list',
        'popularity_factor', 'relative_strength', 'market_history', 'oversold_strategy',
        'rotation_data', 'sector_rotation', 'sentiment', 'sentiment_factor', 'tech_cfg',
    )
    stock_code = cand["code"]
    current_price = cand["current_price"]

    tech_factors = TechnicalFactors(history, tech_cfg)
    all_factors.update(tech_factors.calculate_all())

    flow_data = flow_cache.get(stock_code)
    if flow_data is None:
        flow_data = money_fetcher.get_stock_fund_flow(stock_code)
    mf = money_factor.calculate_all(flow_data)
    all_factors.update(mf)

    stock_news_score = news_factor.score_stock_news(stock_code, news_list)
    all_factors["news_score"] = stock_news_score

    in_hot = stock_code in hot_codes
    sf = sentiment_factor.calculate_all(sentiment, in_hot)
    all_factors.update(sf)

    strategies = StockStrategies.get_triggered_strategies(history)
    all_factors["strategies_triggered"] = strategies
    all_factors["strategy_count"] = len(strategies)

    chip = chip_calc.calculate(history)
    all_factors.update(chip)
    all_factors["chip_score"] = chip_calc.get_chip_score(history)
    # 补 chip_ 前缀映射，与 ml_selector._extract_features 期望的 key 名对齐
    all_factors["chip_profit_ratio"] = chip.get("profit_ratio", 50)
    all_factors["chip_concentration_70"] = chip.get("concentration_70", 50)
    all_factors["chip_above_peak"] = chip.get("price_above_peak", 50)
    all_factors["close_price"] = current_price

    stock_data = cand["row"].to_dict()
    ff = fundamental_factor.calculate_all(stock_data)
    all_factors.update(ff)

    popularity_data = {
        "hot_rank": hot_df[hot_df["股票代码"].astype(str) == stock_code]["排名"].iloc[0] if not hot_df.empty and stock_code in hot_df.get("股票代码", pd.Series()).astype(str).values else 0,
        "in_hot_list": stock_code in hot_codes,
        "limit_up": sentiment.get("limit_up", 0),
        "limit_down": sentiment.get("limit_down", 0),
        "max_board": sentiment.get("max_board", 0),
    }
    pf = popularity_factor.calculate_all(popularity_data, stock_code)
    all_factors.update(pf)

    inf = institutional_factor.calculate_all(stock_code, stock_data)
    all_factors.update(inf)

    rs = relative_strength.calculate_all(history, market_df=market_history)
    all_factors.update(rs)

    rebound_result = oversold_strategy.check_oversold_rebound(history)
    all_factors["oversold_rebound_detected"] = rebound_result["detected"]
    all_factors["oversold_rebound_score"] = rebound_result["score"]
    all_factors["oversold_rebound_signals"] = rebound_result["signals"]

    all_factors["rotation_score"] = sector_rotation.get_rotation_score(rotation_data)
    all_factors["tech_sector_strength"] = rotation_data.get("tech_strength", 50)
    all_factors["rotation_signal"] = rotation_data.get("rotation_signal", "neutral")


def _calc_sustain_block(ctx, cand, history, all_factors):
    """反弹持续性与预判块: 持续性/次日预判/预期透支/利好兑现/获利出逃。"""
    (expectation_pricing, news_event_analyzer, news_list, next_day_predictor,
     profit_taking_detector, rebound_sustainability, rotation_data) = ctx.unpack(
        'expectation_pricing', 'news_event_analyzer', 'news_list', 'next_day_predictor',
        'profit_taking_detector', 'rebound_sustainability', 'rotation_data',
    )
    stock_code = cand["code"]
    stock_name = cand["name"]

    sustainability = rebound_sustainability.analyze(history, rotation_data)
    all_factors["sustainability_score"] = sustainability["sustainability_score"]
    all_factors["rebound_type"] = sustainability["rebound_type"]
    all_factors["next_day_bias"] = sustainability["next_day_bias"]
    all_factors["sustainability_signals"] = sustainability["signals"]
    all_factors["sustainability_risks"] = sustainability["risk_factors"]

    next_day = next_day_predictor.predict(history, sustainability, rotation_data)
    all_factors["next_day_prediction"] = next_day

    expectation = expectation_pricing.analyze(history, news_list, stock_code)
    all_factors["expectation_score"] = expectation["expectation_score"]
    all_factors["expectation_overvalued"] = expectation["overvalued"]
    all_factors["expectation_reasons"] = expectation["reasons"]

    event_pricing = news_event_analyzer.analyze_event_pricing(news_list, stock_code, history)
    all_factors["event_priced"] = event_pricing["event_priced"]
    all_factors["event_pricing_degree"] = event_pricing["pricing_degree"]
    all_factors["event_pricing_signals"] = event_pricing["signals"]

    profit_taking = profit_taking_detector.detect(history, code=stock_code, name=stock_name)
    all_factors["profit_taking_detected"] = profit_taking["detected"]
    all_factors["profit_taking_intensity"] = profit_taking["intensity"]
    all_factors["profit_taking_signals"] = profit_taking["signals"]


def _calc_sentiment_event_block(ctx, cand, history, all_factors):
    """题材行为块: A股特色/游资/情绪周期/量化/两融/股东/涨跌停/板块联动/龙头/资金持续/调研/龙虎榜。"""
    (ashare_factors, dragon_tiger, hot_money_tracker, institutional_research,
     limit_quality_factor, lockup_factor, margin_factor,
     money_flow_continuity, quant_detector, sector_leader, sector_linkage,
     sentiment, sentiment_cycle, shareholder_factor) = ctx.unpack(
        'ashare_factors', 'dragon_tiger', 'hot_money_tracker', 'institutional_research',
        'limit_quality_factor', 'lockup_factor', 'margin_factor',
        'money_flow_continuity', 'quant_detector', 'sector_leader', 'sector_linkage',
        'sentiment', 'sentiment_cycle', 'shareholder_factor',
    )
    stock_code = cand["code"]
    stock_name = cand["name"]
    stock_data = cand["row"].to_dict()

    ashare = ashare_factors.calculate_all(history, sentiment, code=stock_code, name=stock_name)
    all_factors.update(ashare)

    hot_money = hot_money_tracker.analyze(history, code=stock_code, name=stock_name)
    all_factors["hot_money_score"] = hot_money["hot_money_score"]
    all_factors["hot_money_signals"] = hot_money["signals"]

    sentiment_analysis = sentiment_cycle.analyze(sentiment)
    all_factors["sentiment_cycle"] = sentiment_analysis

    quant_result = quant_detector.detect_all(history, stock_data)
    all_factors["quant_detected"] = quant_result["quant_detected"]
    all_factors["quant_score"] = quant_result["quant_score"]
    all_factors["quant_signals"] = quant_result["quant_signals"]
    all_factors["quant_risk"] = quant_result["quant_risk"]

    margin = margin_factor.calculate(stock_code)
    all_factors["margin_trend"] = margin["margin_trend"]
    all_factors["margin_score"] = margin.get("margin_score", 50)
    all_factors["margin_signals"] = margin["margin_signals"]
    shareholder = shareholder_factor.calculate(stock_code)
    all_factors["shareholder_trend"] = shareholder["shareholder_trend"]
    all_factors["shareholder_score"] = shareholder.get("shareholder_score", 50)
    all_factors["shareholder_signals"] = shareholder["shareholder_signals"]
    lockup = lockup_factor.calculate(stock_code)
    all_factors["lockup_risk"] = lockup["lockup_risk"]
    all_factors["lockup_score"] = lockup.get("lockup_score", 50)
    all_factors["lockup_signals"] = lockup["lockup_signals"]
    limit_quality = limit_quality_factor.calculate(history, code=stock_code, name=stock_name)
    all_factors["limit_quality"] = limit_quality["limit_quality"]
    all_factors["limit_signals"] = limit_quality["limit_signals"]

    linkage = sector_linkage.analyze_sector_linkage(stock_code)
    all_factors["sector_linkage_score"] = linkage["sector_linkage_score"]
    all_factors["sector_signals"] = linkage["sector_signals"]
    all_factors["hot_concepts"] = linkage["hot_concepts"]
    all_factors["concept_count"] = linkage.get("concept_count", 0)

    leader = sector_leader.detect_leader(history, stock_code)
    all_factors["is_leader"] = leader["is_leader"]
    all_factors["leader_score"] = leader["leader_score"]
    all_factors["leader_signals"] = leader["leader_signals"]

    flow_continuity = money_flow_continuity.analyze(stock_code)
    all_factors["flow_continuity"] = flow_continuity["flow_continuity"]
    all_factors["flow_continuity_score"] = flow_continuity["flow_continuity_score"]
    all_factors["flow_continuity_signals"] = flow_continuity["flow_signals"]

    research = institutional_research.analyze(stock_code)
    all_factors["research_score"] = research["research_score"]
    all_factors["research_signals"] = research["research_signals"]

    lh = dragon_tiger.analyze(stock_code)
    all_factors["lh_score"] = lh["lh_score"]
    all_factors["lh_signals"] = lh["lh_signals"]
    all_factors["hot_money_count"] = lh["hot_money_count"]
    all_factors["institution_count"] = lh["institution_count"]
    all_factors["institution_buy_net"] = lh["institution_buy_net"]


def _calc_policy_deep_block(ctx, cand, history, all_factors):
    """政策与深度研究块: 深度研究/政策事件/资金行为模式/国家队。"""
    (capital_behavior_factor, deep_research_factor, market_history, national_team_factor,
     news_list, policy_event_factor, sentiment) = ctx.unpack(
        'capital_behavior_factor', 'deep_research_factor', 'market_history',
        'national_team_factor', 'news_list', 'policy_event_factor', 'sentiment',
    )
    stock_code = cand["code"]
    stock_name = cand["name"]
    stock_data = cand["row"].to_dict()

    dr = deep_research_factor.analyze(history, news_list, stock_code, stock_data)
    all_factors["deep_research_score"] = dr["deep_research_score"]
    all_factors["deep_research_signals"] = dr["deep_research_signals"]
    all_factors["volatility_regime"] = dr["volatility_regime"]
    all_factors["position_at"] = dr["position_at"]
    all_factors["catalyst_match_score"] = dr["catalyst_match_score"]
    all_factors["big_move_count"] = dr["big_move_count"]
    all_factors["prediction_confidence"] = dr["prediction_confidence"]
    all_factors["historical_reference"] = dr["historical_reference"]
    all_factors["risk_reference"] = dr["risk_reference"]

    # 政策事件因子
    policy_result = policy_event_factor.calculate(
        stock_code=stock_code,
        news_list=news_list,
        stock_sector=stock_data.get("行业", ""),
        stock_concepts=stock_data.get("concepts", []),
        stock_name=stock_name,
    )
    all_factors["policy_score"] = policy_result["policy_score"]
    all_factors["policy_direction"] = policy_result["policy_direction"]
    all_factors["macro_impact"] = policy_result["macro_impact"]
    all_factors["policy_events"] = policy_result["events"]
    all_factors["policy_signals"] = policy_result["signals"]

    # 资金行为模式因子
    cap_behavior = capital_behavior_factor.calculate(history)
    all_factors["capital_behavior"] = cap_behavior["capital_behavior"]
    all_factors["capital_behavior_confidence"] = cap_behavior["capital_behavior_confidence"]
    all_factors["capital_behavior_signals"] = cap_behavior["capital_behavior_signals"]
    all_factors["flow_continuity"] = cap_behavior["flow_continuity"]

    # 国家队因子
    national_team = national_team_factor.calculate(
        history=history,
        stock_info=stock_data,
        market_index=market_history,
    )
    all_factors["national_team_score"] = national_team["national_team_score"]
    all_factors["national_team_holding"] = national_team["national_team_holding"]
    all_factors["national_team_change"] = national_team["national_team_change"]
    all_factors["national_team_signals"] = national_team["national_team_signals"]


def _calc_composite_block(ctx, cand, history, all_factors):
    """合成评分块: ATR止损/股东人数/流通股东/护城河/子评分/结果落袋。"""
    (atr_stop, circulate_factor, holder_factor, position_manager, scorer,
     scored_stocks, trade_cfg) = ctx.unpack(
        'atr_stop', 'circulate_factor', 'holder_factor', 'position_manager', 'scorer',
        'scored_stocks', 'trade_cfg',
    )
    stock_code = cand["code"]
    stock_name = cand["name"]
    current_price = cand["current_price"]

    try:
        atr_series = atr_stop.calculate_atr(history)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        if current_atr > 0 and current_price > 0:
            stop_info = position_manager.calculate_stop_loss(current_price, current_atr, 50)
            take_info = position_manager.calculate_take_profit(current_price, current_atr, 50)
            pos_info = position_manager.calculate_position(50, trade_cfg.get("max_position_pct", 0.25), current_price)
            all_factors["atr"] = round(current_atr, 3)
            all_factors["dynamic_stop_loss"] = stop_info["stop_loss"]
            all_factors["dynamic_stop_loss_pct"] = stop_info["stop_loss_pct"]
            all_factors["dynamic_take_profit"] = take_info["take_profit"]
            all_factors["dynamic_take_profit_pct"] = take_info["take_profit_pct"]
            all_factors["position_pct"] = pos_info["position_pct"]
            all_factors["position_shares"] = pos_info["shares"]
    except Exception as e:
        logger.warning("ATR止损计算失败(code=%s): %s", stock_code, e)

    # 股东人数/筹码集中度因子
    holder = holder_factor.analyze(stock_code)
    all_factors["holder_score"] = holder["holder_score"]
    all_factors["holder_trend"] = holder["holder_trend"]
    all_factors["holder_quarter_change_pct"] = holder["holder_quarter_change_pct"]

    # 流通股东因子
    circulate = circulate_factor.analyze(stock_code)
    all_factors["circulate_score"] = circulate["circulate_score"]
    all_factors["inst_holding_change"] = circulate["inst_holding_change"]
    all_factors["northbound_change"] = circulate["northbound_change"]
    all_factors["top10_holding_pct"] = circulate["top10_holding_pct"]

    # 护城河/不可替代性因子（A层：复用 concept_count / leader / institutional / 市值）
    all_factors.update(MoatFactors().calculate_all(all_factors))

    sub_scores = scorer.get_sub_scores(all_factors)
    expectation_penalty = scorer.penalty_expectation(all_factors)

    scored_stocks.append({
        "code": stock_code,
        "name": stock_name,
        "current_price": current_price,
        "score": 50,  # placeholder, batch_normalize will overwrite
        "sub_scores": sub_scores,
        "expectation_penalty": expectation_penalty,
        "factors": all_factors,
        "strong_signals": 0,
    })



def _pm_deep_loop(ctx):
    """逐票深度因子计算 + 截面Z-Score归一化。

    显式依赖注入: 本阶段只需候选列表/取数器/DB;
    单票计算经 _calc_one_stock → _calc_*_block 各自按需 unpack。
    """
    db, market_fetcher, money_fetcher, top_candidates = ctx.unpack(
        'db', 'market_fetcher', 'money_fetcher', 'top_candidates',
    )
    config = ctx["config"]

    print("  --- 阶段2: 深度因子计算 ---")
    scored_stocks = []
    phase2_start = datetime.now()
    ctx["phase2_start"] = phase2_start

    # 并行预取历史K线 + 资金流（网络I/O并行，CPU计算保持串行）
    prefetch_codes = [c["code"] for c in top_candidates]
    print(f"  🔄 并行预取 {len(prefetch_codes)} 只股票的历史K线和资金流...")
    prefetch_start = datetime.now()
    history_cache, flow_cache = _prefetch_history_and_flow(
        market_fetcher, money_fetcher, prefetch_codes, days=120
    )
    ctx["history_cache"] = history_cache
    ctx["flow_cache"] = flow_cache
    ctx["scored_stocks"] = scored_stocks
    elapsed_prefetch = (datetime.now() - prefetch_start).total_seconds()
    ok_count = sum(1 for v in history_cache.values() if v is not None and not v.empty)
    print(f"  ✅ 预取完成: 历史K线 {ok_count}/{len(prefetch_codes)} 只, 耗时 {elapsed_prefetch:.1f}s")

    # K线落库：ML_TRAINING_PLAN 关键缺口#5——为离线全市场重算回测积累数据。
    # 幂等 upsert，失败不影响主流程
    try:
        saved_rows = 0
        saved_stocks = 0
        for code_k, h_k in history_cache.items():
            n = db.save_klines(code_k, h_k)
            if n > 0:
                saved_rows += n
                saved_stocks += 1
        print(f"  💾 K线落库: {saved_stocks} 只 / {saved_rows} 行")
    except Exception as e:
        logger.warning("K线落库失败(不影响主流程): %s", e)

    # 获取沪深300历史K线，用于 relative_strength 因子计算相对大盘强弱
    # （此前未传入 market_df 导致 relative_strength_5d/20d 永远缺失，sub_relative_strength 恒为 50）
    market_history = None
    try:
        market_history = market_fetcher.get_stock_history("sh000300", days=120)
        if market_history is None or market_history.empty or len(market_history) < 20:
            market_history = None
        else:
            # 指数K线一并落库，供离线 regime 判定/回测基准使用
            try:
                db.save_klines("sh000300", market_history)
            except Exception as e:
                logger.warning("沪深300指数K线落库失败(忽略): %s", e)
    except Exception as e:
        print(f"  ⚠️ 获取沪深300历史失败，relative_strength 将降级为个股自身强弱: {e}")
        market_history = None

    for idx, cand in enumerate(top_candidates):
        result = _calc_one_stock(ctx, idx, cand)
        if result == "timeout":
            print(f"  ⏰ 阶段2已运行超过8分钟，停止计算（已处理{idx}只）")
            break

    print(f"  共计算 {len(scored_stocks)} 只股票的因子")
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _calc_avg_daily_amount_20d(history):
    """20日日均成交额(亿元), apply_liquidity_guard 消费。

    K线"成交额"列单位为元(腾讯主源 row[6]万元*10000 / 新浪备源 close*volume,
    两源口径一致, 见 kline_fetcher.py), 除以 1e8 转亿元与 guard 阈值对齐。
    数据不可用时返回 None——guard 侧对缺字段有跳过+warning 语义, 此处不造数。

    单元测试: tests/test_bugfixes.py::test_avg_daily_amount_20d_producer
    """
    if history is None or history.empty or "成交额" not in history.columns:
        return None
    amt = pd.to_numeric(history["成交额"], errors="coerce").tail(20).mean()
    if pd.notna(amt) and amt > 0:
        return round(float(amt) / 1e8, 4)
    return None
