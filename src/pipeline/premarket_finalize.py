# -*- coding: utf-8 -*-
"""盘前模式收尾段: 归一化筛选 → 行业分散 → 候选池缓存 → 落库开仓 → 报告 → 反馈

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
各阶段显式按需 unpack(禁止巨型键包往返), 状态经 ctx 传递。
"""
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger

logger = get_logger()
from src.engine.backtest import ATRStopLoss, PositionManager
from src.engine.reporter import ReportGenerator
from src.data.database import StockDatabase
from src.engine.scorer import StockScorer
from src.engine.entry_risk_filter import (
    _filter_high_risk_entries)
from src.engine.paper_trade_ops import (
    _update_open_paper_trades, _fetch_close_price_on_date)
from src.pipeline.common import _pack
from src.pipeline.common import save_cache
from src.pipeline.common import write_summary


def _pm_guard(ctx):
    """截面Z-Score归一化 + alpha overlay/chip风控 + 三层筛选 + 择时与健康度降仓。"""
    (
        db, filter_cfg, health_note, regime, regime_filter,
        regime_result, scored_stocks, screener, weights,
    ) = ctx.unpack(
        'db', 'filter_cfg', 'health_note', 'regime', 'regime_filter',
        'regime_result', 'scored_stocks', 'screener', 'weights',
    )
    config = ctx["config"]

    print("  --- 截面Z-Score归一化(替代固定50分基线) ---")
    # 动态正交化对由 feedback_loop IC 分析持久化，跨进程生效
    from src.engine.feedback_loop import FeedbackLoop
    dynamic_pairs = FeedbackLoop._load_dynamic_ortho_pairs()
    if dynamic_pairs:
        print(f"  (动态正交化 {len(dynamic_pairs)} 对已加载)")
    StockScorer.batch_normalize(scored_stocks, weights, extra_ortho_pairs=dynamic_pairs)

    # === Chip 风控排除 + Ashare 弱信号排除 + 美股板块风控(全市场分层验证产物, 实盘/回测同口径) ===
    # alpha_overlay 已禁用(2026-08-26 组合模拟证伪多头可提取性), 不再调用
    from src.engine.scorer import apply_chip_guard, apply_ashare_guard, apply_us_sector_guard
    scored_stocks = apply_chip_guard(scored_stocks, config)
    scored_stocks = apply_ashare_guard(scored_stocks, config)
    # 美股板块风控: 传入行业映射
    industry_map = market_fetcher.get_industry_map() if market_fetcher else None
    scored_stocks = apply_us_sector_guard(scored_stocks, config, industry_map=industry_map)

    # 信号筛选
    print("  --- 第二层: 信号强度筛选 ---")
    qualified = screener.filter_signal_strength(
        scored_stocks, min_strong=filter_cfg.get("min_strong_signals", 1))
    print("  --- 第三层: 综合评分筛选 ---")
    final = screener.filter_by_score(
        qualified, min_score=filter_cfg.get("min_score", 55))

    # === 入场点风险过滤:剔除追涨/高位/量价背离票 ===
    final = _filter_high_risk_entries(final, filter_cfg)

    final.sort(key=lambda x: x["score"], reverse=True)
    max_rec = filter_cfg.get("max_recommend", 2)
    # health_note 可能已被运维告警初始化(见 run_premarket 开头)

    # === 大盘择时层：缩减推荐数 + bear/crash 空仓（评估与权重切换已在 5.2 完成）===
    try:
        if regime_result is not None:
            regime_max = regime_result["max_recommend"]
            final_max_rec = min(max_rec, regime_max) if regime != "bull" else max_rec
            print(f"  🎯 大盘择时: {regime} | 得分 {regime_result['score']}/100 | "
                  f"推荐上限 {max_rec} → {final_max_rec}")
            # crash/bear 模式执行空仓策略：清空推荐，不下场
            if regime_filter is not None and regime_filter.should_empty_portfolio(
                    regime, regime_result["score"]):
                print(f"  🛑 市场风险过高（{regime}模式，得分{regime_result['score']}），今日不推荐任何股票")
                final = []
                final_max_rec = 0
            max_rec = final_max_rec
    except Exception as e:
        logger.warning("大盘择时缩减推荐失败，沿用默认推荐上限: %s", e)

    # === 策略健康度层：自身推荐滚动收益持续负超额时自动降仓(风格失效自保) ===
    try:
        from src.engine.strategy_health import assess_strategy_health, apply_action
        health_today = datetime.now().strftime("%Y-%m-%d")
        health = assess_strategy_health(db, today=health_today,
                                        config=filter_cfg.get("strategy_health"))
        if health.get("status") == "degraded":
            before_health = max_rec
            max_rec = apply_action(max_rec, health["action"])
            warn_txt = (f"⚠️ 策略健康度降级: 近{health['samples']}条推荐 "
                        f"胜率{health['win_rate']}% / 平均d5 {health['avg_d5']:+.2f}% "
                        f"→ 推荐上限 {before_health} → {max_rec}")
            print(f"  🩺 {warn_txt}")
            health_note = health_note + f"【{warn_txt}】" if health_note else f"【{warn_txt}】"
        else:
            print(f"  🩺 策略健康度: {health.get('status')} "
                  f"(样本{health.get('samples', 0)})")
    except Exception as e:
        logger.warning("策略健康度监控失败(不影响主流程): %s", e)

    recommendations = final[:max_rec]
    _pack(ctx, dict(locals()))
    ctx["config"] = config


# 行业→赛道 映射与赛道容量上限(原字面量原样保留, 语义见 _pm_diversify)
_SECTOR_GROUP_MAP = {
    "金融行业": "金融", "房地产": "地产", "开发区": "地产",
    "电子信息": "科技", "电子器件": "科技", "电器行业": "科技",
    "仪器仪表": "科技", "发电设备": "科技",
    "化工行业": "制造", "化纤行业": "制造", "塑料制品": "制造",
    "机械行业": "制造", "纺织机械": "制造", "纺织行业": "制造",
    "服装鞋类": "制造", "印刷包装": "制造", "造纸行业": "制造",
    "家电行业": "制造", "家具行业": "制造", "陶瓷行业": "制造",
    "玻璃行业": "制造", "水泥行业": "制造", "建筑建材": "制造",
    "生物制药": "医药", "医疗器械": "医药",
    "食品行业": "消费", "酿酒行业": "消费",
    "商业百货": "消费", "酒店旅游": "消费",
    "石油行业": "能源", "煤炭行业": "能源", "电力行业": "能源",
    "交通运输": "交通", "公路桥梁": "交通", "供水供气": "公用",
    "钢铁行业": "钢铁", "有色金属": "有色",
    "汽车制造": "汽车", "摩托车": "汽车",
    "飞机制造": "军工", "船舶制造": "军工",
    "农林牧渔": "农业", "农药化肥": "农业",
    "环保行业": "环保", "传媒娱乐": "传媒",
    "物资外贸": "综合", "综合行业": "综合", "其它行业": "综合",
    "次新股": "次新",
}
_SECTOR_CAP_MAP = {
    "科技": 2, "制造": 2, "金融": 1, "消费": 1, "医药": 1,
    "地产": 1, "能源": 1, "交通": 1, "钢铁": 1, "有色": 1,
    "汽车": 1, "军工": 1, "农业": 1, "环保": 1, "传媒": 1,
    "公用": 1, "综合": 1, "次新": 1,
}


def _pm_diversify(ctx):
    """行业分散约束: 同行业/同赛道限流 + 业务画像生成(随缓存持久化)。"""
    (
        final, market_fetcher, max_rec, recommendations, screener,
        scored_stocks,
    ) = ctx.unpack(
        'final', 'market_fetcher', 'max_rec', 'recommendations', 'screener',
        'scored_stocks',
    )
    config = ctx["config"]

    try:
        industry_map = market_fetcher.get_industry_map()
        if industry_map:
            SECTOR_GROUP = _SECTOR_GROUP_MAP
            SECTOR_CAP = _SECTOR_CAP_MAP
            max_per_ind = 2 if max_rec >= 4 else 1
            for s in final:
                code = s.get("code", "")
                ind = industry_map.get(code, "未知")
                s["factors"]["industry"] = ind
                s["factors"]["sector"] = SECTOR_GROUP.get(ind, "其他")
            recommendations = screener.diversify_by_industry(
                final, max_per_industry=max_per_ind,
                industry_map=industry_map, max_total=max_rec
            )
            # 赛道级约束：同赛道最多 N 只
            sector_count = {}
            filtered = []
            for s in recommendations:
                sec = s["factors"].get("sector", "其他")
                cap = SECTOR_CAP.get(sec, 1)
                if sector_count.get(sec, 0) < cap:
                    filtered.append(s)
                    sector_count[sec] = sector_count.get(sec, 0) + 1
            recommendations = filtered
            print(f"  🏭 行业分散后推荐: {len(recommendations)} 只 (赛道上限: {dict(sector_count)})")

            # === B层：规则抽取业务画像 + 复用 MoatFactors 不可替代性 ===
            # 零新网络依赖：仅基于已算好的 industry / hot_concepts / moat 信号做关键词规则抽取
            # 对全量候选股计算，并随缓存持久化，供盘中/午后复用
            try:
                from src.factors.business_profile import BusinessProfileFactor
                bp_engine = BusinessProfileFactor()
                bp_count = 0
                for s in scored_stocks:
                    s["business_profile"] = bp_engine.build(
                        s.get("code", ""), s.get("name", ""), s.get("factors", {})
                    )
                    bp_count += 1
                logger.info("业务画像生成: %d 只（含盘中缓存）", bp_count)
            except Exception as e:
                logger.warning("业务画像生成失败(不影响主流程): %s", e)
    except Exception as e:
        logger.warning("行业分散约束失败，沿用原排序: %s", e)
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_cache_candidates(ctx):
    """候选池落盘 cache/premarket_YYYY-MM-DD.json, 供盘中(9:25)/午后(14:40)复用。"""
    (
        _regime_info, global_markets, index_data, news_analysis, rotation_data,
        score, sentiment, scored_stocks, trade_cfg,
    ) = ctx.unpack(
        '_regime_info', 'global_markets', 'index_data', 'news_analysis', 'rotation_data',
        'score', 'sentiment', 'scored_stocks', 'trade_cfg',
    )
    config = ctx["config"]

    # 注意: _pack 排除 '_' 开头键, 此处 today 以普通名进入 ctx 供后续阶段使用
    today = datetime.now().strftime("%Y-%m-%d")

    cache_data = {
        "date": today,
        "run_at": datetime.now().strftime("%H:%M:%S"),
        "index_data": index_data,
        "global_markets": global_markets,
        "news_analysis": news_analysis,
        "sentiment": sentiment,
        "rotation_data": rotation_data,
        "trade_config": trade_cfg,
        "regime": _regime_info,
        "candidates": [],
    }
    for s in scored_stocks:
        f = s["factors"]
        nd = f.get("next_day_prediction", {})
        if not isinstance(nd, dict):
            nd = {}
        cache_data["candidates"].append({
            "code": s["code"],
            "name": s["name"],
            "score": s["score"],
            "strong_signals": s.get("strong_signals", 0),
            "premarket_price": s["current_price"],
            "signals": {
                "strategies": f.get("strategies_triggered", []),
                "strategy_count": f.get("strategy_count", 0),
                "macd_golden_cross": f.get("macd_golden_cross", 0),
                "ma_bullish": f.get("ma_bullish", 0),
                "volume_price_up": f.get("volume_price_up", 0),
                "breakout_20d_high": f.get("breakout_20d_high", 0),
                "next_day_prediction": str(f.get("next_day_prediction", "neutral")),
                "next_day": nd,
                "gap_open_probability": nd.get("gap_open_probability", 0) or 0,
                "gap_reasons": nd.get("reasons", []) or [],
                "oversold_rebound_detected": f.get("oversold_rebound_detected", False),
                "quant_risk": f.get("quant_risk", "low"),
                "quant_detected": f.get("quant_detected", False),
                "sustainability_score": f.get("sustainability_score", 50),
                "profit_taking_detected": f.get("profit_taking_detected", False),
                "hot_concepts": f.get("hot_concepts", ""),
            },
            "news_score": f.get("news_score", 50),
            "hot_money_score": f.get("hot_money_score", 0),
            "business_profile": s.get("business_profile"),
            "sustainability_risks": f.get("sustainability_risks", []) or [],
        })
    save_cache(f"premarket_{today}.json", cache_data)
    print(f"\n  ✅ 候选池已缓存 → cache/premarket_{today}.json ({len(cache_data['candidates'])}只)")
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_position_plan(ctx):
    """paper_trades 持仓维护 + 为每只推荐计算 ATR止损/止盈/仓位 配置。"""
    market_fetcher, recommendations, trade_cfg = ctx.unpack(
        'market_fetcher', 'recommendations', 'trade_cfg')
    config = ctx["config"]
    today = datetime.now().strftime("%Y-%m-%d")

    db = StockDatabase()
    try:
        _update_open_paper_trades(db, market_fetcher, trade_cfg, today)
    except Exception as _e:
        print(f"  ⚠️ paper_trades 维护失败: {_e}")
    if recommendations:
        atr_calc = ATRStopLoss()
        pos_mgr = PositionManager(max_stop_loss_pct=trade_cfg.get("max_stop_loss_pct", -0.08))
        account_capital = trade_cfg.get("account_capital", 100000)

        rec_configs = {}
        for rec in recommendations:
            code = rec["code"]
            entry_price = rec.get("current_price", 0) or 0
            score = rec["score"]
            atr_value = 0
            if entry_price > 0:
                try:
                    hist = market_fetcher.get_stock_history(code, days=30)
                    if hist is not None and not hist.empty:
                        atr_series = atr_calc.calculate_atr(hist)
                        atr_raw = float(atr_series.iloc[-1]) if not atr_series.empty else 0
                        atr_value = atr_raw if atr_raw > entry_price * 0.005 else 0
                except Exception as e:
                    logger.warning(f"计算ATR失败: {e}")
                    atr_value = 0

            pos_info = pos_mgr.calculate_position(score, account_capital, max(entry_price, 1))
            sl_info = pos_mgr.calculate_stop_loss(entry_price, atr_value, score)
            tp_info = pos_mgr.calculate_take_profit(entry_price, atr_value, score)

            rec["factors"]["position_pct"] = pos_info["position_pct"]

            rec_configs[code] = {
                "stop_loss_price": sl_info["stop_loss"],
                "stop_loss_pct": round(sl_info["stop_loss_pct"] / 100, 4),
                "take_profit_price": tp_info["take_profit"],
                "take_profit_pct": round(tp_info["take_profit_pct"] / 100, 4),
                "atr_multiplier": sl_info["atr_multiplier"],
                "trailing_activated": False,
                "trailing_stop_price": 0,
                "trailing_pct": 3.0,
                "hold_days": trade_cfg.get("hold_days", 3),
                "shares": pos_info["shares"],
            }
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_save_and_open(ctx):
    """推荐落库 + 同步开纸面仓 + 因子快照落库 + 历史推荐收益回填。"""
    (
        _regime_info, market_fetcher, rec_configs, recommendations, scored_stocks,
        trade_cfg,
    ) = ctx.unpack(
        '_regime_info', 'market_fetcher', 'rec_configs', 'recommendations', 'scored_stocks',
        'trade_cfg',
    )
    config = ctx["config"]
    today = datetime.now().strftime("%Y-%m-%d")
    db = StockDatabase()

    for i, rec in enumerate(recommendations, 1):
        db.save_recommendation(
            trade_date=today,
            code=rec["code"],
            name=rec["name"],
            score=rec["score"],
            price=rec.get("current_price", 0),
            rank=i,
            mode="premarket",
            factors=rec.get("factors", {}),
            regime=(_regime_info or {}).get("regime") if _regime_info else None,
        )
        # 同步开 paper_trade 仓位（含ATR动态止损/止盈 + 仓位管理 + 追踪止盈）
        try:
            code = rec["code"]
            # 避免重复建仓：同一股票已有未平仓仓位时跳过
            if db.has_open_position(code):
                continue
            tcfg = rec_configs.get(code, trade_cfg)
            db.open_paper_trade(
                trade_date=today,
                code=code,
                name=rec["name"],
                score=rec["score"],
                entry_price=rec.get("current_price", 0),
                position_pct=rec["factors"].get("position_pct"),
                shares=tcfg.get("shares"),
                trade_config=tcfg,
            )
        except Exception as e:
            logger.warning("开仓失败 %s: %s", code, e)
    for s in scored_stocks:
        db.save_factor_snapshot(
            trade_date=today,
            code=s["code"],
            name=s["name"],
            score=s["score"],
            sub_scores=s.get("sub_scores", {}),
            factors=s.get("factors", {}),
        )
    try:
        for lookback, db_field, param_field in [(1, "d1_return", "d1"), (3, "d3_return", "d3"),
                                                (5, "d5_return", "d5"), (10, "d10_return", "d10"),
                                                (14, "d14_return", "d14")]:
            check_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
            past_recs = db.get_recent_recommendations(days=lookback + 2)
            for past in past_recs:
                if past.get(db_field) is not None:
                    continue  # already updated
                if past["trade_date"] != check_date:
                    continue
                entry = past.get("current_price", 0)
                if entry <= 0:
                    continue
                # 用T+N日的实际收盘价（而不是当日实时价）计算收益
                exit_price = _fetch_close_price_on_date(market_fetcher, past["code"], check_date, lookback)
                if exit_price and exit_price > 0:
                    ret = (exit_price / entry - 1) * 100
                    db.update_performance(
                        check_date, past["code"],
                        **{param_field: round(ret, 2)},
                        mode=past.get("mode") or "premarket",
                    )
    except Exception as e:
        logger.warning("更新历史表现失败: %s", e)
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_report(ctx):
    """盘前研究报告生成与落盘。"""
    (
        db, global_markets, health_note, index_data, news_factor,
        news_list, recommendations, rotation_data, sentiment, trade_cfg,
    ) = ctx.unpack(
        'db', 'global_markets', 'health_note', 'index_data', 'news_factor',
        'news_list', 'recommendations', 'rotation_data', 'sentiment', 'trade_cfg',
    )
    config = ctx["config"]

    reporter = ReportGenerator()
    history = db.get_recent_recommendations(
        days=14, modes=("afternoon", "premarket", "feedback_training"))
    report_content = reporter.generate_report(
        recommendations=recommendations,
        market_overview=index_data,
        global_markets=global_markets,
        news_analysis=news_factor.analyze_news(news_list),
        sentiment=sentiment,
        trade_config=trade_cfg,
        rotation_data=rotation_data,
        history_records=history,
        report_role="research",
        intraday_note=health_note,
    )
    report_path = reporter.save_report(report_content)
    print(f"  ✅ 盘前报告已生成: {report_path}")
    _pack(ctx, dict(locals()))
    ctx["config"] = config


def _pm_feedback(ctx):
    """因子IC权重优化建议 + 反馈循环(归因/IC衰减/自适应调权/ML重训练) + 运行总结。"""
    (
        candidates_df, qualified, recommendations, scored_stocks, t0,
        weights,
    ) = ctx.unpack(
        'candidates_df', 'qualified', 'recommendations', 'scored_stocks', 't0',
        'weights',
    )
    config = ctx["config"]
    _regime_info = ctx.get("_regime_info")

    try:
        from src.engine.ml_selector import FactorOptimizer
        optimizer = FactorOptimizer()
        factor_data = optimizer.load_from_database(lookback_days=60, forward_window=3)
        if not factor_data.empty and "future_return" in factor_data.columns:
            opt_weights = optimizer.optimize_weights(factor_data, return_col="future_return")
            if opt_weights:
                print("\n  📊 权重优化建议（基于因子IC分析）：")
                changed = False
                for name, opt_w in opt_weights.items():
                    cur_w = weights.get(name, 0)
                    if abs(opt_w - cur_w) > 0.02:
                        changed = True
                        print(f"    {name}: {cur_w:.3f} → {opt_w:.4f}")
                if not changed:
                    print("    当前权重与优化结果偏差较小，无需调整")
    except Exception as e:
        logger.warning("权重优化分析失败: %s", e)

    # === 反馈循环：归因 + IC衰减 + 自适应权重 + ML重训练 ===
    try:
        from src.engine.feedback_loop import FeedbackLoop
        fb = FeedbackLoop(config)
        fb_result = fb.run_all(today=datetime.now().strftime("%Y-%m-%d"))
        if fb_result.get("attribution_done") and fb_result.get("n_attributions", 0) > 0:
            print(f"\n  🔄 反馈循环: 归因 {fb_result['n_attributions']} 条")
        if fb_result.get("weight_adjusted"):
            for wc in fb_result.get("weight_changes", []):
                print(f"  ⚖️  调权: {wc['factor']} {wc['old_weight']:.3f}→{wc['new_weight']:.3f} ({wc['reason']})")
        if fb_result.get("model_retrained"):
            m = fb_result.get("model_metrics", {})
            print(f"  🧠 ML重训练: {m.get('model_type','?')} accuracy={m.get('accuracy','?')} "
                  f"samples={m.get('train_samples',0)}+{m.get('test_samples',0)}")
        if fb_result.get("messages"):
            for msg in fb_result["messages"]:
                if "失败" in msg or "错误" in msg:
                    print(f"  ⚠️ {msg}")
    except Exception as e:
        logger.error("反馈循环失败: %s", e)

    write_summary("premarket", len(candidates_df), recommendations,
                   scored=len(scored_stocks), qualified=len(qualified),
                   regime=(_regime_info or {}).get("regime") if _regime_info else None,
                   regime_score=(_regime_info or {}).get("score") if _regime_info else None)

    _elapsed = (datetime.now() - t0).total_seconds()
    print("\n" + "=" * 60)
    print(f"  盘前模式完成! 总耗时: {_elapsed:.0f}s")
    print(f"  候选池: {len(candidates_df)} → 深度计算: {len(scored_stocks)}")
    print(f"  信号筛选: {len(qualified)} → 达标推荐: {len(recommendations)}")
    print("  9:25请执行: python run_daily.py --intraday")
    print("=" * 60)
    _pack(ctx, dict(locals()))
    ctx["config"] = config


def _pm_persist(ctx):
    """持久化编排器: 分散 → 缓存 → 仓位计划 → 落库开仓 → 报告。

    注意各阶段严格单次执行: _update_open_paper_trades/save_recommendation
    等操作非幂等, 禁止重复调用(2026-08 修复双跑缺陷)。
    """
    _pm_diversify(ctx)
    _pm_cache_candidates(ctx)
    _pm_position_plan(ctx)
    _pm_save_and_open(ctx)
    _pm_report(ctx)
