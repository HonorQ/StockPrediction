# -*- coding: utf-8 -*-
"""盘前模式(07:00): 全量因子计算与候选池产出

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import pandas as pd
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger

logger = get_logger()

def _pack(ctx, local_vars):
    """阶段结束时把局部状态写回 ctx(排除私有/参数名)。"""
    for k, v in local_vars.items():
        if not k.startswith('_') and k not in ('ctx', 'config'):
            ctx[k] = v



from src.utils.heartbeat import check_task_alerts
from src.factors.moat import MoatFactors
from src.engine.entry_risk_filter import _filter_high_risk_entries
from src.engine.paper_trade_ops import _update_open_paper_trades
from src.engine.paper_trade_ops import _fetch_close_price_on_date

from src.data.market_data import MarketDataFetcher
from src.data.money_flow import MoneyFlowFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.hot_stocks import HotStocksFetcher
from src.data.concept_map import ConceptMap
from src.factors.technical import TechnicalFactors
from src.factors.money_flow_factor import MoneyFlowFactors
from src.factors.sentiment import SentimentFactors
from src.factors.news_factor import NewsFactor
from src.factors.strategies import StockStrategies
from src.factors.chip import ChipDistribution
from src.factors.fundamental import FundamentalFactors
from src.factors.popularity import PopularityFactors
from src.factors.institutional import InstitutionalFactors
from src.factors.relative_strength import RelativeStrength
from src.factors.sector_rotation import SectorRotation, OversoldReboundStrategy
from src.factors.rebound_analysis import ReboundSustainability, NextDayPredictor
from src.factors.expectation_pricing import ExpectationPricing, NewsEventAnalyzer, ProfitTakingDetector
from src.factors.ashare_specific import AShareCharacteristics, SentimentCycle, HotMoneyTracker
from src.factors.ashare_extended import MarginTradingFactor, ShareholderChangeFactor, LockUpExpiryFactor, SeasonalityFactor, LimitUpQualityFactor
from src.factors.phase2_market_intel import SectorLinkage, MoneyFlowContinuity, InstitutionalResearch, DragonTigerBoard, SectorLeader
from src.factors.quant_behavior import QuantBehaviorDetector
from src.factors.deep_research import DeepResearchFactor
from src.factors.policy_event import PolicyEventFactor
from src.factors.macro_event import MacroEventDetector
from src.factors.capital_behavior import CapitalBehaviorFactor
from src.factors.national_team import NationalTeamFactor
from src.factors.holder_concentration import HolderConcentrationFactor
from src.factors.circular_holder import CirculateHolderFactor
from src.engine.screener import StockScreener
from src.engine.scorer import StockScorer
from src.engine.reporter import ReportGenerator
from src.engine.backtest import ATRStopLoss, PositionManager
from src.data.database import StockDatabase
from src.pipeline.common import (  # noqa: F401
    _DualOutput, setup_logging, write_summary, _prefetch_history_and_flow,
    save_cache, load_cache, _log_source_health, _cleanup_cache,
)


def _pm_init(ctx):
    (
        _alerts, _t0, a, ashare_factors, atr_stop,
        capital_behavior_factor, chip_calc, circulate_factor, db, deep_research_factor,
        dragon_tiger, expectation_pricing, filter_cfg, fundamental_factor, global_market_cfg,
        health_note, holder_factor, hot_fetcher, hot_money_tracker, institutional_factor,
        institutional_research, k, keywords_cfg, latest, limit_quality_factor,
        lockup_factor, macro_detector, margin_factor, market_fetcher, money_factor,
        money_fetcher, money_flow_continuity, national_team_factor, news_event_analyzer, news_factor,
        news_fetcher, next_day_predictor, oversold_strategy, policy_event_factor, popularity_factor,
        position_manager, profit_taking_detector, quant_detector, rebound_sustainability, relative_strength,
        scorer, screener, sector_leader, sector_linkage, sector_rotation,
        sentiment_cycle, sentiment_factor, shareholder_factor, tech_cfg, trade_cfg,
        v, weights,
    ) = ctx.unpack(
        '_alerts', '_t0', 'a', 'ashare_factors', 'atr_stop',
        'capital_behavior_factor', 'chip_calc', 'circulate_factor', 'db', 'deep_research_factor',
        'dragon_tiger', 'expectation_pricing', 'filter_cfg', 'fundamental_factor', 'global_market_cfg',
        'health_note', 'holder_factor', 'hot_fetcher', 'hot_money_tracker', 'institutional_factor',
        'institutional_research', 'k', 'keywords_cfg', 'latest', 'limit_quality_factor',
        'lockup_factor', 'macro_detector', 'margin_factor', 'market_fetcher', 'money_factor',
        'money_fetcher', 'money_flow_continuity', 'national_team_factor', 'news_event_analyzer', 'news_factor',
        'news_fetcher', 'next_day_predictor', 'oversold_strategy', 'policy_event_factor', 'popularity_factor',
        'position_manager', 'profit_taking_detector', 'quant_detector', 'rebound_sustainability', 'relative_strength',
        'scorer', 'screener', 'sector_leader', 'sector_linkage', 'sector_rotation',
        'sentiment_cycle', 'sentiment_factor', 'shareholder_factor', 'tech_cfg', 'trade_cfg',
        'v', 'weights',
    )
    config = ctx["config"]

    print("=" * 60)
    print("  盘前模式 — 全量因子计算")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    _t0 = datetime.now()

    # 失败告警：检测历史运行异常(上次失败/长期静默)，注入报告横幅
    health_note = None
    _alerts = check_task_alerts()
    if _alerts:
        for a in _alerts:
            print(f"  🚨 运维告警: {a}")
        health_note = "🚨 " + "; ".join(_alerts)

    _log_source_health()
    _cleanup_cache(days=7)

    weights = config.get("weights", {})
    filter_cfg = config.get("filter", {})
    tech_cfg = config.get("technical", {})
    trade_cfg = config.get("trade", {})
    keywords_cfg = config.get("keywords", {})
    global_market_cfg = config.get("global_markets", [])

    # 初始化组件
    market_fetcher = MarketDataFetcher()
    money_fetcher = MoneyFlowFetcher()
    news_fetcher = NewsFetcher()
    hot_fetcher = None
    money_factor = MoneyFlowFactors()
    sentiment_factor = SentimentFactors()
    news_factor = NewsFactor(keywords_cfg)
    chip_calc = ChipDistribution()
    fundamental_factor = FundamentalFactors()
    popularity_factor = PopularityFactors()
    institutional_factor = InstitutionalFactors()
    relative_strength = RelativeStrength()
    sector_rotation = SectorRotation()
    oversold_strategy = OversoldReboundStrategy()
    rebound_sustainability = ReboundSustainability()
    next_day_predictor = NextDayPredictor()
    expectation_pricing = ExpectationPricing()
    news_event_analyzer = NewsEventAnalyzer()
    profit_taking_detector = ProfitTakingDetector()
    ashare_factors = AShareCharacteristics()
    sentiment_cycle = SentimentCycle()
    hot_money_tracker = HotMoneyTracker()
    quant_detector = QuantBehaviorDetector()
    margin_factor = MarginTradingFactor()
    shareholder_factor = ShareholderChangeFactor()
    lockup_factor = LockUpExpiryFactor()
    SeasonalityFactor()
    limit_quality_factor = LimitUpQualityFactor()
    sector_linkage = SectorLinkage()
    money_flow_continuity = MoneyFlowContinuity(money_fetcher=money_fetcher)
    institutional_research = InstitutionalResearch()
    dragon_tiger = DragonTigerBoard()
    holder_factor = HolderConcentrationFactor()
    circulate_factor = CirculateHolderFactor()
    deep_research_factor = DeepResearchFactor()
    sector_leader = SectorLeader()
    policy_event_factor = PolicyEventFactor()
    macro_detector = MacroEventDetector()
    capital_behavior_factor = CapitalBehaviorFactor()
    national_team_factor = NationalTeamFactor()
    atr_stop = ATRStopLoss(atr_period=14, multiplier=2.0)
    position_manager = PositionManager(max_position_pct=0.25, min_position_pct=0.05)
    screener = StockScreener(config.get("stock", {}))
    # 规则 §6：config.yaml 是静态先验，反馈循环的调权结果落库。
    # scorer 实例化时从 weight_history 表读最新调权结果，叠加到 config 静态权重上。
    # 这样 config.yaml 不被覆盖写回，同时调权结果对次日生效。
    try:
        db = StockDatabase()
        latest = db.get_latest_weights()
        if latest:
            for k, v in latest.items():
                if k in weights:
                    weights[k] = v
            print(f"  [权重合并] 从 weight_history 加载 {len(latest)} 个因子最新调权结果")
    except Exception as e:
        print(f"  [权重合并] 跳过: {e}")
    scorer = StockScorer(weights)
    _pack(ctx, dict(locals()))
    ctx["config"] = config


def _pm_mkt(ctx):
    (
        a_stock_df, change, change_col, chg_60, code,
        code_col, col_60, concept_map, concept_negative, concept_neutral,
        concept_positive, concept_result, event, filtered_positive, global_market_cfg,
        global_markets, hot_fetcher, icon, index_data, macro_detector,
        macro_events, market_fetcher, news_analysis, news_factor, news_fetcher,
        news_list, overbought_count, row, turnover, turnover_col,
    ) = ctx.unpack(
        'a_stock_df', 'change', 'change_col', 'chg_60', 'code',
        'code_col', 'col_60', 'concept_map', 'concept_negative', 'concept_neutral',
        'concept_positive', 'concept_result', 'event', 'filtered_positive', 'global_market_cfg',
        'global_markets', 'hot_fetcher', 'icon', 'index_data', 'macro_detector',
        'macro_events', 'market_fetcher', 'news_analysis', 'news_factor', 'news_fetcher',
        'news_list', 'overbought_count', 'row', 'turnover', 'turnover_col',
    )
    config = ctx["config"]

    print("\n[1/6] 获取A股实时行情...")
    a_stock_df = market_fetcher.get_a_share_realtime()
    if a_stock_df.empty:
        print("  ❌ 无法获取A股行情，退出")
        return

    hot_fetcher = HotStocksFetcher(market_df=a_stock_df)

    # 2. 外盘
    print("\n[2/6] 获取外盘行情...")
    global_markets = market_fetcher.get_global_market_summary(global_market_cfg)

    # 3. 指数
    print("\n[3/6] 获取指数行情...")
    index_data = market_fetcher.get_index_realtime()

    # 4. 新闻（多源合并）
    print("\n[4/6] 获取新闻和消息面...")
    news_list = news_fetcher.get_all_news(limit_per_source=50)
    news_analysis = news_factor.analyze_news(news_list)

    # 4.0 宏观事件检测（全局检测一次，所有股票共用）
    print("\n[4.0] 检测宏观事件...")
    macro_events = macro_detector.detect_all()
    if macro_events:
        print(f"  📊 检测到 {len(macro_events)} 个宏观事件:")
        for event in macro_events[:5]:
            icon = "📈" if event["direction"] == "positive" else "📉"
            print(f"    {icon} {event['title']}")
    else:
        print("  📊 无重大宏观事件")

    # 4.1 概念映射（信息差检测：从新闻中提前发现潜在热点股）
    print("\n[4.1] 概念映射匹配（含情感分析）...")
    concept_map = ConceptMap(auto_sync=True)  # 自动同步 akshare 概念成分股
    concept_map.build_stock_index(a_stock_df)
    concept_result = concept_map.match_news_to_stocks(news_list, news_factor)
    concept_positive = concept_result.get("positive", [])
    concept_negative = concept_result.get("negative", [])
    concept_neutral = concept_result.get("neutral", [])
    if concept_positive:
        print(f"  📈 利好匹配 {len(concept_positive)} 只 → 检查价格位置（用实时数据）...")
        filtered_positive = []
        overbought_count = 0
        code_col = "代码"
        change_col = "涨跌幅" if "涨跌幅" in a_stock_df.columns else "pctChg"
        for code in concept_positive:
            row = a_stock_df[a_stock_df[code_col].astype(str) == code]
            if row.empty:
                filtered_positive.append(code)
                continue
            row = row.iloc[0]
            change = row.get(change_col, 0) or 0
            change = float(change) if change else 0

            # 今日涨幅过大 → 已price in
            if abs(change) > 5:
                overbought_count += 1
                continue

            # 换手率极低+涨幅大 → 可能无量拉升
            turnover_col = "换手率" if "换手率" in a_stock_df.columns else "turnover"
            turnover = float(row.get(turnover_col, 0) or 0) if turnover_col in a_stock_df.columns else 0
            if abs(change) > 5 and turnover < 0.5:
                overbought_count += 1
                continue

            # 60日涨幅过大 → 长期已透支
            col_60 = "60日涨跌幅" if "60日涨跌幅" in a_stock_df.columns else ""
            if col_60 and col_60 in a_stock_df.columns:
                chg_60 = float(row.get(col_60, 0) or 0)
                if chg_60 > 50:
                    overbought_count += 1
                    continue

            filtered_positive.append(code)
        concept_positive = filtered_positive
        if overbought_count > 0:
            print(f"    📊 已过滤 {overbought_count} 只（今日涨幅过大/无量拉升/60日涨幅透支）")
        print(f"    ✅ 最终利好入热股池: {len(concept_positive)} 只")
    if concept_negative:
        print(f"  📉 利空匹配 {len(concept_negative)} 只 → 不加入（可能需回避）")
    if concept_neutral:
        print(f"  📊 中性匹配 {len(concept_neutral)} 只 → 不处理")
    _pack(ctx, dict(locals()))
    ctx["config"] = config


def _pm_sent(ctx):
    (
        _regime_info, concept_positive, global_markets, hot_codes, hot_df,
        hot_fetcher, index_data, k, market_fetcher, news_list,
        northbound_data, r, regime, regime_filter, regime_result,
        regime_weights, rotation_data, scorer, sector_rotation, sentiment,
        si_cfg, switched, v, weights,
    ) = ctx.unpack(
        '_regime_info', 'concept_positive', 'global_markets', 'hot_codes', 'hot_df',
        'hot_fetcher', 'index_data', 'k', 'market_fetcher', 'news_list',
        'northbound_data', 'r', 'regime', 'regime_filter', 'regime_result',
        'regime_weights', 'rotation_data', 'scorer', 'sector_rotation', 'sentiment',
        'si_cfg', 'switched', 'v', 'weights',
    )
    config = ctx["config"]

    print("\n[5/6] 获取热点和市场情绪...")
    hot_df = hot_fetcher.get_hot_rank()
    hot_codes = set(concept_positive)
    if not hot_df.empty:
        hot_codes.update(hot_df["股票代码"].astype(str).tolist())
    sentiment = hot_fetcher.calculate_market_sentiment()

    # 5.1 板块轮动
    print("\n[5.1] 分析板块轮动...")
    rotation_data = sector_rotation.analyze_sector_rotation(hot_df, news_list)

    # 5.2 大盘择时预评估 + regime 权重切换
    # 必须在 batch_normalize 之前完成：bear/crash 切防守权重、bull 切进攻权重，
    # 使本轮 Z-Score 截面评分直接使用切换后的权重。
    # (此前评估在评分之后才执行，config.yaml 的 regime.weights 从未生效)
    regime_filter = None
    regime_result = None
    _regime_info = None
    regime = "neutral"
    try:
        from src.engine.market_regime import MarketRegimeFilter
        regime_filter = MarketRegimeFilter(market_fetcher=market_fetcher)
        # 北向资金维度此前从未接入(一直传 None)，现从数据层获取；
        # 接口失败时返回中性默认值(net_inflow=0)，不影响择时其余维度
        try:
            northbound_data = market_fetcher.get_north_flow()
        except Exception as nb_err:
            logger.warning("获取北向资金失败，择时忽略北向维度: %s", nb_err)
            northbound_data = None
        regime_result = regime_filter.assess_regime(
            market_overview=index_data,
            sentiment=sentiment,
            northbound_data=northbound_data,
            global_markets=global_markets,
        )
        _regime_info = regime_result
        regime = regime_result["regime"]
        print(f"\n[5.2] 大盘择时预评估: {regime} | 得分 {regime_result['score']}/100")
        for r in regime_result["reasons"][:3]:
            print(f"      • {r}")
        # 在动态调权结果(weight_history 合并后)之上做 regime 部分覆盖，
        # 仅覆盖 24 维评分体系中存在的因子(无效维度由 get_regime_weights 跳过并告警)
        regime_weights = regime_filter.get_regime_weights(regime, weights)
        switched = {
            k: v for k, v in regime_weights.items()
            if abs(v - float(weights.get(k, 0))) > 1e-9
        }
        weights = regime_weights
        if switched:
            print("  ⚖️ 已应用 %s 因子权重: %s" % (
                regime, ", ".join(f"{k}={v}" for k, v in sorted(switched.items()))))
    except Exception as e:
        logger.warning("大盘择时预评估失败，沿用默认权重: %s", e)

    # === regime 条件化信号反转（A/B 实验落地）===
    # IC校准发现 trend/strategy/quant_behavior 负贡献；跨窗口回测证实：
    # neutral/bull 启用反转超额显著改善(neutral d10 +1.87 vs -4.12)，
    # bear 反转更差(-3.21)故不启用——bear 靠 empty_on_bear 清仓防守。
    # 三窗口样本有限，先按 config.signal_invert 配置运行并持续复核。
    try:
        si_cfg = config.get("signal_invert", {}) or {}
        if si_cfg.get("enabled") and regime in (si_cfg.get("regimes") or []):
            scorer.invert_signals = True
            print(f"  🔁 已启用负IC维度反转打分 (regime={regime}, "
                  f"维度: {', '.join(scorer.NEGATIVE_IC_DIMS)})")
    except Exception as e:
        logger.warning("信号反转配置失败(不影响主流程): %s", e)
    _pack(ctx, dict(locals()))
    ctx["config"] = config


def _pm_screen(ctx):
    """硬性条件筛选: ST/停牌/板块排除"""
    (
        QS_CHANGE_CAP, QS_CHANGE_W, QS_HOT, QS_MV_SWEET, QS_NEWS_CAP,
        QS_TURNOVER2, QS_TURNOVER5, QS_VOLUME_RATIO, _, a_stock_df,
        all_factors, ashare, ashare_factors, atr_series, atr_stop,
        before, c, cand, candidates_df, cap_behavior,
        capital_behavior_factor, change_pct, chip, chip_calc, circulate,
        circulate_factor, code_k, current_atr, current_price, db,
        deep_research_factor, dr, dragon_tiger, elapsed_prefetch, event_pricing,
        exclude_boards, expectation, expectation_penalty, expectation_pricing, ff,
        filter_cfg, flow_cache, flow_continuity, flow_data, fundamental_factor,
        h_k, history, history_cache, holder, holder_factor,
        hot_codes, hot_df, hot_money, hot_money_tracker, idx,
        in_hot, inf, institutional_factor, institutional_research, leader,
        lh, limit_quality, limit_quality_factor, linkage, lockup,
        lockup_factor, margin, margin_factor, market_fetcher, market_history,
        mf, money_factor, money_fetcher, money_flow_continuity, mv_yi,
        n, national_team, national_team_factor, news_event_analyzer, news_factor,
        news_list, next_day, next_day_predictor, ns, ok_count,
        oversold_strategy, pf, phase2_start, policy_event_factor, policy_result,
        popularity_data, popularity_factor, pos_info, position_manager, prefetch_codes,
        prefetch_start, profit_taking, profit_taking_detector, qs, qs_cfg,
        quant_detector, quant_result, quick_scored, rebound_result, rebound_sustainability,
        regime, relative_strength, research, rotation_data, row,
        rs, saved_rows, saved_stocks, scored_stocks, scorer,
        screener, sector_leader, sector_linkage, sector_rotation, sentiment,
        sentiment_analysis, sentiment_cycle, sentiment_factor, sf, shareholder,
        shareholder_factor, stock_cfg, stock_code, stock_data, stock_name,
        stock_news_score, stop_info, strategies, sub_scores, sustainability,
        take_info, tech_cfg, tech_factors, top_candidates, top_n,
        total_mv, trade_cfg, turnover, v, volume_ratio,
    ) = ctx.unpack(
        'QS_CHANGE_CAP', 'QS_CHANGE_W', 'QS_HOT', 'QS_MV_SWEET', 'QS_NEWS_CAP',
        'QS_TURNOVER2', 'QS_TURNOVER5', 'QS_VOLUME_RATIO', '_', 'a_stock_df',
        'all_factors', 'ashare', 'ashare_factors', 'atr_series', 'atr_stop',
        'before', 'c', 'cand', 'candidates_df', 'cap_behavior',
        'capital_behavior_factor', 'change_pct', 'chip', 'chip_calc', 'circulate',
        'circulate_factor', 'code_k', 'current_atr', 'current_price', 'db',
        'deep_research_factor', 'dr', 'dragon_tiger', 'elapsed_prefetch', 'event_pricing',
        'exclude_boards', 'expectation', 'expectation_penalty', 'expectation_pricing', 'ff',
        'filter_cfg', 'flow_cache', 'flow_continuity', 'flow_data', 'fundamental_factor',
        'h_k', 'history', 'history_cache', 'holder', 'holder_factor',
        'hot_codes', 'hot_df', 'hot_money', 'hot_money_tracker', 'idx',
        'in_hot', 'inf', 'institutional_factor', 'institutional_research', 'leader',
        'lh', 'limit_quality', 'limit_quality_factor', 'linkage', 'lockup',
        'lockup_factor', 'margin', 'margin_factor', 'market_fetcher', 'market_history',
        'mf', 'money_factor', 'money_fetcher', 'money_flow_continuity', 'mv_yi',
        'n', 'national_team', 'national_team_factor', 'news_event_analyzer', 'news_factor',
        'news_list', 'next_day', 'next_day_predictor', 'ns', 'ok_count',
        'oversold_strategy', 'pf', 'phase2_start', 'policy_event_factor', 'policy_result',
        'popularity_data', 'popularity_factor', 'pos_info', 'position_manager', 'prefetch_codes',
        'prefetch_start', 'profit_taking', 'profit_taking_detector', 'qs', 'qs_cfg',
        'quant_detector', 'quant_result', 'quick_scored', 'rebound_result', 'rebound_sustainability',
        'regime', 'relative_strength', 'research', 'rotation_data', 'row',
        'rs', 'saved_rows', 'saved_stocks', 'scored_stocks', 'scorer',
        'screener', 'sector_leader', 'sector_linkage', 'sector_rotation', 'sentiment',
        'sentiment_analysis', 'sentiment_cycle', 'sentiment_factor', 'sf', 'shareholder',
        'shareholder_factor', 'stock_cfg', 'stock_code', 'stock_data', 'stock_name',
        'stock_news_score', 'stop_info', 'strategies', 'sub_scores', 'sustainability',
        'take_info', 'tech_cfg', 'tech_factors', 'top_candidates', 'top_n',
        'total_mv', 'trade_cfg', 'turnover', 'v', 'volume_ratio',
    )
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
    """实时快评 → TopN候选(regime动态) + 并行K线预取"""
    (
        QS_CHANGE_CAP, QS_CHANGE_W, QS_HOT, QS_MV_SWEET, QS_NEWS_CAP,
        QS_TURNOVER2, QS_TURNOVER5, QS_VOLUME_RATIO, _, a_stock_df,
        all_factors, ashare, ashare_factors, atr_series, atr_stop,
        before, c, cand, candidates_df, cap_behavior,
        capital_behavior_factor, change_pct, chip, chip_calc, circulate,
        circulate_factor, code_k, current_atr, current_price, db,
        deep_research_factor, dr, dragon_tiger, elapsed_prefetch, event_pricing,
        exclude_boards, expectation, expectation_penalty, expectation_pricing, ff,
        filter_cfg, flow_cache, flow_continuity, flow_data, fundamental_factor,
        h_k, history, history_cache, holder, holder_factor,
        hot_codes, hot_df, hot_money, hot_money_tracker, idx,
        in_hot, inf, institutional_factor, institutional_research, leader,
        lh, limit_quality, limit_quality_factor, linkage, lockup,
        lockup_factor, margin, margin_factor, market_fetcher, market_history,
        mf, money_factor, money_fetcher, money_flow_continuity, mv_yi,
        n, national_team, national_team_factor, news_event_analyzer, news_factor,
        news_list, next_day, next_day_predictor, ns, ok_count,
        oversold_strategy, pf, phase2_start, policy_event_factor, policy_result,
        popularity_data, popularity_factor, pos_info, position_manager, prefetch_codes,
        prefetch_start, profit_taking, profit_taking_detector, qs, qs_cfg,
        quant_detector, quant_result, quick_scored, rebound_result, rebound_sustainability,
        regime, relative_strength, research, rotation_data, row,
        rs, saved_rows, saved_stocks, scored_stocks, scorer,
        screener, sector_leader, sector_linkage, sector_rotation, sentiment,
        sentiment_analysis, sentiment_cycle, sentiment_factor, sf, shareholder,
        shareholder_factor, stock_cfg, stock_code, stock_data, stock_name,
        stock_news_score, stop_info, strategies, sub_scores, sustainability,
        take_info, tech_cfg, tech_factors, top_candidates, top_n,
        total_mv, trade_cfg, turnover, v, volume_ratio,
    ) = ctx.unpack(
        'QS_CHANGE_CAP', 'QS_CHANGE_W', 'QS_HOT', 'QS_MV_SWEET', 'QS_NEWS_CAP',
        'QS_TURNOVER2', 'QS_TURNOVER5', 'QS_VOLUME_RATIO', '_', 'a_stock_df',
        'all_factors', 'ashare', 'ashare_factors', 'atr_series', 'atr_stop',
        'before', 'c', 'cand', 'candidates_df', 'cap_behavior',
        'capital_behavior_factor', 'change_pct', 'chip', 'chip_calc', 'circulate',
        'circulate_factor', 'code_k', 'current_atr', 'current_price', 'db',
        'deep_research_factor', 'dr', 'dragon_tiger', 'elapsed_prefetch', 'event_pricing',
        'exclude_boards', 'expectation', 'expectation_penalty', 'expectation_pricing', 'ff',
        'filter_cfg', 'flow_cache', 'flow_continuity', 'flow_data', 'fundamental_factor',
        'h_k', 'history', 'history_cache', 'holder', 'holder_factor',
        'hot_codes', 'hot_df', 'hot_money', 'hot_money_tracker', 'idx',
        'in_hot', 'inf', 'institutional_factor', 'institutional_research', 'leader',
        'lh', 'limit_quality', 'limit_quality_factor', 'linkage', 'lockup',
        'lockup_factor', 'margin', 'margin_factor', 'market_fetcher', 'market_history',
        'mf', 'money_factor', 'money_fetcher', 'money_flow_continuity', 'mv_yi',
        'n', 'national_team', 'national_team_factor', 'news_event_analyzer', 'news_factor',
        'news_list', 'next_day', 'next_day_predictor', 'ns', 'ok_count',
        'oversold_strategy', 'pf', 'phase2_start', 'policy_event_factor', 'policy_result',
        'popularity_data', 'popularity_factor', 'pos_info', 'position_manager', 'prefetch_codes',
        'prefetch_start', 'profit_taking', 'profit_taking_detector', 'qs', 'qs_cfg',
        'quant_detector', 'quant_result', 'quick_scored', 'rebound_result', 'rebound_sustainability',
        'regime', 'relative_strength', 'research', 'rotation_data', 'row',
        'rs', 'saved_rows', 'saved_stocks', 'scored_stocks', 'scorer',
        'screener', 'sector_leader', 'sector_linkage', 'sector_rotation', 'sentiment',
        'sentiment_analysis', 'sentiment_cycle', 'sentiment_factor', 'sf', 'shareholder',
        'shareholder_factor', 'stock_cfg', 'stock_code', 'stock_data', 'stock_name',
        'stock_news_score', 'stop_info', 'strategies', 'sub_scores', 'sustainability',
        'take_info', 'tech_cfg', 'tech_factors', 'top_candidates', 'top_n',
        'total_mv', 'trade_cfg', 'turnover', 'v', 'volume_ratio',
    )
    config = ctx["config"]

    print("  --- 阶段1: 实时行情快速预评分 ---")
    qs_cfg = filter_cfg.get("quick_score", {}) or {}
    QS_CHANGE_W = float(qs_cfg.get("change_weight", 3))
    QS_CHANGE_CAP = float(qs_cfg.get("change_cap", 20))
    QS_TURNOVER2 = float(qs_cfg.get("turnover2_bonus", 10))
    QS_TURNOVER5 = float(qs_cfg.get("turnover5_bonus", 5))
    QS_VOLUME_RATIO = float(qs_cfg.get("volume_ratio_bonus", 10))
    QS_HOT = float(qs_cfg.get("hot_bonus", 15))
    QS_MV_SWEET = float(qs_cfg.get("mv_sweet_bonus", 5))
    QS_NEWS_CAP = float(qs_cfg.get("news_bonus_cap", 10))

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

    组件与上下文经 ctx 注入; 结果 append 到 scored_stocks。
    """
    (ashare_factors, atr_stop, capital_behavior_factor, chip_calc, circulate_factor, deep_research_factor, dragon_tiger, expectation_pricing, flow_cache, fundamental_factor, history_cache, holder_factor, hot_codes, hot_df, hot_money_tracker, institutional_factor, institutional_research, limit_quality_factor, lockup_factor, margin_factor, market_fetcher, market_history, money_factor, money_fetcher, money_flow_continuity, national_team_factor, news_event_analyzer, news_factor, news_list, next_day_predictor, oversold_strategy, phase2_start, policy_event_factor, popularity_factor, position_manager, profit_taking_detector, quant_detector, rebound_sustainability, relative_strength, rotation_data, scored_stocks, scorer, sector_leader, sector_linkage, sector_rotation, sentiment, sentiment_cycle, sentiment_factor, shareholder_factor, tech_cfg, top_candidates, trade_cfg) = ctx.unpack(
        'ashare_factors', 'atr_stop', 'capital_behavior_factor', 'chip_calc'
        'circulate_factor', 'deep_research_factor', 'dragon_tiger', 'expectation_pricing'
        'flow_cache', 'fundamental_factor', 'history_cache', 'holder_factor'
        'hot_codes', 'hot_df', 'hot_money_tracker', 'institutional_factor'
        'institutional_research', 'limit_quality_factor', 'lockup_factor', 'margin_factor'
        'market_fetcher', 'market_history', 'money_factor', 'money_fetcher'
        'money_flow_continuity', 'national_team_factor', 'news_event_analyzer', 'news_factor'
        'news_list', 'next_day_predictor', 'oversold_strategy', 'phase2_start'
        'policy_event_factor', 'popularity_factor', 'position_manager', 'profit_taking_detector'
        'quant_detector', 'rebound_sustainability', 'relative_strength', 'rotation_data'
        'scored_stocks', 'scorer', 'sector_leader', 'sector_linkage'
        'sector_rotation', 'sentiment', 'sentiment_cycle', 'sentiment_factor'
        'shareholder_factor', 'tech_cfg', 'top_candidates', 'trade_cfg'
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
    if history.empty or len(history) < 20:
        print("      ⚠️ 历史数据不足，跳过")
        return "skip"

    tech_factors = TechnicalFactors(history, tech_cfg)
    all_factors = tech_factors.calculate_all()

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

    stock_data = row.to_dict()
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
    except Exception:
        pass

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
    """逐票深度因子计算 + 截面Z-Score归一化"""
    (
        QS_CHANGE_CAP, QS_CHANGE_W, QS_HOT, QS_MV_SWEET, QS_NEWS_CAP,
        QS_TURNOVER2, QS_TURNOVER5, QS_VOLUME_RATIO, _, a_stock_df,
        all_factors, ashare, ashare_factors, atr_series, atr_stop,
        before, c, cand, candidates_df, cap_behavior,
        capital_behavior_factor, change_pct, chip, chip_calc, circulate,
        circulate_factor, code_k, current_atr, current_price, db,
        deep_research_factor, dr, dragon_tiger, elapsed_prefetch, event_pricing,
        exclude_boards, expectation, expectation_penalty, expectation_pricing, ff,
        filter_cfg, flow_cache, flow_continuity, flow_data, fundamental_factor,
        h_k, history, history_cache, holder, holder_factor,
        hot_codes, hot_df, hot_money, hot_money_tracker, idx,
        in_hot, inf, institutional_factor, institutional_research, leader,
        lh, limit_quality, limit_quality_factor, linkage, lockup,
        lockup_factor, margin, margin_factor, market_fetcher, market_history,
        mf, money_factor, money_fetcher, money_flow_continuity, mv_yi,
        n, national_team, national_team_factor, news_event_analyzer, news_factor,
        news_list, next_day, next_day_predictor, ns, ok_count,
        oversold_strategy, pf, phase2_start, policy_event_factor, policy_result,
        popularity_data, popularity_factor, pos_info, position_manager, prefetch_codes,
        prefetch_start, profit_taking, profit_taking_detector, qs, qs_cfg,
        quant_detector, quant_result, quick_scored, rebound_result, rebound_sustainability,
        regime, relative_strength, research, rotation_data, row,
        rs, saved_rows, saved_stocks, scored_stocks, scorer,
        screener, sector_leader, sector_linkage, sector_rotation, sentiment,
        sentiment_analysis, sentiment_cycle, sentiment_factor, sf, shareholder,
        shareholder_factor, stock_cfg, stock_code, stock_data, stock_name,
        stock_news_score, stop_info, strategies, sub_scores, sustainability,
        take_info, tech_cfg, tech_factors, top_candidates, top_n,
        total_mv, trade_cfg, turnover, v, volume_ratio,
    ) = ctx.unpack(
        'QS_CHANGE_CAP', 'QS_CHANGE_W', 'QS_HOT', 'QS_MV_SWEET', 'QS_NEWS_CAP',
        'QS_TURNOVER2', 'QS_TURNOVER5', 'QS_VOLUME_RATIO', '_', 'a_stock_df',
        'all_factors', 'ashare', 'ashare_factors', 'atr_series', 'atr_stop',
        'before', 'c', 'cand', 'candidates_df', 'cap_behavior',
        'capital_behavior_factor', 'change_pct', 'chip', 'chip_calc', 'circulate',
        'circulate_factor', 'code_k', 'current_atr', 'current_price', 'db',
        'deep_research_factor', 'dr', 'dragon_tiger', 'elapsed_prefetch', 'event_pricing',
        'exclude_boards', 'expectation', 'expectation_penalty', 'expectation_pricing', 'ff',
        'filter_cfg', 'flow_cache', 'flow_continuity', 'flow_data', 'fundamental_factor',
        'h_k', 'history', 'history_cache', 'holder', 'holder_factor',
        'hot_codes', 'hot_df', 'hot_money', 'hot_money_tracker', 'idx',
        'in_hot', 'inf', 'institutional_factor', 'institutional_research', 'leader',
        'lh', 'limit_quality', 'limit_quality_factor', 'linkage', 'lockup',
        'lockup_factor', 'margin', 'margin_factor', 'market_fetcher', 'market_history',
        'mf', 'money_factor', 'money_fetcher', 'money_flow_continuity', 'mv_yi',
        'n', 'national_team', 'national_team_factor', 'news_event_analyzer', 'news_factor',
        'news_list', 'next_day', 'next_day_predictor', 'ns', 'ok_count',
        'oversold_strategy', 'pf', 'phase2_start', 'policy_event_factor', 'policy_result',
        'popularity_data', 'popularity_factor', 'pos_info', 'position_manager', 'prefetch_codes',
        'prefetch_start', 'profit_taking', 'profit_taking_detector', 'qs', 'qs_cfg',
        'quant_detector', 'quant_result', 'quick_scored', 'rebound_result', 'rebound_sustainability',
        'regime', 'relative_strength', 'research', 'rotation_data', 'row',
        'rs', 'saved_rows', 'saved_stocks', 'scored_stocks', 'scorer',
        'screener', 'sector_leader', 'sector_linkage', 'sector_rotation', 'sentiment',
        'sentiment_analysis', 'sentiment_cycle', 'sentiment_factor', 'sf', 'shareholder',
        'shareholder_factor', 'stock_cfg', 'stock_code', 'stock_data', 'stock_name',
        'stock_news_score', 'stop_info', 'strategies', 'sub_scores', 'sustainability',
        'take_info', 'tech_cfg', 'tech_factors', 'top_candidates', 'top_n',
        'total_mv', 'trade_cfg', 'turnover', 'v', 'volume_ratio',
    )
    config = ctx["config"]

    print("  --- 阶段2: 深度因子计算 ---")
    scored_stocks = []
    phase2_start = datetime.now()

    # 并行预取历史K线 + 资金流（网络I/O并行，CPU计算保持串行）
    prefetch_codes = [c["code"] for c in top_candidates]
    print(f"  🔄 并行预取 {len(prefetch_codes)} 只股票的历史K线和资金流...")
    prefetch_start = datetime.now()
    history_cache, flow_cache = _prefetch_history_and_flow(
        market_fetcher, money_fetcher, prefetch_codes, days=120
    )
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
            except Exception:
                pass
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
    ctx["config"] = config


    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_guard(ctx):
    (
        _health_today, before_health, db, dynamic_pairs, filter_cfg,
        final, final_max_rec, health, health_note, max_rec,
        qualified, recommendations, regime, regime_filter, regime_max,
        regime_result, scored_stocks, screener, warn_txt, weights,
    ) = ctx.unpack(
        '_health_today', 'before_health', 'db', 'dynamic_pairs', 'filter_cfg',
        'final', 'final_max_rec', 'health', 'health_note', 'max_rec',
        'qualified', 'recommendations', 'regime', 'regime_filter', 'regime_max',
        'regime_result', 'scored_stocks', 'screener', 'warn_txt', 'weights',
    )
    config = ctx["config"]

    print("  --- 截面Z-Score归一化(替代固定50分基线) ---")
    # 动态正交化对由 feedback_loop IC 分析持久化，跨进程生效
    from src.engine.feedback_loop import FeedbackLoop
    dynamic_pairs = FeedbackLoop._load_dynamic_ortho_pairs()
    if dynamic_pairs:
        print(f"  (动态正交化 {len(dynamic_pairs)} 对已加载)")
    StockScorer.batch_normalize(scored_stocks, weights, extra_ortho_pairs=dynamic_pairs)

    # === Alpha overlay + Chip 风控排除(全市场分层验证产物, 实盘/回测同口径) ===
    from src.engine.scorer import apply_alpha_overlay, apply_chip_guard
    apply_alpha_overlay(scored_stocks, config)
    scored_stocks = apply_chip_guard(scored_stocks, config)

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
        _health_today = datetime.now().strftime("%Y-%m-%d")
        health = assess_strategy_health(db, today=_health_today,
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


def _pm_persist_stage1(ctx):
    """阶段4之1"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]

    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]
    _pack(ctx, dict(locals()))
    ctx['config'] = config

def _pm_p2s1(ctx):
    """阶段2之1"""
    """阶段4之2"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]

    """阶段4之2"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]
    try:
        industry_map = market_fetcher.get_industry_map()
        if industry_map:
            SECTOR_GROUP = {
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
            SECTOR_CAP = {
                "科技": 2, "制造": 2, "金融": 1, "消费": 1, "医药": 1,
                "地产": 1, "能源": 1, "交通": 1, "钢铁": 1, "有色": 1,
                "汽车": 1, "军工": 1, "农业": 1, "环保": 1, "传媒": 1,
                "公用": 1, "综合": 1, "次新": 1,
            }
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

def _pm_p2s2(ctx):
    """阶段2之2"""
    """阶段4之2"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]

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
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_persist_stage2(ctx):
    """编排器(原实现按段切分, 状态经ctx传递)。"""
    _pm_p2s1(ctx)
    _pm_p2s2(ctx)

def _pm_p3s1(ctx):
    """阶段2之1"""
    """阶段4之3"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]

    """阶段4之3"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]
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
                except Exception:
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

def _pm_p3s2(ctx):
    """阶段2之2"""
    """阶段4之3"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
    )
    config = ctx["config"]

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
            config = rec_configs.get(code, trade_cfg)
            db.open_paper_trade(
                trade_date=today,
                code=code,
                name=rec["name"],
                score=rec["score"],
                entry_price=rec.get("current_price", 0),
                position_pct=rec["factors"].get("position_pct"),
                shares=config.get("shares"),
                trade_config=config,
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
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_persist_stage3(ctx):
    """编排器(原实现按段切分, 状态经ctx传递)。"""
    _pm_p3s1(ctx)
    _pm_p3s2(ctx)

def _pm_persist_stage4(ctx):
    """阶段4之4"""
    (
        SECTOR_CAP, SECTOR_GROUP, _regime_info, account_capital, atr_calc,
        atr_raw, atr_series, atr_value, bp_count, bp_engine,
        cache_data, cap, check_date, code, db,
        db_field, entry, entry_price, exit_price, f,
        filtered, final, global_markets, health_note, hist,
        history, i, ind, index_data, industry_map,
        lookback, market_fetcher, max_per_ind, max_rec, nd,
        news_analysis, news_factor, news_list, param_field, past,
        past_recs, pos_info, pos_mgr, rec, rec_configs,
        recommendations, report_content, report_path, reporter, ret,
        rotation_data, s, score, scored_stocks, screener,
        sec, sector_count, sentiment, sl_info, today,
        tp_info, trade_cfg,
    ) = ctx.unpack(
        'SECTOR_CAP', 'SECTOR_GROUP', '_regime_info', 'account_capital', 'atr_calc',
        'atr_raw', 'atr_series', 'atr_value', 'bp_count', 'bp_engine',
        'cache_data', 'cap', 'check_date', 'code', 'db',
        'db_field', 'entry', 'entry_price', 'exit_price', 'f',
        'filtered', 'final', 'global_markets', 'health_note', 'hist',
        'history', 'i', 'ind', 'index_data', 'industry_map',
        'lookback', 'market_fetcher', 'max_per_ind', 'max_rec', 'nd',
        'news_analysis', 'news_factor', 'news_list', 'param_field', 'past',
        'past_recs', 'pos_info', 'pos_mgr', 'rec', 'rec_configs',
        'recommendations', 'report_content', 'report_path', 'reporter', 'ret',
        'rotation_data', 's', 'score', 'scored_stocks', 'screener',
        'sec', 'sector_count', 'sentiment', 'sl_info', 'today',
        'tp_info', 'trade_cfg',
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
    _pack(ctx, dict(locals()))
    ctx['config'] = config


def _pm_persist(ctx):
    """编排器(原实现按段切分, 状态经ctx传递)。"""
    _pm_persist_stage1(ctx)
    _pm_persist_stage2(ctx)
    _pm_persist_stage3(ctx)
    _pm_persist_stage4(ctx)


def _pm_feedback(ctx):
    (
        _elapsed, _regime_info, _t0, candidates_df, changed,
        cur_w, factor_data, fb, fb_result, m,
        msg, name, opt_w, opt_weights, optimizer,
        qualified, recommendations, scored_stocks, today, wc,
        weights,
    ) = ctx.unpack(
        '_elapsed', '_regime_info', '_t0', 'candidates_df', 'changed',
        'cur_w', 'factor_data', 'fb', 'fb_result', 'm',
        'msg', 'name', 'opt_w', 'opt_weights', 'optimizer',
        'qualified', 'recommendations', 'scored_stocks', 'today', 'wc',
        'weights',
    )
    config = ctx["config"]

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
        fb_result = fb.run_all(today)
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

    _elapsed = (datetime.now() - _t0).total_seconds()
    print("\n" + "=" * 60)
    print(f"  盘前模式完成! 总耗时: {_elapsed:.0f}s")
    print(f"  候选池: {len(candidates_df)} → 深度计算: {len(scored_stocks)}")
    print(f"  信号筛选: {len(qualified)} → 达标推荐: {len(recommendations)}")
    print("  9:25请执行: python run_daily.py --intraday")
    print("=" * 60)
    _pack(ctx, dict(locals()))
    ctx["config"] = config


class _Ctx(dict):
    """阶段间状态容器。

    unpack 对缺失键返回 None——首阶段解包未初始化名时,
    由原代码立即重新赋值, 语义与原局部变量一致。
    """

    def unpack(self, *names):
        return tuple(self.get(n) for n in names)


def run_premarket(config):
    """盘前模式编排器: 各阶段通过 ctx 传递状态。"""
    ctx = _Ctx({"config": config})
    _pm_init(ctx)
    _pm_mkt(ctx)
    _pm_sent(ctx)
    _pm_screen(ctx)
    _pm_quick_topn(ctx)
    _pm_deep_loop(ctx)
    _pm_guard(ctx)
    _pm_persist(ctx)
    _pm_feedback(ctx)


