# -*- coding: utf-8 -*-
"""盘前模式(07:00): 全量因子计算与候选池产出

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

logger = get_logger()

# 阶段实现位于同目录各模块:
from src.pipeline.premarket_data import _pm_mkt, _pm_sent  # noqa: F401
from src.pipeline.premarket_deep import (  # noqa: F401
    _pm_screen, _pm_quick_topn, _calc_one_stock, _pm_deep_loop)
from src.pipeline.premarket_finalize import (  # noqa: F401
    _pm_guard, _pm_feedback)



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
