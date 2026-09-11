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

from src.factors.ashare_specific import AShareCharacteristics, SentimentCycle, HotMoneyTracker
from src.engine.backtest import ATRStopLoss, PositionManager
from src.factors.capital_behavior import CapitalBehaviorFactor
from src.factors.chip import ChipDistribution
from src.factors.circular_holder import CirculateHolderFactor
from src.factors.deep_research import DeepResearchFactor
from src.factors.expectation_pricing import ExpectationPricing, NewsEventAnalyzer, ProfitTakingDetector
from src.factors.fundamental import FundamentalFactors
from src.factors.holder_concentration import HolderConcentrationFactor
from src.factors.institutional import InstitutionalFactors
from src.factors.macro_event import MacroEventDetector
from src.factors.ashare_extended import MarginTradingFactor, ShareholderChangeFactor, LockUpExpiryFactor, SeasonalityFactor, LimitUpQualityFactor
from src.data.market_data import MarketDataFetcher
from src.factors.money_flow_factor import MoneyFlowFactors
from src.data.money_flow import MoneyFlowFetcher
from src.factors.national_team import NationalTeamFactor
from src.factors.news_factor import NewsFactor
from src.data.news_fetcher import NewsFetcher
from src.factors.policy_event import PolicyEventFactor
from src.factors.popularity import PopularityFactors
from src.factors.quant_behavior import QuantBehaviorDetector
from src.factors.rebound_analysis import ReboundSustainability, NextDayPredictor
from src.factors.relative_strength import RelativeStrength
from src.factors.phase2_market_intel import SectorLinkage, MoneyFlowContinuity, InstitutionalResearch, DragonTigerBoard, SectorLeader
from src.factors.sector_rotation import SectorRotation, OversoldReboundStrategy
from src.factors.sentiment import SentimentFactors
from src.data.database import StockDatabase
from src.engine.scorer import StockScorer
from src.engine.screener import StockScreener
from src.pipeline.common import _Ctx
from src.pipeline.common import _cleanup_cache
from src.pipeline.common import _log_source_health
from src.pipeline.common import _pack
from src.utils.heartbeat import check_task_alerts


# 阶段实现位于同目录各模块:
from src.pipeline.premarket_data import _pm_mkt, _pm_sent  # noqa: F401
from src.pipeline.premarket_deep import (  # noqa: F401
    _pm_screen, _pm_quick_topn, _calc_one_stock, _pm_deep_loop)
from src.pipeline.premarket_finalize import (  # noqa: F401
    _pm_guard, _pm_feedback, _pm_persist)



def _pm_init(ctx):
    """组件初始化阶段: 实例化全部因子引擎/取数器/评分器并写入 ctx。

    显式依赖: 本阶段是生产者, 仅消费 config; 不做占位式 unpack。
    """
    config = ctx["config"]

    print("=" * 60)
    print("  盘前模式 — 全量因子计算")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

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
    sentiment_cycle = SentimentCycle(config)
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
            # 清零因子免疫: config 静态权重为 0 表示已被胜率校准否决
            # (2026-08-27 23 维过拟合清零), 历史/旧逻辑落库的调权结果
            # 不得将其复活(2026-09-11 修复: 曾致 strategy/trend 等静默回归)。
            revived = [k for k, v in latest.items()
                       if k in weights and float(weights[k]) == 0 and float(v) != 0]
            for k, v in latest.items():
                if k in weights and float(weights[k]) != 0:
                    weights[k] = v
            if revived:
                print(f"  [权重合并] 跳过已否决因子的历史调权: {sorted(revived)}")
            print(f"  [权重合并] 从 weight_history 加载 {len(latest)} 个因子最新调权结果")
    except Exception as e:
        print(f"  [权重合并] 跳过: {e}")
    scorer = StockScorer(weights)
    # t0 不带下划线: _pack 排除 '_' 开头键, 收尾阶段 _pm_feedback 依赖此键计算总耗时
    t0 = datetime.now()
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
