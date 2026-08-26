# -*- coding: utf-8 -*-
"""盘前模式(07:00): 全量因子计算与候选池产出

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.log import get_logger

logger = get_logger()
from src.data.concept_map import ConceptMap
from src.data.hot_stocks import HotStocksFetcher
from src.pipeline.common import _pack


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
