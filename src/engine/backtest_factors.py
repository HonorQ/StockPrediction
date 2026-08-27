"""
K 线可重建因子计算器 — 反馈训练专用

职责：
  从 baostock K 线行 (date, close, volume, turn, pctChg) 重建生产 scorer.get_sub_scores
  口径的 24 维子评分。只计算可从历史 K 线重建的因子，无法重建的（新闻/资金流/龙虎榜/护城河等）
  置为 50（中性），不强行造数据。

可重建的因子（9 维）：
  trend / strategy / chip / relative_strength / ashare_specific
  quant_behavior / capital_behavior / deep_research / oversold_rebound

无法重建的因子（15 维，置 50）：
  fundamental / institutional / sector_rotation / news / margin_trading
  shareholder / lockup_risk / research / dragon_tiger / leader
  policy_event / national_team / holder_concentration / circulate_holder

输出：
  {factor_name: sub_score (0-100)} — 与 StockScorer.get_sub_scores 完全一致
"""
import logging
import os
import sys
from typing import Dict, List
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.factors.technical import TechnicalFactors
from src.factors.strategies import StockStrategies
from src.factors.chip import ChipDistribution
from src.factors.relative_strength import RelativeStrength
from src.factors.ashare_specific import AShareCharacteristics
from src.factors.quant_behavior import QuantBehaviorDetector
from src.factors.capital_behavior import CapitalBehaviorFactor
from src.factors.deep_research import DeepResearchFactor
from src.factors.sector_rotation import OversoldReboundStrategy
from src.engine.scorer import StockScorer

logger = logging.getLogger(__name__)


# 24 维因子完整列表（与 StockScorer.get_sub_scores 完全一致，
# 原清单缺 holder_concentration / circulate_holder 两维）
ALL_FACTORS = [
    "trend", "strategy", "chip", "fundamental", "institutional",
    "sector_rotation", "oversold_rebound", "news", "relative_strength",
    "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
    "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
    "policy_event", "national_team", "capital_behavior", "moat",
    "holder_concentration", "circulate_holder",
]

# 可从 K 线重建的因子
RECONSTRUCTABLE_FACTORS = {
    "trend", "strategy", "chip", "relative_strength", "ashare_specific",
    "quant_behavior", "capital_behavior", "deep_research", "oversold_rebound",
}


def baostock_rows_to_dataframe(kline_rows: List[List]) -> pd.DataFrame:
    """把 baostock K 线行转换为生产 DataFrame 格式。

    生产列名（与 MarketDataFetcher.get_stock_history 对齐）：
        日期 / 开盘 / 收盘 / 最高 / 最低 / 成交量 / 成交额 / 换手率 / 涨跌幅

    支持两种输入格式：
    - 5 字段: [date, close, volume, turn, pctChg] — OHL 用收盘价+涨跌幅估算
    - 8 字段: [date, close, volume, turn, pctChg, open, high, low] — 使用真实 OHL
    """
    if not kline_rows:
        return pd.DataFrame()

    records = []
    for row in kline_rows:
        if len(row) < 5:
            continue
        date_str, close_str, vol_str, turn_str, pct_str = row[:5]
        try:
            close = float(close_str) if close_str else 0.0
            vol = float(vol_str) if vol_str else 0.0
            turn = float(turn_str) if turn_str else 0.0
            pct = float(pct_str) if pct_str else 0.0
        except (ValueError, TypeError) as e:
            logger.warning("捕获异常已忽略(src/engine/backtest_factors.py:86): %s", e)
            continue

        if close <= 0:
            continue

        # 当输入行有 8 个字段时，使用真实 OHL 而非估算
        if len(row) >= 8:
            open_val = float(row[5]) if row[5] else close
            high_val = float(row[6]) if row[6] else close
            low_val = float(row[7]) if row[7] else close
        else:
            # 5 字段 fallback：用 close ± pct 反推近似值。
            # 影线必须置为实体边界(平K线)——原 *1.005/*0.995 纯构造影线，
            # 会批量制造虚假的锤头线/上影线等 K 线形态信号污染训练集
            open_val = close / (1 + pct / 100) if pct != 0 else close
            high_val = max(open_val, close)
            low_val = min(open_val, close)

        # baostock volume 单位为「股」，生产口径(腾讯K线)为「手」→ 除以100对齐；
        # 成交额(元) = 收盘价 × 股数。原 close*vol*100 把成交额放大百倍污染量价因子
        amount = close * vol
        vol_lots = vol / 100.0

        records.append({
            "日期": str(date_str),
            "开盘": open_val,
            "收盘": close,
            "最高": high_val,
            "最低": low_val,
            "成交量": vol_lots,
            "成交额": amount,
            "换手率": turn,
            "涨跌幅": pct,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def calc_factors_from_kline(kline_rows: List[List], market_df: pd.DataFrame = None) -> Dict:
    """从 baostock K 线行计算因子字典（用于 StockScorer.get_sub_scores 输入）。

    Args:
        kline_rows: baostock 行列表 [date, close, volume, turn, pctChg]
        market_df: 大盘历史 DataFrame（沪深300等），用于 relative_strength 相对大盘部分；
                   不传时相对强弱仅算个股自身强度（strength_score），相对大盘降级

    Returns:
        factors dict — 与 run_daily.py 中 all_factors 结构对齐（可重建部分）
        无法重建的因子不写入（get_sub_scores 会用默认值 50）
    """
    df = baostock_rows_to_dataframe(kline_rows)
    if df.empty or len(df) < 20:
        return {}

    factors: Dict = {}
    tech_cfg = {
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "rsi_period": 6, "ma_periods": [5, 10, 20, 60],
        "boll_period": 20, "boll_std": 2, "atr_period": 14,
    }

    # 1. 技术因子（trend / strategy 的输入）
    try:
        tech = TechnicalFactors(df, tech_cfg)
        factors.update(tech.calculate_all())
    except Exception as e:
        logger.warning("TechnicalFactors 因子计算失败(按中性处理): %s", e)

    # 2. K 线形态策略（strategy 的输入）
    try:
        strategies = StockStrategies.get_triggered_strategies(df)
        factors["strategies_triggered"] = strategies
        factors["strategy_count"] = len(strategies)
    except Exception as e:
        logger.warning("StockStrategies 因子计算失败(按中性处理): %s", e)
        factors["strategies_triggered"] = []
        factors["strategy_count"] = 0

    # 3. 筹码分布（chip 的输入）
    try:
        chip_calc = ChipDistribution()
        chip_result = chip_calc.calculate(df)
        factors.update(chip_result)
        factors["chip_score"] = chip_calc.get_chip_score(df)
    except Exception as e:
        logger.warning("ChipDistribution 因子计算失败(按中性处理): %s", e)

    # 4. 相对强弱（relative_strength 的输入；传入 market_df 时含相对大盘部分）
    try:
        rs = RelativeStrength()
        rs_result = rs.calculate_all(df, market_df=market_df)
        factors.update(rs_result)
    except Exception as e:
        logger.warning("RelativeStrength 因子计算失败(按中性处理): %s", e)

    # 5. A股特色因子（ashare_specific 的输入）
    try:
        ashare = AShareCharacteristics()
        ashare_result = ashare.calculate_all(df, market_sentiment={})
        factors.update(ashare_result)
    except Exception as e:
        logger.warning("AShareCharacteristics 因子计算失败(按中性处理): %s", e)

    # 6. 量化行为检测（quant_behavior 的输入）
    try:
        quant_detector = QuantBehaviorDetector()
        quant_result = quant_detector.detect_all(df, realtime=None)
        factors["quant_detected"] = quant_result.get("quant_detected", False)
        factors["quant_score"] = quant_result.get("quant_score", 50)
        factors["quant_signals"] = quant_result.get("quant_signals", [])
        factors["quant_risk"] = quant_result.get("quant_risk", "low")
    except Exception as e:
        logger.warning("QuantBehaviorDetector 因子计算失败(按中性处理): %s", e)
        factors["quant_detected"] = False
        factors["quant_score"] = 50
        factors["quant_signals"] = []
        factors["quant_risk"] = "low"

    # 7. 资金行为模式（capital_behavior 的输入）
    try:
        cap_factor = CapitalBehaviorFactor()
        cap_result = cap_factor.calculate(df)
        factors["capital_behavior"] = cap_result.get("capital_behavior", "normal")
        factors["capital_behavior_confidence"] = cap_result.get("capital_behavior_confidence", 0)
        factors["capital_behavior_signals"] = cap_result.get("capital_behavior_signals", [])
        factors["flow_continuity"] = cap_result.get("flow_continuity", "neutral")
    except Exception as e:
        logger.warning("CapitalBehaviorFactor 因子计算失败(按中性处理): %s", e)
        factors["capital_behavior"] = "normal"
        factors["capital_behavior_confidence"] = 0
        factors["flow_continuity"] = "neutral"

    # 8. 深度研究（deep_research 的输入；news_list 传空）
    try:
        dr = DeepResearchFactor()
        dr_result = dr.analyze(df, news_list=[], stock_code=None, stock_data=None)
        factors["deep_research_score"] = dr_result.get("deep_research_score", 50)
        # 其他 deep_research 字段不写入 sub_scores，但保留用于上下文
    except Exception as e:
        logger.warning("DeepResearchFactor 因子计算失败(按中性处理): %s", e)
        factors["deep_research_score"] = 50

    # 9. 超跌反弹（oversold_rebound 的输入）
    try:
        oversold_strategy = OversoldReboundStrategy()
        rebound_result = oversold_strategy.check_oversold_rebound(df)
        factors["oversold_rebound_detected"] = rebound_result.get("detected", False)
        factors["oversold_rebound_score"] = rebound_result.get("score", 0)
        factors["oversold_rebound_signals"] = rebound_result.get("signals", [])
    except Exception as e:
        logger.warning("OversoldReboundStrategy 因子计算失败(按中性处理): %s", e)
        factors["oversold_rebound_detected"] = False
        factors["oversold_rebound_score"] = 0
        factors["oversold_rebound_signals"] = []

    return factors


def calc_sub_scores_from_kline(kline_rows: List[List],
                                weights: Dict[str, float] = None,
                                market_df: pd.DataFrame = None) -> Dict[str, float]:
    """从 baostock K 线行计算 24 维 sub_scores（与 StockScorer.get_sub_scores 完全一致）。

    Args:
        kline_rows: baostock 行列表 [date, close, volume, turn, pctChg]
        weights: 可选权重（仅用于初始化 StockScorer，sub_scores 计算不依赖权重）
        market_df: 大盘历史 DataFrame，透传给 relative_strength（可选）

    Returns:
        {factor_name: sub_score (0-100)} — 24 维齐全
        可重建的因子有真实值，无法重建的因子 = 50（中性）
    """
    factors = calc_factors_from_kline(kline_rows, market_df=market_df)
    if not factors:
        return {}

    scorer = StockScorer(weights or {})
    return scorer.get_sub_scores(factors)


def slice_kline_by_date(kline_rows: List[List], date: str) -> List[List]:
    """切片：返回 date 及之前的所有 K 线行（模拟当日可见数据）"""
    return [r for r in kline_rows if r[0] <= date]
