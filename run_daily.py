"""
每日运行入口 - 双模式
  --premarket  盘前模式下重计算 (建议7:00-8:00跑)
  --intraday   盘中模式快速修正 (建议9:25集合竞价后跑)

无参数时默认跑 --premarket
"""
import os
import sys
import json
import yaml
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.factors.moat import MoatFactors
from src.utils.log import get_logger

logger = get_logger()

# 绕过系统代理：项目数据源均为国内站点，强制全部直连。
# 不设此项时，Windows 残留的系统代理(如 Clash 关闭后仍指向 127.0.0.1:7897)
# 会让 requests 把请求发给不存在的代理导致大面积超时。
_NO_PROXY = "*"
os.environ["NO_PROXY"] = _NO_PROXY
os.environ["no_proxy"] = _NO_PROXY

# 全局网络超时：防止数据源故障时请求无限挂起导致每日任务卡死
# （2026-08-10 案例：东财 push2 连不上，industry_map 重建请求无超时挂起，盘前任务卡住）
import socket as _socket
_socket.setdefaulttimeout(15)

# ======================================================================
# 日志系统：同时输出到控制台和文件
# ======================================================================
class _DualOutput:
    """同时写入控制台和日志文件的输出包装"""
    def __init__(self, logger, original):
        self._logger = logger
        self._original = original

    def write(self, msg):
        if msg and msg.strip():
            self._logger.info(msg.strip())
        self._original.write(msg)

    def flush(self):
        self._original.flush()


def setup_logging(mode: str = "premarket"):
    """配置日志：文件(按日期轮转) + 控制台"""
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{today}_{mode}.log")

    logger = logging.getLogger("stock_prediction")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()

    # 文件handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # 控制台handler（通过StreamHandler复用原始stdout）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # 重定向stdout到日志文件（捕获所有print输出）
    sys.stdout = _DualOutput(logger, sys.stdout)

    # 捕获未处理异常并写入日志（避免静默崩溃无痕）
    def _excepthook(exc_type, exc_val, exc_tb):
        logger.error("未捕获异常导致进程退出", exc_info=(exc_type, exc_val, exc_tb))
    sys.excepthook = _excepthook

    return logger


def write_summary(mode: str, candidates_count: int, recommendations: list, **extra):
    """追加一行JSON到 daily_summary.log，快速查看每次运行结果"""
    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "candidates": candidates_count,
        "recommendations": [
            {"code": r.get("code", ""), "name": r.get("name", ""), "score": r.get("score", 0)}
            for r in recommendations
        ],
    }
    summary.update(extra)
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "daily_summary.log")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def _fetch_close_price_on_date(market_fetcher, code: str, base_date: str, lookback: int):
    """获取 base_date 之后的第 lookback 个交易日的实际收盘价。
    用于历史推荐收益计算，避免使用当日实时价造成的未来函数误差。

    策略：
      1. 用 get_stock_history 取最近 N 天的K线
      2. 优先查找日期 == base_date 对应行，然后向后数 lookback 行
      3. 找不到精确匹配时，返回最后 lookback+1 行中最后一行的收盘价
    """
    try:
        history = market_fetcher.get_stock_history(code, days=lookback + 30)
        if history is None or history.empty or len(history) < lookback + 1:
            return None
        close_col = "收盘" if "收盘" in history.columns else "close"
        date_col = "日期" if "日期" in history.columns else "date"

        # 优先定位 base_date 对应的行
        if date_col in history.columns:
            mask = history[date_col].astype(str) == str(base_date)
            if mask.any():
                base_idx = mask.idxmax()
                exit_idx = base_idx + lookback
                if exit_idx < len(history):
                    return float(history.iloc[exit_idx][close_col])

        # 无法找到精确的 T+N 交易日价格（如跨周末/数据未更新）→ 返回 None
        # 避免降级到最近价导致收益被错误算成 0.0（历史污染根因）
        return None
    except Exception:
        return None


def _prefetch_history_and_flow(market_fetcher, money_fetcher, codes: list, days: int = 120):
    """并行预取多只股票的历史K线和资金流数据。
    返回 (history_cache, flow_cache) 两个 dict[code -> DataFrame]。
    网络 I/O 是瓶颈，并行化可显著加速（受限于数据源限流，默认 8 并发）。
    """
    import concurrent.futures as _cf

    history_cache: Dict[str, object] = {}
    flow_cache: Dict[str, object] = {}

    def _fetch_one(code: str):
        try:
            h = market_fetcher.get_stock_history(code, days=days)
        except Exception:
            h = None
        try:
            f = money_fetcher.get_stock_fund_flow(code)
        except Exception:
            f = None
        return code, h, f

    # 8 并发是 akshare/腾讯 限流下的安全值
    max_workers = min(8, max(2, len(codes)))
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_one, c) for c in codes]
        for fut in _cf.as_completed(futures):
            try:
                code, h, f = fut.result()
                history_cache[code] = h
                flow_cache[code] = f
            except Exception:
                continue

    return history_cache, flow_cache


def _update_open_paper_trades(db, market_fetcher, trade_cfg: dict, today: str):
    """扫描所有未平仓 paper_trades，根据当日收盘价检查：
        - ATR动态止损 / 固定百分比止损
        - ATR动态止盈 / 固定百分比止盈
        - 追踪止盈（trailing stop）
        - 持仓超过 hold_days → 到期平仓
    持仓期间的最大回撤/最大浮盈也会更新。
    """
    import json
    open_trades = db.get_open_paper_trades()
    if not open_trades:
        return

    hold_days = trade_cfg.get("hold_days", 3)

    closed_count = 0
    for t in open_trades:
        entry_price = t.get("entry_price") or 0
        if entry_price <= 0:
            continue
        code = t["code"]
        name = t.get("name", code)
        trade_id = t["id"]
        entry_date = t.get("entry_date") or t["trade_date"]

        # ===== 优先检查持仓到期（不依赖价格） =====
        try:
            d0 = datetime.strptime(entry_date, "%Y-%m-%d")
            d1 = datetime.strptime(today, "%Y-%m-%d")
            if (d1 - d0).days >= hold_days:
                # 持仓到期：尝试取当日收盘价平仓
                try:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    close_price = _fetch_close_price_on_date(market_fetcher, code, today_str, 0)
                except Exception:
                    close_price = None
                # 取不到价格时跳过(下日再试)，不用入场价强行平仓——
                # 那会把"未知收益"记成 0.0，污染纸面交易统计
                if not close_price or close_price <= 0:
                    print(f"    ⏳ {name}: 到期但取不到当日价格，顺延平仓")
                    continue
                ret_pct = (close_price / entry_price - 1) * 100
                db.close_paper_trade(
                    trade_id=trade_id,
                    exit_date=today,
                    exit_price=close_price,
                    exit_reason="hold_expire",
                    max_drawdown_pct=t.get("max_drawdown_pct") or ret_pct,
                    max_profit_pct=t.get("max_profit_pct") or ret_pct,
                )
                closed_count += 1
                print(f"    📤 {name}: hold_expire (收益 {ret_pct:+.1f}%)")
                continue  # 已处理到期，跳过后续止损/止盈检查
        except Exception:
            pass

        # ===== 取当日收盘价用于止损/止盈判断 =====
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            close_price = _fetch_close_price_on_date(market_fetcher, code, today_str, 0)
        except Exception:
            close_price = None
        if not close_price or close_price <= 0:
            continue  # 无法取价，跳过止损/止盈（到期已在上面处理）

        ret_pct = (close_price / entry_price - 1) * 100

        # 解析 trade_config
        try:
            raw = t.get("trade_config")
            stored = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            stored = {}

        prev_max_drawdown = t.get("max_drawdown_pct") or 0
        prev_max_profit = t.get("max_profit_pct") or 0
        new_max_drawdown = min(prev_max_drawdown, ret_pct)
        new_max_profit = max(prev_max_profit, ret_pct)

        db.update_paper_trade_extremes(
            trade_id,
            max_drawdown_pct=new_max_drawdown,
            max_profit_pct=new_max_profit,
        )

        # 判断平仓条件（按优先级）
        exit_reason = None

        # 1. ATR动态止损
        atr_sl = stored.get("stop_loss_price")
        if atr_sl and close_price <= atr_sl:
            exit_reason = "stop_loss"

        # 2. ATR动态止盈
        if not exit_reason:
            atr_tp = stored.get("take_profit_price")
            if atr_tp and close_price >= atr_tp:
                exit_reason = "take_profit"

        # 3. 追踪止盈
        if not exit_reason:
            trailing_activated = stored.get("trailing_activated", False) or new_max_profit >= 5.0
            if trailing_activated:
                current_trailing = stored.get("trailing_stop_price", 0)
                trailing_pct = stored.get("trailing_pct", 3.0)
                trailing_price = round(close_price * (1 - trailing_pct / 100), 2)

                if current_trailing > 0 and close_price <= current_trailing:
                    exit_reason = "trailing_stop"
                elif trailing_price > current_trailing:
                    stored["trailing_activated"] = True
                    stored["trailing_stop_price"] = trailing_price
                    try:
                        db.update_trade_config(trade_id, stored)
                    except Exception:
                        pass

        # 4. 固定百分比止损（fallback）
        if not exit_reason:
            sl_pct = stored.get("stop_loss_pct") or trade_cfg.get("stop_loss_pct", -0.05)
            if ret_pct / 100 <= sl_pct:
                exit_reason = "stop_loss"

        # 5. 固定百分比止盈（fallback）
        if not exit_reason:
            tp_pct = stored.get("take_profit_pct") or trade_cfg.get("take_profit_pct", 0.08)
            if ret_pct / 100 >= tp_pct:
                exit_reason = "take_profit"

        if exit_reason:
            db.close_paper_trade(
                trade_id=trade_id,
                exit_date=today,
                exit_price=close_price,
                exit_reason=exit_reason,
                max_drawdown_pct=new_max_drawdown,
                max_profit_pct=new_max_profit,
            )
            closed_count += 1
            print(f"    📤 {name}: {exit_reason} (收益 {ret_pct:+.1f}%)")

    if closed_count > 0:
        print(f"  📋 paper_trades: 平仓 {closed_count} 笔（共 {len(open_trades)} 个持仓中）")


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


def _check_crash_and_close_positions(fresh, config):
    """检测大盘崩盘(跌停潮/全市场暴跌),对 open paper_trades 止损清仓。
    解决7-24案例:盘前开仓后当天崩盘,纸面交易要到次日premarket才检查止损,损失已扩大。
    intraday模式契约:只用已有实时行情,不触发重计算/重网络取数。
    """
    filter_cfg = config.get("filter", {})
    crash_cfg = filter_cfg.get("intraday_crash", {})
    limit_down_threshold = crash_cfg.get("limit_down_threshold", 30)
    avg_change_threshold = crash_cfg.get("avg_change_threshold", -1.5)

    change_col = "涨跌幅" if "涨跌幅" in fresh.columns else "pctChg"
    if change_col not in fresh.columns:
        return

    changes = pd.to_numeric(fresh[change_col], errors="coerce").dropna()
    if changes.empty:
        return

    limit_down_count = int((changes <= -9.5).sum())
    avg_change = float(changes.mean())

    if limit_down_count < limit_down_threshold and avg_change >= avg_change_threshold:
        return  # 未触发崩盘

    print(f"  🛑 检测到大盘崩盘: 跌停{limit_down_count}家, 全市场均涨跌{avg_change:+.2f}%")

    try:
        db = StockDatabase()
        open_trades = db.get_open_paper_trades()
        if not open_trades:
            return

        # 从实时行情建 代码→价格 索引
        price_map = {}
        for _, r in fresh.iterrows():
            price_map[str(r["代码"])] = float(r.get("最新价", 0) or 0)

        today = datetime.now().strftime("%Y-%m-%d")
        closed_count = 0
        for t in open_trades:
            entry_price = t.get("entry_price") or 0
            if entry_price <= 0:
                continue
            code = str(t["code"])
            trade_id = t["id"]

            close_price = price_map.get(code, 0)
            if close_price <= 0:
                continue

            ret_pct = (close_price / entry_price - 1) * 100
            prev_max_drawdown = t.get("max_drawdown_pct") or 0
            prev_max_profit = t.get("max_profit_pct") or 0
            new_max_drawdown = min(prev_max_drawdown, ret_pct)
            new_max_profit = max(prev_max_profit, ret_pct)

            db.close_paper_trade(
                trade_id=trade_id,
                exit_date=today,
                exit_price=close_price,
                exit_reason="crash_stop",
                max_drawdown_pct=new_max_drawdown,
                max_profit_pct=new_max_profit,
            )
            closed_count += 1
            print(f"    📤 {t.get('name', code)}: crash_stop (收益 {ret_pct:+.1f}%)")

        if closed_count > 0:
            print(f"  📋 崩盘清仓: {closed_count} 笔")
    except Exception as e:
        print(f"  ⚠️ 崩盘清仓失败: {e}")

from src.data.market_data import MarketDataFetcher
from src.data.source_health import health_check, summarize
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


CACHE_DIR = os.path.join(ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_config() -> dict:
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_weights_consistency(config: dict) -> None:
    """启动时校验 config.yaml 的 weights 与评分器维度一致。

    权重有两处来源（config.yaml 与 scorer.py 默认），目前仅靠人工同步。
    一旦改一处忘改另一处，会出现“某因子被静默丢弃”或“废权重恒为 50”
    的行为偏移。此处强制 key 集合一致，否则直接退出，避免带病运行。
    """
    cfg_weights = config.get("weights") or {}
    scorer = StockScorer()
    sub_keys = set(scorer.get_sub_scores({}).keys())   # 实际参与评分的 22 维
    default_keys = set(scorer.weights.keys())          # scorer 默认权重维度
    cfg_keys = set(cfg_weights.keys())

    problems = []
    # 0) 权重和必须为 1（否则评分尺度漂移，entry_score/min_score 实际含义失真）
    total = sum(float(v) for v in cfg_weights.values())
    if abs(total - 1.0) > 0.02:
        problems.append(f"config.weights 总和={total:.4f}，偏离 1.0 超过容差(±0.02)")
    # 1) config 权重必须覆盖全部子评分维度（否则该维度被静默丢弃）
    missing = sub_keys - cfg_keys
    if missing:
        problems.append(f"config.weights 缺失以下子评分维度的权重: {sorted(missing)}")
    # 2) config 不应存在子评分中没有的维度（否则恒为 50 的废权重）
    extra = cfg_keys - sub_keys
    if extra:
        problems.append(f"config.weights 包含子评分中不存在的冗余维度: {sorted(extra)}")
    # 3) scorer 默认权重自身需与子评分维度对齐（防止默认表漂移）
    if default_keys != sub_keys:
        _miss = sorted(sub_keys - default_keys)
        _ext = sorted(default_keys - sub_keys)
        problems.append(f"scorer 默认权重与子评分维度不一致: 缺 {_miss} / 多 {_ext}")

    if problems:
        for p in problems:
            logger.error("权重一致性校验失败 - %s", p)
        raise SystemExit(
            "权重配置 (config.yaml weights) 与评分器维度不一致，请先对齐再运行。"
        )


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        if isinstance(o, datetime):
            return str(o)
        return super().default(o)


def save_cache(name: str, data):
    path = os.path.join(CACHE_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=_NumpyEncoder)
    return path


def load_cache(name: str):
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# 盘前模式：全量计算，保存候选池
# ======================================================================
def _log_source_health():
    """启动数据源健康自检并输出状态日志。"""
    try:
        logger.info(summarize(health_check()))
    except Exception as e:
        logger.warning("数据源健康自检失败: %s", e)


def _cleanup_cache(days: int = 7):
    """清理超过 N 天的缓存文件"""
    import shutil
    cutoff = datetime.now() - timedelta(days=days)
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath, ignore_errors=True)
        except Exception:
            pass


def run_premarket(config):
    print("=" * 60)
    print("  盘前模式 — 全量因子计算")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    _t0 = datetime.now()

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

    # 1. 获取实时行情
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

    # 5. 热点+情绪
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

    # 6. 筛选 + 深度计算
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
        # 超时保护：已运行超过8分钟就退出
        if (datetime.now() - phase2_start).total_seconds() > 480:
            print(f"  ⏰ 阶段2已运行超过8分钟，停止计算（已处理{idx}/{len(top_candidates)}只）")
            break

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
            continue

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

    print(f"  共计算 {len(scored_stocks)} 只股票的因子")

    # 截面Z-Score归一化
    print("  --- 截面Z-Score归一化(替代固定50分基线) ---")
    # 动态正交化对由 feedback_loop IC 分析持久化，跨进程生效
    from src.engine.feedback_loop import FeedbackLoop
    dynamic_pairs = FeedbackLoop._load_dynamic_ortho_pairs()
    if dynamic_pairs:
        print(f"  (动态正交化 {len(dynamic_pairs)} 对已加载)")
    StockScorer.batch_normalize(scored_stocks, weights, extra_ortho_pairs=dynamic_pairs)

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

    recommendations = final[:max_rec]

    # === 行业分散约束：同一行业最多推荐 1 只，同赛道最多 2 只 ===
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

    # === 缓存候选池（去掉大字段，仅存关键数据供盘中用）===
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

    # 保存到数据库
    db = StockDatabase()
    # 先扫描 paper_trades 未平仓仓位，根据当日收盘价检查止损/止盈/到期
    try:
        _update_open_paper_trades(db, market_fetcher, trade_cfg, today)
    except Exception as _e:
        print(f"  ⚠️ paper_trades 维护失败: {_e}")
    # 预先计算推荐股票的 ATR 和仓位
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
    # 保存所有候选的因子截面（用于IC计算）
    for s in scored_stocks:
        db.save_factor_snapshot(
            trade_date=today,
            code=s["code"],
            name=s["name"],
            score=s["score"],
            sub_scores=s.get("sub_scores", {}),
            factors=s.get("factors", {}),
        )

    # 更新历史推荐的表现（用T+N日的实际收盘价计算收益，避免未来函数）
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

    # 生成盘前报告
    reporter = ReportGenerator()
    # 回顾表排除回测行(与午后模式同口径，每日一条最终推荐)
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
    )
    report_path = reporter.save_report(report_content)
    print(f"  ✅ 盘前报告已生成: {report_path}")

    # === 权重优化建议（基于历史因子IC分析）===
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


# ======================================================================
# 盘中模式：读缓存 + 刷实时行情 + 重排序
# ======================================================================
def _open_paper_trades_for_recs(db, market_fetcher, trade_cfg: dict,
                                 recommendations: list, today: str,
                                 score_field: str = "score",
                                 label: str = "午后"):
    """为推荐列表开纸面仓位（ATR动态止损止盈 + 仓位管理 + 追踪止盈参数），
    与盘前模式完全同口径；同一股票已有未平仓仓位时跳过。

    数据闭环意义：此前只有盘前推荐开仓，无法用历史数据回答
    "早盘推荐和午后推荐哪个更准"——午后推荐同样开仓后即可分别归因。
    score_field: 推荐dict中分数键名(盘前=score, 午后=afternoon_score)。
    """
    if not recommendations:
        return
    atr_calc = ATRStopLoss()
    pos_mgr = PositionManager(max_stop_loss_pct=trade_cfg.get("max_stop_loss_pct", -0.08))
    account_capital = trade_cfg.get("account_capital", 100000)
    opened = 0
    for rec in recommendations:
        code = rec["code"]
        entry_price = rec.get("current_price", 0) or 0
        score = rec.get(score_field, 0) or 0
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

        factors = rec.get("factors")
        if factors is None:
            factors = {}
            rec["factors"] = factors
        factors["position_pct"] = pos_info["position_pct"]

        try:
            if db.has_open_position(code):
                continue
            db.open_paper_trade(
                trade_date=today,
                code=code,
                name=rec.get("name", code),
                score=score,
                entry_price=entry_price,
                position_pct=pos_info["position_pct"],
                shares=pos_info["shares"],
                trade_config={
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
                },
            )
            opened += 1
        except Exception as e:
            logger.warning("%s开仓失败 %s: %s", label, code, e)
    if opened:
        print(f"  📋 {label}纸面开仓 {opened} 笔")


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
        return

    print(f"  ✅ 载入盘前候选池: {len(cache['candidates'])} 只 (计算于 {cache['run_at']})")

    # 1. 快速拉取实时行情
    print("\n[1/1] 获取实时开盘行情...")
    market_fetcher = MarketDataFetcher()
    fresh = market_fetcher.get_a_share_realtime()
    if fresh.empty:
        print("  ❌ 获取行情失败")
        return

    # 建立 代码→行 的快速索引
    fresh["代码_str"] = fresh["代码"].astype(str)
    price_map = {}
    for _, r in fresh.iterrows():
        price_map[r["代码_str"]] = r

    # 大盘崩盘检测:对 open 仓位止损清仓(解决7-24盘前开仓后当天崩盘问题)
    _check_crash_and_close_positions(fresh, config)

    # 2. 对候选池刷新评分（仅作观察池快照与预警依据，不产出买入推荐）
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

        # 基础分 = 盘前评分（因子分）
        intraday_score = old_score

        # 价格偏离修正
        old_price = c.get("premarket_price", 0)
        if old_price > 0:
            price_change_pct = (new_price - old_price) / old_price * 100
            # 高开加分，低开减分
            if price_change_pct > 3:
                intraday_score += 3
            elif price_change_pct > 1:
                intraday_score += 1
            elif price_change_pct < -3:
                intraday_score -= 3
            elif price_change_pct < -1:
                intraday_score -= 1

            # 涨停/跌停过滤 → 转为不可交易预警
            if change_pct >= 9.5:
                excluded.append({"code": code, "name": name,
                                 "reason": f"涨停 {change_pct:+.1f}%，买不进"})
                continue
            if change_pct <= -9.5:
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

    # 3. 排序（仅作观察池快照）
    updated.sort(key=lambda x: x["intraday_score"], reverse=True)

    # === 入场风险过滤 → 转为高风险预警标记 ===
    # 盘中模式复用盘前缓存，须与盘前一致的入场风险过滤，避免盘前判死的高风险票被盘中捞回
    filter_cfg = config.get("filter", {})
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

    # 4. 生成预警版报告（不含推荐列表）
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
    report_path = reporter.save_report(report_content)

    write_summary("intraday_watch", len(candidates), [],
                   excluded=len(excluded), risky=len(risky))

    print("\n" + "=" * 60)
    print("  盘中预警完成! (本档不产出买入推荐)")
    print(f"  观察池: {len(updated)} 只 | 不可交易: {len(excluded)} | 高风险: {len(risky)}")
    print(f"  报告文件: {report_path}")
    print("  💡 买入决策请以 14:40 午后《尾盘最终决策版》为准")
    print("=" * 60)


# ======================================================================
# 午后模式：早盘真实成交数据深度修正（建议14:40运行——
# 尾盘最后半小时是主力拉升/出货的揭晓时段，越接近收盘信息越完整；
# 14:40启动约3-5分钟出报告，留足14:45-14:55下单窗口，避免被迫进收盘集合竞价）
# ======================================================================
def run_afternoon(config):
    print("=" * 60)
    print("  午后模式 — 早盘数据深度修正")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    _log_source_health()

    today = datetime.now().strftime("%Y-%m-%d")
    cache = load_cache(f"premarket_{today}.json")
    if cache is None:
        print(f"  ❌ 未找到盘前缓存 cache/premarket_{today}.json")
        print("  → 请先运行盘前模式: python run_daily.py --premarket")
        return

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
        return

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

    # 5. 深度修正评分
    print("\n[5/5] 基于早盘数据深度修正评分...")
    news_factor = NewsFactor(keywords_cfg)
    candidates = cache["candidates"]
    bp_by_code = {c["code"]: c.get("business_profile") for c in candidates}
    updated = []

    for idx, c in enumerate(candidates):
        code = c["code"]
        name = c["name"]
        row = price_map.get(code)
        if row is None:
            continue

        new_price = float(row.get("最新价", 0))
        change_pct = float(row.get("涨跌幅", 0) or 0)
        volume_ratio = float(row.get("量比", 0) or 0)
        turnover = float(row.get("换手率", 0) or 0)

        if new_price <= 0:
            continue

        # 涨停/跌停过滤
        if change_pct >= 9.5 or change_pct <= -9.5:
            print(f"    ⛔ {name}({code}) {'涨停' if change_pct >= 9.5 else '跌停'}，跳过")
            continue

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

        updated.append({
            "code": code,
            "name": name,
            "current_price": new_price,
            "change_pct": round(change_pct, 2),
            "volume_ratio": volume_ratio,
            "turnover": turnover,
            "premarket_score": c["score"],
            "afternoon_score": round(afternoon_score, 1),
            "net_inflow": net_inflow,
            "news_score": ns,
            "strong_signals": signals.get("strategy_count", 0),
        })
        print(f"    [{idx+1}/{len(candidates)}] {name}({code}) "
              f"{c['score']}→{afternoon_score:.1f}分 "
              f"{change_pct:+.1f}% 量比{volume_ratio:.1f} 净流入{net_inflow:.0f}万")

    # === ML模型辅助信号（可选，不影响主流程）===
    # 特征来源必须是盘前 factor_snapshots 的完整原始因子(与训练数据同源)。
    # 此前传 updated 里不存在的 factors 键 → 永远 predict({}) → 特征全默认值50，
    # 属于分布外推理的伪修正，已修复。
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

    # 生成推荐（达标线与盘前/盘中统一取 config，避免三档口径漂移）
    min_score = filter_cfg.get("min_score", 53)
    max_rec = filter_cfg.get("max_recommend", 2)
    qualified = [c for c in updated if c["afternoon_score"] >= min_score]
    final = qualified[:max_rec]

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
    _open_paper_trades_for_recs(db, market_fetcher, trade_cfg, final,
                                 today, score_field="afternoon_score",
                                 label="午后")
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


# ======================================================================
# 因子IC评估
# ======================================================================
def run_factor_ic(config):
    """因子IC评估: 从数据库加载历史因子数据，计算IC/ICIR，输出权重建议"""
    from src.engine.ml_selector import FactorOptimizer

    lookback = config.get("factor_ic", {}).get("lookback_days", 60)
    forward = config.get("factor_ic", {}).get("forward_window", 3)
    weights = config.get("weights", {})

    print("=" * 60)
    print("  因子IC评估")
    print(f"  回溯天数: {lookback}天 | 未来窗口: {forward}天")
    print("=" * 60)
    print()

    optimizer = FactorOptimizer()
    factor_data = optimizer.load_from_database(
        lookback_days=lookback, forward_window=forward
    )

    if factor_data.empty or "future_return" not in factor_data.columns:
        print("  ⚠️ 数据库历史数据不足，无法计算因子IC")
        print("  💡 提示：需要至少运行几次盘前模式积累数据")
        print(f"  📊 当前可用数据: {factor_data.shape if hasattr(factor_data, 'shape') else '0'}")
        return

    print(f"  📊 加载因子截面数据: {len(factor_data)} 条")
    print()

    ic = optimizer.calculate_ic(factor_data, return_col="future_return")
    if not ic:
        print("  ⚠️ 无法计算因子IC")
        return

    # 按日计算ICIR（标准口径：mean(IC_daily) / std(IC_daily)）
    ic_by_day = optimizer.calculate_ic_by_day(factor_data, return_col="future_return")

    print("  📈 因子IC排名（前15）：")
    print("  " + "-" * 60)
    print(f"  {'排名':>4} {'因子':<20} {'IC值':>10} {'ICIR':>10} {'样本日':>8} {'强度':>6}")
    print("  " + "-" * 60)
    sorted_ic = sorted(ic.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, ic_val) in enumerate(sorted_ic[:15], 1):
        daily_ics = ic_by_day.get(name, [])
        icir = optimizer.calculate_icir(daily_ics)
        intensity = "🔵 强" if abs(ic_val) > 0.05 else ("🟡 中" if abs(ic_val) > 0.03 else "⚪ 弱")
        print(f"  {i:>4} {name:<20} {ic_val:>+10.4f} {icir:>10.2f} {len(daily_ics):>8} {intensity:>6}")
    print()

    high_corr = optimizer.analyze_factor_correlation(factor_data)
    if not high_corr.empty and len(high_corr) > 0:
        print("  ⚡ 高度相关因子对（>0.7）：")
        for _, row in high_corr.iterrows():
            print(f"      {row['factor1']} ↔ {row['factor2']}: {row['correlation']:.3f}")

    print()
    opt_weights = optimizer.optimize_weights(factor_data, return_col="future_return")
    if opt_weights:
        print("  🎯 优化权重建议（对比当前权重）：")
        print(f"  {'因子':<20} {'当前权重':>10} {'优化权重':>10} {'调整':>10}")
        print("  " + "-" * 50)
        for name in weights:
            cur = weights.get(name, 0)
            opt = opt_weights.get(name, 0)
            delta = opt - cur
            print(f"  {name:<20} {cur:>10.3f} {opt:>10.4f} {delta:>+10.4f}")
        print()

    report = optimizer.generate_weight_report()
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "factor_ic")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir,
                               f"factor_ic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ 因子IC报告已保存: {report_path}")


# ======================================================================
# 交易日判断
# ======================================================================
def _is_trading_day(d):
    """判断 d 是否为 A 股交易日（含法定节假日）。
    多数据源回退，避免数据源临时故障误判为非交易日：
      1) baostock 官方交易日历
      2) akshare 交易日历（tool_trade_date_hist_sina）
      3) 以上都不可用 → 周末判断兜底（节假日可能误判，但不会漏跑交易日）
    只有 baostock/akshare 明确返回该日期且标记为非交易日时才算非交易日。
    """
    day = d.strftime("%Y-%m-%d")
    weekend_only = lambda: d.weekday() < 5

    def _via_baostock():
        import baostock as bs
        bs.logout()  # 清掉可能残留的连接，防止上次失败状态污染本次判断
        try:
            bs.login()
        except Exception:
            return None
        rs = bs.query_trade_dates(start_date=day, end_date=day)
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 2 and row[0] == day:
                return row[1] == "1"
        return None

    def _via_akshare():
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is None or df.empty:
            return None
        dates = set(df["trade_date"].astype(str))
        if day in dates:
            return True
        # 数据源覆盖范围之外的近期日期视为无法判断（不武断判为休市）
        return None

    for label, fn in (("baostock", _via_baostock), ("akshare", _via_akshare)):
        try:
            result = fn()
        except Exception as e:
            logger.warning(f"{label} 交易日查询失败({e})，换下一个数据源")
            result = None
        if result is not None:
            return result

    logger.warning("交易日历数据源均不可用，回退为周末判断")
    return weekend_only()


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票预测系统")
    parser.add_argument("--premarket", action="store_true", help="盘前模式：全量因子计算（建议7:00-8:00运行）")
    parser.add_argument("--intraday", action="store_true", help="盘中模式：基于盘前缓存快速修正（建议9:25运行）")
    parser.add_argument("--afternoon", action="store_true", help="午后模式：基于全天真实成交+资金流向深度修正（建议14:40运行，尾盘最终决策）")
    parser.add_argument("--factor-ic", action="store_true", help="因子IC评估：从数据库加载历史数据计算因子IC")
    parser.add_argument("--lookback", type=int, default=60, help="因子IC回溯天数")
    args = parser.parse_args()

    # 确定运行模式
    if args.factor_ic:
        mode = "factor_ic"
    elif args.intraday:
        mode = "intraday"
    elif args.afternoon:
        mode = "afternoon"
    else:
        mode = "premarket"

    logger = setup_logging(mode)
    logger.info(f"启动模式: {mode}")

    today = datetime.now().date()
    if not _is_trading_day(today):
        logger.info(f"今日({today})非A股交易日（周末或法定节假日），跳过运行")
        sys.exit(0)

    config = load_config()
    _validate_weights_consistency(config)

    if args.factor_ic:
        config.setdefault("factor_ic", {})["lookback_days"] = args.lookback
        run_factor_ic(config)
    elif args.intraday:
        run_intraday(config)
    elif args.afternoon:
        run_afternoon(config)
    else:
        # 默认走盘前模式
        run_premarket(config)
