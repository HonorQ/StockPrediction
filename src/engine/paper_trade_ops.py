# -*- coding: utf-8 -*-
"""纸面交易运维：持仓维护/崩盘清仓/按推荐开仓。

从 run_daily.py 外迁(规则 AGENTS.md 二: 业务逻辑禁止留在入口文件)。
依赖注入式设计——db/market_fetcher 由调用方传入, 本模块不建连接。
"""
import json
from datetime import datetime

import pandas as pd

from src.utils.log import get_logger
from src.utils.helpers import get_limit_pct
from src.engine.backtest import ATRStopLoss, PositionManager

logger = get_logger()


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
    except Exception as e:
        logger.warning(f"获取历史收盘价失败: {e}")
        return None


def _fetch_today_close(market_fetcher, code):
    """取 code 当日收盘价；失败返回 None（调用方顺延处理，禁止降级为 0.0）。"""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        return _fetch_close_price_on_date(market_fetcher, code, today_str, 0)
    except Exception as e:
        logger.warning(f"获取当日收盘价失败: {e}")
        return None


def _is_before_market_close() -> bool:
    """当前是否早于A股收盘(15:00)。收盘前不存在当日收盘价。"""
    now = datetime.now()
    return (now.hour, now.minute) < (15, 0)


def _try_close_expired(db, market_fetcher, t, hold_days, today):
    """持仓到期平仓检查。返回 'closed'(已平) / 'defer'(顺延) / 'not_due'(未到期)。

    到期但不取得到价时顺延(下日再试), 不用入场价强行平仓——
    那会把"未知收益"记成 0.0, 污染纸面交易统计。
    盘中(收盘前)本来就取不到当日收盘价, 属正常等待, 按未到期处理且不刷屏。
    """
    entry_date = t.get("entry_date") or t["trade_date"]
    d0 = datetime.strptime(entry_date, "%Y-%m-%d")
    d1 = datetime.strptime(today, "%Y-%m-%d")
    if (d1 - d0).days < hold_days:
        return "not_due"

    close_price = _fetch_today_close(market_fetcher, t["code"])
    if not close_price or close_price <= 0:
        if _is_before_market_close():
            logger.debug("%s: 到期但当日未收盘, 平仓推迟到收盘后", t.get("name", t["code"]))
            return "not_due"
        print(f"    ⏳ {t.get('name', t['code'])}: 到期但取不到当日价格，顺延平仓")
        return "defer"

    ret_pct = (close_price / t.get("entry_price") - 1) * 100
    db.close_paper_trade(
        trade_id=t["id"],
        exit_date=today,
        exit_price=close_price,
        exit_reason="hold_expire",
        max_drawdown_pct=t.get("max_drawdown_pct") or ret_pct,
        max_profit_pct=t.get("max_profit_pct") or ret_pct,
    )
    print(f"    📤 {t.get('name', t['code'])}: hold_expire (收益 {ret_pct:+.1f}%)")
    return "closed"


def _judge_exit(stored, close_price, ret_pct, new_max_profit, trade_cfg,
                db=None, trade_id=None):
    """按优先级判定平仓原因: ATR止损 > ATR止盈 > 追踪止盈 > 固定止损 > 固定止盈。

    口径(全项目唯一): ret_pct/new_max_profit 为百分数(1.0=1%);
    stop_loss_pct/take_profit_pct/trailing_pct 为小数比例或百分数见 config。
    追踪止盈激活且上移时会就地更新 stored 并持久化(经 db.update_trade_config)。
    """
    # 1./2. ATR 动态止损/止盈
    atr_sl = stored.get("stop_loss_price")
    if atr_sl and close_price <= atr_sl:
        return "stop_loss"
    atr_tp = stored.get("take_profit_price")
    if atr_tp and close_price >= atr_tp:
        return "take_profit"

    # 3. 追踪止盈: 浮盈达 5% 自动激活, 自高点回撤 trailing_pct 触发
    trailing_activated = stored.get("trailing_activated", False) or new_max_profit >= 5.0
    if trailing_activated:
        current_trailing = stored.get("trailing_stop_price", 0)
        trailing_pct = stored.get("trailing_pct", 3.0)
        trailing_price = round(close_price * (1 - trailing_pct / 100), 2)

        if current_trailing > 0 and close_price <= current_trailing:
            return "trailing_stop"
        if trailing_price > current_trailing:
            stored["trailing_activated"] = True
            stored["trailing_stop_price"] = trailing_price
            if db is not None:
                try:
                    db.update_trade_config(trade_id, stored)
                except Exception as e:
                    logger.warning("追踪止盈位持久化失败(trade=%s): %s", trade_id, e)

    # 4./5. 固定百分比止损/止盈（fallback）
    sl_pct = stored.get("stop_loss_pct") or trade_cfg.get("stop_loss_pct", -0.05)
    if ret_pct / 100 <= sl_pct:
        return "stop_loss"
    tp_pct = stored.get("take_profit_pct") or trade_cfg.get("take_profit_pct", 0.08)
    if ret_pct / 100 >= tp_pct:
        return "take_profit"
    return None


def _update_open_paper_trades(db, market_fetcher, trade_cfg: dict, today: str):
    """扫描所有未平仓 paper_trades，根据当日收盘价检查：
        - ATR动态止损 / 固定百分比止损
        - ATR动态止盈 / 固定百分比止盈
        - 追踪止盈（trailing stop）
        - 持仓超过 hold_days → 到期平仓
    持仓期间的最大回撤/最大浮盈也会更新。
    """
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

        # ===== 优先检查持仓到期（不依赖价格） =====
        try:
            status = _try_close_expired(db, market_fetcher, t, hold_days, today)
        except Exception as e:
            logger.warning("到期检查失败(trade=%s): %s", trade_id, e)
            status = "not_due"
        if status == "closed":
            closed_count += 1
            continue
        if status == "defer":
            continue

        # ===== 取当日收盘价用于止损/止盈判断 =====
        close_price = _fetch_today_close(market_fetcher, code)
        if not close_price or close_price <= 0:
            continue  # 无法取价，跳过止损/止盈（到期已在上面处理）

        ret_pct = (close_price / entry_price - 1) * 100

        # 解析 trade_config
        try:
            raw = t.get("trade_config")
            stored = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as e:
            logger.warning(f"解析交易配置失败: {e}")
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

        exit_reason = _judge_exit(
            stored, close_price, ret_pct, new_max_profit, trade_cfg,
            db=db, trade_id=trade_id,
        )
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

    changes = pd.to_numeric(fresh[change_col], errors="coerce")
    if changes.dropna().empty:
        return

    # 跌停家数按各板块实际幅度判定(20cm/30cm/ST 5%), 容差0.5pct对齐四舍五入
    code_col = "代码" if "代码" in fresh.columns else None
    name_col = "名称" if "名称" in fresh.columns else None
    if code_col:
        def _down_line(r):
            pct = get_limit_pct(str(r[code_col]), str(r.get(name_col) or "") if name_col else "")
            return -(pct - 0.5)
        limit_down_count = int((changes <= fresh.apply(_down_line, axis=1)).sum())
    else:
        limit_down_count = int((changes <= -9.5).sum())  # 兜底: 无代码列时沿用主板口径
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

from src.data.database import StockDatabase


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
            except Exception as e:
                logger.warning(f"计算ATR失败: {e}")
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
