"""
反馈训练 Walk-Forward 主入口
阶段1: 6/29 → 7/31 (A 股交易日)
阶段2: 2026/01 ~ 2026/06 (A 股交易日)

每日流程:
   1. 计算当日所有候选股的 sub_scores (backtest_factors.calc_sub_scores_from_kline)
  2. 加载每只股票的 stock_factor_weights (StockWeightManager.get_weights_batch)
  3. 截面 Z-Score 归一化 + per-stock 权重加权 → 选 top 10 推荐
  4. 写 recommendations + factor_snapshots
  5. 对所有已过 T+N (N=1,3,5,10,14) 的历史推荐补算实际收益
  6. FeedbackLoop.run_all(today, skip_ml=True): 归因 + IC + 全局调权 + 个股调权
"""
from src.utils.log import setup_logging, get_logger
setup_logging()
logger = get_logger()

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import baostock as bs
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import List

# K线本地缓存（避免重复 baostock 查询）
KLINE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "klines")
os.makedirs(KLINE_CACHE_DIR, exist_ok=True)

# 全局缓存键（固定，不依赖具体日期范围）
KLINE_CACHE_KEY = "phase2_full"

def _kline_cache_path(code):
    return os.path.join(KLINE_CACHE_DIR, f"{KLINE_CACHE_KEY}_{hashlib.md5(code.encode()).hexdigest()[:16]}.pkl")


def fetch_kline(code, start, end, use_cache=True):
    """baostock 拉取单只股票 K 线，带本地缓存。"""
    if use_cache:
        cp = _kline_cache_path(code)
        if os.path.exists(cp):
            try:
                with open(cp, "rb") as f:
                    return code, pickle.load(f)
            except Exception:
                pass
    rs = bs.query_history_k_data_plus(
        code, "date,close,volume,turn,pctChg,open,high,low",
        start_date=start, end_date=end, frequency="d", adjustflag="3"
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if use_cache and rows:
        try:
            cp = _kline_cache_path(code)
            with open(cp, "wb") as f:
                pickle.dump(rows, f)
        except Exception:
            import traceback; traceback.print_exc()
    return code, rows


def prefetch(stocks, start, end):
    """串行预取所有股票 K 线。

    注意：baostock 不支持并发查询（多线程/多会话均会挂死），故改为串行。
    已添加本地磁盘缓存，第二次起仅加载本地文件。
    """
    data = {}
    logger.info(f'  预取 {len(stocks)} 只K线(串行)...')
    t0 = time.time()
    for i, (c, _) in enumerate(stocks, 1):
        _, r = fetch_kline(c, start, end)
        if r:
            data[c] = r
        if i % 500 == 0:
            cached = sum(1 for s, _ in stocks[:i] if os.path.exists(_kline_cache_path(s)))
            logger.info(f'    {i}/{len(stocks)} ({time.time() - t0:.0f}s, cache_hit={cached})')
    logger.info(f'  完成 {len(data)} 只 ({time.time() - t0:.0f}s)')
    return data

sys.path.insert(0, '.')
sys.path.insert(0, 'src')

from src.data.database import StockDatabase
from src.engine.stock_weight_manager import StockWeightManager, _normalize, ALL_FACTORS
from src.engine.feedback_loop import FeedbackLoop
from src.engine.backtest_factors import (
    calc_sub_scores_from_kline, calc_factors_from_kline, slice_kline_by_date,
    baostock_rows_to_dataframe,
)
from src.engine.scorer import StockScorer
from src.engine.ml_selector import train_shadow_ml, MLStockSelector


# === 阶段1 日期列表（硬编码 A 股交易日，6/29~7/31） ===
DATES_PHASE1 = [
    '2026-06-29', '2026-06-30', '2026-07-01', '2026-07-02', '2026-07-03',
    '2026-07-06', '2026-07-07', '2026-07-08', '2026-07-09', '2026-07-10',
    '2026-07-13', '2026-07-14', '2026-07-15', '2026-07-16', '2026-07-17',
    '2026-07-20', '2026-07-21', '2026-07-22', '2026-07-23', '2026-07-24',
    '2026-07-27', '2026-07-28', '2026-07-29', '2026-07-30', '2026-07-31',
]

# 阶段2 日期列表（启动时用 baostock 动态获取 2026/01~06 交易日）
DATES_PHASE2: List[str] = []


def get_stocks(date):
    """获取某日全 A 股名单（排除 ST/北交所/科创/指数）"""
    rs = bs.query_all_stock(date)
    stocks = []
    while rs.next():
        stocks.append(rs.get_row_data())
    result = []
    for s in stocks:
        code, status, name = s[0], s[1], s[2]
        if status != "1":
            continue
        if name.startswith(("*", "ST")):
            continue
        if any(code.startswith(p) for p in ["bj", "sh.688", "sh.000", "sh.399", "sz.399", "sz.300"]):
            continue
        result.append((code, name))
    return result


def fetch_trade_dates(start, end):
    """用 baostock 拉取指定范围内的 A 股交易日列表"""
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    dates = []
    while rs.next():
        row = rs.get_row_data()
        # baostock 返回 [date, is_trading_day]
        if len(row) >= 2 and row[1] == "1":
            dates.append(row[0])
    return dates


def compute_sub_scores(date, stocks, kc, market_kc=None):
    """对每只股票切片 K 线并计算 sub_scores。
    返回 [{code, raw_code, name, price, sub_scores}, ...]
    market_kc: 大盘指数 K 线（如沪深300），用于 relative_strength 相对大盘部分
    """
    # 市场基准：切片到当日的大盘 DataFrame（供相对强弱因子使用）
    market_df = None
    if market_kc:
        mrows = slice_kline_by_date(market_kc, date)
        if len(mrows) >= 20:
            market_df = baostock_rows_to_dataframe(mrows)

    results = []
    for code, name in stocks:
        raw = kc.get(code)
        if not raw or len(raw) < 20:
            continue
        # 切片到当日（避免未来函数）
        rows = slice_kline_by_date(raw, date)
        if len(rows) < 20:
            continue
        sub_scores = calc_sub_scores_from_kline(rows, market_df=market_df)
        if not sub_scores:
            continue
        factors = calc_factors_from_kline(rows, market_df=market_df)
        # 当前价 = 切片最后一行的 close
        try:
            price = float(rows[-1][1])
        except (ValueError, IndexError):
            continue
        if price <= 0:
            continue
        results.append({
            "code": code.replace("sh.", "sh").replace("sz.", "sz"),
            "raw_code": code,  # 保留 baostock 格式用于 future_returns
            "name": name,
            "price": price,
            "sub_scores": sub_scores,
            "factors": factors,
        })
    return results


def score_cross_sectional_per_stock(candidates, global_weights):
    """截面排名分归一化 + per-stock 权重加权。
    对每个维度用排名分替代 Z-Score，确保评分均匀分布在 0-100 区间。
    就地写入 candidate['score'] 和 candidate['z_scores']。
    """
    dims = [k for k in global_weights.keys() if global_weights.get(k, 0) > 0]
    if not dims or not candidates:
        return
    n = len(candidates)

    raw = {dim: [] for dim in dims}
    for c in candidates:
        ss = c.get("sub_scores", {})
        for dim in dims:
            raw[dim].append(ss.get(dim, 50))

    for dim in dims:
        series = pd.Series(raw[dim])
        raw[dim] = series.clip(
            series.quantile(0.05), series.quantile(0.95)
        ).tolist()

    if n >= 10:
        for target, regressor in StockScorer.ORTHOGONALIZE_PAIRS:
            if target not in raw or regressor not in raw:
                continue
            t_vals, r_vals = raw[target], raw[regressor]
            mu_t = sum(t_vals) / n
            mu_r = sum(r_vals) / n
            cov_tr = sum((t - mu_t) * (r - mu_r) for t, r in zip(t_vals, r_vals)) / n
            var_r = sum((r - mu_r) ** 2 for r in r_vals) / n
            if var_r < 1e-6:
                continue
            beta = cov_tr / var_r
            raw[target] = [t - beta * (r - mu_r) for t, r in zip(t_vals, r_vals)]

    for i, c in enumerate(candidates):
        score_total = 0.0
        rank_scores = {}
        stock_weights = c.get("stock_weights", global_weights)
        for dim in dims:
            vals = raw[dim]
            sorted_idx = sorted(range(n), key=lambda k: vals[k])
            rank = sorted_idx.index(i)
            pct = rank / (n - 1) if n > 1 else 0.5
            rank_scores[dim] = round(pct * 100, 1)
            score_total += pct * stock_weights.get(dim, 0)
        score = score_total * 100 / max(sum(stock_weights.get(d, 0) for d in dims), 1e-6)
        c["score"] = round(max(0, min(100, score)), 1)
        c["z_scores"] = rank_scores


# 评分分布监控日志路径
TOP_N = 50
SCORE_DIST_LOG = os.path.join("reports", "feedback", "score_distribution.log")


def monitor_score_distribution(candidates):
    """监控每日评分分布，判断因子组合是否有区分能力。"""
    scores = [c["score"] for c in candidates]
    scores.sort(reverse=True)
    top1 = scores[0]
    median = scores[len(scores) // 2]
    bottom10 = scores[len(scores) // 10]
    std = float(np.std(scores))
    top_n_mean = sum(scores[:TOP_N]) / TOP_N

    gap_top_median = top1 - median
    gap_top_bottom10 = top1 - bottom10
    gap_top_n_median = top_n_mean - median

    if gap_top_median > 20 and std > 8:
        status = "HEALTHY"
    elif gap_top_median > 10 and std > 4:
        status = "WARN"
    else:
        status = "ALERT"

    os.makedirs(os.path.dirname(SCORE_DIST_LOG), exist_ok=True)
    with open(SCORE_DIST_LOG, "a", encoding="utf-8") as f:
        log_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{log_date}: candidates={len(candidates)}, "
            f"top1={top1:.1f}, median={median:.1f}, "
            f"gap={gap_top_median:.1f}, std={std:.2f}, "
            f"top{TOP_N}_mean={top_n_mean:.1f}, "
            f"gap_top{TOP_N}_median={gap_top_n_median:.1f}, "
            f"status={status}\n"
        )

    return {
        "top1": top1,
        "median": median,
        "gap_top_median": gap_top_median,
        "gap_top_bottom10": gap_top_bottom10,
        "gap_top_n_median": gap_top_n_median,
        "std": std,
        "status": status,
    }


def update_future_returns(db, kc, today):
    """补算历史推荐的 T+N 收益（N=1,3,5,10,14）。
    只处理 trade_date + N <= today 的推荐，避免未来函数。
    """
    # (offset_days, rec_field, db_param_name)
    offsets = [(1, "d1_return", "d1"), (3, "d3_return", "d3"),
               (5, "d5_return", "d5"), (10, "d10_return", "d10"),
               (14, "d14_return", "d14")]
    count = 0

    all_recs = db.get_recent_recommendations(days=365)
    for rec in (r for r in all_recs if r.get("trade_date", "") <= today):
        date_str = rec.get("trade_date", "")
        code = rec.get("code", "")
        if not date_str or not code:
            continue
        try:
            trade_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        # kc 的键为 baostock 原始格式（含点，如 "sh.510060"），
        # 而存储/候选的 code 已去点（如 "sh510060"），这里补回点以匹配 kc
        raw_code = code if "." in code else code[:2] + "." + code[2:]
        rows = kc.get(raw_code)
        if not rows:
            continue
        entry = rec.get("current_price", 0)
        if entry <= 0:
            continue
        future = [r for r in rows if r[0] > date_str]

        for off, field, param in offsets:
            # 未来函数检查：trade_date + N 必须 <= today
            future_date = (trade_dt + timedelta(days=off)).strftime("%Y-%m-%d")
            if future_date > today:
                continue
            # 幂等：已有值则跳过
            if rec.get(field) is not None:
                continue
            if len(future) >= off:
                try:
                    ep = float(future[off - 1][1])
                except (ValueError, IndexError):
                    continue
                if ep > 0:
                    db.update_performance(date_str, code,
                                          **{param: round((ep / entry - 1) * 100, 2)})
                    count += 1
    return count


def generate_report(db, phase_result, all_weight_changes, output_path):
    """生成 markdown 报告到 reports/feedback_training_<phase>.md"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    daily = phase_result["daily"]
    total_recs = len(daily) * 10
    dates = phase_result["dates"]

    # 推荐表现（T+14 主信号 + 多窗口）
    recs = db.get_recent_recommendations(days=120)
    recs_in_phase = [r for r in recs if dates[0] <= r.get("trade_date", "") <= dates[-1]]

    def _win_stats(field):
        items = [r for r in recs_in_phase if r.get(field) is not None]
        if not items:
            return 0, 0, 0.0
        wins = sum(1 for r in items if r[field] > 0)
        avg = sum(r[field] for r in items) / len(items)
        return len(items), wins, avg

    d1_n, d1_w, d1_avg = _win_stats("d1_return")
    d3_n, d3_w, d3_avg = _win_stats("d3_return")
    d5_n, d5_w, d5_avg = _win_stats("d5_return")
    d10_n, d10_w, d10_avg = _win_stats("d10_return")
    d14_n, d14_w, d14_avg = _win_stats("d14_return")

    # 因子归因统计
    attr_stats = db.get_attribution_stats(days=120)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 反馈训练报告 - {phase_result['phase']}\n\n")
        f.write(f"**阶段**: {dates[0]} → {dates[-1]} ({len(daily)} 天)\n\n")

        f.write("## 阶段汇总\n\n")
        f.write(f"- 总推荐: {total_recs} 只\n\n")
        f.write("## T+N 多窗口胜率\n\n")
        f.write("| 窗口 | 有收益数 | 胜数 | 胜率 | 平均收益 |\n")
        f.write("|---|---|---|---|---|\n")
        for name, n, w, avg in [("T+1", d1_n, d1_w, d1_avg),
                                 ("T+3", d3_n, d3_w, d3_avg),
                                 ("T+5", d5_n, d5_w, d5_avg),
                                 ("T+10", d10_n, d10_w, d10_avg),
                                 ("T+14", d14_n, d14_w, d14_avg)]:
            if n > 0:
                f.write(f"| {name} | {n} | {w} | {w/n*100:.1f}% | {avg:.2f}% |\n")
            else:
                f.write(f"| {name} | 0 | 0 | N/A | N/A |\n")

        f.write("\n## 每日推荐明细\n\n")
        f.write(f"| 日期 | 候选数 | Top{TOP_N}均分 | 收益更新 | 归因数 | 个股调权数 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d in daily:
            f.write(f"| {d['date']} | {d['n_candidates']} | {d['top_n_score']} | "
                    f"{d['n_returns_updated']} | {d['n_attributions']} | "
                    f"{d['n_stock_weight_changes']} |\n")

        f.write("\n## 个股权重变化 Top10\n\n")
        if all_weight_changes:
            # 已按时间顺序累积，取变化幅度最大的 Top10
            sorted_changes = sorted(all_weight_changes, key=lambda x: abs(x.get("change", 0)), reverse=True)
            f.write("| 股票 | 因子 | 旧权重 | 新权重 | 变化 | 原因 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for w in sorted_changes[:10]:
                f.write(f"| {w.get('code','')} | {w.get('factor','')} | "
                        f"{w.get('old',0):.4f} | {w.get('new',0):.4f} | "
                        f"{w.get('change',0):+.4f} | {w.get('reason','')} |\n")
        else:
            f.write("无个股权重变化（可能样本数不足 MIN_SAMPLES=3）\n")

        f.write("\n## 因子归因统计\n\n")
        if attr_stats and attr_stats.get("factor_accuracy"):
            f.write("| 因子 | 命中 | 失败 | 准确率 |\n")
            f.write("|---|---|---|---|\n")
            for factor, data in sorted(attr_stats["factor_accuracy"].items(),
                                       key=lambda x: x[1]["correct"] + x[1]["wrong"], reverse=True):
                total = data["correct"] + data["wrong"]
                acc = data["correct"] / total if total > 0 else 0
                f.write(f"| {factor} | {data['correct']} | {data['wrong']} | {acc*100:.1f}% |\n")
        else:
            f.write("无归因数据\n")

    logger.info(f'  报告已生成: {output_path}')

def run_phase(dates, phase_name, db, stock_mgr, global_weights, all_weight_changes,
              shadow=False, alpha=0.8):
    """执行一个阶段的 walk-forward 训练。"""
    logger.info('%s', '=' * 60)
    logger.info(f'  Walk-Forward: {dates[0]} → {dates[-1]} ({len(dates)}天) [{phase_name}]')
    logger.info('%s', '=' * 60)

    # 1. 获取股票名单：从最晚训练日向前回溯，取首个能取到非空名单的日期。
    #    未来/休市日期 query_all_stock 会返回空，需回退到最近的交易日，
    #    否则 all_stocks 为空导致全程无数据。
    today_str = datetime.now().strftime("%Y-%m-%d")
    all_stocks = []
    stock_ref_date = None
    for d in sorted(dates, reverse=True):
        if d > today_str:
            continue
        s = get_stocks(d)
        if s:
            all_stocks = s
            stock_ref_date = d
            break
    if not all_stocks:
        logger.info('  [警告] 未能取到任何股票名单（可能所选日期均无行情），阶段跳过')
        logger.info(f'  共获取股票名单: {len(all_stocks)} 只')
        return {"phase": phase_name, "dates": dates, "daily": []}

    # 2. 预取 K 线（start=阶段首-90天, end=阶段尾+21天，覆盖 60 交易日回看 + 14 交易日未来）
    # 注意：prefetch_first 用原始阶段首日（而非 from-date 过滤后的 dates[0]），
    # 确保缓存key一致 + 历史推荐收益更新不受限
    if phase_name == "2026年上半年":
        prefetch_first = "2026-01-05"
        prefetch_last = "2026-06-30"
    else:
        prefetch_first = dates[0]
        prefetch_last = dates[-1]
    start = (datetime.strptime(prefetch_first, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    end = (datetime.strptime(prefetch_last, "%Y-%m-%d") + timedelta(days=21)).strftime("%Y-%m-%d")
    kc = prefetch(all_stocks, start, end)

    # 2b. 预取大盘指数 K 线（沪深300）作为相对强弱的市场基准
    market_kc = None
    try:
        _, market_kc = fetch_kline("sh.000300", start, end)
    except Exception:
        logger.warning("  指数 K 线获取失败，相对强弱降级为仅个股自身强度")
        market_kc = None

    # 3. 预热 per-stock 权重缓存
    stock_mgr.warmup_cache()

    # 4. 初始化 FeedbackLoop（传入 stock_weight_manager）
    fb = FeedbackLoop({}, stock_weight_manager=stock_mgr)

    # 5. 逐日 walk-forward
    daily_results = []
    for i, today in enumerate(dates):
        logger.info(f'\n  [{i + 1}/{len(dates)}] {today}:')

        # 5a. 计算 21 维 sub_scores
        candidates = compute_sub_scores(today, all_stocks, kc, market_kc)
        if not candidates:
            logger.info(' 无数据')
            continue

        # 5b. 批量加载 per-stock 权重
        codes = [c["code"] for c in candidates]
        weights_by_code = stock_mgr.get_weights_batch(codes)
        for c in candidates:
            c["stock_weights"] = weights_by_code.get(c["code"], global_weights)

        # 5c. 截面评分（per-stock 权重加权）
        score_cross_sectional_per_stock(candidates, global_weights)

        # 5c2. 评分区分度监控
        dist = monitor_score_distribution(candidates)
        logger.info(f' 评分分布: gap_top_median={dist["gap_top_median"]:.1f}, '
                    f'std={dist["std"]:.2f}, status={dist["status"]}')

        # 5c3. P7B: ML 评分融合（加载已训练模型，综合评分 = 现有 × α + ML × (1-α)）
        ml_selector = MLStockSelector()
        logger.info(f'  ML加载状态: model={type(ml_selector.model).__name__ if ml_selector.model is not None else "None"}, '
                    f'n_features={len(ml_selector.feature_names)}')
        if ml_selector.model is not None:
            ml_count = 0
            for c in candidates:
                pred = ml_selector.predict(c.get("factors", {}))
                ml_score = pred.get("ml_score", 50)
                c["ml_score"] = ml_score
                c["score"] = round(c["score"] * alpha + ml_score * (1 - alpha), 1)
                if pred.get("ml_prediction") == "bullish":
                    ml_count += 1
            logger.info(f'  ML融合(α={alpha}): {ml_count}/{len(candidates)} bullish, '
                        f'模型={ml_selector.model_type}')

        # 5d. 选 top N + 写 recommendations + factor_snapshots
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:TOP_N]
        for rank, c in enumerate(selected, 1):
            db.save_recommendation(
                today, c["code"], c["name"], c["score"], c["price"],
                rank=rank, mode="feedback_training",
                factors={"z_scores": c.get("z_scores", {})},
            )
            db.save_factor_snapshot(
                today, c["code"], c["name"], c["score"],
                sub_scores=c["sub_scores"],
                factors=c.get("factors", {}),
            )

        # 5e. 补算历史推荐 T+N 收益
        n_updated = update_future_returns(db, kc, today)

        # 5f. 反馈循环（归因 + IC + 个股调权+ML训练，skip_weight_adjustment=True）
        fb_result = fb.run_all(today=today, skip_ml=False, skip_weight_adjustment=True)

        # 5f2. 影子 ML 模式（不参与决策，仅记录对比数据）
        if shadow:
            shadow_model = train_shadow_ml(today, db=db, kc=kc)
            if shadow_model is not None:
                ml_scores = []
                for c in candidates:
                    pred = shadow_model.predict(c.get("factors", {}))
                    ml_scores.append({
                        "code": c["code"],
                        "name": c["name"],
                        "ml_score": pred["ml_score"],
                        "ml_probability": pred["ml_probability"],
                        "ml_prediction": pred["ml_prediction"],
                        "existing_score": c.get("score", 50),
                        "existing_rank": 0,
                        "ml_rank": 0,
                    })
                ml_scores.sort(key=lambda x: x["ml_score"], reverse=True)
                for i, s in enumerate(ml_scores):
                    s["ml_rank"] = i + 1
                existing_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
                rank_map = {c["code"]: i + 1 for i, c in enumerate(existing_sorted)}
                for s in ml_scores:
                    s["existing_rank"] = rank_map.get(s["code"], 999)
                db.save_shadow_ml_scores(today, ml_scores)
                ml_top_n = [s["code"] for s in ml_scores[:TOP_N]]
                existing_top_n = [c["code"] for c in selected]
                overlap = len(set(ml_top_n) & set(existing_top_n))
                logger.info(f'  影子ML: top{TOP_N}重合={overlap}/{TOP_N}, '
                            f'ML_最高={ml_scores[0]["ml_score"]}, '
                            f'样本数={len(ml_scores)}')

        # 5g. 收集个股权重变化（用于报告）
        stock_changes = fb_result.get("stock_weight_changes") or {}
        for code, changes in stock_changes.items():
            for ch in changes:
                all_weight_changes.append({
                    "code": code,
                    "factor": ch.get("factor", ""),
                    "old": ch.get("old_weight", 0),
                    "new": ch.get("new_weight", 0),
                    "change": (ch.get("new_weight", 0) or 0) - (ch.get("old_weight", 0) or 0),
                    "reason": ch.get("reason", ""),
                    "date": today,
                })

        model_metrics = fb_result.get("model_metrics")
        if model_metrics:
            acc = model_metrics.get("accuracy", 0)
            logger.info(f'  ML模型重训练: acc={acc:.2%}, type={model_metrics.get("model_type", "?")}')
        daily_results.append({
            "date": today,
            "n_candidates": len(candidates),
            "top_n_score": round(sum(c["score"] for c in selected) / TOP_N, 1),
            "n_returns_updated": n_updated,
            "n_attributions": fb_result.get("n_attributions", 0),
            "n_stock_weight_changes": sum(len(v) for v in stock_changes.values()),
            "score_gap": dist["gap_top_median"],
            "score_std": round(dist["std"], 2),
            "score_status": dist["status"],
        })
        logger.info(f" 候选{len(candidates)}→推荐10, 收益更新{n_updated}, 归因{fb_result.get('n_attributions', 0)}, 调权{sum((len(v) for v in stock_changes.values()))}")  # 6. 生成报告
    phase_result = {"phase": phase_name, "dates": dates, "daily": daily_results}
    report_path = os.path.join("reports", "feedback", f"feedback_training_{phase_name}.md")
    generate_report(db, phase_result, all_weight_changes, report_path)

    return phase_result


def main():
    parser = argparse.ArgumentParser(description="反馈训练 Walk-Forward")
    parser.add_argument("--smoke-test", action="store_true",
                        help="小样本验证（10只×3天）")
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="运行指定阶段")
    parser.add_argument("--all", action="store_true",
                        help="运行全部两阶段")
    parser.add_argument("--shadow", action="store_true",
                        help="启用影子 ML 模式（训练独立模型，不参与决策）")
    parser.add_argument("--ml-alpha", type=float, default=0.8,
                        help="ML 评分融合系数: 综合=现有×α + ML×(1-α) (默认0.8)")
    parser.add_argument("--from-date", type=str, default=None,
                        help="从指定日期开始（跳过之前的日期，用于续跑）")
    args = parser.parse_args()

    if not (args.smoke_test or args.phase or args.all):
        parser.print_help()
        return

    # 初始化 baostock
    bs.login()

    try:
        db = StockDatabase()

        # 加载全局权重（config.yaml）
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        global_weights = dict(config.get("weights", {}))
        # 保证 21 维齐全
        for fac in ALL_FACTORS:
            global_weights.setdefault(fac, 0.0)
        global_weights = _normalize(global_weights)
        logger.info(f'  全局权重: {len(global_weights)} 维, sum={sum(global_weights.values()):.4f}')
        stock_mgr = StockWeightManager(db, global_weights)
        all_weight_changes = []  # 跨阶段累积个股权重变化（用于报告）

        if args.smoke_test:
            # 小样本：10 只股票 × 3 天（6/29, 6/30, 7/1）
            smoke_dates = DATES_PHASE1[:3]
            logger.warning('\n[冒烟测试] 10 只股票 × 3 天')
            # 覆盖 get_stocks 只返回前 10 只
            _orig_get_stocks = get_stocks
            globals()["get_stocks"] = lambda d: _orig_get_stocks(d)[:10]
            try:
                run_phase(smoke_dates, "冒烟测试", db, stock_mgr,
                          global_weights, all_weight_changes,
                          shadow=args.shadow, alpha=args.ml_alpha)
            finally:
                globals()["get_stocks"] = _orig_get_stocks
        elif args.phase == 1:
            run_phase(DATES_PHASE1, "2026年7月", db, stock_mgr,
                      global_weights, all_weight_changes,
                      shadow=args.shadow, alpha=args.ml_alpha)
        elif args.phase == 2:
            if not DATES_PHASE2:
                logger.info('\n[阶段2] 获取 2026/01~06 交易日列表...')
                DATES_PHASE2.extend(fetch_trade_dates("2026-01-01", "2026-06-30"))
                logger.info(f'  共 {len(DATES_PHASE2)} 个交易日')
            phase2_dates = DATES_PHASE2
            if args.from_date:
                phase2_dates = [d for d in DATES_PHASE2 if d >= args.from_date]
                logger.info(f'  从{args.from_date}开始，共{len(phase2_dates)}个交易日')
            run_phase(phase2_dates, "2026年上半年", db, stock_mgr,
                      global_weights, all_weight_changes,
                      shadow=args.shadow, alpha=args.ml_alpha)
        elif args.all:
            run_phase(DATES_PHASE1, "2026年7月", db, stock_mgr,
                      global_weights, all_weight_changes,
                      shadow=args.shadow, alpha=args.ml_alpha)
            if not DATES_PHASE2:
                logger.info('\n[阶段2] 获取 2026/01~06 交易日列表...')
                DATES_PHASE2.extend(fetch_trade_dates("2026-01-01", "2026-06-30"))
                logger.info(f'  共 {len(DATES_PHASE2)} 个交易日')
            run_phase(DATES_PHASE2, "2026年上半年", db, stock_mgr,
                      global_weights, all_weight_changes,
                      shadow=args.shadow, alpha=args.ml_alpha)
    finally:
        bs.logout()
        logger.info('\n✅ 完成!')

if __name__ == "__main__":
    main()
