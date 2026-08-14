"""
Walk-Forward 回测（纯随机选股基线）
与 run_backtest.py 完全相同，唯一区别：选股逻辑改为随机选10只
"""
from src.utils.log import setup_logging, get_logger
setup_logging()
logger = get_logger()

import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'src')
import os, json, time, numpy as np, pandas as pd, baostock as bs
import yaml, atexit, random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.database import StockDatabase
from src.engine.factor_attribution import AttributionAggregator
from src.engine.feedback_loop import FeedbackLoop
from src.factors.technical import TechnicalFactors
from src.engine.scorer import StockScorer

db = StockDatabase()
bs.login()
atexit.register(bs.logout)

DATES = ['2026-06-29','2026-06-30','2026-07-01','2026-07-02','2026-07-03',
         '2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10',
         '2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17',
         '2026-07-20','2026-07-21','2026-07-22']

PREFETCH_START = (datetime.strptime(DATES[0], "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    WEIGHTS = yaml.safe_load(_f).get("weights", {})
scorer = StockScorer(WEIGHTS)

def fetch_kline(code, start, end):
    rs = bs.query_history_k_data_plus(code, "date,close,volume,turn,pctChg,open,high,low",
        start_date=start, end_date=end, frequency="d", adjustflag="3")
    rows = []
    while rs.next(): rows.append(rs.get_row_data())
    return code, rows

def prefetch(stocks, start, end):
    data = {}
    logger.info(f'  预取 {len(stocks)} 只K线...')
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(fetch_kline, c, start, end): c for c, _ in stocks}
        for i, f in enumerate(as_completed(fut), 1):
            c, r = f.result()
            if r: data[c] = r
            if i % 1000 == 0: logger.info(f'    {i}/{len(stocks)} ({time.time() - t0:.0f}s)')
    logger.info(f'  完成 {len(data)} 只 ({time.time() - t0:.0f}s)')
    return data

def get_stocks(date):
    rs = bs.query_all_stock(date); stocks = []
    while rs.next(): stocks.append(rs.get_row_data())
    result = []
    for s in stocks:
        code, status, name = s[0], s[1], s[2]
        if status != "1": continue
        if name.startswith(("*", "ST")): continue
        if any(code.startswith(p) for p in ["bj","sh.688","sh.000","sh.399","sz.399","sz.300"]): continue
        result.append((code, name))
    return result

def factors(date, stocks, kc):
    results = []
    for code, name in stocks:
        raw = kc.get(code)
        if not raw: continue
        rows = [r for r in raw if r[0] <= date]
        if len(rows) < 20: continue
        df = pd.DataFrame({
            "开盘": [float(r[5]) for r in rows],
            "最高": [float(r[6]) for r in rows],
            "最低": [float(r[7]) for r in rows],
            "收盘": [float(r[1]) for r in rows],
            "成交量": [float(r[2]) or 0 for r in rows],
        })
        close = df["收盘"].values
        n = len(close)

        all_factors = TechnicalFactors(df).calculate_all()
        all_factors["relative_strength_5d"] = (close[-1] / close[-6] - 1) * 100 if n >= 6 else 0
        all_factors["relative_strength_20d"] = (close[-1] / close[-21] - 1) * 100 if n >= 21 else 0

        sub_scores = scorer.get_sub_scores(all_factors)
        expectation_penalty = scorer.penalty_expectation(all_factors)
        results.append({
            "code": code.replace("sh.", "sh").replace("sz.", "sz"),
            "name": name,
            "price": float(close[-1]),
            "sub_scores": sub_scores,
            "expectation_penalty": expectation_penalty,
        })

    StockScorer.batch_normalize(results, scorer.weights)
    return results

def future_returns(kc):
    for rec in db.get_recent_recommendations(days=60):
        date, code = rec.get("trade_date",""), rec.get("code","")
        if rec.get("d3_return") is not None: continue
        rows = kc.get(code.replace("sh","sh.").replace("sz","sz."))
        if not rows: continue
        entry = rec.get("current_price", 0)
        if entry <= 0: continue
        future = [r for r in rows if r[0] > date]
        for off, col in [(3,"d3_return"),(5,"d5_return"),(10,"d10_return")]:
            if len(future) >= off:
                ep = float(future[off-1][1])
                if ep > 0: db.update_performance(date, code, **{col: round((ep/entry-1)*100, 2)})

def main():
    logger.info('%s', '=' * 60)
    logger.info(f'  Walk-Forward 随机基线: {DATES[0]} → {DATES[-1]} ({len(DATES)}天)')
    logger.info('%s', '=' * 60)

    logger.info('\n[1/4] 获取股票名单')
    all_stocks = get_stocks(DATES[-1])
    logger.info(f'  候选: {len(all_stocks)} 只')

    logger.info(f'\n[2/4] 预取K线 ({PREFETCH_START}~{DATES[-1]})')
    kc = prefetch(all_stocks, PREFETCH_START, DATES[-1])

    logger.info('\n[3/4] 逐日模拟')
    for i, d in enumerate(DATES):
        logger.info(f'\n  [{i + 1}/{len(DATES)}] {d}: 计算因子...')
        items = factors(d, all_stocks, kc)
        if not items:
            logger.info('无数据')
            continue

        # 唯一与 run_backtest.py 不同：随机选10只，而非按评分排序选Top10
        selected = random.sample(items, min(10, len(items)))
        logger.info(f' {len(items)}只→随机选{len(selected)}只')
        for rank, s in enumerate(selected, 1):
            db.save_recommendation(d, s["code"], s["name"], s["score"], s["price"],
                                   rank=rank, mode="backtest_random")

    logger.info('\n[4/4] 计算收益+反馈循环')
    future_returns(kc)
    bt_today = DATES[-1]
    fb = FeedbackLoop({})
    n_attr = fb._run_attribution(today=bt_today)
    logger.info(f'  归因: {n_attr} 条')
    attr = db.get_attribution_history(days=60)
    if attr:
        agg = AttributionAggregator(attr)
        logger.info(f'\n  {agg.summarize()}')

    ic = fb._run_ic_analysis(today=bt_today)
    if ic.get("status") == "ok":
        top = sorted(ic.get("factors",{}).items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        logger.info(f"\n  Top5 IC: {', '.join((f'{k}={v:.3f}' for k, v in top))}")

    recs = db.get_recent_recommendations(days=60)
    r5 = [r for r in recs if r.get("d5_return") is not None]
    if r5:
        wins = sum(1 for r in r5 if r["d5_return"] > 0)
        avg_r = sum(r["d5_return"] for r in r5) / len(r5)
        logger.info(f'\n  总推荐: {len(recs)}, 有d5收益: {len(r5)}, 胜率: {wins}/{len(r5)}={wins / len(r5) * 100:.1f}%, 平均d5: {avg_r:.2f}%')
    logger.info('\n✅ 完成!')

if __name__ == "__main__":
    main()
