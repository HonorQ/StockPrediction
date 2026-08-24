"""
Walk-Forward 回测 + 反馈循环 (优化版)
一次性预取K线，内存中切片模拟逐日仿真

说明：回测复用实盘 StockScorer 评分链路（get_sub_scores → penalty_expectation →
batch_normalize），保证回测与实盘打分逻辑一致，避免"两套代码漂移"导致回测失真。
"""
from src.utils.log import setup_logging, get_logger
setup_logging()
logger = get_logger()

import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'src')
import os, time, pandas as pd, baostock as bs
import yaml, atexit, pickle, argparse
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

# 预取起始日：回测首日往前约100个自然日，保证 EMA26/MACD/BOLL/MA20 等指标
# 在回测首日即有足够历史收敛（否则首日因子失真，与实盘打分不一致）。
PREFETCH_START = (datetime.strptime(DATES[0], "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")

# 加载与实盘同源的权重（config.yaml 为静态先验，不被反馈循环改写）
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


def _to_float(v, default=0.0):
    """baostock 停牌日返回空字符串/None，直接 float() 会崩"""
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# K线本地缓存：全市场预取约需45分钟(baostock限流+断连)，落盘后重跑秒级恢复
def _kline_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cache", f"bt_kline_{DATES[0]}_{DATES[-1]}.pkl")


def _load_fetch1year_cache(codes):
    """复用 scripts/fetch_1year_data.py 已落盘的按股K线缓存(cache/klines/*.pkl)。
    文件名为 md5(code)[:16]，格式与 baostock 行一致。
    仅采用完整覆盖 [PREFETCH_START, DATES[-1]] 的文件。"""
    import hashlib
    kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "klines")
    out = {}
    for code in codes:
        key = hashlib.md5(code.encode()).hexdigest()[:16]
        p = os.path.join(kdir, f"1year_{key}.pkl")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                rows = pickle.load(f)
            if rows and rows[0][0] <= PREFETCH_START and rows[-1][0] >= DATES[-1]:
                out[code] = rows
        except Exception:
            continue
    return out


def prefetch_with_cache(stocks, start, end, refresh=False) -> dict:
    path = _kline_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            kc = pickle.load(f)
        logger.info(f'  从缓存加载 {len(kc)} 只K线: {path}')
        return kc

    # 一级复用：fetch_1year_data 的按股缓存，只补拉缺失股票
    local = _load_fetch1year_cache([c for c, _ in stocks]) if not refresh else {}
    missing = [(c, n) for c, n in stocks if c not in local]
    logger.info(f'  本地复用 {len(local)} 只, 需在线拉取 {len(missing)} 只')
    fetched = prefetch(missing, start, end) if missing else {}

    kc = dict(local)
    for c, rows in fetched.items():
        if rows:
            kc[c] = rows

    with open(path, "wb") as f:
        pickle.dump(kc, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'  K线已缓存({len(kc)} 只): {path}')
    return kc

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
    """回测因子计算：复用实盘 StockScorer 评分链路，保证回测与实盘打分一致。

    - K线类因子用实盘同一个 TechnicalFactors 计算（数据截至回测日，杜绝未来函数）；
    - 相对强弱用K线收益补充；
    - 新闻/基本面/资金流/龙虎榜等回测历史时点拿不到的维度，走 get_sub_scores 内部
      默认中性值（与实盘"数据源缺失降级中性"规则一致）；
    - 最终得分经 batch_normalize 截面归一化（与实盘 run_premarket 完全相同的逻辑）。
    """
    results = []
    for code, name in stocks:
        raw = kc.get(code)
        if not raw: continue
        rows = [r for r in raw if r[0] <= date]  # 仅取截至回测日的K线，无前视
        if len(rows) < 20: continue
        # baostock 字段顺序: date,close,volume,turn,pctChg,open,high,low
        # 停牌日 volume/turn 等字段为空字符串 → 统一经 _to_float 防崩；
        # 收盘价无效(停牌未开盘)的行整行跳过，避免0价格污染因子
        parsed = []
        for r in rows:
            c = _to_float(r[1], default=0.0)
            if c <= 0:
                continue
            parsed.append({
                "date": r[0], "close": c,
                "volume": _to_float(r[2]), "turn": _to_float(r[3]),
                "pctChg": _to_float(r[4]),
                "open": _to_float(r[5], default=c),
                "high": _to_float(r[6], default=c),
                "low": _to_float(r[7], default=c),
            })
        if len(parsed) < 20:
            continue
        df = pd.DataFrame({
            "开盘": [x["open"] for x in parsed],
            "最高": [x["high"] for x in parsed],
            "最低": [x["low"] for x in parsed],
            "收盘": [x["close"] for x in parsed],
            "成交量": [x["volume"] for x in parsed],
        })
        close = df["收盘"].values
        n = len(close)

        # 复用实盘技术因子（与 run_premarket 同一个类）
        all_factors = TechnicalFactors(df).calculate_all()
        # 相对强弱（回测可由K线收益计算）
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

    # 与实盘一致：截面Z-Score归一化（含正交化），写入最终综合分 score
    StockScorer.batch_normalize(results, scorer.weights)
    return results

def future_returns(kc):
    """用 T+N 真实收盘价回看收益（r[1]=close，仅取 date 之后的K线，无前视）。"""
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
                ep = _to_float(future[off-1][1])
                if ep > 0: db.update_performance(date, code, **{col: round((ep/entry-1)*100, 2)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="强制重新预取K线(忽略本地缓存)")
    args = ap.parse_args()

    logger.info('%s', '=' * 60)
    logger.info(f'  Walk-Forward: {DATES[0]} → {DATES[-1]} ({len(DATES)}天)')
    logger.info('%s', '=' * 60)

    logger.info('\n[1/4] 获取股票名单')
    all_stocks = get_stocks(DATES[-1])
    logger.info(f'  候选: {len(all_stocks)} 只')

    logger.info(f'\n[2/4] 预取K线 ({PREFETCH_START}~{DATES[-1]})')
    kc = prefetch_with_cache(all_stocks, PREFETCH_START, DATES[-1], refresh=args.refresh)

    logger.info('\n[3/4] 逐日模拟')
    for i, d in enumerate(DATES):
        logger.info(f'\n  [{i + 1}/{len(DATES)}] {d}: 计算因子...')
        items = factors(d, all_stocks, kc)
        if not items:
            logger.info('无数据')
            continue
        # 按实盘综合评分排序（batch_normalize 已写入 score）
        items.sort(key=lambda x: x["score"], reverse=True)
        top10 = items[:10]
        logger.info(f' {len(items)}只→推荐10只')
        for rank, s in enumerate(top10, 1):
            db.save_recommendation(d, s["code"], s["name"], s["score"], s["price"],
                                   rank=rank, mode="backtest")

    logger.info('\n[4/4] 计算收益+反馈循环')
    future_returns(kc)
    # 回测日传入反馈循环，归因/IC/调权仅引用截至当日的记录，杜绝权重前视
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
        decay = ic.get("decay_factors", {})
        if decay:
            for n, d in decay.items():
                logger.info(f"  IC衰减: {n} {d['ic_all']:.3f}→{d['ic_recent']:.3f}")

    ch = fb._adjust_weights(today=bt_today)
    if ch:
        for c in ch:
            logger.info(f"  调权: {c['factor']} {c['old_weight']}→{c['new_weight']} ({c['reason']})")

    # 回测跳过 ML 重训练：避免用回测期快照污染生产模型库
    logger.info('  (回测模式跳过 ML 重训练)')

    recs = db.get_recent_recommendations(days=60)
    r5 = [r for r in recs if r.get("d5_return") is not None]
    if r5:
        wins = sum(1 for r in r5 if r["d5_return"] > 0)
        avg_r = sum(r["d5_return"] for r in r5) / len(r5)
        logger.info(f'\n  总推荐: {len(recs)}, 有d5收益: {len(r5)}, 胜率: {wins}/{len(r5)}={wins / len(r5) * 100:.1f}%, 平均d5: {avg_r:.2f}%')
    logger.info('\n✅ 完成!')

if __name__ == "__main__":
    main()
