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
import yaml, atexit, pickle, argparse, json
import socket as _socket
from datetime import datetime, timedelta

# 全局网络超时：防止数据源半死连接导致整轮回测无限挂起
# (2026-08-25 案例：baostock login 后 query 无响应，进程 CPU 0 闲置挂起)
_socket.setdefaulttimeout(20)
from src.data.database import StockDatabase
from src.engine.factor_attribution import AttributionAggregator
from src.engine.feedback_loop import FeedbackLoop
from src.factors.technical import TechnicalFactors
from src.engine.scorer import StockScorer

db = StockDatabase()
bs.login()
atexit.register(bs.logout)

# 默认回测窗口（可用 --start/--end 覆盖）
DEFAULT_START, DEFAULT_END = '2026-06-29', '2026-07-22'
WINDOW_START, WINDOW_END = DEFAULT_START, DEFAULT_END

# 逐日模拟的交易日列表，main() 中从K线数据的日期集合推导
DATES = [DEFAULT_START]

# 预取起始日：回测首日往前约100个自然日，保证 EMA26/MACD/BOLL/MA20 等指标
# 在回测首日即有足够历史收敛（否则首日因子失真，与实盘打分不一致）。
PREFETCH_START = (datetime.strptime(DEFAULT_START, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")


def _set_window(start: str, end: str):
    """设置回测窗口（--start/--end），同步预取起点"""
    global WINDOW_START, WINDOW_END, PREFETCH_START, DATES
    WINDOW_START, WINDOW_END = start, end
    DATES = [start]
    PREFETCH_START = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")

# 加载与实盘同源的权重（config.yaml 为静态先验，不被反馈循环改写）
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    WEIGHTS = yaml.safe_load(_f).get("weights", {})
scorer = StockScorer(WEIGHTS)

# 推荐写入模式：--invert-signals 时用 backtest_invert，
# 与常规基线(backtest)并存于 recommendations 表便于 A/B 对比
REC_MODE = "backtest"


def enable_invert_signals():
    """反转假设实验：对负 IC 维度(trend/strategy/quant_behavior)取反打分"""
    global scorer, REC_MODE
    scorer = StockScorer(WEIGHTS, invert_signals=True)
    REC_MODE = "backtest_invert"

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
    except (ValueError, TypeError) as e:
        logger.warning("捕获异常已忽略(run_backtest.py:81): %s", e)
        return default


# K线本地缓存：全市场预取约需45分钟(baostock限流+断连)，落盘后重跑秒级恢复
def _kline_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cache", f"bt_kline_{WINDOW_START}_{WINDOW_END}.pkl")


def _load_fetch1year_cache(codes):
    """复用 scripts/fetch_1year_data.py 已落盘的按股K线缓存(cache/klines/*.pkl)。
    文件名为 md5(code)[:16]，格式与 baostock 行一致。
    覆盖校验放宽为：首日 <= 窗口起点-85自然日(≈60个交易日，技术指标收敛
    所需)，且尾部 >= WINDOW_END。原严格校验(PREFETCH_START=窗口前100天)
    导致 2026 年起的1年缓存对所有更早窗口全部失效。
    code→hash 映射优先读 _code_index.json(由反查工具生成)，否则现场计算。"""
    import hashlib
    kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "klines")
    index_path = os.path.join(kdir, "_code_index.json")
    known_codes = None
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                known_codes = set(json.load(f))
        except Exception as e:
            logger.warning("读取指数已知代码文件失败, 回测将不按已知列表过滤: %s", e)
            known_codes = None

    min_start = (datetime.strptime(WINDOW_START, "%Y-%m-%d") - timedelta(days=85)).strftime("%Y-%m-%d")
    out = {}
    for code in codes:
        if known_codes is not None and code not in known_codes:
            continue
        key = hashlib.md5(code.encode()).hexdigest()[:16]
        p = os.path.join(kdir, f"1year_{key}.pkl")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                rows = pickle.load(f)
            if rows and rows[0][0] <= min_start and rows[-1][0] >= WINDOW_END:
                out[code] = rows
        except Exception as e:
            logger.warning("加载缓存K线失败 %s: %s", p, e)
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

    if not kc:
        # 全部拉取失败(如 baostock 限流)：不落盘坏缓存，下次直接重试
        logger.info('  ⚠️ 未获得任何K线数据，跳过缓存落盘')
        return kc

    with open(path, "wb") as f:
        pickle.dump(kc, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'  K线已缓存({len(kc)} 只): {path}')
    return kc

def prefetch(stocks, start, end):
    """预取K线。必须串行：baostock 的 query/next 共享一个全局 socket，
    多线程并发读取会产生 utf-8 解码错误与 10054 断连(实测)，且失败重试
    反而拖慢总耗时。单线程顺序拉取约 0.2s/只，3000 只约 10 分钟。"""
    data = {}
    logger.info(f'  预取 {len(stocks)} 只K线(串行)...')
    t0 = time.time()
    fails = 0
    for i, (c, _) in enumerate(stocks, 1):
        try:
            _, r = fetch_kline(c, start, end)
            if r:
                data[c] = r
        except Exception as e:
            logger.warning("获取 %s K线失败: %s", c, e)
            fails += 1
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1)
            eta = (len(stocks) - i) / max(rate, 0.01)
            logger.info(f'    {i}/{len(stocks)} ({time.time() - t0:.0f}s, {rate:.1f}只/s, ETA {eta:.0f}s, 失败{fails})')
    logger.info(f'  完成 {len(data)} 只 ({time.time() - t0:.0f}s, 失败 {fails})')
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
        # update_performance 的关键字是 d3/d5/d10(非列名 dN_return)
        for off, kw in [(3, "d3"), (5, "d5"), (10, "d10")]:
            if len(future) >= off:
                ep = _to_float(future[off-1][1])
                if ep > 0: db.update_performance(date, code, **{kw: round((ep/entry-1)*100, 2)})

def _load_local_stock_list():
    """baostock 限流时的兜底：读取反查工具生成的本地缓存股票清单"""
    idx = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cache", "klines", "_code_index.json")
    if not os.path.exists(idx):
        return []
    try:
        with open(idx, "r", encoding="utf-8") as f:
            codes = json.load(f)
        return [(c, "") for c in codes]
    except Exception as e:
        logger.warning("读取指数成分文件失败: %s", e)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="强制重新预取K线(忽略本地缓存)")
    ap.add_argument("--start", default=DEFAULT_START, help="回测窗口起点 YYYY-MM-DD")
    ap.add_argument("--end", default=DEFAULT_END, help="回测窗口终点 YYYY-MM-DD")
    ap.add_argument("--invert-signals", action="store_true",
                    help="反转实验: 负IC维度(trend/strategy/quant)取反打分, "
                         "推荐写入 backtest_invert 模式并跳过反馈循环(不污染调权)")
    args = ap.parse_args()
    _set_window(args.start, args.end)
    if args.invert_signals:
        enable_invert_signals()
        logger.info('  🔁 反转模式已启用: 负IC维度取反打分, mode=backtest_invert')

    logger.info('%s', '=' * 60)
    logger.info(f'  Walk-Forward: {WINDOW_START} → {WINDOW_END}')
    logger.info('%s', '=' * 60)

    # 缓存优先：窗口K线已全部落盘时，名单直接从缓存推导，完全零网络。
    # (2026-08-25 案例：query_all_stock 对部分历史日期挂死且 baostock 内部
    #  吞掉 socket 超时，进程无限等待——绕开是唯一可靠解)
    # 匹配规则：精确文件名优先；否则扫描全部 bt_kline_*.pkl 解析日期范围，
    # 取能覆盖 [WINDOW_START, WINDOW_END] 的最大缓存——支持单日分片复用整窗缓存
    import glob
    import re as _re
    cache_path = _kline_cache_path()
    kc = None
    if not os.path.exists(cache_path) or args.refresh:
        best_path, best_n = None, -1
        for p in glob.glob(os.path.join(os.path.dirname(cache_path), "bt_kline_*.pkl")):
            m = _re.match(r"bt_kline_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.pkl",
                          os.path.basename(p))
            if not m:
                continue
            cs, ce = m.group(1), m.group(2)
            if cs <= WINDOW_START and ce >= WINDOW_END:
                n = os.path.getsize(p)
                if n > best_n:
                    best_path, best_n = p, n
        if best_path:
            cache_path = best_path

    if os.path.exists(cache_path) and not args.refresh:
        try:
            with open(cache_path, "rb") as f:
                kc = pickle.load(f)
            logger.info(f'\n[1/4] 从K线缓存加载 {len(kc)} 只: {cache_path}')
        except Exception as e:
            logger.info(f'  缓存读取失败({e})，转在线模式')
            kc = None

    if kc:
        all_stocks = [(c, "") for c in sorted(kc.keys())]
    else:
        logger.info('\n[1/4] 获取股票名单(在线)')
        all_stocks = get_stocks(WINDOW_END)
        if not all_stocks:
            local = _load_local_stock_list()
            if local:
                logger.info(f'  ⚠️ baostock 名单不可用，回退本地K线缓存清单: {len(local)} 只')
                all_stocks = local
        if not all_stocks:
            raise SystemExit('无法获取股票名单(baostock不可用且无本地缓存)，中止')
        logger.info(f'  候选: {len(all_stocks)} 只')
        logger.info(f'\n[2/4] 预取K线 ({PREFETCH_START}~{WINDOW_END})')
        kc = prefetch_with_cache(all_stocks, PREFETCH_START, WINDOW_END, refresh=args.refresh)

    if not kc:
        raise SystemExit('未获得任何K线数据，中止')
    if not kc:
        raise SystemExit('未获得任何K线数据，中止')

    # 从K线数据推导窗口内的交易日列表
    global DATES
    trade_days = sorted({r[0] for rows in kc.values() for r in rows
                         if r[0] and WINDOW_START <= r[0] <= WINDOW_END})
    DATES = trade_days

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
                                   rank=rank, mode=REC_MODE)

    logger.info('\n[4/4] 计算收益+反馈循环')
    future_returns(kc)
    # 回测日传入反馈循环，归因/IC/调权仅引用截至当日的记录，杜绝权重前视
    bt_today = DATES[-1]
    if REC_MODE == "backtest_invert":
        # 反转实验：只产出对比数据，不触发归因/调权(避免污染生产 weight_history)
        logger.info('  (反转实验模式: 跳过归因/IC/调权)')
    else:
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
