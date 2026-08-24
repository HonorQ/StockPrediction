"""
批量回算历史因子截面 + 训练ML模型
流程:
  1. 从腾讯行情获取A股日K线（~2年，复用 MarketDataFetcher）
  2. 逐日回放计算因子（TechnicalFactors + 相对强弱等价格类因子）
  3. 计算T+N前向收益作为标注
  4. 保存到 database.factor_snapshots
  5. 训练LightGBM+RF集成模型，保存到 models/
"""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import yaml

from src.data.market_data import MarketDataFetcher
from src.data.database import StockDatabase
from src.factors.technical import TechnicalFactors
from src.engine.scorer import StockScorer
from src.engine.ml_selector import MLStockSelector

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, "config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    WEIGHTS = yaml.safe_load(f).get("weights", {})
scorer = StockScorer(WEIGHTS)
db = StockDatabase()

END_DATE = "2026-07-24"
START_DATE = "2025-07-01"
FORWARD_DAYS = [3, 5, 10, 14]


def get_stock_list():
    """从新浪实时行情获取股票列表（与 run_premarket 同源）"""
    mf = MarketDataFetcher()
    df = mf.get_a_share_realtime()
    if df.empty:
        raise RuntimeError("无法获取股票列表")
    stocks = []
    for _, row in df.iterrows():
        code = str(row.get("代码", "")).strip()
        name = str(row.get("名称", "")).strip()
        if not code or not name:
            continue
        if name.startswith(("*", "ST")): continue
        if code.startswith(("bj", "BJ")): continue
        if code.startswith(("300", "301")): continue
        if code.startswith(("688", "689")): continue
        if "ETF" in name.upper() or "指数" in name or "LOF" in name.upper():
            continue
        stocks.append((code, name, float(row.get("成交额", 0) or 0)))
    # 按成交额排序取前1500只（流动性过滤）
    stocks.sort(key=lambda x: x[2], reverse=True)
    stocks = stocks[:1500]
    return [(c, n) for c, n, _ in stocks]


def prefetch_history(stocks, days=500, max_workers=8):
    """并行预取K线"""
    mf = MarketDataFetcher()
    data = {}
    t0 = time.time()
    total = len(stocks)

    def fetch_one(code):
        try:
            df = mf.get_stock_history(code, days=days)
            if df is not None and not df.empty:
                return code, df
        except Exception:
            pass
        return code, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(fetch_one, c): c for c, _ in stocks}
        for i, f in enumerate(as_completed(fut), 1):
            code, df = f.result()
            if df is not None:
                data[code] = df
            if i % 100 == 0 or i == total:
                pct = i / total * 100
                print(f"  K线预取: {i}/{total} ({pct:.0f}%) | OK={len(data)} | {time.time()-t0:.0f}s")
    return data


def compute_factors_and_future_returns(code, kline_df, trade_date, kc):
    kline = kc.get(code)
    if kline is None or kline.empty:
        return None
    rows_upto = kline[kline["日期"] <= trade_date]
    if len(rows_upto) < 30:
        return None

    df = rows_upto.tail(400).reset_index(drop=True)
    close = df["收盘"].values
    n = len(close)

    factors = TechnicalFactors(df).calculate_all()
    factors["relative_strength_5d"] = (close[-1] / close[-6] - 1) * 100 if n >= 6 else 0
    factors["relative_strength_20d"] = (close[-1] / close[-21] - 1) * 100 if n >= 21 else 0

    all_kline = kc.get(code)
    future_prices = all_kline[all_kline["日期"] > trade_date]["收盘"].values
    labels = {}
    for fd in FORWARD_DAYS:
        if len(future_prices) >= fd:
            labels[f"future_{fd}d"] = (future_prices[fd - 1] / close[-1] - 1) * 100
        else:
            labels[f"future_{fd}d"] = None

    sub_scores = scorer.get_sub_scores(factors)
    score = sum(sub_scores.get(k, 50) * WEIGHTS.get(k, 0) for k in WEIGHTS)
    score = max(0, min(100, score))
    return {
        "code": code,
        "sub_scores": sub_scores,
        "factors": {**factors, **labels},
        "score": round(score, 1),
    }


def main():
    print("=" * 60)
    print("  批量回算训练数据生成")
    print(f"  日期: {START_DATE} → {END_DATE}")
    print("=" * 60)

    print("\n[1/5] 获取股票列表（新浪实时行情）...")
    stocks = get_stock_list()
    print(f"  {len(stocks)} 只股票")

    print("\n[2/5] 预取K线（腾讯行情, ~500天）...")
    kc = prefetch_history(stocks, days=600, max_workers=8)
    print(f"  ✅ 完成: {len(kc)} 只股票")

    print("\n[3/5] 生成交易日历...")
    all_dates = sorted(set(
        d for k in kc for d in kc[k]["日期"].tolist()
        if START_DATE <= d <= END_DATE
    ))
    print(f"  {len(all_dates)} 个交易日")

    print("\n[4/5] 逐日回算因子...")
    total_saved = 0
    t_start = time.time()
    for di, date in enumerate(all_dates):
        day_start = time.time()
        day_items = []
        for code, name in stocks:
            if code not in kc:
                continue
            result = compute_factors_and_future_returns(code, kc[code], date, kc)
            if result is None: continue
            result["name"] = name
            day_items.append(result)
        try:
            batch = []
            for item in day_items:
                batch.append({
                    "trade_date": date,
                    "code": item["code"],
                    "name": item["name"],
                    "score": item["score"],
                    "sub_scores": item["sub_scores"],
                    "factors": item["factors"],
                })
            db.save_factor_snapshots_batch(batch)
        except Exception as e:
            print(f"  ⚠️ 批次写入失败 {date}: {e}")
        total_saved += len(day_items)
        elapsed = time.time() - day_start
        if (di + 1) % 20 == 0 or di == 0 or di == len(all_dates) - 1:
            rate = total_saved / (time.time() - t_start + 0.1)
            pct = (di + 1) / len(all_dates) * 100
            print(f"  [{di+1}/{len(all_dates)}] {pct:.0f}% | {date} | {len(day_items)} stocks "
                  f"| total={total_saved} | {elapsed:.1f}s | {rate:.0f} rows/s")

    total_time = time.time() - t_start
    print(f"\n✅ 因子回算完成: {total_saved} 行, {total_time:.0f}s ({total_time/60:.1f}min)")

    print("\n[5/5] 训练ML模型...")
    try:
        selector = MLStockSelector()
        rows = db.get_factor_history(days=400)
        if not rows:
            print("  ⚠️ 数据库无因子快照，跳过训练")
            return
        print(f"  加载 {len(rows)} 条因子快照")
        training_data = []
        for r in rows:
            factors = r.get("factors", {})
            if isinstance(factors, str):
                factors = json.loads(factors)
            # 标签周期对齐实盘持有期（hold_days=3），用 future_3d 而非 future_14d
            future_3d = factors.get("future_3d")
            if future_3d is None: continue
            training_data.append({
                "factors": factors,
                "future_return": future_3d,
                "date": r.get("trade_date", ""),
            })
        print(f"  有效标注数据: {len(training_data)} 条")
        if training_data:
            # 市场中性化：按交易日分组，用当日个股 future_return 减去市场中位数
            # 使标签反映"跑赢市场"的选股 alpha，而非市场 beta（普涨日全正、普跌日全负）
            day_returns = defaultdict(list)
            for d in training_data:
                day_returns[d["date"]].append(d["future_return"])
            day_median = {
                day: float(np.median(rets))
                for day, rets in day_returns.items()
                if len(rets) >= 5  # 截面样本过少时基准不可靠，保留原值
            }
            neutralized = 0
            for d in training_data:
                median = day_median.get(d["date"])
                if median is not None:
                    d["future_return"] = d["future_return"] - median
                    neutralized += 1
            print(f"  市场中性化: {neutralized}/{len(training_data)} 条样本已减去当日中位数收益")

            df = selector.prepare_training_data(training_data, min_return=1.0)
            if not df.empty:
                # 传 dates 走时序验证（规则第6条要求，禁止 random_split 降级）
                dates = [d["date"] for d in training_data]
                train_result = selector.train(df, dates=dates)
                print("  ✅ 模型训练完成")
                print(f"    准确率: {train_result.get('accuracy', 'N/A')}")
                print(f"    F1: {train_result.get('f1', 'N/A')}")
                print(f"    特征: {len(train_result.get('feature_names', []))}")
    except Exception as e:
        print(f"  ⚠️ ML训练失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
