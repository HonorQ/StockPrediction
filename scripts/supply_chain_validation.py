# -*- coding: utf-8 -*-
"""美股→A股供应链联动验证

用全年美股个股数据 + A股供应链公司数据, 验证:
1. 英伟达涨>2% → A股英伟达供应链次日涨?
2. 特斯拉涨>2% → A股特斯拉供应链次日涨?
3. 苹果涨>2% → A股果链次日涨?

用法: python scripts/supply_chain_validation.py
"""
import os, json, numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# 加载供应链映射
with open(os.path.join(ROOT, "data", "us_a_supply_chain.json"), encoding="utf-8") as f:
    sc_data = json.load(f)
supply_chain = sc_data["supply_chain"]

# 加载美股数据
us_dir = os.path.join(ROOT, "cache", "us_daily")
us_sector = pd.read_csv(os.path.join(us_dir, "us_sector_returns.csv"), index_col=0)

# 加载A股面板
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

# 美股关键公司
us_stocks = {
    "NVDA": "英伟达",
    "TSLA": "特斯拉",
    "AAPL": "苹果",
    "AMD": "AMD",
    "MSFT": "微软",
    "META": "Meta",
}

def get_us_returns(symbol, days=300):
    """获取美股个股日收益率"""
    path = os.path.join(us_dir, f"{symbol}_daily.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    ret = df["close"].pct_change() * 100
    return ret.tail(days)

def get_a_share_returns(code, days=300):
    """获取A股日收益率(fwd5)"""
    # code格式: sh600000 → 面板中可能是 sh.600000
    panel_code = code[:2] + "." + code[2:]
    stock = panel[panel["code"] == panel_code]
    if len(stock) == 0:
        return None
    ret = stock.set_index("date")["fwd5"]
    return ret.tail(days)

print("=== 美股→A股供应链联动验证 ===\n")

results = []

for us_sym, us_name in us_stocks.items():
    if us_sym not in supply_chain:
        continue

    # 获取美股收益率
    us_ret = get_us_returns(us_sym)
    if us_ret is None:
        print(f"{us_name}: 无美股数据")
        continue

    suppliers = supply_chain[us_sym]["a_share_suppliers"]
    confirmed = [s for s in suppliers if s["degree"] == "confirmed"]

    print(f"--- {us_name}({us_sym}) → {len(confirmed)}家确认供应商 ---")

    # 收集所有供应商的平均收益
    supplier_daily_rets = []
    for s in confirmed:
        a_ret = get_a_share_returns(s["code"])
        if a_ret is not None:
            # 对齐日期
            common_dates = sorted(set(us_ret.index) & set(a_ret.index))
            if len(common_dates) < 50:
                continue
            aligned_us = us_ret.loc[common_dates]
            aligned_a = a_ret.loc[common_dates]
            supplier_daily_rets.append({
                "name": s["name"],
                "link": s["link"],
                "us": aligned_us,
                "a": aligned_a,
                "n": len(common_dates),
            })

    if not supplier_daily_rets:
        print("  无足够A股数据")
        continue

    # 分析1: 美股涨>2% → A股供应链次日
    print(f"\n  供应链分析({len(supplier_daily_rets)}家有数据):")

    all_lifts = []
    for sd in supplier_daily_rets:
        us_arr = sd["us"].values
        a_arr = sd["a"].values
        # 隔夜: 美股T日 → A股T+1日
        us_up = us_arr[:-1] > 2  # 美股涨>2%
        us_down = us_arr[:-1] < -2  # 美股跌>2%
        a_next = a_arr[1:]  # A股次日

        if us_up.sum() < 3 or us_down.sum() < 3:
            continue

        p_a_up = (a_next > 0).mean()
        p_a_up_given_up = ((a_next > 0) & us_up).sum() / us_up.sum()
        p_a_up_given_down = ((a_next > 0) & us_down).sum() / us_down.sum()
        lift = p_a_up_given_up - p_a_up

        avg_a_up = a_next[us_up].mean()
        avg_a_down = a_next[us_down].mean()

        results.append({
            "us_name": us_name,
            "us_sym": us_sym,
            "supplier": sd["name"],
            "link": sd["link"],
            "us_up_days": int(us_up.sum()),
            "us_down_days": int(us_down.sum()),
            "p_a_up_given_us_up": round(p_a_up_given_up, 3),
            "p_a_up_given_us_down": round(p_a_up_given_down, 3),
            "avg_a_given_us_up": round(float(avg_a_up), 3),
            "avg_a_given_us_down": round(float(avg_a_down), 3),
            "lift": round(float(lift), 3),
        })

        all_lifts.append(lift)

        flag = "**" if abs(lift) > 0.1 else ("*" if abs(lift) > 0.05 else "")
        print(f"    {sd['name']:<10} {sd['link']:<20} "
              f"美股涨>A股涨={p_a_up_given_up:.1%}({us_up.sum()}次) "
              f"跌>A股涨={p_a_up_given_down:.1%}({us_down.sum()}次) "
              f"lift={lift:+.1%} {flag}")

    # 汇总
    if all_lifts:
        avg_lift = np.mean(all_lifts)
        print(f"  平均lift: {avg_lift:+.1%}")

# 汇总表
print(f"\n{'='*80}")
print("供应链信号汇总")
print(f"{'='*80}")
print(f"{'美股':<8}{'供应商':<10}{'关系':<20}{'美股涨A涨':>10}{'美股跌A涨':>10}{'提升':>8}")
print("-" * 70)
for r in sorted(results, key=lambda x: -x["lift"]):
    flag = "**" if abs(r["lift"]) > 0.1 else ("*" if abs(r["lift"]) > 0.05 else "")
    print(f"  {r['us_name']:<6}{r['supplier']:<8}{r['link']:<18}"
          f"{r['p_a_up_given_us_up']:>9.1%}{r['p_a_up_given_us_down']:>9.1%}"
          f"{r['lift']:>+7.1%} {flag}")

# 结论
print("\n=== 结论 ===")
if results:
    strong = [r for r in results if r["lift"] > 0.1]
    print(f"lift>10%的供应链对: {len(strong)}/{len(results)}")
    for r in strong:
        print(f"  {r['us_name']}→{r['supplier']}({r['link']}): lift={r['lift']:+.1%}")

    # 按美股公司汇总
    by_us = {}
    for r in results:
        key = r["us_name"]
        if key not in by_us:
            by_us[key] = []
        by_us[key].append(r["lift"])
    print("\n按美股公司汇总平均lift:")
    for us, lifts in sorted(by_us.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(lifts)
        sig = "✅" if avg > 0.05 else ("⚠️" if avg > 0 else "❌")
        print(f"  {sig} {us}: {avg:+.1%} ({len(lifts)}家供应商)")

    # 保存
    outp = os.path.join(ROOT, "reports", "supply_chain_validation.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")
