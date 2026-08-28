# -*- coding: utf-8 -*-
"""供应链完整验证: 拉取38家A股供应链公司数据 + 分析"""
import os, sys, json, time
import numpy as np, pandas as pd
import baostock as bs

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_DIR = os.path.join(ROOT, "cache", "us_daily")

# 加载供应链映射
with open(os.path.join(ROOT, "data", "us_a_supply_chain.json"), encoding="utf-8") as f:
    sc_data = json.load(f)
supply_chain = sc_data["supply_chain"]

# 初始化baostock
bs.login()

# 收集所有A股代码
a_codes = set()
for us_code, info in supply_chain.items():
    for s in info["a_share_suppliers"]:
        a_codes.add(s["code"])

print(f"供应链A股公司: {len(a_codes)}家\n")

# 拉取A股历史K线
print("拉取A股供应链公司历史K线...")
a_data = {}
for code in sorted(a_codes):
    # 转换为baostock格式
    bs_code = code[:2] + "." + code[2:]
    rs = bs.query_history_k_data_plus(
        bs_code, "date,close,volume,turn,pctChg",
        start_date="2025-08-01", end_date="2026-12-31",
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        df = pd.DataFrame(rows, columns=["date", "close", "volume", "turn", "pctChg"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["pctChg"] = pd.to_numeric(df["pctChg"], errors="coerce")
        df = df.dropna(subset=["close"])
        if len(df) > 50:
            a_data[code] = df
            print(f"  {code}: {len(df)}行")
    time.sleep(0.2)

bs.logout()
print(f"\n成功: {len(a_data)}/{len(a_codes)}家\n")

# 加载美股数据
us_stocks = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "META", "GOOGL", "AMZN"]
us_returns = {}
for sym in us_stocks:
    path = os.path.join(US_DIR, f"{sym}_daily.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.set_index("date")
        us_returns[sym] = df["close"].pct_change() * 100

# 分析
print("=" * 80)
print("供应链隔夜联动分析")
print("=" * 80)

all_results = []
for us_code, info in supply_chain.items():
    if us_code not in us_returns:
        continue
    us_ret = us_returns[us_code]
    suppliers = info["a_share_suppliers"]
    confirmed = [s for s in suppliers if s["degree"] == "confirmed"]

    print(f"\n--- {info['name']}({us_code}) → {len(confirmed)}家确认供应商 ---")

    for s in confirmed:
        code = s["code"]
        if code not in a_data:
            continue

        a_df = a_data[code]
        a_dates = set(a_df["date"].tolist())
        us_dates = set(us_ret.index.tolist())
        common = sorted(a_dates & us_dates)

        if len(common) < 30:
            continue

        us_arr = us_ret.loc[common].values
        a_arr = a_df.set_index("date").loc[common]["pctChg"].values

        # 隔夜: 美股T日 → A股T+1日
        us_t = us_arr[:-1]
        a_next = a_arr[1:]

        if len(us_t) < 30:
            continue

        # 统计
        us_up = us_t > 2
        us_down = us_t < -2
        us_up_1 = us_t > 1
        us_down_1 = us_t < -1

        p_base = (a_next > 0).mean()

        # 美股涨>2%
        if us_up.sum() >= 3:
            p_up2 = ((a_next > 0) & us_up).sum() / us_up.sum()
            avg_up2 = a_next[us_up].mean()
        else:
            p_up2, avg_up2 = None, None

        # 美股跌>2%
        if us_down.sum() >= 3:
            p_down2 = ((a_next > 0) & us_down).sum() / us_down.sum()
            avg_down2 = a_next[us_down].mean()
        else:
            p_down2, avg_down2 = None, None

        # 美股涨>1%
        if us_up_1.sum() >= 5:
            p_up1 = ((a_next > 0) & us_up_1).sum() / us_up_1.sum()
            avg_up1 = a_next[us_up_1].mean()
        else:
            p_up1, avg_up1 = None, None

        # lift
        lift2 = (p_up2 - p_base) if p_up2 is not None else None
        lift1 = (p_up1 - p_base) if p_up1 is not None else None

        # Pearson相关性
        corr = np.corrcoef(us_t, a_next)[0, 1]
        t_val = corr * np.sqrt(len(us_t) - 2) / np.sqrt(1 - corr ** 2) if abs(corr) < 1 else 0

        all_results.append({
            "us_name": info["name"],
            "us_sym": us_code,
            "supplier": s["name"],
            "code": code,
            "link": s["link"],
            "n": len(us_t),
            "pearson": round(corr, 4),
            "t_stat": round(t_val, 2),
            "us_up_2": {"p_a_up": round(p_up2, 3) if p_up2 else None,
                        "avg": round(avg_up2, 3) if avg_up2 else None,
                        "days": int(us_up.sum())} if p_up2 else None,
            "us_down_2": {"p_a_up": round(p_down2, 3) if p_down2 else None,
                          "avg": round(avg_down2, 3) if avg_down2 else None,
                          "days": int(us_down.sum())} if p_down2 else None,
            "us_up_1": {"p_a_up": round(p_up1, 3) if p_up1 else None,
                        "avg": round(avg_up1, 3) if avg_up1 else None,
                        "days": int(us_up_1.sum())} if p_up1 else None,
            "lift_2": round(lift2, 3) if lift2 else None,
            "lift_1": round(lift1, 3) if lift1 else None,
        })

        # 输出
        up2_str = f"{p_up2:.1%}({us_up.sum()}次)" if p_up2 else "N/A"
        down2_str = f"{p_down2:.1%}({us_down.sum()}次)" if p_down2 else "N/A"
        lift_str = f"{lift2:+.1%}" if lift2 else "N/A"
        sig = "*" if abs(t_val) > 1.64 else ""

        print(f"  {s['name']:<10} {s['link']:<20} "
              f"涨>2%→A涨={up2_str} 跌>2%→A涨={down2_str} "
              f"lift={lift_str} t={t_val:+.2f} {sig}")

# 汇总
print(f"\n{'='*80}")
print("信号排名(按lift>2%排序)")
print(f"{'='*80}")
valid = [r for r in all_results if r["lift_2"] is not None]
valid.sort(key=lambda x: x["lift_2"], reverse=True)

print(f"{'美股':<8}{'供应商':<10}{'关系':<18}{'样本':>5}{'涨>2%lift':>10}{'t值':>8}")
print("-" * 60)
for r in valid:
    sig = "*" if abs(r["t_stat"]) > 1.64 else ""
    print(f"  {r['us_name']:<6}{r['supplier']:<8}{r['link']:<16}{r['n']:>5}"
          f"{r['lift_2']:>+9.1%}{r['t_stat']:>+7.2f} {sig}")

# 结论
print(f"\n=== 结论 ===")
strong = [r for r in valid if r["lift_2"] and r["lift_2"] > 0.05]
if strong:
    print(f"lift>5%的供应链对: {len(strong)}家")
    for r in strong:
        print(f"  ✅ {r['us_name']}→{r['name']}({r['link']}): lift={r['lift_2']:+.1%} t={r['t_stat']:+.2f}")
else:
    print("未发现显著供应链联动信号")

# 保存
outp = os.path.join(ROOT, "reports", "supply_chain_full_validation.json")
with open(outp, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {outp}")
