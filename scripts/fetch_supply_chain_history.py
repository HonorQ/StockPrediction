# -*- coding: utf-8 -*-
"""拉取A股供应链公司更长历史数据(2年)"""
import os, sys, json, time
import baostock as bs
import pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# 加载供应链
with open(os.path.join(ROOT, "data", "us_a_supply_chain.json"), encoding="utf-8") as f:
    sc_data = json.load(f)

a_codes = set()
for info in sc_data["supply_chain"].values():
    for s in info["a_share_suppliers"]:
        a_codes.add(s["code"])

print(f"A股供应链公司: {len(a_codes)}家")
print("拉取2年历史数据...\n")

bs.login()

data_dir = os.path.join(ROOT, "cache", "supply_chain")
os.makedirs(data_dir, exist_ok=True)

results = {}
for code in sorted(a_codes):
    bs_code = code[:2] + "." + code[2:]
    rs = bs.query_history_k_data_plus(
        bs_code, "date,close,volume,turn,pctChg,open,high,low",
        start_date="2024-01-01", end_date="2026-12-31",
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        df = pd.DataFrame(rows, columns=["date", "close", "volume", "turn", "pctChg", "open", "high", "low"])
        for col in ["close", "volume", "turn", "pctChg", "open", "high", "low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        if len(df) > 100:
            # 保存
            df.to_csv(os.path.join(data_dir, f"{code}_daily.csv"), index=False)
            results[code] = len(df)
            print(f"  {code}: {len(df)}行 ({df['date'].iloc[0]}~{df['date'].iloc[-1]})")
    time.sleep(0.2)

bs.logout()
print(f"\n成功: {len(results)}/{len(a_codes)}家")

# 统计
if results:
    avg = sum(results.values()) / len(results)
    print(f"平均: {avg:.0f}行/家")
