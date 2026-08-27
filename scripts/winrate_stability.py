# -*- coding: utf-8 -*-
"""胜率稳定性检查: ashare_specific 分时段/分参数胜率"""
import os, pandas as pd, numpy as np

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

print(f"面板: {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只\n")

# ── 1. 分月胜率 ──
print("=== ashare_specific 分月胜率 (Top10, hold=5) ===")
panel["month"] = panel["date"].str[:7]
monthly = []
for month, g in panel.groupby("month"):
    daily_wr = []
    for date, dg in g.groupby("date"):
        dg = dg.dropna(subset=["fwd5", "ashare_specific"])
        if len(dg) < 100:
            continue
        top = dg.nlargest(10, "ashare_specific")
        daily_wr.append(float(top["fwd5"].mean()))
    arr = np.array(daily_wr)
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    monthly.append({"month": month, "wr": wr, "avg": avg, "days": len(arr)})
    print(f"  {month}: 胜率={wr:.1f}%  均值={avg:+.3f}%  天数={len(arr)}")

# ── 2. 分Top-N胜率 ──
print("\n=== ashare_specific 分Top-N胜率 (hold=5) ===")
for n in [3, 5, 10, 20, 30]:
    daily_wr = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=["fwd5", "ashare_specific"])
        if len(g) < 100:
            continue
        top = g.nlargest(n, "ashare_specific")
        daily_wr.append(float(top["fwd5"].mean()))
    arr = np.array(daily_wr)
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    print(f"  Top{n:<3}: 胜率={wr:.1f}%  均值={avg:+.3f}%")

# ── 3. 分持有期胜率 ──
print("\n=== ashare_specific 分持有期胜率 (Top10) ===")
for h in [3, 5, 10]:
    col = f"fwd{h}"
    if col not in panel.columns:
        continue
    daily_wr = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=[col, "ashare_specific"])
        if len(g) < 100:
            continue
        top = g.nlargest(10, "ashare_specific")
        daily_wr.append(float(top[col].mean()))
    arr = np.array(daily_wr)
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    sharpe = arr.mean() / arr.std() * np.sqrt(252/h) if arr.std() > 0 else 0
    print(f"  hold={h}日: 胜率={wr:.1f}%  均值={avg:+.3f}%  Sharpe={sharpe:.2f}")

# ── 4. 样本外验证: 前80天训练/后95天测试 ──
print("\n=== 样本外验证 (前80天训练/后95天测试) ===")
dates = sorted(panel["date"].unique())
split = dates[80]
train = panel[panel["date"] < split]
test = panel[panel["date"] >= split]
print(f"  训练期: {dates[0]}~{split} ({train['date'].nunique()}天)")
print(f"  测试期: {split}~{dates[-1]} ({test['date'].nunique()}天)")

for name, sub in [("训练期", train), ("测试期", test)]:
    daily_wr = []
    for date, g in sub.groupby("date"):
        g = g.dropna(subset=["fwd5", "ashare_specific"])
        if len(g) < 100:
            continue
        top = g.nlargest(10, "ashare_specific")
        daily_wr.append(float(top["fwd5"].mean()))
    arr = np.array(daily_wr)
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    print(f"  {name}: 胜率={wr:.1f}%  均值={avg:+.3f}%  天数={len(arr)}")

# ── 5. 所有因子的样本内外胜率对比 ──
print("\n=== 因子样本内外胜率对比 (Top10, hold=5) ===")
print(f"{'因子':<22}{'训练胜率':>10}{'测试胜率':>10}{'差异':>8}")
print("─" * 50)
dims = ["ashare_specific", "quant_behavior", "strategy", "capital_behavior",
        "chip", "deep_research", "oversold_rebound", "trend", "relative_strength"]
for dim in dims:
    results = {}
    for name, sub in [("train", train), ("test", test)]:
        daily_wr = []
        for date, g in sub.groupby("date"):
            g = g.dropna(subset=["fwd5", dim])
            if len(g) < 50:
                continue
            top = g.nlargest(10, dim)
            daily_wr.append(float(top["fwd5"].mean()))
        arr = np.array(daily_wr)
        results[name] = (arr > 0).mean() * 100 if len(arr) > 0 else 0
    diff = results["test"] - results["train"]
    flag = "✓" if abs(diff) < 5 else "✗"
    print(f"  {dim:<20}{results['train']:>9.1f}%{results['test']:>9.1f}%{diff:>+7.1f}% {flag}")
