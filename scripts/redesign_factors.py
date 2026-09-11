# -*- coding: utf-8 -*-
"""因子体系重新设计 — 诊断+方案对比

核心问题: 24维中只有ashare_specific稳定, 其余23维过拟合。
调查: ashare_specific为什么有效? 能否提取类似稳定因子?

用法: python scripts/redesign_factors.py
"""
import os, json
import numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

print(f"面板: {len(panel)} 行 / {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只\n")

# ── 1. ashare_specific 子因子稳定性分析 ──
print("=" * 60)
print("1. ashare_specific 子因子贡献分析")
print("=" * 60)

# ashare_specific 的子因子: mean_reversion_score, volatility_score,
# position_score, hot_money_score, sentiment_cycle
# 这些不在 market_panel 中, 但从 factor_snapshots 可取
import sqlite3
db = os.path.join(ROOT, "data", "stock_history.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# 取所有有 ashare_specific 真实值的记录
cur = conn.cursor()
cur.execute("""
    SELECT fs.trade_date, fs.code, fs.sub_ashare_specific, fs.factors
    FROM factor_snapshots fs
    WHERE fs.sub_ashare_specific IS NOT NULL AND fs.sub_ashare_specific != 50
    ORDER BY fs.trade_date
""")
rows = cur.fetchall()
print(f"  ashare_specific 真实值记录: {len(rows)} 条")

# 解析子因子
sub_data = []
for r in rows:
    try:
        f = json.loads(r["factors"]) if r["factors"] else {}
    except:
        continue
    sub_data.append({
        "date": r["trade_date"],
        "code": r["code"],
        "ashare_score": r["sub_ashare_specific"],
        "reversion_score": f.get("mean_reversion_score", 50),
        "volatility_score": f.get("volatility_score", 50),
        "position_score": f.get("position_score", 50),
        "hot_money_score": f.get("hot_money_score", 50),
    })

sub_df = pd.DataFrame(sub_data)
print("  子因子分布:")
for col in ["reversion_score", "volatility_score", "position_score", "hot_money_score"]:
    vals = sub_df[col]
    print(f"    {col:<22} 均值={vals.mean():.1f}  标准差={vals.std():.1f}  "
          f"范围=[{vals.min():.0f}, {vals.max():.0f}]  方差{'充足' if vals.std()>10 else '不足'}")

# ── 2. 从K线提取稳定因子候选 ──
print(f"\n{'=' * 60}")
print("2. K线可重建因子的样本外胜率")
print("=" * 60)

# 这些因子可以从K线计算, 不依赖外部数据
kline_factors = {
    "rev20": "20日反转",
    "open_gap": "竞价情绪",
    "pos_250": "一年位置",
    "max20": "20日最大涨幅",
    "ivol20": "20日波动率",
    "turn_chg5": "换手率变化",
    "vpcorr20": "量价相关性",
    "amihud20": "非流动性",
}

# 检查哪些在面板中
avail = {k: v for k, v in kline_factors.items() if k in panel.columns}
print(f"  面板中可用的K线因子: {list(avail.keys())}")

# 分月+样本外
dates = sorted(panel["date"].unique())
split = dates[80]
train = panel[panel["date"] < split]
test = panel[panel["date"] >= split]

print(f"\n  {'因子':<14}{'训练胜率':>10}{'测试胜率':>10}{'差异':>8}{'稳定':>6}")
print("  " + "─" * 50)
for dim, desc in avail.items():
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
    stable = "✓" if abs(diff) < 5 else "✗"
    print(f"  {dim:<14}{results['train']:>9.1f}%{results['test']:>9.1f}%{diff:>+7.1f}%  {stable}")

# ── 3. 与 ashare_specific 的相关性 ──
print(f"\n{'=' * 60}")
print("3. 新因子与 ashare_specific 的相关性")
print("=" * 60)

for dim in avail:
    if dim in panel.columns:
        corr = panel[["ashare_specific", dim]].corr().iloc[0, 1]
        print(f"  {dim:<14} 与 ashare_specific 相关性: {corr:+.3f}")

# ── 4. 方案对比 ──
print(f"\n{'=' * 60}")
print("4. 评分系统方案对比")
print("=" * 60)

def compute_winrate(sub_panel, signal_col, top_n=10, hold=5):
    daily = []
    for date, g in sub_panel.groupby("date"):
        g = g.dropna(subset=[f"fwd{hold}", signal_col])
        if len(g) < 100:
            continue
        top = g.nlargest(top_n, signal_col)
        daily.append(float(top[f"fwd{hold}"].mean()))
    arr = np.array(daily)
    return (arr > 0).mean() * 100, arr.mean(), arr.std()

# 方案A: 纯ashare_specific
wr_a, avg_a, std_a = compute_winrate(panel, "ashare_specific")
sharpe_a = avg_a / std_a * np.sqrt(252/5) if std_a > 0 else 0

# 方案B: ashare_specific + 风控排除(chip)
panel_b = panel.copy()
# 简化: 排除 chip > 80 的票(高获利盘)
panel_b = panel_b[panel_b["chip"] <= 80]
wr_b, avg_b, std_b = compute_winrate(panel_b, "ashare_specific")
sharpe_b = avg_b / std_b * np.sqrt(252/5) if std_b > 0 else 0

# 方案C: ashare_specific + pos_250排除(高位)
panel_c = panel.copy()
if "pos_250" in panel_c.columns:
    panel_c = panel_c[panel_c["pos_250"] <= 0.8]
wr_c, avg_c, std_c = compute_winrate(panel_c, "ashare_specific")
sharpe_c = avg_c / std_c * np.sqrt(252/5) if std_c > 0 else 0

# 方案D: ashare_specific + vol排除(高波动)
panel_d = panel.copy()
if "ivol20" in panel_d.columns:
    vol_median = panel_d["ivol20"].median()
    panel_d = panel_d[panel_d["ivol20"] <= vol_median]
wr_d, avg_d, std_d = compute_winrate(panel_d, "ashare_specific")
sharpe_d = avg_d / std_d * np.sqrt(252/5) if std_d > 0 else 0

print(f"\n  {'方案':<30}{'胜率':>8}{'均值':>8}{'Sharpe':>8}")
print("  " + "─" * 54)
print(f"  {'A: 纯ashare_specific':<28}{wr_a:>7.1f}%{avg_a:>+7.3f}%{sharpe_a:>8.2f}")
print(f"  {'B: +chip风控排除':<28}{wr_b:>7.1f}%{avg_b:>+7.3f}%{sharpe_b:>8.2f}")
if "pos_250" in panel.columns:
    print(f"  {'C: +pos_250高位排除':<28}{wr_c:>7.1f}%{avg_c:>+7.3f}%{sharpe_c:>8.2f}")
if "ivol20" in panel.columns:
    print(f"  {'D: +低波动排除':<28}{wr_d:>7.1f}%{avg_d:>+7.3f}%{sharpe_d:>8.2f}")
print(f"  {'E: 当前24维(对照)':<28}  42.9%  -0.526%    -0.82")

conn.close()
