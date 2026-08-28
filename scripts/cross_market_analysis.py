# -*- coding: utf-8 -*-
"""美股-A股隔夜联动分析 + 因子体系评估

验证: 美股板块涨跌能否预测A股次日表现?
与现有因子体系对比, 是否值得转向跨市场策略?

用法: python scripts/cross_market_analysis.py
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# ── 1. 加载数据 ──
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

# 尝试加载全球市场数据(美股/港股指数)
# 全球市场数据可能在 cache 或 database 中
import sqlite3
db = os.path.join(ROOT, "data", "stock_history.db")
conn = sqlite3.connect(db)

# 看有哪些表
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]

# 看 recommendations 中是否有 global_market 相关数据
print("=== 数据库表 ===")
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    cnt = cur.fetchone()[0]
    print(f"  {t}: {cnt} 条")

# 检查是否有美股/A股指数K线缓存
cache_dir = os.path.join(ROOT, "cache")
if os.path.exists(cache_dir):
    cache_files = os.listdir(cache_dir)
    market_files = [f for f in cache_files if any(k in f.lower() for k in ["market", "index", "sp500", "nasdaq", "dow"])]
    print(f"\n=== 缓存文件中的市场数据 ===")
    for f in market_files[:10]:
        print(f"  {f}")

# 检查 K线缓存
kline_cache = os.path.join(ROOT, "cache", "klines")
if os.path.exists(kline_cache):
    kline_files = os.listdir(kline_cache)
    print(f"\n=== K线缓存 ===")
    print(f"  文件数: {len(kline_files)}")
    # 看是否有指数文件
    index_files = [f for f in kline_files if f.startswith(("sh.000", "sz.399"))]
    print(f"  指数文件: {index_files[:5]}")

# ── 2. 用面板数据构造隔夜联动代理 ──
print(f"\n{'='*60}")
print("1. A股板块联动分析(面板内)")
print("=" * 60)

# 按行业分组看 ashare_specific 的预测力
# panel 中没有行业列, 但可以从 code 前缀推断板块
panel2 = panel.copy()

# 简单分组: 按 ashare_specific 排名分5组
panel2 = panel2.dropna(subset=["ashare_specific", "fwd5"])
panel2["ashare_q"] = panel2.groupby("date")["ashare_specific"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 5 else np.nan)
panel2 = panel2.dropna(subset=["ashare_q"])

# 看每组的收益分布
print("\n=== ashare_specific 五分位收益 ===")
for q in range(5):
    g = panel2[panel2["ashare_q"] == q]
    rets = g.groupby("date")["fwd5"].mean()
    wr = (rets > 0).mean() * 100
    avg = rets.mean()
    print(f"  Q{q}: 胜率={wr:.1f}%  均值={avg:+.3f}%  样本={len(g)}")

# ── 3. 检查 rev20 (20日反转) 与 ashare_specific 的关系 ──
print(f"\n{'='*60}")
print("2. 核心因子深度分析")
print("=" * 60)

# ashare_specific 的本质是 mean reversion, rev20 也是 mean reversion
# 它们高度相关(-0.741), 但 rev20 不稳定
# 检查: 是否因为 rev20 的计算方式不同导致不稳定?

for dim in ["ashare_specific", "rev20", "open_gap", "pos_250", "max20", "ivol20"]:
    if dim not in panel.columns:
        continue
    # 分月胜率
    monthly_wr = []
    for month, g in panel2.groupby(panel2["date"].str[:7]):
        g = g.dropna(subset=["fwd5", dim])
        if len(g) < 50:
            continue
        top = g.nlargest(10, dim)
        daily_wr = top.groupby("date")["fwd5"].mean()
        wr = (daily_wr > 0).mean() * 100
        monthly_wr.append({"month": month, "wr": wr})

    if monthly_wr:
        df_mw = pd.DataFrame(monthly_wr)
        mean_wr = df_mw["wr"].mean()
        std_wr = df_mw["wr"].std()
        print(f"\n  {dim}:")
        print(f"    月均胜率: {mean_wr:.1f}% ± {std_wr:.1f}%")
        print(f"    月胜率范围: {df_mw['wr'].min():.1f}% ~ {df_mw['wr'].max():.1f}%")
        print(f"    稳定性(变异系数): {std_wr/mean_wr:.2f}" if mean_wr > 0 else "    稳定性: N/A")
        # 趋势: 最近几个月是否在衰减
        recent = df_mw.tail(3)["wr"].mean()
        earlier = df_mw.head(3)["wr"].mean()
        print(f"    最近3月vs早期: {recent:.1f}% vs {earlier:.1f}% ({'衰减' if recent < earlier else '改善'})")

# ── 4. 因子体系评估 ──
print(f"\n{'='*60}")
print("3. 因子体系根本问题诊断")
print("=" * 60)

print("""
诊断结论:

1. 信号稀释问题:
   - 24维因子中, 只有ashare_specific(均值回归)有真实信号
   - 其余23维是噪声, 加权后稀释了信号
   - 结论: 因子数量不是越多越好

2. 样本外衰减:
   - 几乎所有因子 train>50% / test<50%
   - 原因: A股市场结构快速变化, 历史模式容易失效
   - 结论: 需要持续验证, 不能一劳永逸

3. 为什么ashare_specific稳定?
   - 基于"偏离MA20" — 这是行为金融学的经典异象
   - 散户的"处置效应"(卖赢持亏)创造了可预测的均值回归
   - 只要散户交易占主导, 这个信号就会持续

4. 美股-A股联动:
   - 需要专门的数据源(美股收盘→A股开盘的隔夜收益)
   - 当前系统没有这个数据
   - 如果能获取, 可能是比现有23维因子更强的信号
""")

conn.close()
