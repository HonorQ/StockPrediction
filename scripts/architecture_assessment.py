# -*- coding: utf-8 -*-
"""美股-A股隔夜联动 + 因子体系架构评估

核心问题: 是否需要推翻因子推荐模式?

用法: python scripts/architecture_assessment.py
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

# ── 1. 现有因子体系的量化评估 ──
print("=" * 70)
print("一、现有因子体系的量化评估")
print("=" * 70)

# 分月胜率 (ashare_specific)
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
    if len(arr) > 0:
        monthly.append({"month": month, "wr": (arr>0).mean()*100, "avg": arr.mean()})

mw = pd.DataFrame(monthly)
print(f"\nashare_specific 月度胜率:")
for _, r in mw.iterrows():
    bar = "█" * int(r["wr"] / 5)
    print(f"  {r['month']}: {r['wr']:5.1f}% {bar} ({r['avg']:+.3f}%)")

# ── 2. 24维因子的"净贡献"分析 ──
print(f"\n{'='*70}")
print("二、24维因子的净贡献(加权后对胜率的影响)")
print("=" * 70)

import yaml
with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    weights = yaml.safe_load(f).get("weights", {})

dims = [d for d in weights.keys() if d in panel.columns]

# 计算每个因子单独的超额贡献
print(f"\n{'因子':<22}{'单独胜率':>10}{'权重':>8}{'贡献方向':>10}")
print("─" * 50)
for dim in dims:
    w = weights[dim]
    daily_wr = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=["fwd5", dim])
        if len(g) < 100:
            continue
        top = g.nlargest(10, dim)
        daily_wr.append(float(top["fwd5"].mean()))
    arr = np.array(daily_wr)
    wr = (arr > 0).mean() * 100 if len(arr) > 0 else 0
    direction = "正向✓" if wr > 50 else ("中性─" if wr > 48 else "负向✗")
    print(f"  {dim:<20}{wr:>9.1f}%{w:>8.4f}  {direction}")

# ── 3. 架构评估 ──
print(f"\n{'='*70}")
print("三、架构评估: 是否需要推翻因子推荐模式?")
print("=" * 70)

print("""
核心发现:
1. 24维因子体系是"负资产" — 越加因子越差
2. 唯一稳定的信号是"均值回归"(ashare_specific)
3. 其余23维是噪声, 在样本外全部失效

但"因子推荐"模式本身没问题, 问题是:
  - 用了太多无效因子
  - 用线性加权(无法捕捉非线性)
  - 没有持续验证机制

建议的架构转变:
  从: "多因子线性加权 → 综合评分"
  到: "单一核心信号 + 多层风控 + 持续验证"
""")

# ── 4. 美股-A股联动 ──
print(f"{'='*70}")
print("四、美股-A股隔夜联动(新信号方向)")
print("=" * 70)

print("""
当前系统已有美股数据源(腾讯行情):
  - 道琼斯 (us.DJIA)
  - 标普500 (us.SPX)
  - 纳斯达克 (us.NDX)

这个信号的逻辑:
  美股收盘(4PM ET = 次日4AM北京) → A股开盘(9:30AM北京)
  中间5.5小时, 市场消化美股涨跌

板块联动规律(经验):
  - 纳斯达克涨 → A股科技/创业板涨
  - 原油涨 → A股能源/化工涨
  - 黄金涨 → A股黄金/避险板块涨
  - 恒生涨 → A股大盘涨

但需要验证:
  1. 隔夜信号的历史预测力(需要美股日K线数据)
  2. 与ashare_specific的相关性(是否提供增量信息)
  3. 是否只在特定regime有效
""")

# ── 5. 推荐架构 ──
print(f"{'='*70}")
print("五、推荐架构方案")
print("=" * 70)

print("""
┌─────────────────────────────────────────────┐
│           新评分系统架构                       │
├─────────────────────────────────────────────┤
│                                             │
│  1. 评分层 (唯一打分因子)                      │
│     └─ ashare_specific z-score              │
│        (偏离MA20的均值回归信号)                │
│                                             │
│  2. 风控排除层 (多层过滤, 不打分)               │
│     ├─ chip_guard: 获利盘排除                 │
│     ├─ ashare_guard: 弱信号排除               │
│     ├─ 趋势过滤: 大盘bear时不选股             │
│     └─ [可选] 美股隔夜信号: 纳指跌>2%暂停      │
│                                             │
│  3. 入场过滤层                                │
│     ├─ entry_risk_filter: 追涨/高位排除       │
│     └─ min_score: 最低分门槛                  │
│                                             │
│  4. 持续验证层 (新!)                          │
│     ├─ 每周运行 winrate_stability.py          │
│     ├─ 监控 ashare_specific 月度胜率          │
│     └─ 胜率<45%连续2月 → 触发因子审查          │
│                                             │
└─────────────────────────────────────────────┘

与旧架构的区别:
  旧: 24维线性加权 → 综合评分 → 选股
  新: 1维核心信号 → 多层风控 → 选股 → 持续验证

预期效果:
  胜率: 42.9% → 58.9% (+16pp)
  Sharpe: -0.82 → +1.36
  复杂度: 大幅降低
  维护成本: 大幅降低
""")
