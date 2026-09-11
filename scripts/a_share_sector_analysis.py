# -*- coding: utf-8 -*-
"""A股行业动量分析 - 哪些行业有最强的板块效应?

用 industry_mapper 将个股分组到行业, 计算行业动量。
动量强的行业更可能跟随美股板块联动。
"""
import os, sys, json, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# 加载面板
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
panel["month"] = panel["date"].str[:7]

# 加载行业映射
industry_path = os.path.join(ROOT, "data", "industry_map.json")
if not os.path.exists(industry_path):
    print("无行业映射数据, 跳过")
    sys.exit(0)

with open(industry_path, "r", encoding="utf-8") as f:
    industry_map = json.load(f)

print(f"行业映射: {len(industry_map)} 只股票 / {len(set(industry_map.values()))} 个行业")
print(f"面板: {len(panel)} 行 / {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只\n")

# 给面板加行业列 (panel用sh.510030, industry_map用sh600176格式)
def normalize_code(code):
    """sh.510030 → sh510030"""
    return code.replace(".", "")

panel["norm_code"] = panel["code"].apply(normalize_code)
panel["industry"] = panel["norm_code"].map(industry_map)
panel_with_ind = panel.dropna(subset=["industry"])

print(f"有行业标签的股票: {panel_with_ind['code'].nunique()} 只")
print(f"覆盖行数: {len(panel_with_ind)} / {len(panel)} ({len(panel_with_ind)/len(panel)*100:.0f}%)\n")

# ── 1. 行业板块效应排名 ──
# 行业动量 = 行业内股票收益的一致性
# 计算方法: 每日行业内股票收益的截面标准差→低标准差=高一致性=强板块效应
print("=" * 60)
print("1. 行业板块效应排名(一致性越高=板块联动越强)")
print("=" * 60)

industry_stats = []
for ind, g in panel_with_ind.groupby("industry"):
    if len(g) < 100:
        continue
    # 每日行业内股票fwd5的标准差
    daily_std = g.groupby("date")["fwd5"].std()
    # 行业内股票数
    n_stocks = g["code"].nunique()
    # 行业平均收益
    avg_ret = g.groupby("date")["fwd5"].mean().mean()
    # 一致性(1-平均标准差, 标准差越小一致性越高)
    consistency = 1 - daily_std.mean() if daily_std.mean() > 0 else 0

    industry_stats.append({
        "industry": ind,
        "n_stocks": n_stocks,
        "avg_daily_std": round(daily_std.mean(), 3),
        "consistency": round(consistency, 3),
        "avg_return": round(avg_ret, 3),
    })

df_ind = pd.DataFrame(industry_stats).sort_values("consistency", ascending=False)
print(f"\n{'行业':<18}{'股票数':>6}{'平均标准差':>10}{'一致性':>8}{'平均收益':>10}")
print("─" * 52)
for _, r in df_ind.head(20).iterrows():
    print(f"  {r['industry']:<16}{r['n_stocks']:>6}{r['avg_daily_std']:>10.3f}"
          f"{r['consistency']:>8.3f}{r['avg_return']:>+9.3f}%")

# ── 2. 与美股板块的关键词匹配 ──
print(f"\n{'='*60}")
print("2. 与美股板块关键词匹配")
print("=" * 60)

US_TO_A = {
    "半导体": ["芯片", "半导体", "集成电路"],
    "液冷服务器": ["液冷", "服务器", "算力", "数据中心"],
    "光通信/CPO": ["光模块", "CPO", "光通信"],
    "电源/电力": ["电源", "电力设备", "特高压"],
    "新能源": ["新能源车", "锂电池", "光伏", "储能", "风电"],
    "AI应用": ["AI", "人工智能", "软件"],
    "医药": ["医药", "创新药", "生物"],
}

all_industries = set(df_ind["industry"].values)
for us_sector, keywords in US_TO_A.items():
    matched = set()
    for ind in all_industries:
        for kw in keywords:
            if kw in ind:
                matched.add(ind)
    if matched:
        print(f"\n  {us_sector} → A股行业:")
        for ind in sorted(matched):
            row = df_ind[df_ind["industry"] == ind].iloc[0] if len(df_ind[df_ind["industry"] == ind]) > 0 else None
            if row is not None:
                print(f"    {ind}  一致性={row['consistency']:.3f}  股票数={row['n_stocks']}")
    else:
        print(f"\n  {us_sector} → 未找到匹配行业")

# ── 3. 行业动量策略回测 ──
print(f"\n{'='*60}")
print("3. 行业动量策略(Top3行业等权持有5日)")
print("=" * 60)

# 每日计算行业平均收益, 选Top3
daily_ind_ret = panel_with_ind.groupby(["date", "industry"])["fwd5"].mean().reset_index()
daily_ind_ret.columns = ["date", "industry", "ind_ret"]

# 行业动量信号: 过去5天的平均收益
daily_ind_ret = daily_ind_ret.sort_values(["industry", "date"])
daily_ind_ret["momentum_5d"] = daily_ind_ret.groupby("industry")["ind_ret"].transform(
    lambda x: x.rolling(5, min_periods=3).mean())

# 每日选Top3行业
dates = sorted(daily_ind_ret["date"].unique())
hold = 5
results = []
for i in range(60, len(dates), hold):
    d = dates[i]
    day = daily_ind_ret[daily_ind_ret["date"] == d].dropna(subset=["momentum_5d"])
    if len(day) < 10:
        continue
    top3 = day.nlargest(3, "momentum_5d")
    # 计算接下来5天的平均收益
    future_dates = dates[i+1:i+hold+1]
    if not future_dates:
        continue
    future = daily_ind_ret[daily_ind_ret["date"].isin(future_dates)]
    top3_ret = future[future["industry"].isin(top3["industry"])]["ind_ret"].mean()
    bench_ret = future["ind_ret"].mean()
    results.append({"date": d, "top3_ret": top3_ret, "bench_ret": bench_ret})

if results:
    df_r = pd.DataFrame(results)
    top3_wr = (df_r["top3_ret"] > 0).mean() * 100
    top3_avg = df_r["top3_ret"].mean()
    bench_avg = df_r["bench_ret"].mean()
    excess = top3_avg - bench_avg
    print(f"  Top3行业持有5日: 胜率={top3_wr:.1f}%  均值={top3_avg:+.3f}%  vs基准={excess:+.3f}%")
    print(f"  基准(全行业等权): 均值={bench_avg:+.3f}%")
