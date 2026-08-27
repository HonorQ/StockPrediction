# -*- coding: utf-8 -*-
"""基于胜率的权重重新校准

计算全部24维因子的 Top10 胜率, 然后基于胜率提出新权重方案并验证。

用法: python scripts/recalibrate_by_winrate.py
"""
import os, sys, json, yaml
import numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# ── 1. 加载数据和权重 ──
panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    old_weights = yaml.safe_load(f).get("weights", {})

HOLD = 5
TOP_N = 10
ret_col = f"fwd{HOLD}"

print(f"面板: {len(panel)} 行 / {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只")
print(f"选股: Top{TOP_N} | 持有: {HOLD}日\n")

# ── 2. 逐因子计算胜率和指标 ──
dims = [d for d in old_weights.keys() if d in panel.columns]
results = []

for dim in dims:
    daily_rets, daily_excess = [], []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=[ret_col, dim])
        if len(g) < 100:
            continue
        top = g.nlargest(TOP_N, dim)
        bench = g[ret_col].mean()
        daily_rets.append(float(top[ret_col].mean()))
        daily_excess.append(float(top[ret_col].mean()) - bench)

    arr = np.array(daily_rets)
    exc = np.array(daily_excess)
    if len(arr) == 0:
        continue

    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    exc_avg = exc.mean()
    sharpe = arr.mean() / arr.std() * np.sqrt(252/HOLD) if arr.std() > 0 else 0

    # 胜率贡献度: 胜率-50 的绝对值(离随机越远越有用)
    wr_score = abs(wr - 50)
    # 方向: 胜率>50 为正向, <50 为负向
    direction = 1 if wr > 50 else -1

    results.append({
        "dim": dim,
        "winrate": round(wr, 1),
        "avg_ret": round(avg, 4),
        "excess": round(exc_avg, 4),
        "sharpe": round(sharpe, 2),
        "old_weight": old_weights.get(dim, 0),
        "wr_score": round(wr_score, 1),
        "direction": direction,
    })

df = pd.DataFrame(results).sort_values("winrate", ascending=False)

print(f"{'因子':<22}{'胜率':>7}{'均值':>8}{'超额':>8}{'Sharpe':>8}{'旧权重':>8}{'方向':>6}")
print("─" * 67)
for _, r in df.iterrows():
    flag = "✅" if r["winrate"] >= 50 else ("⚠️" if r["winrate"] >= 48 else "❌")
    print(f"  {r['dim']:<20}{r['winrate']:>6.1f}%{r['avg_ret']:>+7.3f}%{r['excess']:>+7.3f}%"
          f"{r['sharpe']:>8.2f}{r['old_weight']:>8.4f}  {flag}")

# ── 3. 提出新权重方案 ──
print(f"\n{'='*67}")
print("权重校准方案")
print(f"{'='*67}")

# 方案: 基于胜率的方向和强度赋权
# 胜率 >= 50: 正权重, 权重 ∝ (胜率-48)/100
# 胜率 48-50: 不参与(中性)
# 胜率 < 48: 反向权重(取反), 权重 ∝ (48-胜率)/100
# 归一化使总权重 = 1.0

new_weights = {}
for _, r in df.iterrows():
    wr = r["winrate"]
    dim = r["dim"]
    if wr >= 50:
        # 正向, 权重 ∝ 胜率-48
        new_weights[dim] = max(0, (wr - 48) / 100)
    elif wr >= 48:
        # 中性, 不参与
        new_weights[dim] = 0
    else:
        # 反向, 权重 ∝ 48-胜率 (负权重)
        new_weights[dim] = -max(0, (48 - wr) / 100)

# 归一化
total_pos = sum(v for v in new_weights.values() if v > 0)
total_neg = sum(abs(v) for v in new_weights.values() if v < 0)
if total_pos > 0:
    for dim in new_weights:
        if new_weights[dim] > 0:
            new_weights[dim] = new_weights[dim] / total_pos * 0.85  # 正权重占85%
        elif new_weights[dim] < 0:
            new_weights[dim] = new_weights[dim] / total_neg * 0.15  # 负权重占15%

print(f"\n{'因子':<22}{'旧权重':>10}{'新权重':>10}{'变化':>10}")
print("─" * 52)
for dim in sorted(new_weights.keys(), key=lambda d: -new_weights[d]):
    old = old_weights.get(dim, 0)
    new = new_weights[dim]
    diff = new - old
    flag = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "─")
    print(f"  {dim:<20}{old:>10.4f}{new:>10.4f}{diff:>+10.4f} {flag}")

print(f"\n新权重总和: {sum(new_weights.values()):.4f}")

# ── 4. 验证新权重 ──
def zscore_col(g, col):
    mu, sd = g[col].mean(), g[col].std()
    return (g[col] - mu) / sd if sd > 1e-9 else 0

def compute_composite(panel, weights, label):
    """计算composite并返回胜率指标"""
    p = panel.copy()
    p["composite"] = 0.0
    for dim, w in weights.items():
        if dim not in p.columns or w == 0:
            continue
        col = dim
        if dim == "trend" and w < 0:
            p["_tmp"] = 100 - p[dim]
            col = "_tmp"
        # 按日截面Z-Score
        def _zs_func(g, c=col):
            mu, sd = g[c].mean(), g[c].std()
            g["_zs"] = (g[c] - mu) / sd if sd > 1e-9 else 0
            return g
        p = p.groupby("date", group_keys=False).apply(_zs_func)
        p["composite"] += p["_zs"] * abs(w)
        if "_tmp" in p.columns:
            p.drop("_tmp", axis=1, inplace=True)
        p.drop("_zs", axis=1, inplace=True)

    daily = []
    for date, g in p.groupby("date"):
        g = g.dropna(subset=[ret_col, "composite"])
        if len(g) < 100:
            continue
        top = g.nlargest(TOP_N, "composite")
        bench = g[ret_col].mean()
        daily.append({"ret": float(top[ret_col].mean()), "exc": float(top[ret_col].mean()) - bench})

    d = pd.DataFrame(daily)
    if len(d) == 0:
        return None
    wr = (d["ret"] > 0).mean() * 100
    avg = d["ret"].mean()
    sharpe = d["ret"].mean() / d["ret"].std() * np.sqrt(252/HOLD) if d["ret"].std() > 0 else 0
    exc = d["exc"].mean()
    return {"label": label, "winrate": wr, "avg": avg, "sharpe": sharpe, "excess": exc}

# 基线: 旧权重
old_result = compute_composite(panel, old_weights, "旧权重(24维IC)")

# 新方案: 基于胜率
new_result = compute_composite(panel, new_weights, "新权重(胜率校准)")

# 对照: 等权
equal_weights = {d: 1.0/len(dims) for d in dims}
eq_result = compute_composite(panel, equal_weights, "等权基准")

# 对照: 只用胜率>50的因子
wr50_weights = {d: (1.0/sum(1 for r in results if r["winrate"]>=50)
                     if r["winrate"] >= 50 else 0) for r in results for d in [r["dim"]]}
wr50_result = compute_composite(panel, wr50_weights, "仅胜率>50因子")

print(f"\n{'='*67}")
print("验证结果对比")
print(f"{'='*67}")
print(f"{'方案':<22}{'胜率':>8}{'均值':>8}{'Sharpe':>8}{'超额':>8}")
print("─" * 54)
for r in [old_result, new_result, eq_result, wr50_result]:
    if r:
        print(f"  {r['label']:<20}{r['winrate']:>7.1f}%{r['avg']:>+7.3f}%{r['sharpe']:>8.2f}{r['excess']:>+7.3f}%")

# 保存
out = {
    "factor_winrates": df.to_dict("records"),
    "old_weights": old_weights,
    "new_weights": {k: round(v, 4) for k, v in new_weights.items()},
    "comparison": {r["label"]: r for r in [old_result, new_result, eq_result, wr50_result] if r},
}
outp = os.path.join(ROOT, "reports", "winrate_recalibration.json")
with open(outp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {outp}")
