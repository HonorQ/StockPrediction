# -*- coding: utf-8 -*-
"""当前评分体系胜率分析

从 market_panel_daily.csv 重建 composite score, 按当前 config.yaml 权重加权,
统计 Top-N 选股的胜率/收益分布。

用法: python scripts/winrate_analysis.py [--top-n 10] [--hold 5]
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--hold", type=int, default=5)
    args = ap.parse_args()

    # 加载权重
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    weights = cfg.get("weights", {})

    # 加载面板
    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    hold = args.hold
    ret_col = f"fwd{hold}"

    # 可用维度
    avail_dims = [d for d in weights.keys() if d in panel.columns]
    missing = [d for d in weights.keys() if d not in panel.columns]
    if missing:
        print(f"警告: 以下维度缺失, 按50中性处理: {missing}")

    # 截面 Z-Score 归一化 + 加权 composite
    panel = panel.copy()
    panel["composite"] = 0.0
    total_weight = 0
    for dim in avail_dims:
        w = weights.get(dim, 0)
        if w == 0:
            continue
        # trend 已在 NEGATIVE_IC_DIMS 中取反: composite 中也取反
        col = dim
        if dim == "trend":
            panel["_tmp"] = 100 - panel[dim]
            col = "_tmp"
        # 按日截面 Z-Score
        def _zscore_group(g, c=col):
            mu, sd = g[c].mean(), g[c].std()
            g["_zs"] = (g[c] - mu) / sd if sd > 1e-9 else 0
            return g
        panel = panel.groupby("date", group_keys=False).apply(_zscore_group)
        panel["composite"] += panel["_zs"] * abs(w)
        total_weight += abs(w)
        if "_tmp" in panel.columns:
            panel.drop("_tmp", axis=1, inplace=True)
        panel.drop("_zs", axis=1, inplace=True)

    # 每日选 Top-N
    results = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=[ret_col, "composite"])
        if len(g) < 50:
            continue
        top = g.nlargest(args.top_n, "composite")
        bot = g.nsmallest(args.top_n, "composite")
        avg_ret = float(top[ret_col].mean())
        bot_ret = float(bot[ret_col].mean())
        bench_ret = float(g[ret_col].mean())

        # 逐只统计
        for _, row in top.iterrows():
            results.append({
                "date": date,
                "code": row["code"],
                "ret": float(row[ret_col]),
                "composite": float(row["composite"]),
                "group": "top",
            })

    df = pd.DataFrame(results)

    # 统计
    print(f"\n=== 当前评分体系胜率分析 ===")
    print(f"  权重维度: {len(avail_dims)} / {len(weights)}")
    print(f"  选股: Top{args.top_n} | 持有: {hold}日")
    print(f"  面板: {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只")

    # Top-N 整体
    top_rets = df["ret"]
    wins = (top_rets > 0).sum()
    total = len(top_rets)
    winrate = wins / total * 100 if total > 0 else 0
    avg = top_rets.mean()
    median = top_rets.median()
    std = top_rets.std()
    sharpe = avg / std * np.sqrt(252 / hold) if std > 0 else 0

    # 分日统计
    daily_rets = df.groupby("date")["ret"].mean()
    daily_winrate = (daily_rets > 0).mean() * 100
    daily_avg = daily_rets.mean()
    daily_std = daily_rets.std()
    daily_sharpe = daily_avg / daily_std * np.sqrt(252 / hold) if daily_std > 0 else 0

    # 分位统计
    q25 = top_rets.quantile(0.25)
    q75 = top_rets.quantile(0.75)
    max_win = top_rets.max()
    max_loss = top_rets.min()

    # 超额收益
    daily_excess = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=[ret_col, "composite"])
        if len(g) < 50:
            continue
        top = g.nlargest(args.top_n, "composite")
        bench = g[ret_col].mean()
        daily_excess.append(float(top[ret_col].mean()) - bench)
    excess_arr = np.array(daily_excess)
    excess_avg = excess_arr.mean()
    excess_std = excess_arr.std()
    excess_sharpe = excess_avg / excess_std * np.sqrt(252 / hold) if excess_std > 0 else 0
    excess_winrate = (excess_arr > 0).mean() * 100

    print(f"\n{'指标':<22}{'数值':>12}")
    print(f"{'─'*34}")
    print(f"{'总选股数':<22}{total:>12}")
    print(f"{'胜率(个股)':<22}{winrate:>11.1f}%")
    print(f"{'胜率(日度)':<22}{daily_winrate:>11.1f}%")
    print(f"{'平均收益(个股)':<22}{avg:>+11.3f}%")
    print(f"{'平均收益(日度)':<22}{daily_avg:>+11.3f}%")
    print(f"{'中位数收益':<22}{median:>+11.3f}%")
    print(f"{'标准差':<22}{std:>11.3f}%")
    print(f"{'Sharpe(年化)':<22}{sharpe:>12.2f}")
    print(f"{'最大单笔盈利':<22}{max_win:>+11.3f}%")
    print(f"{'最大单笔亏损':<22}{max_loss:>+11.3f}%")
    print(f"{'Q25':<22}{q25:>+11.3f}%")
    print(f"{'Q75':<22}{q75:>+11.3f}%")
    print(f"{'─'*34}")
    print(f"{'超额收益均值':<22}{excess_avg:>+11.3f}%")
    print(f"{'超额收益标准差':<22}{excess_std:>11.3f}%")
    print(f"{'超额Sharpe(年化)':<22}{excess_sharpe:>12.2f}")
    print(f"{'超额胜率(日度)':<22}{excess_winrate:>11.1f}%")

    # 按 hold 分档
    print(f"\n=== 分持有期胜率 ===")
    for h in [3, 5, 10]:
        col = f"fwd{h}"
        if col not in panel.columns:
            continue
        day_wr = []
        for date, g in panel.groupby("date"):
            g = g.dropna(subset=[col, "composite"])
            if len(g) < 50:
                continue
            top = g.nlargest(args.top_n, "composite")
            day_wr.append(float(top[col].mean()))
        arr = np.array(day_wr)
        wr = (arr > 0).mean() * 100
        print(f"  hold={h}日: 胜率={wr:.1f}%  平均={arr.mean():+.3f}%  Sharpe={arr.mean()/arr.std()*np.sqrt(252/h):.2f}")


if __name__ == "__main__":
    main()
