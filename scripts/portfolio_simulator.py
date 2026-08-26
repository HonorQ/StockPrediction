# -*- coding: utf-8 -*-
"""positive2 多头组合模拟器 —— 从统计信号到可信资金曲线。

方法(非重叠持有段, 与体检的成本结论一致)：
  - 每隔 hold_days 个交易日为一个调仓日
  - 该日选 positive2 截面前 top_n 只(默认50), 段收益 = 组内 fwd{hold} 等权均值
  - 扣除双边成本(佣金+印花税+滑点 ≈ 0.36%)
  - 分段连乘成资金曲线; 基准 = 全池等权同口径

输出: 年化/最大回撤/夏普/胜率 + reports/portfolio_sim.json

用法: python scripts/portfolio_simulator.py [--top-n 50] [--hold 10] [--capital 1000000]
"""
import os
import sys
import json
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

COST_ROUNDTRIP_PCT = 0.36   # 双边成本%(佣金万2.5x2+印花税0.05+滑点)


def max_drawdown(curve):
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / peak * 100
    return float(dd.max())


def simulate(panel, signal="positive2", ret_col="fwd10", top_n=50,
             hold=10, capital=1_000_000, min_universe=500):
    dates = sorted(panel["date"].unique())
    # 调仓日: 每 hold 个交易日
    rebalance_days = dates[60::hold]          # 前60日留作指标预热
    segs = []
    bench_segs = []
    for d in rebalance_days:
        g = panel[panel["date"] == d].dropna(subset=[ret_col, signal])
        if len(g) < min_universe:
            continue
        top = g.nlargest(top_n, signal)
        seg_ret = float(top[ret_col].mean())          # 组合段收益%
        bench_ret = float(g[ret_col].mean())          # 全池等权基准
        segs.append({"date": d, "gross": seg_ret, "net": seg_ret - COST_ROUNDTRIP_PCT,
                     "bench": bench_ret})
        bench_segs.append(bench_ret)
    return segs, bench_segs


def metrics(seg_rets: list, hold: int, capital: float):
    """分段复利 → 绩效"""
    curve = [capital]
    for r in seg_rets:
        curve.append(curve[-1] * (1 + r / 100))
    arr = np.array(seg_rets) / 100
    total = (curve[-1] / capital - 1) * 100
    n_years = len(seg_rets) * hold / 252 if seg_rets else 0
    annual = ((curve[-1] / capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    mdd = max_drawdown(np.array(curve))
    sharpe = (arr.mean() / arr.std() * (252 / hold) ** 0.5) if len(arr) > 2 and arr.std() > 0 else 0
    win = float((arr > 0).mean() * 100) if len(arr) else 0
    return {
        "segments": len(seg_rets),
        "final_capital": round(curve[-1], 0),
        "total_return_pct": round(total, 1),
        "annualized_pct": round(annual, 1),
        "max_drawdown_pct": round(mdd, 1),
        "sharpe_per_year": round(sharpe, 2),
        "segment_winrate": round(win, 1),
    }, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--signal", default="positive_v2",
                    help="positive2/positive_v2/alpha_comb")
    args = ap.parse_args()

    panel_path = os.path.join(ROOT, "reports", "market_panel_daily.csv")
    panel = pd.read_csv(panel_path)

    def zscore(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 1e-9 else s * 0

    # positive2 若面板无该列则现场构造
    def _build(sig):
        nonlocal panel
        if sig == "positive2":
            panel = panel.groupby("date", group_keys=False).apply(
                lambda g: g.assign(positive2=zscore(g["quant_behavior"])
                                   + zscore(g["oversold_rebound"])))
        elif sig == "positive_v2":
            pos = ["quant_behavior", "oversold_rebound",
                   "rev20", "open_gap", "deep_research"]
            have = [c for c in pos if c in panel.columns]
            panel = panel.groupby("date", group_keys=False).apply(
                lambda g: g.assign(positive_v2=g[have].mean(axis=1)))
        elif sig == "alpha_comb":
            neg = ["trend", "strategy", "chip", "relative_strength",
                   "capital_behavior", "ashare_specific"]
            pos = ["quant_behavior", "oversold_rebound"]
            panel = panel.groupby("date", group_keys=False).apply(
                lambda g: g.assign(alpha_comb=zscore(g[pos].mean(axis=1))
                                   - zscore(g[neg].mean(axis=1))))

    _build(args.signal)

    print(f"=== 组合模拟: {args.signal} | Top{args.top_n} | "
          f"持有{args.hold}日 | 初始 {args.capital:,.0f} ===")

    segs, bench = simulate(panel, args.signal, f"fwd{args.hold}",
                           args.top_n, args.hold, args.capital)
    if not segs:
        print("无有效分段")
        return

    net_rets = [s["net"] for s in segs]
    gross_rets = [s["gross"] for s in segs]
    m_net, _ = metrics(net_rets, args.hold, args.capital)
    m_gross, _ = metrics(gross_rets, args.hold, args.capital)
    m_bench, _ = metrics(bench, args.hold, args.capital)

    print(f"\n{'指标':<14}{'净组合':>12}{'毛组合':>12}{'全池基准':>12}")
    for k in ("total_return_pct", "annualized_pct", "max_drawdown_pct",
              "sharpe_per_year", "segment_winrate"):
        print(f"{k:<14}{m_net[k]:>12}{m_gross[k]:>12}{m_bench[k]:>12}")

    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"signal": args.signal, "top_n": args.top_n,
                   "hold_days": args.hold, "cost_roundtrip": COST_ROUNDTRIP_PCT},
        "net": m_net, "gross": m_gross, "benchmark": m_bench,
        "excess_vs_bench_annual": round(
            m_net["annualized_pct"] - m_bench["annualized_pct"], 1),
    }
    outp = os.path.join(ROOT, "reports", "portfolio_sim.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
