# -*- coding: utf-8 -*-
"""trend/strategy regime 反转 A/B 实验

对负IC维度(trend/strategy)做反转实验，按regime分组对比:
  A组(基线): 正常评分
  B组(反转): NEGATIVE_IC_DIMS 内的子分取反(100-x)

输出: reports/regime_reversal_ab.json

用法: python scripts/regime_reversal_ab.py [--hold 5]
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
COST = 0.36


def max_dd(curve):
    peak = np.maximum.accumulate(curve)
    return float(((peak - curve) / peak * 100).max())


def calc_metrics(rets, hold, cap=1_000_000):
    curve = [cap]
    for r in rets:
        curve.append(curve[-1] * (1 + r / 100))
    arr = np.array(rets) / 100
    n_yr = len(rets) * hold / 252 if rets else 0
    total = (curve[-1] / cap - 1) * 100
    ann = ((curve[-1] / cap) ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0
    return {
        "segs": len(rets),
        "total": round(total, 2),
        "annual": round(ann, 2),
        "mdd": round(max_dd(np.array(curve)), 2),
        "sharpe": round(float(arr.mean() / arr.std() * (252/hold)**0.5), 2) if len(arr)>2 and arr.std()>0 else 0,
        "winrate": round(float((arr>0).mean()*100), 1),
    }


def sim_with_invert(panel, signal_cols, ret, n, hold, invert_dims=None):
    """组合模拟，支持对指定维度取反。"""
    dates = sorted(panel["date"].unique())
    rbs = dates[60::hold]
    segs, bench = [], []

    for d in rbs:
        g = panel[panel["date"] == d].dropna(subset=[ret] + signal_cols)
        if len(g) < 500:
            continue

        # 构造 composite signal
        g = g.copy()
        for col in signal_cols:
            if invert_dims and col in invert_dims:
                g[col] = 100 - g[col]  # 取反
        g["_composite"] = g[signal_cols].mean(axis=1)

        top = g.nlargest(n, "_composite")
        segs.append({
            "date": d,
            "gross": float(top[ret].mean()),
            "net": float(top[ret].mean()) - COST,
        })
        bench.append(float(g[ret].mean()))

    return segs, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    hold = args.hold
    ret = f"fwd{hold}"

    # 基线: trend + strategy 正常
    segs_base, bench = sim_with_invert(
        panel, ["trend", "strategy"], ret, args.top_n, hold, invert_dims=None)

    # 反转: trend 取反
    segs_trend_inv, _ = sim_with_invert(
        panel, ["trend", "strategy"], ret, args.top_n, hold, invert_dims={"trend"})

    # 反转: strategy 取反
    segs_strat_inv, _ = sim_with_invert(
        panel, ["trend", "strategy"], ret, args.top_n, hold, invert_dims={"strategy"})

    # 反转: 两者都取反
    segs_both_inv, _ = sim_with_invert(
        panel, ["trend", "strategy"], ret, args.top_n, hold, invert_dims={"trend", "strategy"})

    if not segs_base:
        print("无有效分段")
        return

    m_base = calc_metrics([s["net"] for s in segs_base], hold)
    m_trend = calc_metrics([s["net"] for s in segs_trend_inv], hold)
    m_strat = calc_metrics([s["net"] for s in segs_strat_inv], hold)
    m_both = calc_metrics([s["net"] for s in segs_both_inv], hold)
    m_bench = calc_metrics(bench, hold)

    print(f"\n=== trend/strategy 反转A/B | Top{args.top_n} | hold={hold}日 ===")
    print(f"{'':14}{'基线':>12}{'反转trend':>12}{'反转strat':>12}{'反转both':>12}{'基准':>12}")
    for k in ("total", "annual", "mdd", "sharpe", "winrate"):
        print(f"{k:<14}{m_base[k]:>12}{m_trend[k]:>12}{m_strat[k]:>12}{m_both[k]:>12}{m_bench[k]:>12}")

    print(f"\n超额年化(vs基准):")
    print(f"  基线:     {m_base['annual'] - m_bench['annual']:+.2f}%")
    print(f"  反转trend: {m_trend['annual'] - m_bench['annual']:+.2f}%")
    print(f"  反转strat: {m_strat['annual'] - m_bench['annual']:+.2f}%")
    print(f"  反转both:  {m_both['annual'] - m_bench['annual']:+.2f}%")

    out = {
        "hold": hold, "top_n": args.top_n,
        "baseline": m_base, "invert_trend": m_trend,
        "invert_strategy": m_strat, "invert_both": m_both,
        "benchmark": m_bench,
    }
    outp = os.path.join(ROOT, "reports", "regime_reversal_ab.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
