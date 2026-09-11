# -*- coding: utf-8 -*-
"""ashare_specific 单边组合模拟 + 反转A/B

方法: 从 market_panel_daily.csv 取 ashare_specific 列:
  A组(Top-N正向): 每调仓日选 ashare_specific 最高 N 只, 持有 hold 天
  B组(Bottom-N反转): 每调仓日选 ashare_specific 最低 N 只(反转逻辑), 持有 hold 天
  基准: 全池等权

用法: python scripts/ashare_sim_ab.py [--top-n 30] [--hold 5] [--hold 10]
"""
import os, json, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.36  # 双边成本%


def max_dd(curve):
    peak = np.maximum.accumulate(curve)
    return float(((peak - curve) / peak * 100).max())


def sim(panel, sig, ret, n, hold, cost=COST):
    dates = sorted(panel["date"].unique())
    rbs = dates[60::hold]
    segs, bench = [], []
    for d in rbs:
        g = panel[panel["date"] == d].dropna(subset=[ret, sig])
        if len(g) < 500:
            continue
        top = g.nlargest(n, sig)
        bot = g.nsmallest(n, sig)
        segs.append({
            "date": d,
            "long_gross": float(top[ret].mean()),
            "long_net": float(top[ret].mean()) - cost,
            "short_gross": -float(bot[ret].mean()),  # 做空端收益
            "short_net": -float(bot[ret].mean()) - cost,
            "ls_gross": float(top[ret].mean()) - float(bot[ret].mean()),
            "ls_net": float(top[ret].mean()) - float(bot[ret].mean()) - cost * 2,
        })
        bench.append(float(g[ret].mean()))
    return segs, bench


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--hold", type=int, default=5)
    args = ap.parse_args()

    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    if "ashare_specific" not in panel.columns:
        print("market_panel_daily.csv 缺少 ashare_specific 列")
        return

    for hold in ([args.hold] if args.hold else [5, 10]):
        segs, bench = sim(panel, "ashare_specific", "fwd" + str(hold), args.top_n, hold)
        if not segs:
            print(f"hold={hold}: 无有效分段")
            continue

        m_long = calc_metrics([s["long_net"] for s in segs], hold)
        m_short = calc_metrics([s["short_net"] for s in segs], hold)
        m_ls = calc_metrics([s["ls_net"] for s in segs], hold)
        m_bench = calc_metrics(bench, hold)

        print(f"\n=== ashare_specific A/B | Top{args.top_n} | hold={hold}日 ===")
        print(f"{'':14}{'多头(正向)':>14}{'空头(反转)':>14}{'多空价差':>14}{'全池基准':>14}")
        for k in ("total", "annual", "mdd", "sharpe", "winrate"):
            print(f"{k:<14}{m_long[k]:>14}{m_short[k]:>14}{m_ls[k]:>14}{m_bench[k]:>14}")

        excess = round(m_long["annual"] - m_bench["annual"], 2)
        print(f"\n多头超额年化: {excess:+.2f}%")
        if excess > 0:
            print("结论: 正向可提取 (多头有alpha)")
        else:
            print("结论: 正向不可提取 (alpha在空头端)")

        out = {
            "signal": "ashare_specific", "hold": hold, "top_n": args.top_n,
            "long": m_long, "short": m_short, "ls": m_ls, "benchmark": m_bench,
            "excess_annual": excess,
        }
        outp = os.path.join(ROOT, "reports", f"ashare_ab_hold{hold}.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"已保存: {outp}")


if __name__ == "__main__":
    main()
