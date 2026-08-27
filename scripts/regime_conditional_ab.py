# -*- coding: utf-8 -*-
"""regime 条件化反转 A/B 实验

与 regime_reversal_ab.py 的区别: 本脚本按 regime 分组后,
仅在 neutral/bull 窗口反转 trend, bear/crash 保持基线。

这与 config.yaml signal_invert 配置一致, 验证当前配置是否最优。

输出: reports/regime_conditional_ab.json

用法: python scripts/regime_conditional_ab.py [--hold 5]
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


def get_regime(date_str):
    """简易regime判断: 基于当日全市场平均涨跌幅"""
    # 这里用一个简化版: 从 market_panel_daily.csv 构造 regime
    # 实际应从 MarketRegimeFilter 获取, 但为了解耦用全市场均值近似
    return None  # 占位, 实际在 main() 中构建


def sim_regime_conditional(panel, hold, top_n, invert_regimes=None):
    """regime 条件化模拟: 仅在指定 regime 窗口反转 trend。"""
    from src.engine.market_regime import MarketRegimeFilter
    regime_filter = MarketRegimeFilter()

    dates = sorted(panel["date"].unique())
    rbs = dates[60::hold]

    # 按日构建 regime 映射: 用全市场平均 fwd5 作为当日市场状态代理
    regime_by_date = {}
    for d in dates:
        day_data = panel[panel["date"] == d]
        if len(day_data) == 0:
            continue
        avg_fwd = day_data["fwd5"].mean() if "fwd5" in day_data.columns else 0
        if avg_fwd > 0.5:
            regime_by_date[d] = "bull"
        elif avg_fwd > -0.2:
            regime_by_date[d] = "neutral"
        elif avg_fwd > -1.5:
            regime_by_date[d] = "bear"
        else:
            regime_by_date[d] = "crash"

    segs_base, segs_cond, bench = [], [], []
    for d in rbs:
        g = panel[panel["date"] == d].dropna(subset=["fwd" + str(hold), "trend"])
        if len(g) < 500:
            continue

        rg = regime_by_date.get(d, "neutral")
        should_invert = invert_regimes and rg in invert_regimes

        g = g.copy()
        if should_invert:
            g["trend"] = 100 - g["trend"]

        g["_composite"] = g[["trend"]].mean(axis=1)
        top = g.nlargest(top_n, "_composite")

        segs_cond.append({
            "date": d,
            "regime": rg,
            "inverted": should_invert,
            "net": float(top["fwd" + str(hold)].mean()) - COST,
        })

        # 基线: 不反转
        g_base = panel[panel["date"] == d].dropna(subset=["fwd" + str(hold), "trend"])
        if len(g_base) >= 500:
            g_base = g_base.copy()
            g_base["_composite"] = g_base[["trend"]].mean(axis=1)
            top_base = g_base.nlargest(top_n, "_composite")
            segs_base.append({"date": d, "net": float(top_base["fwd" + str(hold)].mean()) - COST})

        bench.append(float(g["fwd" + str(hold)].mean()))

    return segs_base, segs_cond, bench, regime_by_date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    hold = args.hold

    # 方案A: 全局反转 trend (当前 NEGATIVE_IC_DIMS 行为)
    # 方案B: 仅 neutral/bull 反转 trend (当前 config signal_invert 行为)
    # 方案C: 仅 neutral 反转 trend
    # 方案D: 不反转(纯基线)

    configs = {
        "baseline": None,
        "neutral_bull": {"neutral", "bull"},
        "neutral_only": {"neutral"},
        "all_regimes": {"neutral", "bull", "bear", "crash"},
    }

    print(f"=== regime 条件化反转 A/B | Top{args.top_n} | hold={hold}日 ===\n")

    results = {}
    for name, inv_regimes in configs.items():
        segs_base, segs_cond, bench, regime_map = sim_regime_conditional(
            panel, hold, args.top_n, inv_regimes)

        if not segs_cond:
            print(f"  {name}: 无有效分段")
            continue

        m = calc_metrics([s["net"] for s in segs_cond], hold)
        m_bench = calc_metrics(bench, hold)
        excess = round(m["annual"] - m_bench["annual"], 2)

        # 按 regime 分组统计
        regime_stats = {}
        for rg in ("bull", "neutral", "bear", "crash"):
            rg_segs = [s for s in segs_cond if s["regime"] == rg]
            if rg_segs:
                rg_m = calc_metrics([s["net"] for s in rg_segs], hold)
                regime_stats[rg] = {"segs": len(rg_segs), "annual": rg_m["annual"]}

        inv_desc = "无" if inv_regimes is None else "+".join(sorted(inv_regimes))
        print(f"  {name:<20} 反转: {inv_desc:<20} 超额: {excess:+.2f}%  "
              f"年化: {m['annual']:.2f}%  MDD: {m['mdd']:.1f}%  Sharpe: {m['sharpe']:.2f}")

        results[name] = {
            "invert_regimes": sorted(inv_regimes) if inv_regimes else None,
            "metrics": m, "benchmark": m_bench,
            "excess_annual": excess, "regime_stats": regime_stats,
        }

    out = {"hold": hold, "top_n": args.top_n, "configs": results}
    outp = os.path.join(ROOT, "reports", "regime_conditional_ab.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
