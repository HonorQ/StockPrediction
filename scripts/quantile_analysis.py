# -*- coding: utf-8 -*-
"""十分位分层分析 —— 全市场面板直读版。

2026-10-16 重写: 旧版从 SQLite 按 K 线行数取 `LIMIT 100` 小样本重算因子
(选样偏差), 且跨日样本直接混合聚合(忽略截面日相关性), 与全市场 verdict
口径矛盾(见 reports/research/feedback_summary_2026-10-15.md 技术债2)。
新版直读 rebuild 产出的 reports/market_panel_daily.csv(全市场 ~5200 只),
先按截面日算每日十分位收益, 再对日序列求均值/t —— 与 verdict 同一日频口径。

收益口径: 面板 fwdN 列为百分数(1.0 = 1%), 分位收益同为百分数。
注意: 分层多空价差 ≠ 多头可提取 alpha(空头端信号在A股不可交易,
见 AGENTS 红线5), 判定以顶部十分位单边收益为准。

用法:
  python scripts/quantile_analysis.py                        # 全部有效因子, h10
  python scripts/quantile_analysis.py --horizon 5            # 全部有效因子, h5
  python scripts/quantile_analysis.py --factor quant_behavior --horizon 3
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PANEL_PATH = os.path.join(ROOT, "reports", "market_panel_daily.csv")
OUT_DEFAULT = os.path.join(ROOT, "reports", "quantile_analysis.json")
META_COLS = ("date", "code")


def load_panel():
    df = pd.read_csv(PANEL_PATH)
    if "fwd10" not in df.columns:
        raise SystemExit("面板缺少 fwd10 列, 请先跑 scripts/rebuild_market_panel.py")
    return df


def list_factors(panel):
    """面板中真实有区分度的因子列(nunique>10 且非全 NaN)"""
    dims = [c for c in panel.columns
            if c not in META_COLS and not c.startswith("fwd")]
    return [c for c in dims if panel[c].nunique(dropna=True) > 10]


def daily_quantiles(panel, factor, ret_col, n_quantiles=10, min_universe=50):
    """逐日按 factor 分十分位 → [{date, quantile, ret, count}]

    跳过样本 < min_universe 或当日因子取值不足 n_quantiles 个的截面。
    """
    rows = []
    sub = panel[["date", factor, ret_col]].dropna()
    for date, g in sub.groupby("date"):
        if len(g) < min_universe or g[factor].nunique() < n_quantiles:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g[factor].rank(method="first"), n_quantiles, labels=False)
        for q, qg in g.groupby("q"):
            rows.append({"date": date, "quantile": int(q),
                         "ret": float(qg[ret_col].mean()), "count": int(len(qg))})
    return pd.DataFrame(rows)


def daily_by_level(panel, factor, ret_col, base_level=None, min_universe=100):
    """逐日按 factor 的档位取值分组 → [{date, quantile(档位值), ret, count}]

    离散/分档语义因子(如 quant_behavior 的 38/48/50/58 档)专用:
    qcut 十分位会把"并列众数档"按行序乱切, 必须按档位原值分组。
    超额以 base_level(缺省=全期众数档, 即中性档) 当日的均值为基准。
    """
    if base_level is None:
        base_level = panel[factor].mode().iloc[0]
    sub = panel[["date", factor, ret_col]].dropna()
    rows = []
    for date, g in sub.groupby("date"):
        if len(g) < min_universe:
            continue
        base = g[g[factor] == base_level][ret_col]
        if len(base) < 20:
            continue
        base_mean = float(base.mean())
        for lvl, lg in g.groupby(factor):
            rows.append({"date": str(date), "quantile": float(lvl),
                         "ret": float(lg[ret_col].mean()) - base_mean,
                         "count": int(len(lg))})
    return pd.DataFrame(rows), float(base_level)


def _series_stats(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return {}
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    t = float(arr.mean() / sd * np.sqrt(n)) if sd > 1e-9 else 0.0
    return {
        "mean": round(float(arr.mean()), 4),
        "t_stat": round(t, 2),
        "positive_days": round(float((arr > 0).mean()), 3),
        "days": n,
    }


def analyze_by_quantile(panel, factor, ret_col, n_quantiles=10):
    """连续因子: 十分位统计 → dict(quantiles/spread/top_decile/monotonicity)"""
    qdf = daily_quantiles(panel, factor, ret_col, n_quantiles)
    if qdf.empty:
        return None

    stats = []
    for q in range(n_quantiles):
        s = qdf[qdf["quantile"] == q]["ret"]
        st = _series_stats(s)
        if st:
            st["quantile"] = q
            st["avg_count"] = int(qdf[qdf["quantile"] == q]["count"].mean())
            stats.append(st)

    q1 = qdf[qdf["quantile"] == 0].set_index("date")["ret"]
    q10 = qdf[qdf["quantile"] == n_quantiles - 1].set_index("date")["ret"]
    spread = (q10 - q1).dropna()
    top = _series_stats(q10)

    means = [s["mean"] for s in stats]
    mono = float(pd.Series(range(len(means))).corr(pd.Series(means), method="spearman")) \
        if len(means) >= 3 else None

    if top.get("t_stat", 0) >= 2.0:
        verdict = "正信号(顶部十分位显著为正)"
    elif top.get("t_stat", 0) <= -2.0:
        verdict = "负信号(顶部十分位显著为负, 宜作排除条件)"
    else:
        verdict = "无效(顶部十分位不显著)"

    return {
        "factor": factor,
        "horizon": ret_col,
        "mode": "quantile",
        "quantiles": stats,
        "spread_q10_q1": _series_stats(spread),
        "top_decile": top,
        "monotonicity_rho": round(mono, 3) if mono is not None else None,
        "verdict": verdict,
    }


def analyze_by_level(panel, factor, ret_col, min_level_count=10):
    """离散/分档因子: 按档位原值分组, 统计各档 vs 中性档(众数)超额"""
    ldf, base = daily_by_level(panel, factor, ret_col)
    if ldf.empty:
        return None

    levels = []
    for lvl, s in ldf.groupby("quantile"):
        st = _series_stats(s["ret"])
        if not st:
            continue
        st["level"] = float(lvl)
        st["avg_count"] = int(s["count"].mean())
        levels.append(st)
    levels.sort(key=lambda s: s["level"])

    pos = [s["level"] for s in levels
           if s["t_stat"] >= 2.0 and s["avg_count"] >= min_level_count]
    neg = [s["level"] for s in levels
           if s["t_stat"] <= -2.0 and s["avg_count"] >= min_level_count]
    if pos and neg:
        verdict = f"混合(正档 {pos} / 负档 {neg})"
    elif pos:
        verdict = f"档位正信号: {pos}"
    elif neg:
        verdict = f"档位负信号(排除用): {neg}"
    else:
        verdict = "无效(无显著档位)"

    return {
        "factor": factor,
        "horizon": ret_col,
        "mode": "level",
        "base_level": base,
        "levels": levels,
        "positive_levels": pos,
        "negative_levels": neg,
        "verdict": verdict,
    }


def analyze_factor(panel, factor, ret_col, n_quantiles=10, discrete_max=30):
    """自适应分层: 档位少(<=discrete_max)的离散因子按档位, 否则十分位"""
    nun = panel[factor].nunique(dropna=True)
    if nun <= discrete_max:
        return analyze_by_level(panel, factor, ret_col)
    return analyze_by_quantile(panel, factor, ret_col, n_quantiles)


def print_result(res):
    f, h = res["factor"], res["horizon"]
    if res.get("mode") == "level":
        print(f"\n=== {f} ({h}) [档位模式 base={res['base_level']:.0f}] "
              f"| {res['verdict']} ===")
        print(f"{'档位':>5}{'超额%':>10}{'t':>7}{'正日占比':>9}{'日均只数':>9}")
        for s in res["levels"]:
            print(f"{s['level']:>5.0f}{s['mean']:>+10.4f}{s['t_stat']:>7.2f}"
                  f"{s['positive_days']:>9.0%}{s['avg_count']:>9}")
        return
    print(f"\n=== {f} ({h}) | 单调性 rho={res['monotonicity_rho']} | {res['verdict']} ===")
    print(f"{'分位':<5}{'日均收益%':>10}{'t':>7}{'正日占比':>9}{'日均样本':>9}")
    for s in res["quantiles"]:
        print(f"Q{s['quantile']:<4}{s['mean']:>10.4f}{s['t_stat']:>7.2f}"
              f"{s['positive_days']:>9.0%}{s['avg_count']:>9}")
    sp = res["spread_q10_q1"]
    td = res["top_decile"]
    print(f"Q10-Q1: {sp['mean']:+.4f}% t={sp['t_stat']:+.2f} | "
          f"顶部十分位: {td['mean']:+.4f}% t={td['t_stat']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default=None, help="因子列名; 缺省=全部有效因子")
    ap.add_argument("--horizon", type=int, default=10, choices=(3, 5, 10))
    ap.add_argument("--n-quantiles", type=int, default=10)
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    t0 = time.time()
    panel = load_panel()
    ret_col = f"fwd{args.horizon}"
    print(f"面板: {len(panel)} 行 x {panel['code'].nunique()} 只 "
          f"x {panel['date'].nunique()} 天 | 收益列 {ret_col}")

    factors = [args.factor] if args.factor else list_factors(panel)
    results = {}
    for f in factors:
        res = analyze_factor(panel, f, ret_col, args.n_quantiles)
        if res is None:
            print(f"\n=== {f} ({ret_col}) 跳过: 有效截面不足 ===")
            continue
        results[f] = res
        print_result(res)

    if not results:
        raise SystemExit("无有效结果")

    def _best_t(res):
        if res.get("mode") == "level":
            ts = [s["t_stat"] for s in res["levels"] if s["avg_count"] >= 10]
            return max(ts) if ts else 0.0
        return res["top_decile"].get("t_stat", 0.0)

    ranked = sorted(results.items(), key=lambda kv: -_best_t(kv[1]))
    print("\n按最佳档位/分位 t 排名: " +
          ", ".join(f"{k}({_best_t(v):+.1f})" for k, v in ranked))

    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "panel_rows": int(len(panel)),
        "horizon": args.horizon,
        "n_quantiles": args.n_quantiles,
        "note": "逐日截面分位→日序列t; 收益为百分数; 多空价差≠可提取alpha, 看顶部十分位",
        "results": results,
    }
    outp = args.out or (OUT_DEFAULT if args.factor is None
                        else os.path.join(ROOT, "reports",
                                          f"quantile_{args.factor}_h{args.horizon}.json"))
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp} (耗时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
