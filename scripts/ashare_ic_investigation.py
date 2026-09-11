# -*- coding: utf-8 -*-
"""ashare_specific 多源IC差异调查

全市场IC=-0.062(t=-5.47) vs 推荐子集IC=+0.006(t=0.3), 差6个标准误。
调查根因: 是否小票效应/市值分层/子因子贡献差异。

输出: reports/ashare_ic_investigation.json

用法: python scripts/ashare_ic_investigation.py
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rank_ic(group, factor_col, ret_col):
    """截面 Rank IC"""
    valid = group[[factor_col, ret_col]].dropna()
    if len(valid) < 5 or valid[ret_col].nunique() < 2:
        return np.nan
    return valid[factor_col].rank().corr(valid[ret_col].rank())


def main():
    # 1. 加载全市场面板
    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    print(f"全市场面板: {len(panel)} 行 / {panel['date'].nunique()} 天 / {panel['code'].nunique()} 只")

    # 2. 按日计算 ashare_specific 的 Rank IC (fwd5)
    ic_by_date = []
    for date, g in panel.groupby("date"):
        ic = rank_ic(g, "ashare_specific", "fwd5")
        if not np.isnan(ic):
            ic_by_date.append({"date": date, "ic": ic})

    ic_series = pd.DataFrame(ic_by_date)
    overall_ic = ic_series["ic"].mean()
    overall_t = ic_series["ic"].mean() / ic_series["ic"].std() * np.sqrt(len(ic_series)) if ic_series["ic"].std() > 0 else 0
    print("\n=== 全市场 ashare_specific IC (fwd5) ===")
    print(f"  均值IC: {overall_ic:+.4f}")
    print(f"  t值: {overall_t:+.2f}")
    print(f"  正IC天占比: {(ic_series['ic']>0).mean():.1%}")
    print(f"  样本天数: {len(ic_series)}")

    # 3. 按十分位分析: ashare_specific 高低组的收益差异
    print("\n=== 十分位分层 (ashare_specific → fwd5) ===")
    results_q = []
    for date, g in panel.groupby("date"):
        if len(g) < 100:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["ashare_specific"].rank(method="first"), 10, labels=False)
        for q in range(10):
            qg = g[g["q"] == q]
            if len(qg) > 0:
                results_q.append({"date": date, "q": q, "fwd5": qg["fwd5"].mean()})

    q_df = pd.DataFrame(results_q)
    q_stats = q_df.groupby("q")["fwd5"].agg(["mean", "std", "count"])
    q_stats["t"] = q_stats["mean"] / q_stats["std"] * np.sqrt(q_stats["count"])

    for q in range(10):
        row = q_stats.loc[q]
        print(f"  Q{q}: 均值={row['mean']:+.3f}% t={row['t']:+.2f} n={int(row['count'])}")

    q10_q1 = q_stats.loc[9, "mean"] - q_stats.loc[0, "mean"]
    print(f"  Q10-Q1 价差: {q10_q1:+.3f}%")

    # 4. 子因子贡献分析: 哪些子因子驱动 ashare_specific
    print("\n=== ashare_specific 子因子 IC 贡献 ===")
    sub_factors = ["reversion_signal", "mean_reversion_score", "volatility_score",
                   "position_score", "sentiment_cycle", "hot_money_score"]
    # 这些字段在 market_panel_daily.csv 中可能不存在, 尝试检测
    available = [f for f in sub_factors if f in panel.columns]
    if not available:
        # 从 factor_snapshots 补充
        print("  (market_panel_daily 无子因子列, 从 factor_snapshots 分析)")
        import sqlite3
        db_path = os.path.join(ROOT, "data", "stock_history.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # 取最近一天的 factor_snapshots
        cur = conn.cursor()
        cur.execute("SELECT MAX(trade_date) FROM factor_snapshots")
        max_date = cur.fetchone()[0]
        cur.execute("""
            SELECT fs.code, fs.sub_ashare_specific, fs.factors
            FROM factor_snapshots fs
            WHERE fs.trade_date = ? AND fs.sub_ashare_specific IS NOT NULL
              AND fs.sub_ashare_specific != 50
        """, (max_date,))
        rows = cur.fetchall()
        print(f"  最近一天({max_date}): {len(rows)} 只有真实ashare_specific值")

        # 解析 factors JSON 中的子因子
        factor_vals = {}
        for r in rows:
            try:
                f = json.loads(r["factors"]) if r["factors"] else {}
            except:
                continue
            for sf in ["reversion_signal", "mean_reversion_score", "volatility_score",
                        "position_score", "hot_money_score"]:
                v = f.get(sf)
                if v is not None and not isinstance(v, (list, dict)):
                    if sf not in factor_vals:
                        factor_vals[sf] = []
                    factor_vals[sf].append(float(v) if isinstance(v, (int, float)) else 50)

        for sf, vals in factor_vals.items():
            print(f"  {sf}: 均值={np.mean(vals):.1f} 标准差={np.std(vals):.1f} 范围=[{np.min(vals):.1f}, {np.max(vals):.1f}]")
        conn.close()
    else:
        for sf in available:
            ic = rank_ic(panel, sf, "fwd5")
            print(f"  {sf} → fwd5 IC: {ic:+.4f}")

    # 5. 结论
    print("\n=== 结论 ===")
    if overall_ic > 0:
        print(f"  全市场IC={overall_ic:+.4f}(正), 但t={overall_t:+.2f}{'不显著' if abs(overall_t)<2 else '显著'}")
        print("  推荐子集IC=+0.006(t=0.3, 不显著)")
        print("  → ashare_specific 在两个子群方向一致(弱正), 但统计不显著")
        print("  → 之前的 market_ic_verdict 全市场IC=-0.062(t=-5.47) 可能受极端值驱动")
    else:
        print(f"  全市场IC={overall_ic:+.4f}(负), t={overall_t:+.2f}")
        print("  推荐子集IC=+0.006(正, 不显著)")
        print("  → IC方向在子群间翻转, 可能是小票效应")
        print("  → 推荐池过滤掉了小票, 剩余大票的ashare_specific行为不同")

    # 保存
    out = {
        "overall_ic": round(overall_ic, 4),
        "overall_t": round(overall_t, 2),
        "positive_day_ratio": round(float((ic_series["ic"]>0).mean()), 3),
        "ic_days": len(ic_series),
        "q10_q1_spread": round(q10_q1, 3),
        "quantile_stats": {str(q): {"mean": round(q_stats.loc[q, "mean"], 3),
                                     "t": round(q_stats.loc[q, "t"], 2)}
                           for q in range(10)},
    }
    outp = os.path.join(ROOT, "reports", "ashare_ic_investigation.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
