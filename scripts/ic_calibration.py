# -*- coding: utf-8 -*-
"""因子 IC 方向校准分析。

基于 factor_snapshots(24维 sub_* 独立列) JOIN recommendations(premarket, 百分数
future_return) 的截面面板，按日计算 Rank IC → 均值IC / ICIR / t近似，
输出负 IC 因子清单与方向校准建议，落盘 reports/ic_calibration.json。

用法: python scripts/ic_calibration.py [--days 200] [--min-days 15]
"""
import os
import sys
import json
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--min-days", type=int, default=15,
                    help="IC 样本天数低于该值的因子不做结论")
    args = ap.parse_args()

    from src.data.database import StockDatabase
    db = StockDatabase()
    panel = db.get_sub_score_panel(days=args.days)
    if not panel:
        print("无面板数据")
        return
    df = pd.DataFrame(panel)
    df["trade_date"] = df["trade_date"].astype(str)
    dims = [c for c in df.columns if c not in ("trade_date", "code", "future_return")]
    for c in dims + ["future_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["future_return"])

    print(f"面板: {len(df)} 行 / {df['trade_date'].nunique()} 天 / {df['code'].nunique()} 只 "
          f"({df['trade_date'].min()} ~ {df['trade_date'].max()})")

    # 按日 Rank IC
    ic_by_dim = {d: [] for d in dims}
    for date, g in df.groupby("trade_date"):
        if len(g) < 5 or g["future_return"].nunique() < 2:
            continue
        ret_rank = g["future_return"].rank()
        for d in dims:
            v = g[d]
            if v.nunique() < 2:
                continue
            ic = v.rank().corr(ret_rank)
            if pd.notna(ic):
                ic_by_dim[d].append((date, ic))

    results = []
    for d, series in ic_by_dim.items():
        if len(series) < args.min_days:
            results.append({"factor": d, "ic_days": len(series), "verdict": "insufficient_samples"})
            continue
        ics = np.array([v for _, v in series])
        mean_ic = float(ics.mean())
        std_ic = float(ics.std())
        icir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
        # t 近似: ICIR * sqrt(天数)
        t_stat = icir * (len(ics) ** 0.5)
        positive_ratio = float((ics > 0).mean())
        recent = np.array([v for dt, v in series[-20:]])
        recent_ic = float(recent.mean())

        if abs(mean_ic) < 0.02 or abs(t_stat) < 1.0:
            verdict = "weak"
        elif mean_ic > 0:
            verdict = "positive_keep"
        else:
            verdict = "negative_reverse_or_downweight"
        results.append({
            "factor": d,
            "ic_days": len(ics),
            "mean_ic": round(mean_ic, 4),
            "icir": round(icir, 4),
            "t_stat": round(t_stat, 2),
            "positive_day_ratio": round(positive_ratio, 3),
            "recent_20d_ic": round(recent_ic, 4),
            "verdict": verdict,
        })

    out = sorted(results, key=lambda r: r.get("mean_ic", 0))
    neg = [r for r in out if r.get("verdict") == "negative_reverse_or_downweight"]
    pos = [r for r in reversed(out) if r.get("verdict") == "positive_keep"]

    print(f"\n=== 显著负 IC 因子({len(neg)}) — 候选降权/取反 ===")
    for r in neg:
        print(f"  {r['factor']:<22} IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} "
              f"t={r['t_stat']:+.1f} 正IC天占比={r['positive_day_ratio']:.0%}")

    print(f"\n=== 显著正 IC 因子({len(pos)}) ===")
    for r in pos:
        print(f"  {r['factor']:<22} IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} "
              f"t={r['t_stat']:+.1f} 正IC天占比={r['positive_day_ratio']:.0%}")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out_path = os.path.join(ROOT, "reports", "ic_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "panel_rows": len(df),
            "results": out,
            "negative_factors": [r["factor"] for r in neg],
            "positive_factors": [r["factor"] for r in pos],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
