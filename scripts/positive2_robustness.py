# -*- coding: utf-8 -*-
"""positive2 合成信号稳健性体检。

上线前尽职调查三问：
  1. 市值中性：多空价差是否被小市值组贡献？(小票不可重仓)
  2. 时间稳定性：按月分段 Q10-Q1，失效月占比多少？
  3. 成本后净值：单边多头(Q10相对全池超额)扣交易成本还剩多少？

用法: python scripts/positive2_robustness.py
"""
import os
import sys
import glob
import pickle
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

PANEL = os.path.join(ROOT, "reports", "market_panel_daily.csv")
COST_PER_SIDE_PCT = 0.18   # 单边成本%(佣金+印花税+滑点近似)
HOLD = 10                  # 持有期(天)，轮动成本 = 双边/持有天数


def load_size_proxy():
    """从K线缓存取每只股票的平均日成交额(close×volume)作为规模/流动性代理"""
    out = {}
    for p in glob.glob(os.path.join(ROOT, "cache", "bt_kline_*.pkl")):
        try:
            with open(p, "rb") as f:
                for c, rows in pickle.load(f).items():
                    if c in out:
                        continue
                    vals = []
                    for r in rows:
                        try:
                            amt = float(r[1]) * float(r[2])
                            if amt > 0:
                                vals.append(amt)
                        except (ValueError, TypeError, IndexError):
                            continue
                    if vals:
                        out[c] = float(np.median(vals))
        except Exception:
            continue
    return out


def q10_q1_by_group(df: pd.DataFrame, group_col: str, ret_col="fwd10",
                    n_bins=10) -> pd.DataFrame:
    """在 group_col 各组内做十分位分层，返回 {组: (Q10-Q1均值bp, t)}"""
    res = []

    def _one(g):
        try:
            rk = g[signal_col].rank(method="first")
            b = pd.qcut(rk, n_bins, labels=False)
        except ValueError:
            return None
        gg = g.assign(_b=b)
        hi = gg[gg["_b"] == n_bins - 1][ret_col]
        lo = gg[gg["_b"] == 0][ret_col]
        if len(hi) < 5 or len(lo) < 5:
            return None
        return hi.mean() - lo.mean()

    signal_col = "positive2"
    for key, g in df.groupby(group_col):
        if len(g) < 100:
            continue
        d = _one(g)
        if d is not None and np.isfinite(d):
            res.append({"group": str(key), "ls_bp": round(d * 100, 1)})
    return pd.DataFrame(res)


def main():
    panel = pd.read_csv(PANEL)
    print(f"面板: {len(panel)} 行 / {panel['code'].nunique()} 只")

    # 构造 positive2(截面z和)
    def zscore(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 1e-9 else s * 0

    panel = panel.groupby("date", group_keys=False).apply(
        lambda g: g.assign(
            positive2=zscore(g["quant_behavior"]) + zscore(g["oversold_rebound"])))
    # 多头超额: Q10组均值 - 全体均值
    panel["excess_ret"] = np.nan

    print("\n=== ① 市值中性检验(按流动性五分位, fwd10 Q10-Q1 bp) ===")
    size_map = load_size_proxy()
    panel["amt_proxy"] = panel["code"].map(size_map)
    has_size = panel.dropna(subset=["amt_proxy"]).copy()
    # 规模跨期稳定: 每股取中位成交额→全局五分位(避免按日qcut的重复边界问题)
    code_amt = has_size.groupby("code")["amt_proxy"].median()
    code_q = pd.qcut(code_amt.rank(method="first"), 5, labels=False)
    has_size["size_q"] = has_size["code"].map(code_q).astype("Int64")
    t_all = q10_q1_by_group(has_size, "size_q")
    print(t_all.to_string(index=False))
    small = t_all[t_all["group"].isin(["0"])]["ls_bp"]
    large = t_all[t_all["group"].isin(["4"])]["ls_bp"]
    if len(small):
        print(f"→ 最小市值组 Q10-Q1 = {small.values[0]}bp | "
              f"最大市值组 = {large.values[0] if len(large) else '-'}bp")

    print("\n=== ② 月度时间稳定性(fwd10 Q10-Q1 bp) ===")
    panel["month"] = panel["date"].str[:7]
    monthly = q10_q1_by_group(panel, "month")
    monthly = monthly.sort_values("group")
    print(monthly.to_string(index=False))
    neg_months = monthly[monthly["ls_bp"] < 0]
    print(f"→ 失效月(负价差): {len(neg_months)}/{len(monthly)}")

    print("\n=== ③ 成本后单边多头净值(fwd10, 日度轮动口径) ===")
    def _q10_excess(g):
        try:
            rk = g["positive2"].rank(method="first")
            b = pd.qcut(rk, 10, labels=False)
        except ValueError:
            return None
        top = g[b == 9]["fwd10"]
        if len(top) < 5:
            return None
        return top.mean() - g["fwd10"].mean()   # 相对全池超额(%)

    daily = panel.groupby("date", group_keys=False).apply(_q10_excess).dropna()
    daily = daily.clip(-15, 15)
    gross_ann = daily.mean() * 252
    # 每日全换手轮动的成本: 双边0.36% × 252; 若持有10日则摊薄为 0.36%×252/10
    cost_full = COST_PER_SIDE_PCT * 2 * 252
    cost_hold = COST_PER_SIDE_PCT * 2 * (252 / HOLD)
    print(f"有效截面日: {len(daily)} | 多头超额均值: {daily.mean():+.3f}%/日 "
          f"(胜率 {(daily > 0).mean()*100:.0f}%)")
    print(f"毛年化(简单): {gross_ann:+.1f}%")
    print(f"成本情景[每日全轮动]: -{cost_full:.0f}%/年 → 净 {gross_ann - cost_full:+.1f}%")
    print(f"成本情景[持有{HOLD}日轮动]: -{cost_hold:.0f}%/年 → 净 {gross_ann - cost_hold:+.1f}%")
    print("注: 多头组合实际只买Q10约500只, 等权小仓位可行; 大资金需再看流动性容量")

    out = {
        "size_neutral": t_all.to_dict("records"),
        "monthly": monthly.to_dict("records"),
        "daily_excess_mean_pct": round(float(daily.mean()), 4),
        "daily_winrate": round(float((daily > 0).mean()) * 100, 1),
        "gross_annualized": round(float(gross_ann), 1),
        "net_annual_daily_rotate": round(float(gross_ann - cost_full), 1),
        f"net_annual_hold{HOLD}": round(float(gross_ann - cost_hold), 1),
    }
    outp = os.path.join(ROOT, "reports", "positive2_robustness.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
