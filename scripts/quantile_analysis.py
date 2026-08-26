# -*- coding: utf-8 -*-
"""分位数分层分析 + 合成 alpha 因子检验。

基于 market_panel_daily.csv(每日截面, ~5000只×245天)：
  1. 单因子十分位分层：Q1(最低)..Q10(最高)，看各组 fwd 收益单调性与多空价差(Q10-Q1)
  2. 合成信号：
       reversal6  = -mean(z(trend, strategy, chip, relative_strength,
                            capital_behavior, ashare_specific))   # 六维负IC取反
       positive2  = mean(z(quant_behavior, oversold_rebound))        # 双正因子
       alpha_comb = standardize(reversal6) + standardize(positive2)  # 合成alpha
  3. 对每个信号输出: Q10-Q1 年化、单调性得分(Spearman of 组均值)、IC

用法: python scripts/quantile_analysis.py [--panel reports/market_panel_daily.csv]
"""
import os
import sys
import json
import argparse
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

NEG_DIMS = ["trend", "strategy", "chip", "relative_strength",
            "capital_behavior", "ashare_specific"]
POS_DIMS = ["quant_behavior", "oversold_rebound"]
# 经典K线因子扩展(2026-08): rev20/open_gap 强正; pos_250/max20 强负
POS_DIMS_V2 = POS_DIMS + ["rev20", "open_gap", "deep_research"]
NEG_DIMS_V2 = NEG_DIMS + ["pos_250", "max20"]


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 1e-9 else s * 0


def build_signals(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    for c in NEG_DIMS + POS_DIMS:
        if c not in df.columns:
            raise SystemExit(f"面板缺少维度: {c}")
    # 截面内 z-score（按日）
    df = df.groupby("date", group_keys=False).apply(
        lambda g: g.assign(**{f"z_{c}": zscore(g[c]) for c in NEG_DIMS + POS_DIMS}))
    neg_cols = [f"z_{c}" for c in NEG_DIMS]
    pos_cols = [f"z_{c}" for c in POS_DIMS]
    df["reversal6"] = -df[neg_cols].mean(axis=1)
    df["positive2"] = df[pos_cols].mean(axis=1)
    # v2 合成: 纳入经典因子(全市场IC终审显著者)
    have_pos = [c for c in POS_DIMS_V2 if f"z_{c}" in df.columns]
    have_neg = [c for c in NEG_DIMS_V2 if f"z_{c}" in df.columns]
    df["positive_v2"] = df[have_pos].mean(axis=1)
    if have_neg:
        df["alpha_v2"] = df.groupby("date", group_keys=False).apply(
            lambda g: zscore(g["positive_v2"]) - zscore(g[have_neg].mean(axis=1)))
    else:
        df["alpha_v2"] = df["positive_v2"]
    # 合成：两信号各自按日标准化后相加
    df["alpha_comb"] = df.groupby("date", group_keys=False).apply(
        lambda g: zscore(g["reversal6"]) + zscore(g["positive2"]))
    return df


def quantile_table(df: pd.DataFrame, signal: str, ret_col: str,
                   n_bins: int = 10) -> Dict:
    """按日分十组 → 各组平均收益 → 多空与单调性"""
    def _bins(g):
        # rank 分桶: 对大量重复默认值(如中性50)的因子远比 qcut 值域分桶稳健
        rk = g[signal].rank(method="first")
        return pd.qcut(rk, n_bins, labels=False)

    df = df.copy()
    df["bin"] = df.groupby("date", group_keys=False).apply(_bins)
    df = df.dropna(subset=["bin"])
    grp = df.groupby(["date", "bin"])[ret_col].mean().unstack()
    if grp.shape[1] < n_bins:
        return {}
    q_low = grp.iloc[:, 0]      # Q1 信号最低
    q_high = grp.iloc[:, -1]    # Q10 信号最高
    ls = (q_high - q_low).dropna()          # 日度多空价差(%)
    # winsorize: 涨跌停/流动性极差的票会制造单日极端价差，
    # 截断到±8%(约2个跌停板幅度)防止均值被少数日期拉爆
    ls = ls.clip(-8, 8)
    group_means = grp.mean().clip(-15, 15)
    # 单调性: 组均值与组序号的 spearman
    mono = float(pd.Series(group_means.values).rank().corr(
        pd.Series(group_means.index.astype(float)).rank()))
    ic = float(pd.Series(ls).rank().corr(pd.Series(range(1, len(ls) + 1)).rank())) \
        if len(ls) > 3 else np.nan
    mean_ls = float(ls.mean())
    std_ls = float(ls.std())
    t_stat = mean_ls / (std_ls / len(ls) ** 0.5) if std_ls > 1e-9 else 0
    # 简单年化(日均%×252)：复利外推对短线价差会天文数字化，无意义
    ann = mean_ls * 252
    win = float((ls > 0).mean() * 100)
    return {
        "signal": signal, "ret_col": ret_col,
        "q_groups": {str(int(k)): round(float(v), 3)
                     for k, v in group_means.items()},
        "monotonicity": round(mono, 3),
        "ls_mean_pct": round(mean_ls, 4),
        "ls_tstat": round(t_stat, 2),
        "ls_winrate": round(win, 1),
        "ls_annualized_pct": round(ann, 1),
        "days": int(len(ls)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",
                    default=os.path.join(ROOT, "reports", "market_panel_daily.csv"))
    ap.add_argument("--ret", default="fwd5")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    print(f"面板: {len(panel)} 行 | {panel['code'].nunique()} 只 | "
          f"{panel['date'].nunique()} 截面")

    df = build_signals(panel)

    signals = (NEG_DIMS + POS_DIMS + ["reversal6", "positive2", "alpha_comb"]
               + ["positive_v2", "alpha_v2"])
    results = []
    for sig in signals:
        for ret_col in ("fwd3", "fwd5", "fwd10"):
            r = quantile_table(df, sig, ret_col)
            if r:
                results.append(r)

    # 输出主表(fwd5) + 全量json
    print(f"\n{'='*88}\n十分位分层 Q10-Q1 多空价差 (持有期={args.ret}, "
          f"日度价差均值→年化)\n{'='*88}")
    print(f"{'信号':<18}{'Q1':>8}{'Q5':>8}{'Q10':>8}{'多空bp':>9}"
          f"{'t':>7}{'胜率':>7}{'年化%':>10}{'单调性':>8}")
    for r in sorted(results, key=lambda x: -x.get("ls_tstat", 0)):
        if r["ret_col"] != args.ret:
            continue
        g = r["q_groups"]
        print(f"{r['signal']:<18}{g.get('0','-'):>8}{g.get('4','-'):>8}"
              f"{g.get('9','-'):>8}{r['ls_mean_pct']*100:>9.1f}"
              f"{r['ls_tstat']:>7.2f}{r['ls_winrate']:>6.0f}%"
              f"{r['ls_annualized_pct']:>10.1f}{r['monotonicity']:>8.2f}")

    outp = os.path.join(ROOT, "reports", "quantile_analysis.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
