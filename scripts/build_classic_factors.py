# -*- coding: utf-8 -*-
"""经典学术因子构造 + IC 批量验证。

从全市场K线缓存构造 A 股文献反复验证过的价量因子，追加到每日面板，
复用 Rank IC 框架批量终审：

  rev5        过去5日收益取反(短期反转)
  rev20       过去20日收益取反(中期反转)
  max20       近20日最大单日涨幅(MAX/博彩偏好, 文献: 负预测)
  ivol20      20日收益波动率(高波动异象, 文献: 负预测)
  amihud20    mean(|ret|/成交额)(非流动性溢价, 文献: 正预测)
  vpcorr20    价量相关20日(文献: 负预测)
  open_gap    当日开盘涨幅(open/昨收-1, 竞价情绪代理)
  turn_chg5   换手率5日变化(关注度升温, 文献: 负预测)
  pos_250     现价在一年区间位置(高位动量/低位反转)

用法: python scripts/build_classic_factors.py [--workers 6]
输出: 追加列到 reports/market_panel_daily.csv + IC报告
"""
import logging
import os
import sys
import glob
import pickle
import json
import argparse
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PANEL = os.path.join(ROOT, "reports", "market_panel_daily.csv")
SECTION_STEP = 1


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        logging.warning("捕获异常已忽略(scripts/build_classic_factors.py:38): %s", e)
        return None


def stock_series(rows):
    """baostock行 → dict of 时间序列"""
    n = len(rows)
    dates = [r[0] for r in rows]
    close = [_f(r[1]) for r in rows]
    vol = [_f(r[2]) for r in rows]
    turn = [_f(r[3]) for r in rows]
    pct = [_f(r[4]) for r in rows]
    op = [_f(r[5]) if len(r) > 5 else None for r in rows]
    return {"dates": dates, "close": close, "vol": vol,
            "turn": turn, "pct": pct, "open": op, "n": n}


def classic_factors_for(code_rows):
    """对单只股票计算全部新因子的逐日序列"""
    code, rows = code_rows
    sr = stock_series(rows)
    n, close, pct, vol, turn, op, dates = (
        sr["n"], sr["close"], sr["pct"], sr["vol"], sr["turn"], sr["open"], sr["dates"])

    import math
    out = []
    prev_close = [None] * n
    for i in range(1, n):
        if close[i - 1] and close[i - 1] > 0:
            prev_close[i] = close[i - 1]

    def safe(v):
        return round(v, 5) if v is not None and math.isfinite(v) else None

    for i in range(60, n):
        rec = {"date": dates[i], "code": code}
        c = close[i]

        # rev5 / rev20: 反转 = -(区间收益)
        def ret(k):
            if i >= k and close[i - k] and close[i - k] > 0 and c:
                return (c / close[i - k] - 1) * 100
            return None
        r5, r20 = ret(5), ret(20)
        rec["rev5"] = safe(-r5) if r5 is not None else None
        rec["rev20"] = safe(-r20) if r20 is not None else None

        # max20: 近20日最大单日涨幅
        window = [p for p in pct[max(0, i - 19):i + 1] if p is not None]
        rec["max20"] = safe(max(window)) if len(window) >= 15 else None

        # ivol20: 20日收益标准差
        if len(window) >= 15:
            m = sum(window) / len(window)
            var = sum((x - m) ** 2 for x in window) / len(window)
            rec["ivol20"] = safe(var ** 0.5)
        else:
            rec["ivol20"] = None

        # amihud20: mean(|ret| / 成交额), 成交额≈close*vol
        am = []
        for j in range(max(1, i - 19), i + 1):
            amt = (close[j] or 0) * (vol[j] or 0)
            if amt > 0 and pct[j] is not None:
                am.append(abs(pct[j]) / amt * 1e9)
        rec["amihud20"] = safe(sum(am) / len(am)) if len(am) >= 15 else None

        # vpcorr20: 收益与成交量的20日相关
        pairs = [(pct[j], vol[j]) for j in range(max(0, i - 19), i + 1)
                 if pct[j] is not None and vol[j] is not None]
        if len(pairs) >= 15:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((x - mx) * (y - my) for x, y in pairs)
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            rec["vpcorr20"] = safe(cov / ((vx * vy) ** 0.5)) if vx > 0 and vy > 0 else None
        else:
            rec["vpcorr20"] = None

        # open_gap: 当日开盘涨幅(竞价情绪)
        if op[i] and prev_close[i]:
            rec["open_gap"] = safe((op[i] / prev_close[i] - 1) * 100)
        else:
            rec["open_gap"] = None

        # turn_chg5: 换手率5日变化(关注度升温); 缺失值多的备源跳过
        t_now = turn[i]
        t_prev = turn[i - 5] if i >= 5 else None
        if t_now and t_prev and t_prev > 0:
            rec["turn_chg5"] = safe(t_now / t_prev - 1)
        else:
            rec["turn_chg5"] = None

        # pos_250: 一年价格位置(0=最低 1=最高)
        w = [x for x in close[max(0, i - 249):i + 1] if x]
        if len(w) >= 120:
            lo, hi = min(w), max(w)
            rec["pos_250"] = safe((c - lo) / (hi - lo)) if hi > lo else 0.5
        else:
            rec["pos_250"] = None

        out.append(rec)
    return out


def _load_kc():
    kc = {}
    for p in glob.glob(os.path.join(ROOT, "cache", "bt_kline_*.pkl")):
        try:
            with open(p, "rb") as f:
                for c, rows in pickle.load(f).items():
                    if rows and (c not in kc or len(rows) > len(kc[c])):
                        kc[c] = rows
        except Exception as e:
            logging.warning("捕获异常已忽略(scripts/build_classic_factors.py:153): %s", e)
            continue
    return kc


_KC = None


def _init():
    global _KC
    _KC = _load_kc()


def _work(codes):
    out = []
    kc = _KC or _load_kc()
    for c in codes:
        rows = kc.get(c)
        if rows and len(rows) >= 60:
            try:
                out.extend(classic_factors_for((c, rows)))
            except Exception as e:
                logging.warning("捕获异常已忽略(scripts/build_classic_factors.py:174): %s", e)
                continue
    return out


def rank_ic(panel, dim, ret_col, min_stocks=30):
    import pandas as pd
    ics = []
    for _, g in panel.groupby("date"):
        if len(g) < min_stocks or g[ret_col].nunique() < 2:
            continue
        ic = g[dim].rank().corr(g[ret_col].rank())
        if pd.notna(ic):
            ics.append(ic)
    return ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    panel = pd.read_csv(PANEL)
    codes = sorted(panel["code"].unique())
    print(f"面板股票: {len(codes)}")

    chunks = [codes[i::args.workers] for i in range(args.workers)]
    rows = []
    with Pool(args.workers, initializer=_init) as pool:
        for part in pool.imap_unordered(_work, chunks):
            rows.extend(part)

    ext = pd.DataFrame(rows)
    print(f"新因子行: {len(ext)}, 列: {[c for c in ext.columns if c not in ('date','code')]}")

    merged = panel.merge(ext, on=["date", "code"], how="left")
    merged.to_csv(PANEL, index=False)
    print(f"面板已更新: {PANEL} ({len(merged)}行)")

    new_dims = ["rev5", "rev20", "max20", "ivol20", "amihud20",
                "vpcorr20", "open_gap", "turn_chg5", "pos_250"]
    verdicts = []
    for dim in new_dims:
        if dim not in merged.columns:
            continue
        sub = merged.dropna(subset=[dim])
        cover = len(sub) / len(merged)
        for h in (3, 5, 10):
            ret_col = f"fwd{h}"
            ics = rank_ic(sub.dropna(subset=[ret_col]), dim, ret_col)
            if len(ics) < 30:
                continue
            arr = np.array(ics)
            mean_ic, std = float(arr.mean()), float(arr.std())
            icir = mean_ic / std if std > 1e-9 else 0
            t = icir * len(arr) ** 0.5
            verdicts.append({
                "factor": dim, "horizon": h, "ic_days": len(arr),
                "coverage": round(cover, 2),
                "mean_ic": round(mean_ic, 4), "t_stat": round(t, 2),
                "positive_ratio": round(float((arr > 0).mean()), 2),
            })

    verdicts.sort(key=lambda v: -abs(v.get("t_stat", 0)))
    print(f"\n{'='*78}\n经典因子 IC 批量终审\n{'='*78}")
    for v in verdicts:
        tag = "★显著" if abs(v["t_stat"]) >= 2 else ("(边缘)" if abs(v["t_stat"]) >= 1.5 else "")
        print(f"{v['factor']:<12} 覆盖{v['coverage']:.0%} h={v['horizon']:<2} "
              f"IC={v['mean_ic']:+.4f} t={v['t_stat']:+.2f} "
              f"正占比={v['positive_ratio']:.0%} {tag}")

    outp = os.path.join(ROOT, "reports", "classic_factors_ic.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
