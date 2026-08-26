# -*- coding: utf-8 -*-
"""全市场因子截面重建 + 大样本 IC 终审。

目的：
  现有 IC 校准基于"被推荐过的票"(115天×284只)存在幸存者偏差。
  本脚本利用全市场K线缓存(5000+只)离线重建因子截面，
  将样本扩大 ~40 倍，对 9 个可从K线重建的维度做终审：
    trend / strategy / chip / relative_strength / ashare_specific /
    quant_behavior / capital_behavior / deep_research / oversold_rebound

方法：
  - 截面采样：每5个交易日取1日(约40个截面)，避免逐日计算的时间爆炸
  - 每截面：全部股票取截至当日K线 → backtest_factors 9维子评分
            → 与 T+3/5/10 真实收益配对
  - Rank IC：按截面日 spearman，输出均值/ICIR/t统计/正IC占比

用法: python scripts/rebuild_market_panel.py [--workers 6] [--fresh]
输出: reports/market_panel.csv (断点续存) + reports/market_ic_verdict.json
"""
import os
import sys
import glob
import pickle
import json
import argparse
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

PANEL_PATH = os.path.join(ROOT, "reports", "market_panel.csv")
SECTION_STEP = 5          # 每 N 个交易日一个截面
FWD_HORIZONS = (3, 5, 10)

# ---- worker 全局(lazy init, spawn 后每进程加载一次) ----
_KC = None
_SECTIONS = None


def _load_all_klines():
    """合并 bt_kline 整窗缓存 + fetch_1year 按股缓存 → {code: rows}"""
    kc = {}
    for p in glob.glob(os.path.join(ROOT, "cache", "bt_kline_*.pkl")):
        try:
            with open(p, "rb") as f:
                for c, rows in pickle.load(f).items():
                    if rows and (c not in kc or len(rows) > len(kc[c])):
                        kc[c] = rows
        except Exception:
            continue

    # 一级补充：fetch_1year 缓存(标准 baostock 行)
    idx_path = os.path.join(ROOT, "cache", "klines", "_code_index.json")
    if os.path.exists(idx_path):
        import hashlib
        kdir = os.path.join(ROOT, "cache", "klines")
        with open(idx_path, "r", encoding="utf-8") as f:
            codes = json.load(f)
        for code in codes:
            if code in kc:
                continue
            key = hashlib.md5(code.encode()).hexdigest()[:16]
            p = os.path.join(kdir, f"1year_{key}.pkl")
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        rows = pickle.load(f)
                    if rows and len(rows) >= 60:
                        kc[code] = rows
                except Exception:
                    continue
    return kc


def _init_worker():
    global _KC
    if _KC is None:
        _KC = _load_all_klines()


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def process_chunk(codes):
    """处理一批股票：返回 [(date, code, dim:score..., fwd3, fwd5, fwd10)]"""
    from src.engine.backtest_factors import calc_sub_scores_from_kline

    out = []
    kc = _KC or _load_all_klines()
    for code in codes:
        rows = kc.get(code)
        if not rows or len(rows) < 60:
            continue
        closes = [None] * len(rows)
        for i, r in enumerate(rows):
            closes[i] = _f(r[1])
        for si in range(60, len(rows), SECTION_STEP):
            date = rows[si][0]
            sub = calc_sub_scores_from_kline(rows[:si + 1])
            if not sub:
                continue
            rec = {"date": date, "code": code}
            rec.update(sub)
            ok = True
            for h in FWD_HORIZONS:
                fwd = None
                if si + h < len(rows):
                    c0, c1 = closes[si], closes[si + h]
                    if c0 and c1 and c0 > 0:
                        fwd = (c1 / c0 - 1) * 100
                rec[f"fwd{h}"] = round(fwd, 4) if fwd is not None else None
                if fwd is None:
                    ok = False
            if ok:
                out.append(rec)
    return out


def rank_ic(panel, dim, ret_col):
    """按截面日 spearman rank ic 序列"""
    import pandas as pd
    ics = []
    for date, g in panel.groupby("date"):
        if len(g) < 30 or g[ret_col].nunique() < 2:
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

    kc = _load_all_klines()
    codes = sorted(kc.keys())
    all_days = sorted({r[0] for rows in kc.values() for r in rows if r[0]})
    n_sections = max(0, (len(all_days) - 60)) // SECTION_STEP
    print(f"股票: {len(codes)} | 交易日: {len(all_days)} "
          f"({all_days[0]}~{all_days[-1]}) | 截面数≈{n_sections}")

    # 断点续跑：已有面板则只算缺失股票
    done_codes = set()
    if os.path.exists(PANEL_PATH):
        old = pd.read_csv(PANEL_PATH)
        done_codes = set(old["code"].unique())
        print(f"断点: 已有 {len(done_codes)} 只, 补算 {len(codes) - len(done_codes & set(codes))} 只")
    todo = [c for c in codes if c not in done_codes]

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    all_rows = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, part in enumerate(pool.imap_unordered(process_chunk, chunks), 1):
            all_rows.extend(part)
            print(f"  chunk {i}/{len(chunks)} 完成, 累计 {len(all_rows)} 行", flush=True)

    if not all_rows:
        print("无产出")
        return
    new_df = pd.DataFrame(all_rows)
    if os.path.exists(PANEL_PATH) and done_codes:
        new_df = pd.concat([pd.read_csv(PANEL_PATH), new_df], ignore_index=True)
    new_df.to_csv(PANEL_PATH, index=False)
    print(f"面板已保存: {PANEL_PATH} ({len(new_df)} 行)")

    # ===== IC 终审 =====
    dims = [c for c in new_df.columns if c not in ("date", "code")
            and not c.startswith("fwd")]
    verdicts = []
    for dim in dims:
        for h in FWD_HORIZONS:
            ret_col = f"fwd{h}"
            ics = rank_ic(new_df, dim, ret_col)
            if len(ics) < 10:
                continue
            arr = np.array(ics)
            mean_ic = float(arr.mean())
            std = float(arr.std())
            icir = mean_ic / std if std > 1e-9 else 0.0
            t = icir * (len(arr) ** 0.5)
            verdicts.append({
                "factor": dim, "horizon": h, "ic_days": len(arr),
                "mean_ic": round(mean_ic, 4), "icir": round(icir, 3),
                "t_stat": round(t, 2),
                "positive_ratio": round(float((arr > 0).mean()), 3),
            })

    verdicts.sort(key=lambda v: v["mean_ic"])
    print(f"\n{'='*70}\n全市场大样本 IC 终审 (截面≈40天, 股票~5000)\n{'='*70}")
    sig_pos, sig_neg = [], []
    for v in verdicts:
        tag = ""
        if abs(v["t_stat"]) >= 2.0:
            tag = "  ★显著"
            (sig_neg if v["mean_ic"] < 0 else sig_pos).append(v["factor"])
        elif abs(v["t_stat"]) >= 1.5:
            tag = "  (边缘)"
        print(f"{v['factor']:<22} h={v['horizon']:<2} IC={v['mean_ic']:+.4f} "
              f"ICIR={v['icir']:+.3f} t={v['t_stat']:+.2f} "
              f"正占比={v['positive_ratio']:.0%}{tag}")

    out = {
        "panel_rows": len(new_df),
        "codes": new_df["code"].nunique(),
        "verdicts": verdicts,
        "significantly_positive": sorted(set(sig_pos)),
        "significantly_negative": sorted(set(sig_neg)),
        "note": "全市场截面(每5交易日采样), 无幸存者偏差; |t|>=2 为显著",
    }
    outp = os.path.join(ROOT, "reports", "market_ic_verdict.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
