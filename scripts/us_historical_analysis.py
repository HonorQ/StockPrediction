# -*- coding: utf-8 -*-
"""美股历史数据拉取 + 板块联动分析

用akshare拉取美股20只个股历史K线, 计算板块日收益,
与A股行业做隔夜相关性分析。

用法: python scripts/us_historical_analysis.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
import akshare as ak

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_DIR = os.path.join(ROOT, "cache", "us_daily")
os.makedirs(US_DIR, exist_ok=True)

# 美股板块映射
SECTOR_MAP = {
    "半导体": [("NVDA", "英伟达"), ("AMD", "AMD"), ("MU", "美光"), ("AVGO", "博通"), ("INTC", "英特尔")],
    "液冷服务器": [("SMCI", "超微电脑"), ("DELL", "戴尔"), ("HPE", "慧与")],
    "光通信/CPO": [("COHR", "Coherent")],
    "电源/电力": [("VRT", "Vertiv"), ("ETN", "伊顿")],
    "新能源": [("TSLA", "特斯拉"), ("ENPH", "Enphase"), ("FSLR", "First Solar"), ("RIVN", "Rivian")],
    "AI应用": [("MSFT", "微软"), ("GOOGL", "谷歌"), ("META", "Meta")],
    "医药": [("ABBV", "艾伯维"), ("LLY", "礼来")],
}

# 美股→A股行业映射
US_TO_A = {
    "半导体": ["电子信息", "电子器件"],
    "液冷服务器": ["电子信息", "机械行业"],
    "光通信/CPO": ["电子信息"],
    "电源/电力": ["电力行业", "发电设备", "电器行业"],
    "新能源": ["电力行业", "发电设备", "汽车制造"],
    "AI应用": ["电子信息", "机械行业"],
    "医药": ["生物制药", "医疗器械"],
}


def fetch_us_history(symbol, days=300):
    """拉取美股历史K线"""
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df.tail(days)
    except Exception as e:
        print(f"  拉取{symbol}失败: {e}")
    return None


def compute_daily_returns(us_data, days=175):
    """计算美股各板块每日收益率"""
    # 取最近N天的日期范围
    all_dates = set()
    for sym, df in us_data.items():
        all_dates.update(df["date"].tolist())
    dates = sorted(all_dates)[-days:]

    # 计算个股日收益率
    stock_returns = {}
    for sym, df in us_data.items():
        df = df.set_index("date")
        close = df["close"].reindex(dates)
        stock_returns[sym] = close.pct_change() * 100

    # 按板块聚合(等权)
    sector_returns = {}
    for sector, stocks in SECTOR_MAP.items():
        rets = []
        for sym, _ in stocks:
            if sym in stock_returns:
                rets.append(stock_returns[sym])
        if rets:
            sector_df = pd.concat(rets, axis=1)
            sector_returns[sector] = sector_df.mean(axis=1)

    return pd.DataFrame(sector_returns, index=dates).dropna()


def load_a_share_industry_returns(us_sector_df):
    """加载A股行业收益(与美股同期)"""
    industry_path = os.path.join(ROOT, "data", "industry_map.json")
    panel = pd.read_csv(os.path.join(ROOT, "reports", "market_panel_daily.csv"))

    with open(industry_path, encoding="utf-8") as f:
        industry_map = json.load(f)

    panel["norm_code"] = panel["code"].apply(lambda c: c.replace(".", ""))
    panel["industry"] = panel["norm_code"].map(industry_map)
    panel = panel.dropna(subset=["industry"])

    # 计算每日行业平均收益(fwd5)
    ind_ret = panel.groupby(["date", "industry"])["fwd5"].mean().unstack()
    return ind_ret


def main():
    print("=== 美股历史数据拉取 + 板块联动分析 ===\n")

    # 1. 拉取美股历史
    print("1. 拉取美股历史K线...")
    us_data = {}
    all_syms = [(sym, name) for stocks in SECTOR_MAP.values() for sym, name in stocks]
    for i, (sym, name) in enumerate(all_syms):
        df = fetch_us_history(sym, days=300)
        if df is not None:
            us_data[sym] = df
            print(f"  [{i+1}/{len(all_syms)}] {name}: {len(df)}行")
        time.sleep(0.5)  # 避免请求过快

    print(f"成功: {len(us_data)}/{len(all_syms)}只\n")

    if len(us_data) < 10:
        print("数据不足, 退出")
        return

    # 2. 计算美股板块日收益
    print("2. 计算美股板块日收益...")
    us_sector_df = compute_daily_returns(us_data, days=175)
    print(f"  日期范围: {us_sector_df.index[0]} ~ {us_sector_df.index[-1]}")
    print(f"  天数: {len(us_sector_df)}")
    print(f"  板块: {list(us_sector_df.columns)}")

    # 保存美股数据
    us_out = os.path.join(US_DIR, "us_sector_returns.csv")
    us_sector_df.to_csv(us_out)
    print(f"  已保存: {us_out}\n")

    # 3. 加载A股行业收益
    print("3. 加载A股行业收益...")
    a_ind_ret = load_a_share_industry_returns(us_sector_df)
    # 对齐日期(A股日期 vs 美股日期)
    # 美股T日收益 → A股T+1日收益(隔夜联动)
    us_dates = us_sector_df.index.tolist()
    a_dates = a_ind_ret.index.tolist()

    # 找共同日期
    common_dates = sorted(set(us_dates) & set(a_dates))
    print(f"  A股行业数: {len(a_ind_ret.columns)}")
    print(f"  共同日期: {len(common_dates)}\n")

    if len(common_dates) < 20:
        print("共同日期不足20天, 退出")
        return

    # 4. 计算隔夜相关性: 美股T日→A股T+1日
    print("=" * 60)
    print("4. 美股板块→A股行业 隔夜相关性分析")
    print("=" * 60)
    print("  (美股T日涨跌 → A股T+1日同行业涨跌)\n")

    results = []
    for us_sector, a_industries in US_TO_A.items():
        if us_sector not in us_sector_df.columns:
            continue
        for a_ind in a_industries:
            if a_ind not in a_ind_ret.columns:
                continue

            # 构造配对: 美股T日 vs A股T+1日
            us_rets = []
            a_rets_next = []
            for i, d in enumerate(common_dates[:-1]):
                next_d = common_dates[i+1]
                us_r = us_sector_df.loc[d, us_sector] if d in us_sector_df.index else np.nan
                a_r = a_ind_ret.loc[next_d, a_ind] if next_d in a_ind_ret.index else np.nan
                if not np.isnan(us_r) and not np.isnan(a_r):
                    us_rets.append(us_r)
                    a_rets_next.append(a_r)

            if len(us_rets) < 20:
                continue

            us_arr = np.array(us_rets)
            a_arr = np.array(a_rets_next)

            # Pearson相关性
            corr = np.corrcoef(us_arr, a_arr)[0, 1]
            # Spearman秩相关
            from scipy import stats
            spearman, sp_pval = stats.spearmanr(us_arr, a_arr)
            # T统计量
            t_stat = corr * np.sqrt(len(us_arr) - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0

            # 美股涨时A股是否也涨(条件概率)
            us_up = us_arr > 0
            a_up = a_arr > 0
            p_a_up_given_us_up = (a_up & us_up).sum() / us_up.sum() if us_up.sum() > 0 else 0
            p_a_up_given_us_down = (a_up & ~us_up).sum() / (~us_up).sum() if (~us_up).sum() > 0 else 0

            results.append({
                "us_sector": us_sector,
                "a_industry": a_ind,
                "samples": len(us_rets),
                "pearson_corr": round(corr, 4),
                "spearman_corr": round(spearman, 4),
                "t_stat": round(t_stat, 2),
                "p_a_up_given_us_up": round(p_a_up_given_us_up, 3),
                "p_a_up_given_us_down": round(p_a_up_given_us_down, 3),
                "lift": round(p_a_up_given_us_up - p_a_up_given_us_down, 3),
            })

    df_r = pd.DataFrame(results).sort_values("pearson_corr", ascending=False)

    print(f"{'美股板块':<12}{'A股行业':<12}{'样本':>6}{'Pearson':>10}{'Spearman':>10}{'t值':>8}{'↑概率':>8}{'↓概率':>8}{'提升':>8}")
    print("─" * 82)
    for _, r in df_r.iterrows():
        sig = "***" if abs(r["t_stat"]) > 2.58 else ("**" if abs(r["t_stat"]) > 1.96 else ("*" if abs(r["t_stat"]) > 1.64 else ""))
        print(f"  {r['us_sector']:<10}{r['a_industry']:<10}{r['samples']:>6}"
              f"{r['pearson_corr']:>+9.4f}{r['spearman_corr']:>+9.4f}{r['t_stat']:>+7.2f}"
              f"{r['p_a_up_given_us_up']:>7.1%}{r['p_a_up_given_us_down']:>7.1%}{r['lift']:>+7.1%} {sig}")

    # 5. 结论
    print(f"\n{'='*60}")
    print("5. 结论")
    print("=" * 60)
    sig_results = df_r[df_r["t_stat"].abs() > 1.64]
    if len(sig_results) > 0:
        print(f"  显著相关(***p<0.01/**p<0.05/*p<0.1): {len(sig_results)}对")
        for _, r in sig_results.iterrows():
            direction = "正相关" if r["pearson_corr"] > 0 else "负相关"
            print(f"    {r['us_sector']}→{r['a_industry']}: {direction} "
                  f"t={r['t_stat']:+.2f} lift={r['lift']:+.1%}")
    else:
        print("  未发现显著隔夜相关性")

    # 保存
    outp = os.path.join(ROOT, "reports", "us_a_sector_correlation.json")
    df_r.to_json(outp, orient="records", force_ascii=False, indent=2)
    print(f"\n已保存: {outp}")


if __name__ == "__main__":
    main()
