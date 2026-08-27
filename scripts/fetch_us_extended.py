# -*- coding: utf-8 -*-
"""美股扩展板块数据 - 50只个股+ETF

扩充每个板块到5-10只代表股, 加入板块ETF作为聚合信号。
"""
import os, sys, json, time
import numpy as np, pandas as pd
import akshare as ak

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_DIR = os.path.join(ROOT, "cache", "us_daily")
os.makedirs(US_DIR, exist_ok=True)

# ── 扩展美股板块映射 ──
# 每个板块: 个股 + ETF
SECTOR_MAP = {
    "半导体": {
        "etf": [("SMH", "半导体ETF"), ("SOXX", "芯片ETF")],
        "stocks": [
            ("NVDA", "英伟达"), ("AMD", "AMD"), ("MU", "美光"), ("AVGO", "博通"),
            ("INTC", "英特尔"), ("QCOM", "高通"), ("TXN", "德州仪器"),
            ("MRVL", "Marvell"), ("AMAT", "应用材料"), ("LRCX", "泛林"),
        ],
    },
    "液冷/服务器": {
        "etf": [("IGV", "软件ETF")],
        "stocks": [
            ("SMCI", "超微电脑"), ("DELL", "戴尔"), ("HPE", "慧与"),
            ("IBM", "IBM"), ("CRAY", "Cray"), ("HPQ", "惠普"),
        ],
    },
    "光通信/CPO": {
        "etf": [],
        "stocks": [
            ("COHR", "Coherent"), ("AAOI", "AOI"), ("IIVI", "II-VI"),
            ("LITE", "Lumentum"), ("CIEN", "Ciena"),
        ],
    },
    "电源/电力": {
        "etf": [],
        "stocks": [
            ("VRT", "Vertiv"), ("ETN", "伊顿"), ("EMR", "艾默生"),
            ("ITW", "伊利诺伊工具"), ("ROK", "罗克韦尔"),
        ],
    },
    "新能源/电车": {
        "etf": [("IDRV", "电动车ETF")],
        "stocks": [
            ("TSLA", "特斯拉"), ("RIVN", "Rivian"), ("LCID", "Lucid"),
            ("NIO", "蔚来"), ("XPEV", "小鹏"), ("LI", "理想"),
            ("ENPH", "Enphase"), ("FSLR", "First Solar"), ("SEDG", "SolarEdge"),
        ],
    },
    "AI/云计算": {
        "etf": [("SKYY", "云计算ETF"), ("CLOU", "云ETF")],
        "stocks": [
            ("MSFT", "微软"), ("GOOGL", "谷歌"), ("META", "Meta"),
            ("AMZN", "亚马逊"), ("CRM", "Salesforce"), ("SNOW", "Snowflake"),
            ("PLTR", "Palantir"), ("AI", "C3.ai"),
        ],
    },
    "医药/生物科技": {
        "etf": [("XBI", "生物科技ETF"), ("IBB", "生物科技指数")],
        "stocks": [
            ("ABBV", "艾伯维"), ("LLY", "礼来"), ("PFE", "辉瑞"),
            ("MRK", "默沙东"), ("AMGN", "安进"), ("GILD", "吉利德"),
            ("BIIB", "百健"), ("REGN", "再生元"),
        ],
    },
    "金融": {
        "etf": [("XLF", "金融ETF")],
        "stocks": [
            ("JPM", "摩根大通"), ("BAC", "美国银行"), ("GS", "高盛"),
            ("MS", "摩根士丹利"), ("WFC", "富国银行"), ("V", "Visa"),
        ],
    },
    "消费": {
        "etf": [("XLY", "可选消费ETF"), ("XLP", "必需消费ETF")],
        "stocks": [
            ("WMT", "沃尔玛"), ("COST", "好市多"), ("NKE", "耐克"),
            ("SBUX", "星巴克"), ("MCD", "麦当劳"), ("TGT", "塔吉特"),
        ],
    },
}

# 美股→A股行业映射
US_TO_A = {
    "半导体": ["电子信息", "电子器件"],
    "液冷/服务器": ["电子信息", "机械行业"],
    "光通信/CPO": ["电子信息"],
    "电源/电力": ["电力行业", "发电设备", "电器行业"],
    "新能源/电车": ["电力行业", "发电设备", "汽车制造"],
    "AI/云计算": ["电子信息", "机械行业"],
    "医药/生物科技": ["生物制药", "医疗器械"],
    "金融": ["金融行业"],
    "消费": ["食品行业", "酿酒行业", "商业百货"],
}


def fetch_us(symbol, days=300):
    """拉取美股历史K线"""
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df.tail(days)
    except Exception as e:
        print(f"  {symbol}: {e}")
    return None


def main():
    print("=== 美股扩展板块数据拉取 ===\n")

    # 统计
    total_stocks = sum(len(s["stocks"]) for s in SECTOR_MAP.values())
    total_etf = sum(len(s["etf"]) for s in SECTOR_MAP.values())
    total = total_stocks + total_etf
    print(f"板块数: {len(SECTOR_MAP)}")
    print(f"个股: {total_stocks}只  ETF: {total_etf}只  合计: {total}只\n")

    # 拉取
    all_data = {}
    i = 0
    for sector, info in SECTOR_MAP.items():
        print(f"--- {sector} ---")
        # ETF
        for sym, name in info["etf"]:
            i += 1
            df = fetch_us(sym, days=300)
            if df is not None:
                all_data[sym] = df
                print(f"  [{i}/{total}] {name}(ETF): {len(df)}行")
            time.sleep(0.3)
        # 个股
        for sym, name in info["stocks"]:
            i += 1
            df = fetch_us(sym, days=300)
            if df is not None:
                all_data[sym] = df
                print(f"  [{i}/{total}] {name}: {len(df)}行")
            time.sleep(0.3)

    print(f"\n成功: {len(all_data)}/{total}只")

    # 保存原始数据
    for sym, df in all_data.items():
        df.to_csv(os.path.join(US_DIR, f"{sym}_daily.csv"), index=False)

    # 计算板块日收益
    print(f"\n计算板块日收益...")
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df["date"].tolist())
    dates = sorted(all_dates)[-175:]

    # 个股收益
    stock_ret = {}
    for sym, df in all_data.items():
        df = df.set_index("date")
        close = df["close"].reindex(dates)
        stock_ret[sym] = close.pct_change() * 100

    # 板块收益(ETF优先, 没有ETF用个股等权)
    sector_ret = {}
    for sector, info in SECTOR_MAP.items():
        etf_rets = [stock_ret[s] for s, _ in info["etf"] if s in stock_ret]
        stock_rets = [stock_ret[s] for s, _ in info["stocks"] if s in stock_ret]

        if etf_rets:
            # 有ETF: 用ETF
            sector_ret[sector] = pd.concat(etf_rets, axis=1).mean(axis=1)
        elif stock_rets:
            # 无ETF: 用个股等权
            sector_ret[sector] = pd.concat(stock_rets, axis=1).mean(axis=1)

    sector_df = pd.DataFrame(sector_ret, index=dates).dropna()
    print(f"板块日收益: {len(sector_df)}天 x {len(sector_df.columns)}个板块")

    # 保存
    out_path = os.path.join(US_DIR, "us_sector_returns.csv")
    sector_df.to_csv(out_path)
    print(f"已保存: {out_path}")

    # 描述统计
    print(f"\n=== 板块描述统计 ===")
    print(f"{'板块':<14}{'均值':>8}{'标准差':>8}{'正涨天':>8}{'天数':>6}")
    for col in sector_df.columns:
        s = sector_df[col].dropna()
        print(f"  {col:<12}{s.mean():>+7.2f}%{s.std():>7.2f}%"
              f"{(s>0).sum():>7}/{len(s)}{len(s):>6}")


if __name__ == "__main__":
    main()
