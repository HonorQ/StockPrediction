# -*- coding: utf-8 -*-
"""拉取全年美股数据(365天)"""
import os, sys, json, time
import akshare as ak
import pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_DIR = os.path.join(ROOT, "cache", "us_daily")
os.makedirs(US_DIR, exist_ok=True)

SECTOR_MAP = {
    "半导体": {
        "etf": [("SMH", "半导体ETF"), ("SOXX", "芯片ETF")],
        "stocks": [("NVDA", "英伟达"), ("AMD", "AMD"), ("MU", "美光"), ("AVGO", "博通"),
                   ("INTC", "英特尔"), ("QCOM", "高通"), ("TXN", "德州仪器"),
                   ("MRVL", "Marvell"), ("AMAT", "应用材料"), ("LRCX", "泛林")],
    },
    "液冷/服务器": {
        "etf": [("IGV", "软件ETF")],
        "stocks": [("SMCI", "超微电脑"), ("DELL", "戴尔"), ("HPE", "慧与"),
                   ("IBM", "IBM"), ("CRAY", "Cray"), ("HPQ", "惠普")],
    },
    "光通信/CPO": {
        "etf": [],
        "stocks": [("COHR", "Coherent"), ("AAOI", "AOI"), ("IIVI", "II-VI"),
                   ("LITE", "Lumentum"), ("CIEN", "Ciena")],
    },
    "电源/电力": {
        "etf": [],
        "stocks": [("VRT", "Vertiv"), ("ETN", "伊顿"), ("EMR", "艾默生"),
                   ("ITW", "伊利诺伊工具"), ("ROK", "罗克韦尔")],
    },
    "新能源/电车": {
        "etf": [("IDRV", "电动车ETF")],
        "stocks": [("TSLA", "特斯拉"), ("RIVN", "Rivian"), ("LCID", "Lucid"),
                   ("NIO", "蔚来"), ("XPEV", "小鹏"), ("LI", "理想"),
                   ("ENPH", "Enphase"), ("FSLR", "First Solar"), ("SEDG", "SolarEdge")],
    },
    "AI/云计算": {
        "etf": [("SKYY", "云计算ETF"), ("CLOU", "云ETF")],
        "stocks": [("MSFT", "微软"), ("GOOGL", "谷歌"), ("META", "Meta"),
                   ("AMZN", "亚马逊"), ("CRM", "Salesforce"), ("SNOW", "Snowflake"),
                   ("PLTR", "Palantir"), ("AI", "C3.ai")],
    },
    "医药/生物科技": {
        "etf": [("XBI", "生物科技ETF"), ("IBB", "生物科技指数")],
        "stocks": [("ABBV", "艾伯维"), ("LLY", "礼来"), ("PFE", "辉瑞"),
                   ("MRK", "默沙东"), ("AMGN", "安进"), ("GILD", "吉利德"),
                   ("BIIB", "百健"), ("REGN", "再生元")],
    },
    "金融": {
        "etf": [("XLF", "金融ETF")],
        "stocks": [("JPM", "摩根大通"), ("BAC", "美国银行"), ("GS", "高盛"),
                   ("MS", "摩根士丹利"), ("WFC", "富国银行"), ("V", "Visa")],
    },
    "消费": {
        "etf": [("XLY", "可选消费ETF"), ("XLP", "必需消费ETF")],
        "stocks": [("WMT", "沃尔玛"), ("COST", "好市多"), ("NKE", "耐克"),
                   ("SBUX", "星巴克"), ("MCD", "麦当劳"), ("TGT", "塔吉特")],
    },
}

total = sum(len(s["etf"]) + len(s["stocks"]) for s in SECTOR_MAP.values())
print("拉取 %d 只美股全年数据(365天)...\n" % total)

all_data = {}
i = 0
for sector, info in SECTOR_MAP.items():
    for sym, name in info["etf"] + info["stocks"]:
        i += 1
        try:
            df = ak.stock_us_daily(symbol=sym, adjust="qfq")
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df.tail(365)
                all_data[sym] = df
                print("  [%d/%d] %s: %d行" % (i, total, name, len(df)))
        except Exception as e:
            print("  [%d/%d] %s: 失败 %s" % (i, total, name, e))
        time.sleep(0.3)

print("\n成功: %d/%d" % (len(all_data), total))

# 保存个股CSV
for sym, df in all_data.items():
    df.to_csv(os.path.join(US_DIR, "%s_daily.csv" % sym), index=False)

# 计算板块日收益
print("计算板块日收益...")
all_dates = set()
for df in all_data.values():
    all_dates.update(df["date"].tolist())
dates = sorted(all_dates)[-365:]

stock_ret = {}
for sym, df in all_data.items():
    df = df.set_index("date")
    close = df["close"].reindex(dates)
    stock_ret[sym] = close.pct_change() * 100

sector_ret = {}
for sector, info in SECTOR_MAP.items():
    etf_rets = [stock_ret[s] for s, _ in info["etf"] if s in stock_ret]
    stock_rets = [stock_ret[s] for s, _ in info["stocks"] if s in stock_ret]
    if etf_rets:
        sector_ret[sector] = pd.concat(etf_rets, axis=1).mean(axis=1)
    elif stock_rets:
        sector_ret[sector] = pd.concat(stock_rets, axis=1).mean(axis=1)

sector_df = pd.DataFrame(sector_ret, index=dates).dropna()
sector_path = os.path.join(US_DIR, "us_sector_returns.csv")
sector_df.to_csv(sector_path)

print("板块日收益: %d天 x %d板块" % (len(sector_df), len(sector_df.columns)))
print("已保存: %s" % sector_path)
print("个股CSV: %d个文件" % len(all_data))
