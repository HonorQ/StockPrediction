# -*- coding: utf-8 -*-
"""美股-A股板块联动分析框架

功能:
  1. 读取 cache/us_daily/ 的美股板块历史数据
  2. 读取 A股行业映射 (industry_mapper)
  3. 计算隔夜美股→A股的板块相关性
  4. 输出各板块的预测力排名

需要: 30天以上美股历史数据
"""
import os, sys, json, glob
import numpy as np, pandas as pd

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_DIR = os.path.join(ROOT, "cache", "us_daily")

# 美股板块→A股行业关键词映射
US_TO_A = {
    "半导体": ["芯片", "半导体", "集成电路", "处理器"],
    "液冷服务器": ["液冷", "服务器", "算力", "数据中心", "IDC"],
    "光通信/CPO": ["光模块", "CPO", "光通信", "光纤"],
    "电源/电力": ["电源", "电力设备", "特高压", "智能电网"],
    "新能源": ["新能源车", "锂电池", "光伏", "储能", "风电"],
    "AI应用": ["AI", "人工智能", "软件", "算力"],
    "医药": ["医药", "创新药", "生物制药", "CXO"],
}


def load_us_history():
    """加载美股板块历史"""
    files = sorted(glob.glob(os.path.join(US_DIR, "*.json")))
    if not files:
        return None
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if "sector_returns" in data:
                records.append({
                    "date": data["date"],
                    **data["sector_returns"]
                })
    if not records:
        return None
    return pd.DataFrame(records)


def load_a_share_sectors():
    """加载A股行业映射"""
    industry_path = os.path.join(ROOT, "data", "industry_map.json")
    if not os.path.exists(industry_path):
        return None
    with open(industry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_us_a_correlation(us_df):
    """分析美股板块历史与A股相关性"""
    print("=== 美股板块历史数据 ===")
    print(f"天数: {len(us_df)}")
    print(f"板块: {[c for c in us_df.columns if c != 'date']}")
    print()

    for col in us_df.columns:
        if col == "date":
            continue
        vals = us_df[col].dropna()
        if len(vals) < 5:
            continue
        print(f"  {col}: 均值={vals.mean():+.2f}%  标准差={vals.std():.2f}%  "
              f"范围=[{vals.min():+.2f}%, {vals.max():+.2f}%]  "
              f"正涨天数={(vals>0).sum()}/{len(vals)}")


def main():
    # 加载美股历史
    us_df = load_us_history()
    if us_df is None or len(us_df) < 5:
        print(f"美股历史数据不足(当前: {len(us_df) if us_df is not None else 0}天)")
        print("需要至少30天数据才能做相关性分析。")
        print(f"\n当前已采集: {len(glob.glob(os.path.join(US_DIR, '*.json')))} 天")
        print("请每天运行 python scripts/fetch_us_daily.py 采集数据。")

        # 即使数据不足, 也输出分析框架
        print(f"\n{'='*60}")
        print("分析框架(等数据足够后自动运行)")
        print("="*60)
        print("""
步骤1: 每天采集美股7个板块20只个股收盘数据
  → python scripts/fetch_us_daily.py

步骤2: 积累30天后, 计算:
  a) 美股各板块涨跌幅序列
  b) A股各行业次日涨跌幅序列
  c) 板块对板块的Pearson/Spearman相关性
  d) 隔夜美股→A股的预测力(T-stat)

步骤3: 输出:
  - 哪些美股板块对A股有预测力
  - 滞后1天的相关性强度
  - 是否仅在特定regime有效

步骤4: 若有效, 建立:
  - 美股板块→A股板块的映射表
  - 作为regime过滤器接入评分系统
""")
        return

    analyze_us_a_correlation(us_df)


if __name__ == "__main__":
    main()
