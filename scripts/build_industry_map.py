# -*- coding: utf-8 -*-
"""用新浪板块数据构建行业映射"""
import json, sys
import akshare as ak
import pandas as pd

sectors = ak.stock_sector_spot()
labels = sectors['label'].tolist()
names = sectors['板块'].tolist()

industry_map = {}
for i, (label, name) in enumerate(zip(labels, names)):
    try:
        df = ak.stock_sector_detail(sector=label)
        for _, row in df.iterrows():
            symbol = row['symbol']
            industry_map[symbol] = name
        print(f"[{i+1}/49] {name}: {len(df)} 只")
    except Exception as e:
        print(f"[{i+1}/49] {name}: Failed - {e}")

with open('data/industry_map.json', 'w', encoding='utf-8') as f:
    json.dump(industry_map, f, ensure_ascii=False, indent=2)

print(f"\n完成！共 {len(industry_map)} 条映射，保存到 data/industry_map.json")
