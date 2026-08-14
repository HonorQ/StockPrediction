import sys; sys.path.insert(0, '.')
from src.data.market_data import MarketDataFetcher

fetcher = MarketDataFetcher()
industry_map = fetcher.get_industry_map()

print(f'行业映射条数: {len(industry_map)}')
print(f'行业数: {len(set(industry_map.values()))}')

codes = ['sh600519', 'sh600176', 'sz000001', 'sz002415', 'sz300750', 'sh601318']
for code in codes:
    ind = industry_map.get(code, '未知')
    print(f'  {code} -> {ind}')

print("\n所有行业列表:")
for ind_name in sorted(set(industry_map.values())):
    cnt = sum(1 for v in industry_map.values() if v == ind_name)
    print(f'  {ind_name}: {cnt}只')

print("\n完成！")
