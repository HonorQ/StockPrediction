"""
行业映射数据获取

由 market_data.py 拆分(AGENTS.md 技术债偿还): 行业地图构建/查询。
"""
import akshare as ak
import time
from typing import Dict, Optional

try:
    from src.utils.log import get_logger
    logger = get_logger()
except Exception as e:
    import logging
    logging.warning("get_logger 不可用, 降级到标准 logging: %s", e)
    logger = logging.getLogger("stock_prediction")


class IndustryMapper:
    """行业地图构建与查询（东财→新浪双源 failover）"""

    def __init__(self, cache_get=None, cache_put=None):
        self._cache_get = cache_get or (lambda k: None)
        self._cache_put = cache_put or (lambda k, v: None)
        self._industry_map: Optional[Dict[str, str]] = None

    def get_industry_map(self, force_refresh: bool = False) -> Dict[str, str]:
        """获取 code -> industry 映射。
        本地缓存 data/industry_map.json，有效期 7 天。
        首次或过期时先尝试东财 push2，失败则回退到新浪板块。
        """
        import os
        import json

        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "industry_map.json"
        )

        # 1) 内存缓存
        if self._industry_map is not None and not force_refresh:
            return self._industry_map

        # 2) 本地文件缓存（7天有效期）
        if not force_refresh and os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                if (time.time() - mtime) < 7 * 86400:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        self._industry_map = json.load(f)
                    if self._industry_map:
                        return self._industry_map
            except Exception as e:
                logger.warning("[market_data.py:112] MarketDataFetcher/get_industry_map 失败(忽略): %s", e)

        # 3) 重新构建：先试东财 push2，失败则回退到新浪
        logger.info('  🔄 构建行业映射缓存...')
        industry_map = self._build_industry_map_em()
        if not industry_map:
            logger.info('  ⚠️ 东财行业接口不可用，切换至新浪板块...')
            industry_map = self._build_industry_map_sina()

        if industry_map:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(industry_map, f, ensure_ascii=False)
            except Exception as e:
                logger.warning("[market_data.py:127] MarketDataFetcher/get_industry_map 失败(忽略): %s", e)

        self._industry_map = industry_map
        return industry_map

    def _build_industry_map_em(self) -> Dict[str, str]:
        try:
            board_df = ak.stock_board_industry_name_em()
            if board_df is None or board_df.empty:
                return {}
            industry_map = {}
            board_names = board_df["板块名称"].tolist()
            for board_name in board_names:
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=board_name)
                    if cons is None or cons.empty:
                        continue
                    code_col = "代码" if "代码" in cons.columns else "code"
                    for code in cons[code_col].astype(str):
                        if code not in industry_map:
                            industry_map[code] = board_name
                except Exception as e:
                    # 单板块拉取失败跳过该板块继续, debug留痕(外层有整体降级warning)
                    logger.debug(f'东财行业成分获取失败({board_name}): {e}')
                    continue
            logger.info(f'  ✅ 东财行业映射: {len(industry_map)} 只, {len(set(industry_map.values()))} 个行业')
            return industry_map
        except Exception as e:
            logger.warning("东财行业映射失败: %s", e)
            return {}

    def _build_industry_map_sina(self) -> Dict[str, str]:
        try:
            sectors = ak.stock_sector_spot()
            labels = sectors["label"].tolist()
            names = sectors["板块"].tolist()
            industry_map = {}
            for label, name in zip(labels, names):
                try:
                    df = ak.stock_sector_detail(sector=label)
                    for _, row in df.iterrows():
                        symbol = row["symbol"]
                        if symbol not in industry_map:
                            industry_map[symbol] = name
                except Exception as e:
                    # 单板块拉取失败跳过该板块继续, debug留痕(外层有整体降级error)
                    logger.debug(f'新浪行业成分获取失败({name}): {e}')
                    continue
            logger.info(f'  ✅ 新浪行业映射: {len(industry_map)} 只, {len(set(industry_map.values()))} 个行业')
            return industry_map
        except Exception as e:
            logger.error("新浪行业映射也失败: %s", e)
            return {}

    def get_stock_industry(self, code: str) -> str:
        """获取单只股票的行业（依赖 get_industry_map 缓存）。"""
        if self._industry_map is None:
            self.get_industry_map()
        return (self._industry_map or {}).get(code, "未知")
