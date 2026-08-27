"""
资金流向数据获取
数据源：akshare（东方财富）
"""
from src.utils.log import get_logger
logger = get_logger()

import akshare as ak
import pandas as pd
import threading
import time
from typing import Dict


class MoneyFlowFetcher:
    """资金流向获取器"""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._em_down = False
        self._em_down_at = 0
        self._warned = False
        self._ths_cache: Dict[str, float] = {}  # THS资金流向缓存
        self._ths_cache_time = 0
        # akshare 内部依赖 py_mini_racer(V8)，非线程安全，
        # 多线程并发调用会触发 native 崩溃。用锁强制串行。
        self._ths_lock = threading.Lock()

    def _is_em_down(self) -> bool:
        """半衰期熔断：失败后60秒内跳过，之后自动重试。

        原熔断时长300秒过于激进——单次网络抖动即放弃东财源5分钟，
        期间全部个股资金流降级到无细分数据的同花顺缓存。
        """
        if self._em_down and time.time() - self._em_down_at < 60:
            return True
        if self._em_down:
            self._em_down = False
        return False

    def _load_ths_cache(self):
        """加载同花顺全市场资金流向（缓存5分钟）"""
        now = time.time()
        if self._ths_cache and now - self._ths_cache_time < 300:
            return
        with self._ths_lock:
            # double-check：拿到锁后再次检查，可能已被其他线程加载
            now = time.time()
            if self._ths_cache and now - self._ths_cache_time < 300:
                return
            try:
                df = ak.stock_fund_flow_individual(symbol="即时")
                if df is not None and not df.empty:
                    self._ths_cache = {}
                    for _, row in df.iterrows():
                        code = str(row.get("股票代码", ""))
                        net = row.get("净额", "0")
                        if isinstance(net, str):
                            net = net.replace("亿", "00000000").replace("万", "0000")
                            try:
                                net = float(net)
                            except Exception as e:
                                logger.warning(f"解析资金流向数值失败: {e}")
                                net = 0
                        self._ths_cache[code] = float(net) if pd.notna(net) else 0
                    self._ths_cache_time = now
                    logger.info(f'  ✅ 同花顺资金流向缓存: {len(self._ths_cache)} 只')
            except Exception as e:
                logger.info(f'  ⚠️ 同花顺资金流向失败: {e}')

    # ------------------------------------------------------------------
    # 个股资金流向
    # ------------------------------------------------------------------
    def get_stock_fund_flow(self, stock_code: str) -> Dict:
        """
        获取个股资金流向（超时保护）
        返回: main_flow(主力净流入), super_large(超大单), large(大单),
              medium(中单), small(小单)
        优先级: 东财push2 → 同花顺(净额) → 全零
        """
        code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")

        # 方式1: 直接调用东财push2 API
        if not self._is_em_down():
            try:
                import requests as _req
                # 东财 secid 市场: 1=沪(6/9开头) 0=深(0/2/3开头)；北交所为 0.
                # 原实现北交所(4/8开头)被归入"1.沪"导致查询必错
                raw = code.lstrip("shzjb") or code
                if stock_code.lower().startswith(("sh",)) or raw.startswith(("6", "9")):
                    market = "1"
                elif raw.startswith(("4", "8")) or stock_code.lower().startswith("bj"):
                    market = "0"
                else:
                    market = "0"
                url = "https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get"
                params = {
                    "lmt": 0, "klt": 101,
                    "secid": f"{market}.{code}",
                    "fields1": "f1,f2,f3,f7",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                    "ut": "b2884a393a59ad64002292a3e90d46a5",
                }
                headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com"}
                resp = _req.get(url, params=params, headers=headers, timeout=8)
                data = resp.json()
                d = data.get("data") or {}
                if d.get("klines"):
                    klines = d["klines"]
                    if klines:
                        latest = klines[-1].split(",")
                        return {
                            "date": latest[0] if len(latest) > 0 else "",
                            "main_flow": float(latest[1]) if len(latest) > 1 else 0,
                            "main_pct": float(latest[6]) if len(latest) > 6 else 0,
                            "super_large": float(latest[5]) if len(latest) > 5 else 0,
                            "large": float(latest[4]) if len(latest) > 4 else 0,
                            "medium": float(latest[3]) if len(latest) > 3 else 0,
                            "small": float(latest[2]) if len(latest) > 2 else 0,
                        }
            except Exception as e:
                if not self._warned:
                    logger.info(f'      ⚠️ 资金流向API失败(push2): {e}')
                    self._warned = True
                self._em_down = True
                self._em_down_at = time.time()

        # 方式2: 同花顺缓存（有净额但无主力/大单细分）
        self._load_ths_cache()
        net = self._ths_cache.get(code, 0)
        if net != 0:
            # 同花顺净额 = 流入-流出，无主力细分，统一填 main_flow
            return {
                "main_flow": net,
                "main_pct": 0,
                "super_large": 0,
                "large": 0,
                "medium": 0,
                "small": 0,
                "source": "ths",
            }

        return {"main_flow": 0, "main_pct": 0, "super_large": 0, "large": 0, "medium": 0, "small": 0}

    # ------------------------------------------------------------------
    # 行业资金流向（板块热度参考）
    # ------------------------------------------------------------------
    def get_sector_fund_flow(self) -> pd.DataFrame:
        """获取行业板块资金流向"""
        cache_key = "sector_flow"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if df is not None and not df.empty:
                self._cache[cache_key] = df
                logger.info(f'  ✅ 获取行业资金流向: {len(df)} 个行业')
                return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.info(f'  ⚠️ 获取行业资金流向失败: {e}')
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 北向资金（个股级别）
    # ------------------------------------------------------------------
    def get_north_holdings_change(self) -> pd.DataFrame:
        """获取北向资金持股变化"""
        # 方式1: 尝试akshare（参数已更新）
        for market in ["沪股通", "深股通"]:
            for indicator in ["5日排行", "10日排行"]:
                try:
                    df = ak.stock_hsgt_hold_stock_em(market=market, indicator=indicator)
                    if df is not None and not df.empty:
                        logger.info(f'  ✅ 获取北向持股({market}): {len(df)} 条')
                        return df
                except Exception as e:
                    logger.warning("[money_flow.py:176] MoneyFlowFetcher/get_north_holdings_change 失败(忽略): %s", e)

        # 方式2: 直接调用东财datacenter API（全局代理已通过环境变量设置）
        try:
            import requests as _req
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                "reportName": "RPT_MUTUAL_HOLDSTOCKNORTH_STA",
                "columns": "ALL",
                "pageSize": "500",
                "sortColumns": "HOLD_MARKET_CAP",
                "sortTypes": "-1",
                "source": "WEB",
                "client": "WEB",
            }
            r = _req.get(url, params=params,
                         headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
                         timeout=10)
            data = r.json().get("result", {}).get("data", [])
            if data:
                df = pd.DataFrame(data)
                logger.info(f'  ✅ 获取北向持股(东财API): {len(df)} 条')
                return df
            # 空响应时 df 从未赋值，直接 return df 会抛 NameError 被 except 吞掉
            return pd.DataFrame()
        except Exception as e:
            logger.info(f'  ⚠️ 获取北向持股失败: {e}')
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 龙虎榜
    # ------------------------------------------------------------------
    def get_dragon_tiger(self) -> pd.DataFrame:
        """获取龙虎榜数据"""
        from datetime import datetime
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(
                start_date=date_str, end_date=date_str
            )
            if df is not None and not df.empty:
                logger.info(f'  ✅ 获取龙虎榜: {len(df)} 条')
                return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.info(f'  ⚠️ 获取龙虎榜失败: {e}')
            return pd.DataFrame()
