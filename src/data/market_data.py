"""
A股 + 外盘行情数据获取
数据源：akshare（免费）
"""
import akshare as ak
import pandas as pd
import requests
import re
import time
import atexit
import threading
from typing import Optional, Dict, List
from datetime import datetime

try:
    from src.utils.log import get_logger
    logger = get_logger()
except Exception:
    import logging
    logger = logging.getLogger("stock_prediction")


from src.data.industry_mapper import IndustryMapper  # noqa: F401
from src.data.global_market_fetcher import GlobalMarketFetcher  # noqa: F401
from src.data.kline_fetcher import KlineFetcher  # noqa: F401

class _BaostockConnection:
    """baostock 进程级连接单例。
    避免每只股票都 login/logout，节省 0.5-1 秒/次。
    线程安全：baostock 本身非线程安全，加锁串行调用。
    """
    _instance = None
    _lock = threading.Lock()
    _logged_in = False

    @classmethod
    def get_connection(cls):
        import baostock as bs
        with cls._lock:
            if cls._instance is None:
                cls._instance = bs
                try:
                    bs.login()
                    cls._logged_in = True
                    atexit.register(cls._cleanup)
                except Exception as e:
                    logger.warning("[market_data.py:43] _BaostockConnection/get_connection 失败(忽略): %s", e)
        return cls._instance

    @classmethod
    def _cleanup(cls):
        if cls._logged_in:
            try:
                cls._instance.logout()
                cls._logged_in = False
            except Exception as e:
                logger.warning("[market_data.py:54] _BaostockConnection/_cleanup 失败(忽略): %s", e)


class MarketDataFetcher:
    """行情数据获取器"""

    # 内存缓存有效期(秒)：原实现进程内永不刷新，盘中重复请求拿到陈旧K线。
    # 盘前任务单次运行影响小，但常驻的盘中监控/午后重算必须感知数据老化
    CACHE_TTL = 300

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, float] = {}
        self._warned_bs = False
        self._industry_map: Optional[Dict[str, str]] = None

    # === 委托方法: 逻辑已拆至独立模块, 此处保持向后兼容接口 ===

    def get_industry_map(self, force_refresh=False):
        """委托给 IndustryMapper"""
        if self._industry_map and not force_refresh:
            return self._industry_map
        mapper = IndustryMapper(cache_get=self._cache_get, cache_put=self._cache_put)
        self._industry_map = mapper.get_industry_map(force_refresh=force_refresh)
        return self._industry_map

    def get_stock_industry(self, code):
        """委托给 IndustryMapper"""
        if not self._industry_map:
            self.get_industry_map()
        return self._industry_map.get(code, "未知")

    def get_global_market_summary(self, config_markets):
        """委托给 GlobalMarketFetcher"""
        fetcher = GlobalMarketFetcher()
        return fetcher.get_global_market_summary(config_markets)

    def get_index_realtime(self):
        """委托给 GlobalMarketFetcher"""
        fetcher = GlobalMarketFetcher()
        return fetcher.get_index_realtime()

    def get_stock_history(self, code, days=120):
        """历史K线获取: 腾讯→新浪→baostock 三源 failover"""
        fetcher = KlineFetcher(cache_get=self._cache_get, cache_put=self._cache_put,
                               cache_ts_put=lambda k, v: self._cache_ts.__setitem__(k, v))
        return fetcher.get_stock_history(code, days=days)

    def _fetch_tencent_quotes(self, codes):
        """腾讯实时行情批量查询"""
        return MarketDataFetcher._fetch_tencent_quotes_static(codes)

    def _cache_get(self, key: str):
        """带 TTL 的缓存读取，过期返回 None"""
        import time as _time
        ts = self._cache_ts.get(key)
        if ts is not None and _time.time() - ts > self.CACHE_TTL:
            self._cache.pop(key, None)
            self._cache_ts.pop(key, None)
            return None
        return self._cache.get(key)

    def _cache_put(self, key: str, value):
        import time as _time
        self._cache[key] = value
        self._cache_ts[key] = _time.time()


    @staticmethod
    def _fetch_with_retry(fetcher_fn, max_retries=2, delay=3):
        """指数退避重试（代理已由 ProxyManager 通过全局环境变量控制）"""
        for attempt in range(max_retries):
            try:
                return fetcher_fn()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = delay * (2 ** attempt)
                    logger.info(f'    ⏳ 重试 {attempt + 1}/{max_retries} ({wait}s)...')
                    time.sleep(wait)
                else:
                    raise e
        return None

    # ------------------------------------------------------------------
    # A股实时行情
    # ------------------------------------------------------------------
    _COLUMN_MAP = {
        "pctChg": "涨跌幅", "changepercent": "涨跌幅",
        "trade": "最新价", "settlement": "昨收",
        "buy": "买入", "sell": "卖出",
        "volume": "成交量", "amount": "成交额",
        "code": "代码", "name": "名称",
        "turnoverratio": "换手率",
        "mktcap": "总市值", "nmc": "流通市值",
        "pe": "市盈率", "pb": "市净率",
    }
    _REQUIRED_COLS = {"代码", "名称", "最新价", "涨跌幅", "成交量", "成交额", "换手率", "总市值"}

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验关键列是否存在，统一列名，转换数值类型"""
        if df.empty:
            return df

        # 统一列名（东财/新浪/其他源的列名差异）
        rename_map = {}
        for col in df.columns:
            if col in self._COLUMN_MAP:
                rename_map[col] = self._COLUMN_MAP[col]
        if rename_map:
            df = df.rename(columns=rename_map)

        # 检查缺失的关键列
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("行情数据缺少关键列: %s, 部分因子可能无法计算", missing)

        # 统一股票代码为6位纯数字（兼容新浪带前缀"sh600519"/"bj920000"与东财纯数字"600519"）
        if "代码" in df.columns:
            df["代码"] = df["代码"].astype(str).str.strip().str.lower()
            df["代码"] = df["代码"].str.replace(r'^(sh|sz|bj)', '', regex=True)

        # 数值列强制转换
        num_cols = ["最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "换手率",
                     "总市值", "流通市值", "市盈率", "市净率", "量比", "振幅"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_a_share_realtime(self) -> pd.DataFrame:
        """
        获取全A股实时行情
        返回列：代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额,
                振幅, 最高, 最低, 今开, 昨收, 量比, 换手率, 市盈率,
                市净率, 总市值, 流通市值, 涨速, 5分钟涨跌, 60日涨跌幅
        """
        cache_key = "a_spot"
        if cache_key in self._cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        # 主源：东财（带重试+代理轮换）
        try:
            df = self._fetch_with_retry(
                lambda: ak.stock_zh_a_spot_em(),
                max_retries=2, delay=3
            )
            if df is not None:
                df = df.rename(columns={"序号": "idx"})
                df = self._validate_and_normalize(df)
                self._cache_put(cache_key, df)
                logger.info(f'  ✅ 获取A股实时行情(东财): {len(df)} 只')
                return df
        except Exception as e:
            logger.warning("东财行情失败: %s, 尝试新浪源...", e)

        # 备用源：新浪
        try:
            df = ak.stock_zh_a_spot()
            df = self._validate_and_normalize(df)
            # 新浪源缺少总市值/换手率时，批量补充
            missing = self._REQUIRED_COLS - set(df.columns)
            if missing:
                df = self._batch_supplement_market_cap(df)
            self._cache_put(cache_key, df)
            logger.info(f'  ✅ 获取A股实时行情(新浪): {len(df)} 只')
            return df
        except Exception as e:
            logger.error("获取A股行情全部失败: %s", e)
            return pd.DataFrame()

    def _batch_supplement_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量补充新浪源缺失的总市值/换手率/量比。

        优先东财个股接口（前200只，限流保护）；东财不可达或填不满时，
        用腾讯 qt.gtimg.cn 全市场补充（这批字段缺失会拖累预评分排序）。
        """
        if df.empty:
            return df
        # 快检：东财是否可达（单股探针，带代理）
        probe_code = df["代码"].iloc[0]
        cap_map = {}
        eastmoney_ok = False
        try:
            import concurrent.futures as _cf2
            def _probe():
                return ak.stock_individual_info_em(symbol=probe_code)
            with _cf2.ThreadPoolExecutor() as _ex2:
                probe = _ex2.submit(_probe).result(timeout=8)
            eastmoney_ok = not (probe is None or probe.empty)
        except Exception as e:
            logger.warning("东财个股API不可达，将尝试腾讯源补充市值: %s", e)

        if eastmoney_ok:
            try:
                import concurrent.futures as _cf
                codes = df["代码"].iloc[:200].tolist()
                results = {}
                def _fetch_info(code):
                    try:
                        info = ak.stock_individual_info_em(symbol=code)
                        if info is not None and not info.empty:
                            mcap_row = info[info["item"] == "总市值"]
                            shares_row = info[info["item"] == "流通股本"]
                            mcap = float(mcap_row["value"].iloc[0]) * 1e4 if not mcap_row.empty else 0
                            shares = float(shares_row["value"].iloc[0]) * 1e4 if not shares_row.empty else 0
                            return code, {"总市值": mcap, "流通股本": shares}
                    except Exception as e:
                        logger.warning("个股信息获取失败(%s): %s", code, e)
                    return code, {}
                with _cf.ThreadPoolExecutor(max_workers=5) as _ex:
                    futures = {_ex.submit(_fetch_info, c): c for c in codes}
                    for f in _cf.as_completed(futures, timeout=30):
                        c, info = f.result()
                        results[c] = info
                cap_map = {str(c): v.get("总市值", 0) for c, v in results.items()}
                filled = sum(1 for v in cap_map.values() if v > 0)
                if filled > 0:
                    logger.info(f'    ✅ 东财补充总市值: {filled} 只')
            except Exception as e:
                logger.warning("东财补充总市值失败: %s", e)

        # fallback/兜底：用腾讯行情补充全市场市值/换手率/量比（免费、稳定）
        # 东财补充也可能只覆盖前200只，故只要未填满就继续用腾讯补满全市场。
        filled = sum(1 for v in cap_map.values() if v > 0)
        if filled < len(df):
            try:
                all_codes = df["代码"].astype(str).tolist()
                tencent = self._fetch_tencent_quotes(all_codes)
                if tencent:
                    t_mv = {c: v.get("总市值", 0) for c, v in tencent.items()}
                    t_turn = {c: v.get("换手率", 0.0) for c, v in tencent.items() if v.get("换手率")}
                    t_vr = {c: v.get("量比", 0.0) for c, v in tencent.items() if v.get("量比")}
                    # 合并：东财既有值优先，其余用腾讯填充
                    for c, mv in t_mv.items():
                        if c not in cap_map or cap_map.get(c, 0) <= 0:
                            cap_map[c] = mv
                    filled = sum(1 for v in cap_map.values() if v > 0)
                    logger.info("腾讯补充总市值+换手率+量比: %d 只", filled)
                    if t_turn:
                        df["换手率"] = df["代码"].astype(str).map(t_turn).fillna(df.get("换手率", 0.0))
                    if t_vr:
                        df["量比"] = df["代码"].astype(str).map(t_vr).fillna(df.get("量比", 0.0))
            except Exception as e:
                logger.warning("腾讯补充总市值也失败: %s", e)

        if filled > 0:
            df["总市值"] = df["代码"].astype(str).map(cap_map).fillna(0)
        if "换手率" not in df.columns:
            df["换手率"] = 0.0
        if "量比" not in df.columns:
            df["量比"] = 0.0
        return df

    def _fetch_tencent_quotes(self, codes: List[str]) -> Dict[str, Dict]:
        """腾讯行情补充总市值/换手率/量比（东财不可用时的 fallback）。

        返回 {6位代码: {"总市值": 元, "换手率": %, "量比": float}}。
        腾讯 qt.gtimg.cn 行情字段：p[45]=总市值(亿元)，p[38]=换手率(%)，p[49]=量比。
        免费、稳定，用于东财不可达时保证市值/换手率/量比维度不缺失。
        """
        result: Dict[str, Dict] = {}
        if not codes:
            return result
        syms = []
        for c in codes:
            c = str(c).strip().lower()
            c = re.sub(r'^(sh|sz|bj)', '', c)  # 去前缀，统一为6位纯数字
            prefix = "sh" if c.startswith("6") else ("bj" if c.startswith(("4", "8", "9")) else "sz")
            syms.append(prefix + c)
        batch = 150
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            try:
                r = requests.get("https://qt.gtimg.cn/q=" + ",".join(chunk),
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                for line in r.text.strip().split(";"):
                    if "=" not in line or "none_match" in line:
                        continue
                    seg = line.split('"')
                    if len(seg) < 2:
                        continue
                    raw = seg[1]
                    if not raw:
                        continue
                    # 提取6位股票代码（key 形如 v_sh600519 / v_sz000001，可能带换行）
                    m = re.search(r'v_([a-z]{2})(\d{6})', line)
                    if not m:
                        continue
                    code = m.group(2)
                    p = raw.split("~")
                    if len(p) <= 45:
                        continue
                    try:
                        mv_yi = float(p[45])  # 总市值(亿元)
                    except (ValueError, IndexError):
                        mv_yi = 0.0
                    rate = 0.0
                    try:
                        rate = float(p[38])  # 换手率(%)
                    except (ValueError, IndexError):
                        pass
                    vratio = 0.0
                    try:
                        vratio = float(p[49])  # 量比
                    except (ValueError, IndexError):
                        pass
                    if not (1 <= mv_yi <= 200000):  # 合理性校验，过滤脏数据
                        continue
                    result[code] = {"总市值": mv_yi * 1e8, "换手率": rate, "量比": vratio}
            except Exception as e:
                logger.warning("腾讯行情批次 %d 获取失败: %s", i, e)
        return result

    def _fetch_market_cap_tencent(self, codes: List[str]) -> Dict[str, float]:
        """兼容旧接口：仅返回 {6位代码: 总市值(元)}。见 _fetch_tencent_quotes。"""
        return {c: v.get("总市值", 0) for c, v in self._fetch_tencent_quotes(codes).items()}



    _turnover_warned = False


    _TURN_CACHE_TTL = 7 * 86400


    # ------------------------------------------------------------------
    # 外盘期货 / 指数
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 外盘期货 / 指数（主源：腾讯 qt.gtimg.cn；免费源覆盖美股/港股/商品）
    # ------------------------------------------------------------------
    # 展示名 -> (腾讯代码, 解析类型)
    #   us : s_usDJI / s_usINX / s_usIXIC  ("~"分: [3]最新价 [5]涨跌幅%)
    #   hk : r_hkHSI                      ("~"分: [3]最新价 [4]昨收 => 算涨跌幅)
    #   hf : hf_GC / hf_CL / hf_SI / hf_NG / hf_HG (","分: [0]最新价 [1]涨跌幅%)
    _GLOBAL_MAP = {
        "道琼斯": ("s_usDJI", "us"),
        "标普500": ("s_usINX", "us"),
        "纳斯达克": ("s_usIXIC", "us"),
        "恒生指数": ("r_hkHSI", "hk"),
        "纽约黄金": ("hf_GC", "hf"),
        "纽约原油": ("hf_CL", "hf"),
        "纽约白银": ("hf_SI", "hf"),
        "美国天然气": ("hf_NG", "hf"),
        "美铜": ("hf_HG", "hf"),
    }


    @staticmethod
    def _fetch_tencent(codes: List[str]) -> Dict[str, tuple]:
        """批量拉取腾讯行情，返回 {腾讯代码: (最新价:float, 涨跌幅%:float)}。"""
        result: Dict[str, tuple] = {}
        if not codes:
            return result
        try:
            url = "https://qt.gtimg.cn/q=" + ",".join(codes)
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            for line in resp.text.strip().split(";"):
                if "=" not in line or "none_match" in line:
                    continue
                key = line.split("=")[0].replace("v_", "").strip()
                seg = line.split('"')
                raw = seg[1] if len(seg) > 1 else ""
                if not raw:
                    continue
                try:
                    if key.startswith("hf"):
                        p = raw.split(",")
                        if len(p) >= 2:
                            result[key] = (float(p[0]), float(p[1]))
                    elif key.startswith("s_us"):
                        p = raw.split("~")
                        if len(p) >= 6:
                            result[key] = (float(p[3]), float(p[5]))
                    elif key.startswith("r_hk"):
                        p = raw.split("~")
                        if len(p) >= 5:
                            price, prev = float(p[3]), float(p[4])
                            pct = (price - prev) / prev * 100 if prev else 0.0
                            result[key] = (price, round(pct, 2))
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.warning("腾讯外盘行情获取失败: %s", e)
        return result


    # ------------------------------------------------------------------
    # 涨停板 / 跌停板
    # ------------------------------------------------------------------
    def get_limit_up_pool(self, date: str = None) -> pd.DataFrame:
        """获取涨停板股票池"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_em(date=date)
            logger.info(f'  ✅ 涨停板: {len(df)} 只')
            return df
        except Exception as e:
            logger.warning("获取涨停池失败: %s", e)
            return pd.DataFrame()

    def get_limit_down_pool(self, date: str = None) -> pd.DataFrame:
        """获取跌停板股票池"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_dtgc_em(date=date)
            logger.info(f'  ✅ 跌停板: {len(df)} 只')
            return df
        except Exception as e:
            logger.warning("获取跌停池失败: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 北向资金
    # ------------------------------------------------------------------
    def get_north_flow(self, lookback_days: int = 10) -> Dict:
        """获取北向资金净流入及连续流向天数。

        返回 {
          "date": 最新交易日,
          "net_inflow": 当日净流入(数值化, 单位与数据源一致),
          "consecutive_days": 连续同向天数(正值=连续流入, 负值=连续流出, 0=中性/中断),
          "net_flow": 兼容旧调用方
        }
        market_regime.assess_regime 的北向维度依赖 net_inflow/consecutive_days，
        此前生产侧一直传 None 导致该维度从未参与择时判定。
        """
        result = {"date": "", "net_inflow": 0.0, "consecutive_days": 0, "net_flow": 0.0}
        try:
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is None or df.empty:
                return result
            date_col = "日期" if "日期" in df.columns else None
            flow_col = next(
                (c for c in ("当日净流入", "北向净流入", "净流入") if c in df.columns), None
            )
            if not (date_col and flow_col):
                logger.warning("北向资金接口列结构变化: %s", list(df.columns))
                return result

            df = df.sort_values(date_col)
            recent = df.tail(lookback_days)
            flows = pd.to_numeric(recent[flow_col], errors="coerce").dropna()
            if flows.empty:
                return result

            latest = float(flows.iloc[-1])
            sign_now = 1 if latest > 0 else (-1 if latest < 0 else 0)
            consec = 0
            if sign_now != 0:
                for v in reversed(flows.tolist()):
                    s = 1 if v > 0 else (-1 if v < 0 else 0)
                    if s == sign_now:
                        consec += 1
                    else:
                        break
            result.update({
                "date": str(recent[date_col].iloc[-1]),
                "net_inflow": latest,
                "consecutive_days": consec * sign_now,
                "net_flow": latest,
            })
        except Exception as e:
            logger.warning("获取北向资金失败: %s", e)
        return result
