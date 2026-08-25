"""
A股 + 外盘行情数据获取
数据源：akshare（免费）
"""
import akshare as ak
import pandas as pd
import numpy as np
import requests
import re
import os
import time
import atexit
import threading
from typing import Optional, Dict, List
from datetime import datetime, timedelta

try:
    from src.utils.log import get_logger
    logger = get_logger()
except Exception:
    import logging
    logger = logging.getLogger("stock_prediction")


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
                except Exception:
                    # 登录失败时返回模块本身，调用方自行处理
                    pass
        return cls._instance

    @classmethod
    def _cleanup(cls):
        if cls._logged_in:
            try:
                cls._instance.logout()
                cls._logged_in = False
            except Exception:
                pass


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
            except Exception:
                pass

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
            except Exception:
                pass

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
                except Exception:
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
                except Exception:
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
        except Exception:
            logger.warning("东财个股API不可达，将尝试腾讯源补充市值")

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
                    except:
                        pass
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

    # ------------------------------------------------------------------
    # 个股历史日线（技术因子计算用）
    # ------------------------------------------------------------------
    def get_stock_history(
        self, stock_code: str, days: int = 120, retries: int = 1
    ) -> pd.DataFrame:
        """
        获取个股日K线（前复权）
        主源: 腾讯行情（免费快速无封禁）
        备选: baostock
        """
        import re

        cache_key = f"hist_{stock_code}"
        if cache_key in self._cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        clean_code = re.sub(r'^(sh|sz|bj|SH|SZ|BJ)', '', stock_code)
        clean_code = clean_code.replace(".", "")

        if stock_code.startswith(('sh', 'SH')) or clean_code.startswith(('6', '5', '9')):
            tx_symbol = f"sh{clean_code}"
        elif stock_code.startswith(('bj', 'BJ')):
            tx_symbol = f"bj{clean_code}"
        else:
            tx_symbol = f"sz{clean_code}"

        # === 主源: 腾讯行情（多域名 failover）===
        # ifzq.gtimg.cn 与 web.ifzq.gtimg.cn 会被腾讯间歇性限流/WAF拦截，
        # 且两者不同时失效，故按顺序逐个尝试。
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 50)).strftime("%Y-%m-%d")
        lookback = min(800, max(30, int(days * 1.8) + 20))
        for host in ("ifzq.gtimg.cn", "web.ifzq.gtimg.cn"):
            df = self._fetch_tencent_kline(
                host, tx_symbol, start, end, lookback, days
            )
            if df is not None and not df.empty:
                df = self._supplement_turnover(clean_code, df, days)
                self._cache_put(cache_key, df)
                return df

        # === 备选: 新浪K线 ===
        df = self._fetch_sina_kline(tx_symbol, days)
        if df is not None and not df.empty:
            df = self._supplement_turnover(clean_code, df, days)
            self._cache_put(cache_key, df)
            return df

        # === 备选: baostock（带超时，复用进程级连接）===
        bs_code = (f"sh.{clean_code}" if clean_code.startswith('6') else f"sz.{clean_code}")
        import concurrent.futures as _cf
        _bs_ok = False
        _bs_data = None

        def _run_bs():
            nonlocal _bs_ok, _bs_data
            bs = _BaostockConnection.get_connection()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days + 50)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
            data = []
            while (rs.error_code == '0') and rs.next():
                data.append(rs.get_row_data())
            if data:
                _bs_data = data
                _bs_ok = True

        try:
            with _cf.ThreadPoolExecutor() as _ex:
                _ex.submit(_run_bs).result(timeout=8)
            if _bs_ok and _bs_data:
                df = pd.DataFrame(_bs_data, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "涨跌幅"])
                for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "涨跌幅"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.tail(days).reset_index(drop=True)
                self._cache_put(cache_key, df)
                return df
        except Exception as e:
            if not self._warned_bs:
                logger.warning("baostock(%s)也失败: %s, 该股历史数据不可用", bs_code, e)
                self._warned_bs = True

        return pd.DataFrame()

    def _fetch_tencent_kline(
        self, host: str, tx_symbol: str, start: str, end: str,
        lookback: int, days: int,
    ) -> Optional[pd.DataFrame]:
        """从指定腾讯域名获取前复权日K线，成功返回标准化 DataFrame，失败返回 None。"""
        try:
            resp = requests.get(
                f"https://{host}/appstock/app/fqkline/get",
                params={"param": f"{tx_symbol},day,{start},{end},{lookback},qfq"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                    "Referer": "https://gu.qq.com/",
                },
                timeout=8,
            )
            payload = resp.json()
            item = payload.get("data", {}).get(tx_symbol, {})
            rows = item.get("qfqday") or item.get("day") or []
            if not rows:
                return None
            records = []
            for row in rows:
                if len(row) < 6:
                    continue
                # 腾讯API近年格式变更：OHLC和成交量字段可能返回 dict（如 {"pre":x,"post":y}）
                # 用统一提取函数兼容数值/dict 两种格式
                open_val = self._extract_tencent_value(row[1])
                close_val = self._extract_tencent_value(row[2])
                high_val = self._extract_tencent_value(row[3])
                low_val = self._extract_tencent_value(row[4])
                vol_val = self._extract_tencent_value(row[5])
                if close_val <= 0 or vol_val < 0:
                    continue
                # 成交额：优先用 row[6]，否则用 close*vol*100 估算
                amount_val = close_val * vol_val * 100
                if len(row) > 6:
                    amount_val = self._extract_tencent_value(
                        row[6], default=amount_val / 10000
                    ) * 10000
                records.append({
                    "日期": str(row[0]),
                    "开盘": open_val,
                    "收盘": close_val,
                    "最高": high_val,
                    "最低": low_val,
                    "成交量": vol_val * 100,
                    "成交额": amount_val,
                })
            if not records:
                return None
            df = pd.DataFrame(records)
            # 腾讯日K接口无换手率字段：置 NaN 而非硬编码 0——
            # 0 会被下游(chip筹码衰减/换手异动)当作真实"零换手"，NaN 则走降级路径
            df["换手率"] = float("nan")
            self._warn_no_turnover()
            df["涨跌幅"] = df["收盘"].pct_change().fillna(0) * 100
            return df.tail(days).reset_index(drop=True)
        except Exception as e:
            logger.warning("腾讯K线(%s)失败: %s", host, e)
            return None

    _turnover_warned = False

    @classmethod
    def _warn_no_turnover(cls):
        """备源缺换手率仅告警一次，避免刷屏"""
        if not cls._turnover_warned:
            cls._turnover_warned = True
            logger.warning(
                "K线备源(腾讯/新浪)无换手率字段，chip筹码衰减/换手异动因子将降级"
                "(NaN按无衰减处理)。东财主源恢复后自动消除"
            )

    def _fetch_sina_kline(self, tx_symbol: str, days: int) -> Optional[pd.DataFrame]:
        """新浪日K线备用源（quotes.sina.cn，免费稳定）。
        返回标准化 DataFrame，失败返回 None。注意新浪源无成交额，用 close*volume 估算。
        """
        try:
            datalen = min(1023, max(90, int(days * 1.8) + 20))
            resp = requests.get(
                "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData",
                params={"symbol": tx_symbol, "scale": "240", "ma": "no", "datalen": str(datalen)},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                    "Referer": "https://finance.sina.com.cn",
                },
                timeout=8,
            )
            rows = resp.json()
            if not rows or not isinstance(rows, list):
                return None
            records = []
            for row in rows:
                try:
                    records.append({
                        "日期": str(row["day"]),
                        "开盘": float(row["open"]),
                        "收盘": float(row["close"]),
                        "最高": float(row["high"]),
                        "最低": float(row["low"]),
                        # volume 单位为股，此处直接存股数；成交额用 close*volume 估算
                        "成交量": float(row["volume"]),
                        "成交额": float(row["close"]) * float(row["volume"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if not records:
                return None
            df = pd.DataFrame(records)
            # 新浪K线无换手率字段：同腾讯源处理
            df["换手率"] = float("nan")
            self._warn_no_turnover()
            df["涨跌幅"] = df["收盘"].pct_change().fillna(0) * 100
            return df.tail(days).reset_index(drop=True)
        except Exception as e:
            logger.warning("新浪K线(%s)失败: %s", tx_symbol, e)
            return None

    _TURN_CACHE_TTL = 7 * 86400

    def _supplement_turnover(self, clean_code: str, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """备源K线换手率全缺失时，用 baostock turn 字段按日期补充。

        背景：换手率硬编码0曾导致 chip 筹码衰减/换手异动因子静默失效；
        现置 NaN 后由此方法恢复真实数据。带本地缓存(7天)避免每日重复拉取；
        baostock 调用限时 10s，失败静默保持 NaN 不影响主流程。
        """
        import pickle
        import time as _time
        import concurrent.futures as _cf

        col = "换手率"
        if col not in df.columns or df[col].notna().any():
            return df
        if not clean_code.startswith(("6", "5", "9", "0", "2", "3")):
            return df  # 北交所等 baostock 不覆盖

        self._warn_no_turnover()

        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "cache", "turnover")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{clean_code}.pkl")

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        turn_map = {}
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    cached_ts, cached_rows = pickle.load(f)
                if _time.time() - cached_ts < self._TURN_CACHE_TTL:
                    turn_map = {r[0]: _f(r[1]) for r in cached_rows if r[1] not in (None, "")}
            if not turn_map:
                bs_code = (f"sh.{clean_code}" if clean_code.startswith(("6", "5", "9"))
                           else f"sz.{clean_code}")
                start = (datetime.now() - timedelta(days=days + 50)).strftime("%Y-%m-%d")
                rows_box = []

                def _fetch():
                    bs = _BaostockConnection.get_connection()
                    rs = bs.query_history_k_data_plus(
                        bs_code, "date,turn",
                        start_date=start,
                        end_date=datetime.now().strftime("%Y-%m-%d"),
                        frequency="d", adjustflag="2")
                    while rs.error_code == "0" and rs.next():
                        rows_box.append(rs.get_row_data())

                with _cf.ThreadPoolExecutor() as _ex:
                    _ex.submit(_fetch).result(timeout=10)
                turn_map = {r[0]: _f(r[1]) for r in rows_box if r[1] not in (None, "")}
                with open(cache_file, "wb") as f:
                    pickle.dump((_time.time(), rows_box), f)
        except Exception as e:
            logger.debug("换手率补充失败(%s): %s", clean_code, e)
            return df

        if not turn_map:
            return df
        dates = df["日期"].astype(str).str[:10]
        mask = df[col].isna()
        df.loc[mask, col] = dates[mask].map(turn_map)
        return df

    @staticmethod
    def _extract_tencent_value(field, default: float = 0.0) -> float:
        """统一提取腾讯K线字段值，兼容数值/字符串/dict 三种格式。
        dict 格式常见为 {"pre": x, "post": y}，优先取 post（后复权值）。
        """
        if field is None:
            return default
        if isinstance(field, (int, float)):
            return float(field)
        if isinstance(field, dict):
            # 优先 post > pre > amount > value
            for key in ("post", "pre", "amount", "AMOUNT", "value"):
                v = field.get(key)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            return default
        if isinstance(field, str):
            try:
                return float(field)
            except ValueError:
                return default
        return default

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

    def get_global_market_summary(self, config_markets: list) -> Dict:
        """根据配置监控列表提取关键外盘数据。

        主数据源：腾讯行情 qt.gtimg.cn（字段清晰、含涨跌幅、覆盖美股/港股/外盘期货）。
        免费源（新浪/腾讯）无法覆盖的品种（日经225、富时中国A50、LME铜、铁矿石、
        美元指数、美元兑人民币）返回 price='—'，避免显示误导性的 0.00%。
        """
        cache_key = "global_summary"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._filter_global(cached, config_markets)

        all_data: Dict[str, Dict] = {}
        needed = {}
        for m in config_markets:
            name = m.get("name", "")
            if name in self._GLOBAL_MAP:
                needed[self._GLOBAL_MAP[name][0]] = name

        if needed:
            fetched = self._fetch_tencent(list(needed.keys()))
            for tc, (price, pct) in fetched.items():
                name = needed[tc]
                all_data[name] = {"name": name, "price": price, "change": pct, "available": True}

        self._cache_put(cache_key, all_data)
        return self._filter_global(all_data, config_markets)

    @staticmethod
    def _filter_global(all_data: Dict, config_markets: list) -> Dict:
        """按配置顺序输出；未被免费源覆盖的品种标记为 '—'。"""
        out: Dict[str, Dict] = {}
        for m in config_markets:
            name = m.get("name", "")
            if name in all_data:
                out[name] = all_data[name]
            else:
                out[name] = {"name": name, "price": "—", "change": 0.0, "available": False}
        return out

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
    # 指数行情
    # ------------------------------------------------------------------
    def get_index_realtime(self) -> Dict:
        """获取主要指数实时行情 (Sina主力 + akshare备选)"""
        indices = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
        }
        result = {}
        try:
            codes = ",".join(indices.keys())
            url = f"https://hq.sinajs.cn/list={codes}"
            headers = {"Referer": "https://finance.sina.com.cn"}
            resp = requests.get(url, headers=headers, timeout=10)
            for line in resp.text.strip().split('\n'):
                m = re.search(r'hq_str_(\w+)="([^"]+)"', line)
                if m:
                    code = m.group(1)
                    data = m.group(2).split(',')
                    name = indices.get(code, code)
                    if len(data) > 3:
                        price = float(data[3])
                        prev_close = float(data[2])
                        change = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0
                        result[name] = {"price": price, "change": change}
        except Exception as e:
            logger.warning("获取指数行情(Sina)失败: %s, 尝试akshare...", e)

        if not result:
            try:
                df = ak.stock_zh_index_spot_em()
                if df is not None:
                    for code_raw, name in [("000001", "上证指数"), ("399001", "深证成指"), ("399006", "创业板指")]:
                        matched = df[df["代码"] == code_raw]
                        if not matched.empty:
                            row = matched.iloc[0]
                            result[name] = {
                                "price": row.get("最新价", 0),
                                "change": row.get("涨跌幅", 0),
                            }
            except Exception as e2:
                logger.warning("获取指数行情(akshare)也失败: %s", e2)

        if result:
            for name, d in result.items():
                icon = "📈" if d["change"] > 0 else "📉"
                logger.info(f"  {icon} {name}: {d['price']:.2f} ({d['change']:+.2f}%)")
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
