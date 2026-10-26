"""
历史K线获取 + 换手率补充（腾讯/新浪/baostock 三源 failover）

由 market_data.py 拆分(AGENTS.md 技术债偿还): K线获取与换手率补充。
"""
import os
import requests
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import baostock as _bs

try:
    from src.utils.log import get_logger
    logger = get_logger()
except Exception as e:
    import logging
    logging.warning("get_logger 不可用, 降级到标准 logging: %s", e)
    logger = logging.getLogger("stock_prediction")


# 腾讯K线字段映射: 腾讯代码 → 字段列表
_TX_KLINE_FIELDS = "date,open,high,low,close,volume,amount"

# 腾讯K线域名(按优先级)。web.ifzq.gtimg.cn 被腾讯 WAF 拦截时返回 501 + HTML拦截页,
# 而非JSON, 曾导致整轮盘前K线全灭; 故必须支持域名级 failover。
_TX_KLINE_HOSTS = ("web.ifzq.gtimg.cn", "ifzq.gtimg.cn")
# 单域名连续失败达此次数后, 本进程内不再尝试(避免持续触发WAF并刷屏)
_TX_HOST_MAX_FAIL = 3


class KlineFetcher:
    """历史K线获取 + 换手率补充"""

    def __init__(self, cache_get=None, cache_put=None, cache_ts_put=None, db=None):
        self._cache_get = cache_get or (lambda k: None)
        self._cache_put = cache_put or (lambda k, v: None)
        self._cache_ts_put = cache_ts_put or (lambda k, v: None)
        self._db = db
        self._turnover_warned = False

    # 域名级失败计数(类变量, 进程内共享)
    _host_fail: dict = {}

    @classmethod
    def _tencent_hosts(cls):
        """返回当前可用的腾讯K线域名, 跳过已被WAF封禁/连续失败的域名。"""
        return [h for h in _TX_KLINE_HOSTS
                if cls._host_fail.get(h, 0) < _TX_HOST_MAX_FAIL] or list(_TX_KLINE_HOSTS)

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
            # 同一域名只告警一次: WAF 封禁后每只票都报警会刷屏(2026-09-14: 134条同类告警)
            if KlineFetcher._host_fail.get(host, 0) == 0:
                logger.warning("腾讯K线(%s)失败: %s", host, e)
            else:
                logger.debug("腾讯K线(%s)仍失败: %s", host, e)
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

    def get_stock_history(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """获取单只股票前复权日K线历史(腾讯主源, 新浪备源)。

        供因子计算/远期收益评估使用。返回含 日期/开盘/收盘/最高/最低/成交量/
        成交额/换手率/涨跌幅 的标准 DataFrame；主备源都失败返回 None。
        """
        clean = "".join(ch for ch in str(code) if ch.isdigit())
        if not clean:
            return None
        low = str(code).lower()
        if low.startswith(("sh", "sz", "bj")):
            tx_symbol = low
        else:
            first = clean[0]
            prefix = "sh" if first in "69" else ("sz" if first in "03" else ("bj" if first == "8" else "sh"))
            tx_symbol = prefix + clean
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days * 2 + 60)).strftime("%Y-%m-%d")
        df = None
        for host in self._tencent_hosts():
            df = self._fetch_tencent_kline(host, tx_symbol, start, end, days, days)
            if df is not None and not df.empty:
                KlineFetcher._host_fail[host] = 0
                break
            KlineFetcher._host_fail[host] = KlineFetcher._host_fail.get(host, 0) + 1
        if df is None or df.empty:
            df = self._fetch_sina_kline(tx_symbol, days)
        if df is None or df.empty:
            return None
        df = self._supplement_turnover(clean, df, days)
        # 写入本地数据库（增量同步）
        if self._db is not None and not df.empty:
            try:
                self._db.save_klines(clean, df)
            except Exception as e:
                logger.debug("K线写入本地DB失败(%s): %s", clean, e)
        return df

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
                    # 确保 baostock 已登录，避免 "you don't login" 错误
                    from src.data.market_data import _BaostockConnection
                    _BaostockConnection.get_connection()
                    rs = _bs.query_history_k_data_plus(
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
