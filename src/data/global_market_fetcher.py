"""
外盘行情 + 指数行情获取

由 market_data.py 拆分(AGENTS.md 技术债偿还): 外盘腾讯源/指数行情。
"""
import os
import pickle
import re
import time
import requests
import akshare as ak
from typing import Dict, List

try:
    from src.utils.log import get_logger
    logger = get_logger()
except Exception as e:
    import logging
    logging.warning("get_logger 不可用, 降级到标准 logging: %s", e)
    logger = logging.getLogger("stock_prediction")


# 外盘代码映射: 中文名 → 腾讯代码
_GLOBAL_MAP = {
    "道琼斯": "us.DJI", "标普500": "us.SPX", "纳斯达克": "us.IXIC",
    "恒生指数": "hk.HSI", "日经225": "jpi.N225", "富时100": "uk.FTSE",
    "纽约原油": "hf.CL", "纽约黄金": "hf.GC",
}


class GlobalMarketFetcher:
    """外盘行情 + 指数行情（腾讯源）"""

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
            if name in _GLOBAL_MAP:
                needed[_GLOBAL_MAP[name]] = name

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

    _CACHE_TTL = 300

    def _cache_path(self, key: str) -> str:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "cache", "global_market")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{key}.pkl")

    def _cache_get(self, key: str):
        """读取外盘行情缓存(5分钟TTL)；缺失/损坏/过期返回 None。"""
        if not os.path.exists(self._cache_path(key)):
            return None
        try:
            with open(self._cache_path(key), "rb") as f:
                ts, data = pickle.load(f)
            if time.time() - ts < self._CACHE_TTL:
                return data
        except Exception as e:
            logger.warning("读取外盘缓存失败: %s", e)
        return None

    def _cache_put(self, key: str, data) -> None:
        """写入外盘行情缓存。"""
        try:
            with open(self._cache_path(key), "wb") as f:
                pickle.dump((time.time(), data), f)
        except Exception as e:
            logger.warning("写入外盘缓存失败: %s", e)

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
                        # 期货格式: 用逗号分隔
                        p = raw.split(",")
                        if len(p) >= 2:
                            result[key] = (float(p[0]), float(p[1]))
                    elif key.startswith("us.") or key.startswith("s_us"):
                        # 美股格式: 用波浪号分隔
                        # 字段3=最新价, 字段4=开盘价, 字段5=昨收价
                        # 需要计算涨跌幅 = (最新价-昨收价)/昨收价*100
                        p = raw.split("~")
                        if len(p) >= 6:
                            price = float(p[3])
                            prev_close = float(p[5])
                            change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
                            result[key] = (price, round(change_pct, 2))
                    elif key.startswith("hk.") or key.startswith("r_hk"):
                        # 港股格式: 用波浪号分隔
                        # 字段3=最新价, 字段4=昨收价
                        p = raw.split("~")
                        if len(p) >= 5:
                            price = float(p[3])
                            prev = float(p[4])
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
