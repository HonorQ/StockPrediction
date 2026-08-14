"""
数据源健康自检（轻量、可离线测试）

在 pipeline 启动时探测各数据源可达性，输出状态日志，避免「东财挂了但全程静默降级」。
各探测用最小请求 + 短超时，不影响主流程；失败即视为该源不可用。
"""
import logging

import requests

logger = logging.getLogger("stock_prediction")

_TIMEOUT = 5


def _probe(url: str, ok_substr: str = None, headers: dict = None) -> bool:
    """对 url 发一次短超时 GET，命中 ok_substr 视为可用。"""
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=_TIMEOUT)
        if r.status_code != 200:
            return False
        if ok_substr and ok_substr not in r.text:
            return False
        return True
    except Exception:
        return False


def check_eastmoney() -> bool:
    """探测东方财富（akshare 主源）可用性。"""
    return _probe(
        "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&fs=m:0+t:6",
        ok_substr="data",
    )


def check_tencent() -> bool:
    """探测腾讯行情（外盘/个股 fallback 源）可用性。"""
    return _probe("https://qt.gtimg.cn/q=sh600519", ok_substr="贵州茅台")


def check_sina() -> bool:
    """探测新浪行情可用性。"""
    # 新浪要求 Referer 头，否则返回 403/空内容导致误报
    return _probe(
        "https://hq.sinajs.cn/list=sh600519",
        ok_substr="600519",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
    )


def health_check() -> dict:
    """返回各数据源可用状态字典。"""
    return {
        "eastmoney": check_eastmoney(),
        "tencent": check_tencent(),
        "sina": check_sina(),
    }


def summarize(status: dict) -> str:
    """把状态字典转成一行可读日志。"""
    parts = []
    for name, ok in status.items():
        parts.append(f"{name}={'✅' if ok else '❌'}")
    return "数据源健康: " + " ".join(parts)
