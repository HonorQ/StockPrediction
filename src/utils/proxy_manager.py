"""
HTTP代理轮换管理器
支持静态代理(用户配置) + 免费代理(自动抓取+验证) + 失败自动轮换
"""
from src.utils.log import get_logger
logger = get_logger()

import os
import json
import random
import re
import time
import threading
from contextlib import contextmanager
from typing import Optional, Dict, List
from pathlib import Path


# 即使设置了 HTTP_PROXY，以下域名也不走代理
NO_PROXY_DOMAINS = [
    "ifzq.gtimg.cn", "web.ifzq.gtimg.com",
    "sina.com.cn", "hq.sinajs.cn",
    "cls.cn", "www.cls.cn",
    "baidu.com", "baostock.com",
    "szse.cn", "sse.com.cn",
    "eastmoney.com", "push2.eastmoney.com",
    "datacenter-web.eastmoney.com", "quote.eastmoney.com",
]


class ProxyManager:
    """代理轮换管理器（单例）"""

    FREE_PROXY_SOURCES = [
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http",
    ]

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: Optional[Dict] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self._static_proxies: List[str] = []
        self._free_proxies: List[str] = []
        self._current_index = 0
        self._fail_counts: Dict[str, int] = {}
        self._max_fails = 3
        self._last_fetch_time = 0
        self._cache_file = Path("cache/proxies.json")
        self._enabled = False
        self._test_url = "http://httpbin.org/ip"

        if config and "proxy" in config:
            pc = config["proxy"]
            self._enabled = pc.get("enabled", False)
            self._static_proxies = pc.get("list", [])
            self._max_fails = pc.get("max_fails", 3)

        if self._static_proxies:
            self._enabled = True

        self._load_cache()
        self._refresh_env()

    @property
    def _proxies(self) -> List[str]:
        return self._static_proxies + self._free_proxies

    def get_proxy(self) -> Optional[Dict[str, str]]:
        if not self._enabled or not self._proxies:
            return None
        idx = self._current_index % len(self._proxies)
        proxy_url = self._proxies[idx]
        return {
            "http": f"http://{proxy_url}",
            "https": f"http://{proxy_url}",
        }

    def rotate(self) -> Optional[Dict[str, str]]:
        if not self._proxies:
            return None
        self._current_index = (self._current_index + 1) % len(self._proxies)
        proxy = self.get_proxy()
        self._refresh_env(proxy)
        return proxy

    def _refresh_env(self, proxy: Optional[Dict[str, str]] = None):
        """刷新环境变量（全局生效，供 akshare/requests 自动读取）"""
        if proxy is None:
            proxy = self.get_proxy()
        if proxy:
            os.environ["HTTP_PROXY"] = proxy["http"]
            os.environ["HTTPS_PROXY"] = proxy["https"]
            os.environ["NO_PROXY"] = ",".join(NO_PROXY_DOMAINS)
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("NO_PROXY", None)

    def mark_failed(self, proxy: Optional[Dict[str, str]]):
        if not proxy:
            return
        key = proxy.get("http", "").replace("http://", "")
        if not key:
            return
        self._fail_counts[key] = self._fail_counts.get(key, 0) + 1
        if self._fail_counts[key] >= self._max_fails:
            self.rotate()

    def mark_success(self, proxy: Optional[Dict[str, str]]):
        if not proxy:
            return
        key = proxy.get("http", "").replace("http://", "")
        if key:
            self._fail_counts[key] = 0

    @contextmanager
    def for_request(self):
        yield self.get_proxy()

    def _test_proxy(self, proxy_url: str) -> bool:
        """快速验证代理是否可用（3秒超时）"""
        import requests as _req
        try:
            r = _req.get(
                self._test_url,
                proxies={"http": f"http://{proxy_url}", "https": f"http://{proxy_url}"},
                timeout=3,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _fetch_free_proxies(self):
        import requests as _req
        now = time.time()
        if now - self._last_fetch_time < 600:
            return
        self._last_fetch_time = now

        all_proxies = []
        for url in self.FREE_PROXY_SOURCES:
            try:
                r = _req.get(url, timeout=10)
                if r.status_code == 200:
                    proxies = [p.strip() for p in r.text.strip().split("\n") if p.strip()]
                    all_proxies.extend(proxies)
            except Exception:
                pass

        valid_format = [p for p in all_proxies if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$', p)]
        random.shuffle(valid_format)

        candidates = valid_format[:20]
        working = []
        for p in candidates:
            if self._test_proxy(p):
                working.append(p)
                if len(working) >= 5:
                    break

        self._free_proxies = working

        if working:
            logger.info(f'  🌐 抓取免费代理: {len(valid_format)}→验证通过{len(working)}个')
            return
        elif valid_format:
            logger.info(f'  🌐 抓取免费代理: {len(valid_format)}个, 但均不可用')
            return
        try:
            if self._cache_file.exists():
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                cached = data.get("proxies", [])
                age = time.time() - data.get("time", 0)
                if age < 3600:
                    self._free_proxies = cached
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps({"proxies": self._free_proxies, "time": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
