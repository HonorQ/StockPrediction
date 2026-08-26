# -*- coding: utf-8 -*-
"""盘前/盘中/午后共用的运行时工具

由 run_daily.py 外迁(规则 AGENTS.md 二: 入口只做参数解析/分发)。
"""
import os
import sys
import json
import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from typing import Dict

from src.utils.log import get_logger

logger = get_logger()

from src.data.source_health import health_check, summarize


class _DualOutput:


    """同时写入控制台和日志文件的输出包装"""


    def __init__(self, logger, original):


        self._logger = logger


        self._original = original





    def write(self, msg):


        if msg and msg.strip():


            self._logger.info(msg.strip())


        self._original.write(msg)





    def flush(self):


        self._original.flush()









class _NumpyEncoder(json.JSONEncoder):
    """numpy 类型 → JSON 序列化(run_daily 同款, 缓存保存用)"""
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return super().default(o)


def setup_logging(mode: str = "premarket"):


    """配置日志：文件(按日期轮转) + 控制台"""


    log_dir = os.path.join(ROOT, "logs")


    os.makedirs(log_dir, exist_ok=True)





    today = datetime.now().strftime("%Y-%m-%d")


    log_file = os.path.join(log_dir, f"{today}_{mode}.log")





    logger = logging.getLogger("stock_prediction")


    logger.setLevel(logging.DEBUG)





    # 避免重复添加handler


    if logger.handlers:


        logger.handlers.clear()





    # 文件handler


    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")


    fh.setLevel(logging.DEBUG)


    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))


    logger.addHandler(fh)





    # 控制台handler（通过StreamHandler复用原始stdout）


    ch = logging.StreamHandler(sys.stdout)


    ch.setLevel(logging.INFO)


    ch.setFormatter(logging.Formatter("%(message)s"))


    logger.addHandler(ch)





    # 重定向stdout到日志文件（捕获所有print输出）


    sys.stdout = _DualOutput(logger, sys.stdout)





    # 捕获未处理异常并写入日志（避免静默崩溃无痕）


    def _excepthook(exc_type, exc_val, exc_tb):


        logger.error("未捕获异常导致进程退出", exc_info=(exc_type, exc_val, exc_tb))


    sys.excepthook = _excepthook





    return logger








def write_summary(mode: str, candidates_count: int, recommendations: list, **extra):


    """追加一行JSON到 daily_summary.log，快速查看每次运行结果"""


    summary = {


        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),


        "mode": mode,


        "candidates": candidates_count,


        "recommendations": [


            {"code": r.get("code", ""), "name": r.get("name", ""), "score": r.get("score", 0)}


            for r in recommendations


        ],


    }


    summary.update(extra)


    log_dir = os.path.join(ROOT, "logs")


    os.makedirs(log_dir, exist_ok=True)


    summary_path = os.path.join(log_dir, "daily_summary.log")


    with open(summary_path, "a", encoding="utf-8") as f:


        f.write(json.dumps(summary, ensure_ascii=False) + "\n")














def _prefetch_history_and_flow(market_fetcher, money_fetcher, codes: list, days: int = 120):


    """并行预取多只股票的历史K线和资金流数据。


    返回 (history_cache, flow_cache) 两个 dict[code -> DataFrame]。


    网络 I/O 是瓶颈，并行化可显著加速（受限于数据源限流，默认 8 并发）。


    """


    import concurrent.futures as _cf





    history_cache: Dict[str, object] = {}


    flow_cache: Dict[str, object] = {}





    def _fetch_one(code: str):


        try:


            h = market_fetcher.get_stock_history(code, days=days)


        except Exception:


            h = None


        try:


            f = money_fetcher.get_stock_fund_flow(code)


        except Exception:


            f = None


        return code, h, f





    # 8 并发是 akshare/腾讯 限流下的安全值


    max_workers = min(8, max(2, len(codes)))


    with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:


        futures = [ex.submit(_fetch_one, c) for c in codes]


        for fut in _cf.as_completed(futures):


            try:


                code, h, f = fut.result()


                history_cache[code] = h


                flow_cache[code] = f


            except Exception:


                continue





    return history_cache, flow_cache






































CACHE_DIR = os.path.join(ROOT, "cache")


os.makedirs(CACHE_DIR, exist_ok=True)


def _prefetch_history_and_flow(market_fetcher, money_fetcher, codes: list, days: int = 120):

    """并行预取多只股票的历史K线和资金流数据。

    返回 (history_cache, flow_cache) 两个 dict[code -> DataFrame]。

    网络 I/O 是瓶颈，并行化可显著加速（受限于数据源限流，默认 8 并发）。

    """

    import concurrent.futures as _cf



    history_cache: Dict[str, object] = {}

    flow_cache: Dict[str, object] = {}



    def _fetch_one(code: str):

        try:

            h = market_fetcher.get_stock_history(code, days=days)

        except Exception:

            h = None

        try:

            f = money_fetcher.get_stock_fund_flow(code)

        except Exception:

            f = None

        return code, h, f



    # 8 并发是 akshare/腾讯 限流下的安全值

    max_workers = min(8, max(2, len(codes)))

    with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:

        futures = [ex.submit(_fetch_one, c) for c in codes]

        for fut in _cf.as_completed(futures):

            try:

                code, h, f = fut.result()

                history_cache[code] = h

                flow_cache[code] = f

            except Exception:

                continue



    return history_cache, flow_cache

























CACHE_DIR = os.path.join(ROOT, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)



def save_cache(name: str, data):

    path = os.path.join(CACHE_DIR, name)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, cls=_NumpyEncoder)

    return path



def load_cache(name: str):

    path = os.path.join(CACHE_DIR, name)

    if not os.path.exists(path):

        return None

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)



def _log_source_health():

    """启动数据源健康自检并输出状态日志。"""

    try:

        logger.info(summarize(health_check()))

    except Exception as e:

        logger.warning("数据源健康自检失败: %s", e)



def _cleanup_cache(days: int = 7):

    """清理超过 N 天的缓存文件"""

    import shutil

    cutoff = datetime.now() - timedelta(days=days)

    for fname in os.listdir(CACHE_DIR):

        fpath = os.path.join(CACHE_DIR, fname)

        try:

            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))

            if mtime < cutoff:

                if os.path.isfile(fpath):

                    os.remove(fpath)

                elif os.path.isdir(fpath):

                    shutil.rmtree(fpath, ignore_errors=True)

        except Exception:

            pass




def load_config() -> dict:
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
