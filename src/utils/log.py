"""
统一日志工具（轻量、零依赖）

所有模块共用名为 "stock_prediction" 的 logger。
run_daily.setup_logging 会为该 logger 配置「文件(DEBUG) + 控制台(INFO)」处理器；
若模块在 setup_logging 之前被导入/调用（如单测、独立脚本），
Python 内置的 lastResort 处理器会把 WARNING 及以上日志输出到 stderr，不会丢日志。

用法：
    from src.utils.log import get_logger
    logger = get_logger()          # 返回共用的 stock_prediction 日志器
    logger.info("阶段完成")
    logger.warning("东财失败，切备用源")
    logger.exception("未预期异常")  # 自动附堆栈
"""

import os
import sys
import logging
from datetime import datetime


class _DualOutput:
    """同时写入控制台和日志文件的输出包装
    
    print() → 写入原始 stdout（控制台）并记录到 logger（文件）。
    logger.info() 不会回流（StreamHandler 直接写 _original），避免重复。
    """
    def __init__(self, logger, original):
        self._logger = logger
        self._original = original
        self._logging = False

    def write(self, msg):
        if msg and msg.strip() and not self._logging:
            self._logging = True
            try:
                self._original.write(msg)
            finally:
                self._logging = False

    def flush(self):
        self._original.flush()


def get_logger(name: str = "") -> "logging.Logger":  # noqa: F821
    return logging.getLogger("stock_prediction")


_setup_done = False


def setup_logging(mode: str = "premarket"):
    """配置日志：文件(按日期轮转) + 控制台（全局仅生效一次）"""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(root, "logs", f"{today}_{mode}.log")

    logger = logging.getLogger("stock_prediction")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    sys.stdout = _DualOutput(logger, sys.stdout)

    def _excepthook(exc_type, exc_val, exc_tb):
        logger.error("未捕获异常导致进程退出", exc_info=(exc_type, exc_val, exc_tb))
    sys.excepthook = _excepthook

    return logger
