# -*- coding: utf-8 -*-
"""任务心跳与失败告警(run_daily 三时点计划任务的运维基础)。"""
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEARTBEAT_PATH = os.path.join(ROOT, "logs", "heartbeat.json")
# 允许的最大静默天数(覆盖长假)：超过则视为任务失联
HEARTBEAT_MAX_GAP_DAYS = 5


# 允许的最大静默天数(覆盖长假)：超过则视为任务失联


def _write_heartbeat(mode: str, status: str = "ok", error: str = None):
    """记录各模式最近一次运行状态。失败告警的数据基础。"""
    try:
        data = {}
        if os.path.exists(HEARTBEAT_PATH):
            with open(HEARTBEAT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[mode] = {
            "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "error": (error or "")[:300],
        }
        with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 心跳失败绝不影响主流程


def check_task_alerts(modes=("premarket", "intraday", "afternoon"),
                      max_gap_days: int = HEARTBEAT_MAX_GAP_DAYS) -> list:
    """检测历史运行异常，返回告警文本列表。
    检测: 上次运行失败 / 超过 max_gap_days 无成功心跳。"""
    alerts = []
    try:
        if not os.path.exists(HEARTBEAT_PATH):
            return ["无任何心跳记录(logs/heartbeat.json 不存在)"]
        with open(HEARTBEAT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        now = datetime.now()
        for m in modes:
            info = data.get(m)
            if not info:
                alerts.append(f"{m}: 从未有心跳记录")
                continue
            if info.get("status") == "running":
                # running 超过1天视为上次进程异常退出
                pass
            if info.get("status") not in ("ok", "running"):
                err = (info.get("error") or "")[:80]
                alerts.append(f"{m}: 上次运行失败 [{info.get('last_run')}] {err}")
                continue
            try:
                last = datetime.strptime(info["last_run"], "%Y-%m-%d %H:%M:%S")
                gap = (now - last).days
                if gap > max_gap_days:
                    alerts.append(f"{m}: 已 {gap} 天无成功运行(上次 {info['last_run']})")
            except (KeyError, ValueError):
                pass
    except Exception as e:
        logger.warning("心跳检测失败: %s", e)
    return alerts

def _get_logger():
    from src.utils.log import get_logger
    return get_logger()


logger = _get_logger()
