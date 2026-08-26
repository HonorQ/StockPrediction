"""
策略健康度监控 —— 第三层风控（大盘择时之外）

依据：7窗口滚动验证(2026-01~08)显示体系在部分市场风格下持续负超额
(如08反弹窗高分组跑输大盘2-3pp)。regime 择时只能识别指数级风险，
识别不了"风格失效"——指数正常但选股逻辑被市场抛弃时(反热点风格)，
需要按自身推荐的真实收益自动降仓，让系统学会闭嘴。

规则（对最近 N 日已回填 d5 收益的推荐统计）：
  - 有效样本 < min_samples            → insufficient，不动作
  - 胜率与均值双双破位                → single   (每日推荐上限收紧到1)
  - 仅单项破位                        → halved   (上限减半)
  - 其余                              → ok
"""
from typing import Dict, Optional
from datetime import datetime

from src.utils.log import get_logger

logger = get_logger()

# 默认阈值：d5 口径。win_rate_floor 与 avg_d5_floor 取自7窗口基线的
# 中位水平(整体胜率约40%/均值约-0.5%)之下留缓冲，避免常态误触发
DEFAULTS = {
    "days": 15,           # 回溯自然日
    "min_samples": 8,     # 最少有效样本
    "win_rate_floor": 35.0,
    "avg_d5_floor": -1.5,
}


def assess_strategy_health(db, today: Optional[str] = None,
                           config: Dict = None) -> Dict:
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config or {})

    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    try:
        recs = db.get_recent_recommendations(
            days=cfg["days"],
            modes=("afternoon", "premarket", "feedback_training"))
    except Exception as e:
        logger.warning("健康度监控读取推荐失败: %s", e)
        return {"status": "error", "action": "normal"}

    with_d5 = [r for r in recs
               if r.get("d5_return") is not None
               and r.get("trade_date", "") <= today]
    samples = len(with_d5)
    result = {
        "samples": samples,
        "window_days": cfg["days"],
        "checked_at": today,
        "action": "normal",
        "status": "ok",
    }
    if samples < cfg["min_samples"]:
        result["status"] = "insufficient"
        return result

    wins = sum(1 for r in with_d5 if r["d5_return"] > 0)
    win_rate = wins / samples * 100
    avg_d5 = sum(r["d5_return"] for r in with_d5) / samples
    result.update({
        "win_rate": round(win_rate, 1),
        "avg_d5": round(avg_d5, 2),
    })

    below_wr = win_rate < cfg["win_rate_floor"]
    below_avg = avg_d5 < cfg["avg_d5_floor"]
    if below_wr and below_avg:
        result["action"] = "single"
        result["status"] = "degraded"
    elif below_wr or below_avg:
        result["action"] = "halved"
        result["status"] = "degraded"

    if result["status"] == "degraded":
        logger.warning(
            "策略健康度降级: 近%d条推荐 胜率%.0f%%(阈值%.0f%%) 平均d5%.2f%%(阈值%.2f%%)"
            " → 推荐上限动作=%s",
            samples, win_rate, cfg["win_rate_floor"], avg_d5,
            cfg["avg_d5_floor"], result["action"])
    return result


def apply_action(max_recommend: int, action: str) -> int:
    """按健康度动作收紧推荐上限"""
    if action == "single":
        return min(max_recommend, 1)
    if action == "halved":
        return max(1, max_recommend // 2)
    return max_recommend
