"""
每日运行入口 - 三模式
  --premarket  盘前模式下重计算 (建议7:00-8:00跑)
  --intraday   盘中模式快速修正 (建议9:25集合竞价后跑)
  --afternoon  午后模式最终推荐 (建议14:40跑, 尾盘决策版)

无参数时默认跑 --premarket

规则(AGENTS.md): 本入口只做 参数解析/配置校验/心跳/模式分发;
业务实现见 src/pipeline/ 与各分层模块。
"""
import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# 绕过系统代理：项目数据源均为国内站点，强制全部直连。
# 不设此项时，Windows 残留的系统代理(如 Clash 关闭后仍指向 127.0.0.1:7897)
# 会让 requests 把请求发给不存在的代理导致大面积超时。
_NO_PROXY = "*"
os.environ["NO_PROXY"] = _NO_PROXY
os.environ["no_proxy"] = _NO_PROXY
# requests 会从环境变量读代理，Windows 系统代理(注册表)也会被 urllib3 检测到
# 必须显式清除，否则东财等走代理的接口会 ProxyError 挂死
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

# 全局网络超时：防止数据源故障时请求无限挂起导致每日任务卡死
# （2026-08-10 案例：东财 push2 连不上，industry_map 重建请求无超时挂起，盘前任务卡住）
import socket as _socket  # noqa: E402
_socket.setdefaulttimeout(15)

from src.utils.log import get_logger  # noqa: E402
logger = get_logger()

from src.engine.scorer import StockScorer  # noqa: E402  (权重校验用)
from src.utils.heartbeat import (  # noqa: E402,F401
    HEARTBEAT_PATH, HEARTBEAT_MAX_GAP_DAYS,
    _write_heartbeat, check_task_alerts,
)
from src.pipeline.common import (  # noqa: E402,F401
    setup_logging, write_summary, save_cache, load_cache, load_config,
)
from src.pipeline.premarket import run_premarket  # noqa: E402,F401
from src.pipeline.intraday import run_intraday  # noqa: E402,F401
from src.pipeline.afternoon import run_afternoon  # noqa: E402,F401
from src.pipeline.factor_ic import run_factor_ic  # noqa: E402,F401

# 兼容 re-export: 旧引用经此处转发到分层模块(业务实现见各模块)
from src.engine.paper_trade_ops import (  # noqa: E402,F401
    _fetch_close_price_on_date, _update_open_paper_trades,
    _check_crash_and_close_positions, _open_paper_trades_for_recs,
)
from src.engine.entry_risk_filter import (  # noqa: E402,F401
    _filter_high_risk_entries, _refresh_intraday_gap_signal,
)
from src.data.auction_snapshot import _save_auction_snapshot  # noqa: E402,F401


def _validate_weights_consistency(config) -> None:
    """启动时校验 config.yaml 的 weights 与评分器维度一致。

    权重有两处来源(config.yaml 与 scorer.py 默认)，靠此校验强制对齐：
      0) 总和必须≈1(评分尺度漂移会让阈值实际含义失真)
      1) config 必须覆盖全部子评分维度(缺失=该维度被静默丢弃)
      2) 不应存在子评分中没有的维度(恒50废权重)
    失败直接退出，避免带病运行。
    """
    cfg_weights = config.get("weights") or {}
    scorer = StockScorer()
    sub_keys = set(scorer.get_sub_scores({}).keys())
    default_keys = set(scorer.weights.keys())
    cfg_keys = set(cfg_weights.keys())

    problems = []
    total = sum(float(v) for v in cfg_weights.values())
    if abs(total - 1.0) > 0.02:
        problems.append(f"config.weights 总和={total:.4f}，偏离 1.0 超过容差(±0.02)")
    missing = sub_keys - cfg_keys
    if missing:
        problems.append(f"config.weights 缺失以下子评分维度的权重: {sorted(missing)}")
    extra = cfg_keys - sub_keys
    if extra:
        problems.append(f"config.weights 包含子评分中不存在的冗余维度: {sorted(extra)}")
    if default_keys != sub_keys:
        _miss = sorted(sub_keys - default_keys)
        _ext = sorted(default_keys - sub_keys)
        problems.append(f"scorer 默认权重与子评分维度不一致: 缺 {_miss} / 多 {_ext}")

    if problems:
        for p in problems:
            logger.error("权重一致性校验失败 - %s", p)
        raise SystemExit(
            "权重配置 (config.yaml weights) 与评分器维度不一致，请先对齐再运行。"
        )


def _is_trading_day(d):
    """周末跳过; 法定节假日由数据源自带的停牌过滤兜底。"""
    return d.weekday() < 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票预测系统")
    parser.add_argument("--premarket", action="store_true", help="盘前模式：全量因子计算（建议7:00-8:00运行）")
    parser.add_argument("--intraday", action="store_true", help="盘中模式：基于盘前缓存快速修正（建议9:25运行）")
    parser.add_argument("--afternoon", action="store_true", help="午后模式：基于全天真实成交/资金流向深度修正（建议14:40运行，尾盘最终决策）")
    parser.add_argument("--factor-ic", action="store_true", help="因子IC评估：从数据库加载历史数据计算因子IC")
    parser.add_argument("--lookback", type=int, default=60, help="因子IC回溯天数")
    args = parser.parse_args()

    if args.factor_ic:
        mode = "factor_ic"
    elif args.intraday:
        mode = "intraday"
    elif args.afternoon:
        mode = "afternoon"
    else:
        mode = "premarket"

    logger = setup_logging(mode)
    logger.info(f"启动模式: {mode}")

    # 心跳与失败告警：计划任务静默失败时可在下次运行/报告中感知
    _write_heartbeat(mode, status="running")

    today = datetime.now().date()
    if not _is_trading_day(today):
        logger.info(f"今日({today})非A股交易日（周末或法定节假日），跳过运行")
        sys.exit(0)

    try:
        config = load_config()
        _validate_weights_consistency(config)

        if args.factor_ic:
            config.setdefault("factor_ic", {})["lookback_days"] = args.lookback
            run_factor_ic(config)
        elif args.intraday:
            completed = run_intraday(config)
        elif args.afternoon:
            completed = run_afternoon(config)
        else:
            # 默认走盘前模式
            completed = run_premarket(config)
        # 函数显式返回 False = 前置条件缺失而空跑(如盘前缓存未生成)。
        # 这种"静默未执行"必须落成 failed, 否则心跳全绿而当天实际无产出。
        if completed is False:
            _write_heartbeat(mode, status="failed", error="前置盘前缓存缺失, 任务未执行")
        else:
            _write_heartbeat(mode, status="ok")
    except SystemExit:
        raise
    except Exception as e:
        _write_heartbeat(mode, status="failed", error=f"{type(e).__name__}: {e}")
        raise
