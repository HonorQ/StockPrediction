"""
盘中异动监控 - 闪崩/秒板/尾盘操控/巨量异动预警
"""
from src.utils.log import get_logger
logger = get_logger()

import time
import pandas as pd
from datetime import datetime
from typing import Dict, List
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


class RealtimeMonitor:
    """盘中异动监控器"""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.alerts = []
        self.monitored_stocks = []
        self.last_prices = {}

    def detect_anomalies(self, df: pd.DataFrame, stock_code: str, stock_name: str) -> List[Dict]:
        """
        检测个股异动
        返回异常事件列表
        """
        alerts = []

        if df is None or df.empty or len(df) < 2:
            return alerts

        today = df.iloc[-1]
        change_pct = today.get("涨跌幅", 0) if "涨跌幅" in df.columns else today.get("pct_chg", 0)
        volume = today.get("成交量", 0) if "成交量" in df.columns else today.get("volume", 0)
        amount = today.get("成交额", 0) if "成交额" in df.columns else today.get("amount", 0)
        high = today.get("最高", 0) if "最高" in df.columns else today.get("high", 0)
        low = today.get("最低", 0) if "最低" in df.columns else today.get("low", 0)
        close = today.get("收盘", 0) if "收盘" in df.columns else today.get("close", 0)
        open_price = today.get("开盘", 0) if "开盘" in df.columns else today.get("open", 0)
        pre_close = today.get("昨收", 0)

        # 检查必要数据
        if pre_close is None or pre_close <= 0 or close is None or close <= 0:
            return alerts

        severity = "low"

        # 1. 闪崩检测（跌幅>7%）
        if change_pct < -7:
            alerts.append({
                "type": "flash_crash",
                "severity": "high",
                "signal": f"{stock_name}({stock_code}) 闪崩 {change_pct:.1f}%！",
                "time": datetime.now().strftime("%H:%M:%S"),
            })

        # 2. 秒板检测
        limit_up_price = round(pre_close * 1.1, 2)
        limit_down_price = round(pre_close * 0.9, 2)
        if abs(close - limit_up_price) / limit_up_price < 0.005 and abs(open_price - limit_up_price) / limit_up_price < 0.005:
            alerts.append({
                "type": "instant_limit_up",
                "severity": "medium",
                "signal": f"{stock_name}({stock_code}) 开盘秒板！",
                "time": datetime.now().strftime("%H:%M:%S"),
            })
        elif abs(close - limit_down_price) / limit_down_price < 0.005 and abs(open_price - limit_down_price) / limit_down_price < 0.005:
            alerts.append({
                "type": "instant_limit_down",
                "severity": "high",
                "signal": f"{stock_name}({stock_code}) 开盘秒跌停！",
                "time": datetime.now().strftime("%H:%M:%S"),
            })

        # 3. 天地板检测
        if change_pct > 15 and open_price > 0 and low > 0:
            alerts.append({
                "type": "amplitude_anomaly",
                "severity": "high",
                "signal": f"{stock_name}({stock_code}) 振幅巨大（天地板）！",
                "time": datetime.now().strftime("%H:%M:%S"),
            })

        # 4. 巨量异动检测
        if len(df) >= 10:
            vol_series = df["成交量"] if "成交量" in df.columns else df.get("volume", pd.Series())
            avg_vol = vol_series.tail(10).mean()
            volume_ratio = volume / avg_vol if avg_vol > 0 else 1
            if volume_ratio > 10:
                alerts.append({
                    "type": "volume_surge",
                    "severity": "high",
                    "signal": f"{stock_name}({stock_code}) 巨量异动（量比{volume_ratio:.1f}）",
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
            elif volume_ratio > 5:
                alerts.append({
                    "type": "volume_surge",
                    "severity": "medium",
                    "signal": f"{stock_name}({stock_code}) 放量{volume_ratio:.1f}倍",
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

        # 5. 尾盘异动（如果当前是尾盘时段）
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        if current_hour == 14 and current_minute >= 30:
            if change_pct > 5:
                alerts.append({
                    "type": "tail_manipulation",
                    "severity": "medium",
                    "signal": f"{stock_name}({stock_code}) 尾盘拉升{change_pct:.1f}%",
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
            elif change_pct < -5:
                alerts.append({
                    "type": "tail_manipulation",
                    "severity": "high",
                    "signal": f"{stock_name}({stock_code}) 尾盘砸盘{change_pct:.1f}%",
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

        # 6. 连续大涨/大跌
        if len(df) >= 5:
            change_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
            if change_col in df.columns:
                recent_5d = df[change_col].tail(5)
                if all(x > 5 for x in recent_5d):
                    alerts.append({
                        "type": "consecutive_surge",
                        "severity": "medium",
                        "signal": f"{stock_name}({stock_code}) 5日连涨（游资接力）",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                elif all(x < -5 for x in recent_5d):
                    alerts.append({
                        "type": "consecutive_crash",
                        "severity": "high",
                        "signal": f"{stock_name}({stock_code}) 5日连跌（恐慌抛售）",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })

        # 去重
        seen_signals = set()
        unique_alerts = []
        for a in alerts:
            if a["signal"] not in seen_signals:
                seen_signals.add(a["signal"])
                unique_alerts.append(a)

        return unique_alerts

    def monitor_top_stocks(
        self, top_n: int = 50, duration: int = 60
    ) -> List[Dict]:
        """
        监控Top股票异动
        持续duration秒
        """
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()

        logger.info('\n🔍 开始监控异动...')
        logger.info(f'监控时长: {duration}秒 | 检查间隔: {self.check_interval}秒')
        start_time = time.time()
        all_alerts = []

        while time.time() - start_time < duration:
            try:
                # 获取实时行情
                df = fetcher.get_a_share_realtime()
                if df is not None and not df.empty:
                    # 按涨跌幅排序，取异动股
                    change_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
                    df = df.sort_values(change_col, ascending=False)

                    top_stocks = df.head(top_n)
                    for _, row in top_stocks.iterrows():
                        stock_code = str(row.get("股票代码", "")) or str(row.get("code", ""))
                        stock_name = str(row.get("名称", "")) or str(row.get("name", ""))

                        history = fetcher.get_stock_history(stock_code, days=10)
                        if history.empty:
                            continue

                        alerts = self.detect_anomalies(history, stock_code, stock_name)
                        for a in alerts:
                            if a["signal"] not in [x["signal"] for x in all_alerts]:
                                logger.info(f"  🚨 [{a['severity'].upper()}] {a['signal']}")
                                all_alerts.append(a)

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info('\n监控已停止')
                break
            except Exception as e:
                logger.info(f'监控异常: {e}')
                time.sleep(self.check_interval)

        return all_alerts

    def run_once(self, stock_codes: List[str] = None) -> List[Dict]:
        """单次扫描，返回所有异动"""
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()

        all_alerts = []
        stocks_to_check = stock_codes or []

        if not stocks_to_check:
            # 获取全市场涨幅前50
            df = fetcher.get_all_a_shares()
            if df is not None and not df.empty:
                change_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
                df = df.sort_values(change_col, ascending=False)
                stocks_to_check = (df.head(50)["股票代码"].astype(str).tolist()
                                   if "股票代码" in df.columns
                                   else df.head(50)["code"].astype(str).tolist())

        for code in stocks_to_check:
            try:
                history = fetcher.get_stock_history(code, days=10)
                if history.empty:
                    continue
                name = str(history.iloc[-1].get("名称", code))
                alerts = self.detect_anomalies(history, code, name)
                all_alerts.extend(alerts)
            except Exception:
                continue

        return all_alerts


def main():
    """独立运行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="A股异动监控")
    parser.add_argument("--duration", type=int, default=60, help="监控时长（秒）")
    parser.add_argument("--top_n", type=int, default=50, help="监控股票数")
    parser.add_argument("--once", action="store_true", help="单次扫描")
    args = parser.parse_args()

    monitor = RealtimeMonitor(check_interval=30)

    if args.once:
        alerts = monitor.run_once()
        if alerts:
            logger.info(f'\n共发现 {len(alerts)} 个异动信号:')
            for a in alerts:
                logger.info(f"  [{a['severity'].upper()}] {a['signal']}")
        else:
            logger.info('暂无异动信号')
            monitor.monitor_top_stocks(top_n=args.top_n, duration=args.duration)


if __name__ == "__main__":
    main()
