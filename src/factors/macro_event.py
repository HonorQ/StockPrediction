"""
宏观事件检测器 — 检测重大宏观变化并映射到受影响行业

数据源（全部通过akshare获取）：
  - LPR利率变化（央行货币政策核心指标）
  - M2货币供应量变化（流动性指标）
  - PMI变化（经济景气度）
  - CPI变化（通胀指标）
"""
import pandas as pd
from typing import Dict, List


class MacroEventDetector:
    """宏观事件检测器"""

    def __init__(self):
        self.events = []  # 缓存检测到的事件

    def detect_all(self) -> List[Dict]:
        """检测所有宏观事件，返回事件列表"""
        events = []
        events.extend(self._detect_lpr_events())
        events.extend(self._detect_m2_events())
        events.extend(self._detect_pmi_events())
        events.extend(self._detect_cpi_events())
        self.events = events
        return events

    # ----------------------------------------------------------------
    # LPR利率检测
    # ----------------------------------------------------------------
    def _detect_lpr_events(self) -> List[Dict]:
        """检测LPR利率变化（降息/加息）"""
        events = []
        try:
            import akshare as ak
            df = ak.macro_china_lpr()
            if df is None or df.empty or len(df) < 2:
                return events

            df = df.sort_values("TRADE_DATE").reset_index(drop=True)

            # 取最近两期对比
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            lpr1y_now = float(latest.get("LPR1Y", 0) or 0)
            lpr1y_prev = float(prev.get("LPR1Y", 0) or 0)
            lpr5y_now = float(latest.get("LPR5Y", 0) or 0)
            lpr5y_prev = float(prev.get("LPR5Y", 0) or 0)

            # LPR1Y变化
            if lpr1y_now != lpr1y_prev:
                change = lpr1y_now - lpr1y_prev
                if change < 0:
                    events.append({
                        "type": "lpr_cut",
                        "severity": "high",
                        "direction": "positive",
                        "title": f"LPR1Y降息{abs(change):.0f}bp",
                        "detail": f"1年期LPR从{lpr1y_prev:.2f}%降至{lpr1y_now:.2f}%",
                        "date": str(latest["TRADE_DATE"]),
                        "affected_sectors": ["银行", "券商", "房地产", "保险"],
                        "impact_score": 85,
                    })
                else:
                    events.append({
                        "type": "lpr_hike",
                        "severity": "high",
                        "direction": "negative",
                        "title": f"LPR1Y加息{change:.0f}bp",
                        "detail": f"1年期LPR从{lpr1y_prev:.2f}%升至{lpr1y_now:.2f}%",
                        "date": str(latest["TRADE_DATE"]),
                        "affected_sectors": ["银行", "券商", "房地产", "保险"],
                        "impact_score": 80,
                    })

            # LPR5Y变化（房贷基准利率）
            if lpr5y_now != lpr5y_prev:
                change = lpr5y_now - lpr5y_prev
                if change < 0:
                    events.append({
                        "type": "lpr5y_cut",
                        "severity": "high",
                        "direction": "positive",
                        "title": f"LPR5Y降息{abs(change):.0f}bp",
                        "detail": f"5年期LPR从{lpr5y_prev:.2f}%降至{lpr5y_now:.2f}%",
                        "date": str(latest["TRADE_DATE"]),
                        "affected_sectors": ["房地产", "银行", "建材", "家电"],
                        "impact_score": 85,
                    })

        except Exception:
            pass

        return events

    # ----------------------------------------------------------------
    # M2货币供应检测
    # ----------------------------------------------------------------
    def _detect_m2_events(self) -> List[Dict]:
        """检测M2货币供应量异常变化（流动性过剩/紧缩）"""
        events = []
        try:
            import akshare as ak
            df = ak.macro_china_money_supply()
            if df is None or df.empty or len(df) < 3:
                return events

            # 取最近M2同比增长
            m2_growth_col = "货币和准货币(M2)-同比增长"
            recent = df[m2_growth_col].tail(3).values
            if len(recent) < 3:
                return events

            latest = float(recent[-1]) if pd.notna(recent[-1]) else 0
            avg_3m = sum(float(x) for x in recent if pd.notna(x)) / len(recent)

            # M2同比>12%（流动性过剩）
            if latest > 12:
                events.append({
                    "type": "m2_surge",
                    "severity": "medium",
                    "direction": "positive",
                    "title": f"M2同比增速{latest:.1f}%（流动性过剩）",
                    "detail": f"货币供应量同比增速达{latest:.1f}%，市场流动性充裕",
                    "date": str(df.iloc[-1].get("月份", "")),
                    "affected_sectors": ["券商", "有色金属", "房地产"],
                    "impact_score": 70,
                })
            # M2同比<8%（流动性紧缩）
            elif latest < 8:
                events.append({
                    "type": "m2_contraction",
                    "severity": "medium",
                    "direction": "negative",
                    "title": f"M2同比增速{latest:.1f}%（流动性紧缩）",
                    "detail": f"货币供应量同比增速仅{latest:.1f}%，市场流动性偏紧",
                    "date": str(df.iloc[-1].get("月份", "")),
                    "affected_sectors": ["银行", "券商", "房地产"],
                    "impact_score": 65,
                })

            # M2增速连续下降（收紧趋势）
            if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                events.append({
                    "type": "m2_downward_trend",
                    "severity": "low",
                    "direction": "negative",
                    "title": "M2增速连续下降（流动性收紧趋势）",
                    "detail": "最近3期M2增速持续下降",
                    "date": str(df.iloc[-1].get("月份", "")),
                    "affected_sectors": ["券商", "银行"],
                    "impact_score": 55,
                })

        except Exception:
            pass

        return events

    # ----------------------------------------------------------------
    # PMI景气度检测
    # ----------------------------------------------------------------
    def _detect_pmi_events(self) -> List[Dict]:
        """检测PMI荣枯线突破（经济景气度转折）"""
        events = []
        try:
            import akshare as ak
            df = ak.macro_china_pmi_yearly()
            if df is None or df.empty or len(df) < 2:
                return events

            # 取制造业PMI
            pmi_df = df[df["商品"].str.contains("制造业")].sort_values("日期").reset_index(drop=True)
            if len(pmi_df) < 2:
                return events

            latest_val = float(pmi_df.iloc[-1].get("今值", 0) or 0)
            prev_val = float(pmi_df.iloc[-2].get("今值", 0) or 0)
            forecast_val = float(pmi_df.iloc[-1].get("预测值", 0) or 0)

            # 突破荣枯线50
            if prev_val < 50 and latest_val >= 50:
                events.append({
                    "type": "pmi_break_above_50",
                    "severity": "high",
                    "direction": "positive",
                    "title": f"PMI回升至荣枯线上方{latest_val:.1f}",
                    "detail": f"制造业PMI从{prev_val:.1f}升至{latest_val:.1f}，突破50荣枯线",
                    "date": str(pmi_df.iloc[-1].get("日期", "")),
                    "affected_sectors": ["制造业", "有色金属", "化工", "机械"],
                    "impact_score": 80,
                })
            elif prev_val >= 50 and latest_val < 50:
                events.append({
                    "type": "pmi_break_below_50",
                    "severity": "high",
                    "direction": "negative",
                    "title": f"PMI跌破荣枯线{latest_val:.1f}",
                    "detail": f"制造业PMI从{prev_val:.1f}降至{latest_val:.1f}，跌破50荣枯线",
                    "date": str(pmi_df.iloc[-1].get("日期", "")),
                    "affected_sectors": ["制造业", "有色金属", "化工", "机械"],
                    "impact_score": 75,
                })

            # 超预期/低于预期
            if forecast_val > 0 and latest_val > 0:
                surprise = latest_val - forecast_val
                if surprise > 0.5:
                    events.append({
                        "type": "pmi_beat_expectation",
                        "severity": "medium",
                        "direction": "positive",
                        "title": f"PMI超预期{latest_val:.1f}(预期{forecast_val:.1f})",
                        "detail": "制造业PMI大幅超预期",
                        "date": str(pmi_df.iloc[-1].get("日期", "")),
                        "affected_sectors": ["制造业", "有色金属"],
                        "impact_score": 65,
                    })
                elif surprise < -0.5:
                    events.append({
                        "type": "pmi_miss_expectation",
                        "severity": "medium",
                        "direction": "negative",
                        "title": f"PMI低于预期{latest_val:.1f}(预期{forecast_val:.1f})",
                        "detail": "制造业PMI大幅低于预期",
                        "date": str(pmi_df.iloc[-1].get("日期", "")),
                        "affected_sectors": ["制造业", "有色金属"],
                        "impact_score": 60,
                    })

        except Exception:
            pass

        return events

    # ----------------------------------------------------------------
    # CPI通胀检测
    # ----------------------------------------------------------------
    def _detect_cpi_events(self) -> List[Dict]:
        """检测CPI通胀/通缩信号"""
        events = []
        try:
            import akshare as ak
            df = ak.macro_china_cpi_yearly()
            if df is None or df.empty or len(df) < 2:
                return events

            cpi_df = df.sort_values("日期").reset_index(drop=True)
            latest_val = float(cpi_df.iloc[-1].get("今值", 0) or 0)
            forecast_val = float(cpi_df.iloc[-1].get("预测值", 0) or 0)

            # 通缩信号（CPI同比<0）
            if latest_val < 0:
                events.append({
                    "type": "deflation_risk",
                    "severity": "high",
                    "direction": "negative",
                    "title": f"CPI同比{latest_val:.1f}%（通缩风险）",
                    "detail": "居民消费价格指数同比负增长，经济面临通缩压力",
                    "date": str(cpi_df.iloc[-1].get("日期", "")),
                    "affected_sectors": ["消费", "银行", "房地产"],
                    "impact_score": 70,
                })
            # 高通胀（CPI同比>3%）
            elif latest_val > 3:
                events.append({
                    "type": "high_inflation",
                    "severity": "high",
                    "direction": "negative",
                    "title": f"CPI同比{latest_val:.1f}%（通胀压力）",
                    "detail": "居民消费价格指数同比高企，通胀压力加大",
                    "date": str(cpi_df.iloc[-1].get("日期", "")),
                    "affected_sectors": ["消费", "银行", "有色金属"],
                    "impact_score": 70,
                })

        except Exception:
            pass

        return events

    # ----------------------------------------------------------------
    # 获取事件对个股的影响
    # ----------------------------------------------------------------
    def get_stock_impact(self, stock_code: str, stock_sector: str = "",
                         stock_concepts: List[str] = None) -> Dict:
        """
        计算宏观事件对单只股票的影响

        参数：
            stock_code: 股票代码
            stock_sector: 所属行业
            stock_concepts: 所属概念板块列表

        返回：
            {
                "impact_score": 0-100,
                "direction": "positive"/"negative"/"neutral",
                "events": [...相关事件...],
                "signals": [...描述信号...]
            }
        """
        if not self.events:
            self.detect_all()

        result = {
            "impact_score": 50,
            "direction": "neutral",
            "events": [],
            "signals": [],
        }

        for event in self.events:
            affected = event.get("affected_sectors", [])
            # 检查是否受影响
            matched = False
            if stock_sector and any(s in stock_sector for s in affected):
                matched = True
            if stock_concepts:
                if any(c in affected for c in stock_concepts):
                    matched = True

            if matched:
                result["events"].append(event)
                impact = event.get("impact_score", 50)
                direction = event.get("direction", "neutral")

                if direction == "positive":
                    result["impact_score"] += (impact - 50) * 0.3
                    result["signals"].append(f"{event['title']}（利好）")
                elif direction == "negative":
                    result["impact_score"] -= (impact - 50) * 0.3
                    result["signals"].append(f"{event['title']}（利空）")

        result["impact_score"] = max(0, min(100, result["impact_score"]))
        if result["impact_score"] > 55:
            result["direction"] = "positive"
        elif result["impact_score"] < 45:
            result["direction"] = "negative"

        return result


# ----------------------------------------------------------------
# 便捷函数：获取最近宏观事件摘要
# ----------------------------------------------------------------
def get_macro_summary() -> str:
    """获取最近宏观事件摘要（用于报告输出）"""
    detector = MacroEventDetector()
    events = detector.detect_all()
    if not events:
        return "无重大宏观事件"

    lines = []
    for e in events[:5]:
        icon = "📈" if e["direction"] == "positive" else "📉"
        lines.append(f"{icon} {e['title']}: {e['detail']}")
    return "\n".join(lines)
