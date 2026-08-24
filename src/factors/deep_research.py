"""
深度研究因子
基于历史大涨/大跌模式匹配 + 催化剂对比 + 波动率/位置分析
预测当前股票与历史模式的相似度，给出预测置信度
"""
import pandas as pd
from typing import Dict, List


class DeepResearchFactor:
    """深度研究因子：历史模式匹配 + 催化剂对比"""

    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, news_list: List[Dict] = None,
                stock_code: str = None, stock_data: Dict = None) -> Dict:
        """
        分析历史模式与当前局势
        df: 个股历史数据（至少60天）
        news_list: 当日新闻列表
        stock_code: 股票代码
        stock_data: 实时行情数据
        """
        result = {
            "deep_research_score": 50,
            "deep_research_signals": [],
            "volatility_regime": "normal",
            "position_at": "mid",
            "catalyst_match_score": 50,
            "big_move_count": 0,
            "prediction_confidence": 50,
            "historical_reference": "",
            "risk_reference": "",
        }

        if df is None or df.empty or len(df) < 30:
            return result

        close = df["收盘"] if "收盘" in df.columns else df["close"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]

        score = 50
        signals = []
        risk_factors = []

        # ============================================================
        # 1. 波动率模式分析
        # ============================================================
        vol_result = self._analyze_volatility_regime(close, change_pct)
        result["volatility_regime"] = vol_result["regime"]
        score += vol_result["score_delta"]
        signals.extend(vol_result["signals"])

        # ============================================================
        # 2. 历史位置分析
        # ============================================================
        pos_result = self._analyze_position(close)
        result["position_at"] = pos_result["position"]
        score += pos_result["score_delta"]
        signals.extend(pos_result["signals"])
        risk_factors.extend(pos_result["risk_factors"])

        # ============================================================
        # 3. 历史大涨/大跌事件检测
        # ============================================================
        big_moves = self._detect_big_moves(close, change_pct, volume)
        result["big_move_count"] = len(big_moves)
        move_score = self._score_big_moves(big_moves, close)
        score += move_score["score_delta"]
        signals.extend(move_score["signals"])
        risk_factors.extend(move_score["risk_factors"])

        # ============================================================
        # 4. 催化剂匹配
        # ============================================================
        catalyst = self._match_catalysts(big_moves, news_list, stock_code)
        result["catalyst_match_score"] = catalyst["match_score"]
        score += catalyst["score_delta"]
        signals.extend(catalyst["signals"])
        risk_factors.extend(catalyst["risk_factors"])

        # ============================================================
        # 5. 模式相似度 + 预测
        # ============================================================
        prediction = self._predict_pattern(
            vol_result, pos_result, big_moves, catalyst, close, change_pct
        )
        result["prediction_confidence"] = prediction["confidence"]
        score += prediction["score_delta"]
        signals.extend(prediction["signals"])
        risk_factors.extend(prediction["risk_factors"])

        # 历史参考/风险参考
        if big_moves:
            last_up = next((m for m in reversed(big_moves) if m["type"] == "up"), None)
            last_down = next((m for m in reversed(big_moves) if m["type"] == "down"), None)
            if last_up:
                result["historical_reference"] = (
                    f"{last_up['date']}: {last_up['magnitude']:+.1f}%({last_up['days']}日), "
                    f"驱动: {last_up.get('trigger', '未知')}"
                )
            if last_down:
                result["risk_reference"] = (
                    f"{last_down['date']}: {last_down['magnitude']:+.1f}%({last_down['days']}日), "
                    f"触发: {last_down.get('trigger', '未知')}"
                )

        result["deep_research_score"] = max(0, min(100, score))
        result["deep_research_signals"] = signals + [f"⚠️ {r}" for r in risk_factors]
        return result

    def _analyze_volatility_regime(self, close: pd.Series, change_pct: pd.Series) -> Dict:
        """波动率模式分析"""
        result = {"regime": "normal", "percentile": 50, "score_delta": 0, "signals": []}

        if len(close) < 60:
            return result

        vol_30 = change_pct.tail(30).std()
        vol_60 = change_pct.tail(60).std()

        # 对比历史分位（用滚动窗口估算）
        all_vol = []
        for i in range(20, len(change_pct)):
            all_vol.append(change_pct.iloc[i-20:i].std())
        all_vol = sorted([v for v in all_vol if v > 0])

        if not all_vol:
            return result

        # 计算当前30日波动率在历史中的分位
        rank = sum(1 for v in all_vol if v <= vol_30)
        percentile = rank / len(all_vol) * 100
        result["percentile"] = round(percentile, 1)

        if percentile >= 95:
            result["regime"] = "extreme_high"
            result["score_delta"] = -10
            result["signals"].append(f"波动率极高({percentile:.0f}%分位)，风险加大")
        elif percentile >= 75:
            result["regime"] = "high"
            result["score_delta"] = -5
            result["signals"].append(f"波动率偏高({percentile:.0f}%分位)")
        elif percentile <= 25:
            result["regime"] = "low"
            result["score_delta"] = 5
            result["signals"].append(f"波动率低({percentile:.0f}%分位)，相对安全")
        else:
            result["regime"] = "normal"
            result["score_delta"] = 2

        return result

    def _analyze_position(self, close: pd.Series) -> Dict:
        """历史位置分析"""
        result = {"position": "mid", "score_delta": 0, "signals": [], "risk_factors": []}

        if len(close) < 60:
            return result

        current = close.iloc[-1]
        high_60 = close.tail(60).max()
        low_60 = close.tail(60).min()
        high_120 = close.tail(min(120, len(close))).max()
        low_120 = close.tail(min(120, len(close))).min()

        # 在60日区间中的位置
        if high_60 != low_60:
            pos_60 = (current - low_60) / (high_60 - low_60) * 100
        else:
            pos_60 = 50

        # 距高点回撤
        drawdown_from_high = (current / high_120 - 1) * 100
        up_from_low = (current / low_120 - 1) * 100

        if pos_60 < 25:
            result["position"] = "bottom"
            result["score_delta"] = 10
            result["signals"].append(f"位于60日区间底部({pos_60:.0f}%)，距高点{drawdown_from_high:.1f}%")
        elif pos_60 > 75:
            result["position"] = "top"
            result["score_delta"] = -8
            result["risk_factors"].append(f"位于60日区间顶部({pos_60:.0f}%)，距低点+{up_from_low:.1f}%")
        else:
            result["position"] = "mid"
            result["score_delta"] = 2
            result["signals"].append(f"位于60日区间中部({pos_60:.0f}%)")

        # 距120日高点回撤超过20%的特殊标记
        if drawdown_from_high < -20:
            result["score_delta"] += 5
            result["signals"].append(f"距120日高点回撤{drawdown_from_high:.1f}%（深度调整）")

        return result

    def _detect_big_moves(self, close: pd.Series, change_pct: pd.Series,
                          volume: pd.Series) -> List[Dict]:
        """检测历史大涨/大跌事件"""
        big_moves = []

        if len(close) < 30:
            return big_moves

        # 滚动窗口检测5日涨跌>10%的事件
        for i in range(10, len(close) - 5):
            # 5日累计涨跌
            ret_5d = (close.iloc[i] / close.iloc[i-5] - 1) * 100
            # 10日累计涨跌
            ret_10d = (close.iloc[i] / close.iloc[i-10] - 1) * 100 if i >= 10 else 0

            if abs(ret_5d) >= 10:
                event_type = "up" if ret_5d > 0 else "down"
                # 后续5日走势
                future_5d = (close.iloc[i+5] / close.iloc[i] - 1) * 100

                # 触发前的技术特征
                vol_before = change_pct.iloc[max(0,i-10):i].std()
                vol_ratio = volume.iloc[i] / volume.iloc[max(0,i-5):i].mean() if volume.iloc[max(0,i-5):i].mean() > 0 else 1

                # 获取日期（如果有）
                date_val = ""
                if hasattr(close.index[i], 'strftime'):
                    date_val = close.index[i].strftime('%Y-%m-%d')
                elif '日期' in close.index:
                    date_val = str(close.index[i])
                else:
                    date_val = f"第{i}日"

                big_moves.append({
                    "date": date_val,
                    "type": event_type,
                    "magnitude": round(ret_5d, 1),
                    "days": 5,
                    "ret_10d": round(ret_10d, 1),
                    "future_5d": round(future_5d, 1),
                    "vol_before": round(vol_before, 4),
                    "vol_ratio": round(vol_ratio, 1),
                    "trigger": self._infer_trigger(event_type, ret_5d),
                })

        # 去重：相邻事件只保留幅度最大的
        deduped = []
        for m in big_moves:
            if not deduped or abs(m["magnitude"]) > abs(deduped[-1]["magnitude"]):
                deduped.append(m)
            elif m["date"] != deduped[-1]["date"]:
                deduped.append(m)

        # 最多保留最近5个
        return deduped[-5:]

    def _infer_trigger(self, event_type: str, magnitude: float) -> str:
        """根据涨幅和类型推断触发因素"""
        if event_type == "up":
            if magnitude > 20:
                return "重大利好/行业爆发"
            elif magnitude > 15:
                return "业绩超预期/政策利好"
            else:
                return "板块联动/资金推动"
        else:
            if magnitude < -20:
                return "重大利空/黑天鹅"
            elif magnitude < -15:
                return "业绩暴雷/宏观利空"
            else:
                return "获利回吐/板块调整"

    def _score_big_moves(self, big_moves: List[Dict], close: pd.Series) -> Dict:
        """对历史大涨/大跌事件评分"""
        result = {"score_delta": 0, "signals": [], "risk_factors": []}

        if not big_moves:
            return result

        # 统计大涨vs大跌
        up_moves = [m for m in big_moves if m["type"] == "up"]
        down_moves = [m for m in big_moves if m["type"] == "down"]

        # 大涨历史 = 正面信号（说明该股有爆发力）
        if up_moves:
            best_up = max(up_moves, key=lambda m: m["magnitude"])
            result["score_delta"] += min(len(up_moves) * 5, 15)
            result["signals"].append(
                f"历史有{len(up_moves)}次大涨(最大{best_up['magnitude']:+.1f}%)，具备爆发力"
            )

        # 大跌历史 = 风险提醒
        if down_moves:
            worst_down = min(down_moves, key=lambda m: m["magnitude"])
            result["score_delta"] -= min(len(down_moves) * 3, 10)
            result["risk_factors"].append(
                f"历史有{len(down_moves)}次大跌(最大{worst_down['magnitude']:+.1f}%)，注意回调风险"
            )

        # 最近一次事件的后续表现
        last_move = big_moves[-1]
        if last_move["future_5d"] > 5:
            result["score_delta"] += 5
            result["signals"].append(f"最近大涨后5日继续上涨{last_move['future_5d']:+.1f}%")
        elif last_move["future_5d"] < -5:
            result["score_delta"] -= 5
            result["risk_factors"].append(f"最近大跌后5日继续下跌{last_move['future_5d']:+.1f}%")

        return result

    def _match_catalysts(self, big_moves: List[Dict], news_list: List[Dict],
                         stock_code: str = None) -> Dict:
        """催化剂匹配：对比当前新闻与历史大涨时的触发因素"""
        result = {"match_score": 50, "score_delta": 0, "signals": [], "risk_factors": []}

        if not news_list:
            return result

        # 当前新闻关键词
        current_keywords = set()
        for news in news_list:
            title = news.get("title", "") + news.get("content", "")
            # 提取有意义的关键词（简单分词）
            for kw in ["铜", "铝", "锂", "钢", "煤", "油", "黄金", "芯片", "AI",
                       "新能源", "光伏", "军工", "医药", "消费", "科技", "央行",
                       "降息", "降准", "LPR", "增持", "回购", "业绩", "重组",
                       "库存", "产量", "出口", "进口", "关税", "制裁", "冲突",
                       "涨停", "跌停", "利好", "利空", "超预期", "暴雷"]:
                if kw in title:
                    current_keywords.add(kw)

        # 从历史事件中提取触发关键词
        historical_keywords = set()
        for m in big_moves:
            trigger = m.get("trigger", "")
            for kw in ["利好", "利空", "业绩", "政策", "板块", "资金", "黑天鹅",
                       "宏观", "行业", "爆发", "预期", "调整", "回吐"]:
                if kw in trigger:
                    historical_keywords.add(kw)

        # 也从股票所属行业提取
        industry_kws = {"铜": ["铜", "铜矿", "电解铜"], "铝": ["铝", "电解铝", "铝箔"],
                       "锂": ["锂", "碳酸锂"], "芯片": ["芯片", "半导体"]}
        if stock_code:
            for industry, kws in industry_kws.items():
                for kw in kws:
                    for news in news_list:
                        if kw in news.get("title", "") + news.get("content", ""):
                            current_keywords.add(industry)

        # 匹配度计算
        if current_keywords and historical_keywords:
            overlap = current_keywords & historical_keywords
            union = current_keywords | historical_keywords
            match_ratio = len(overlap) / len(union) * 100 if union else 0
            result["match_score"] = round(match_ratio, 1)

            if match_ratio > 30:
                result["score_delta"] = 10
                result["signals"].append(
                    f"催化剂匹配度高({match_ratio:.0f}%): {', '.join(list(overlap)[:3])}"
                )
            elif match_ratio > 15:
                result["score_delta"] = 5
                result["signals"].append(f"催化剂部分匹配({match_ratio:.0f}%)")
            else:
                result["score_delta"] = -3
                result["risk_factors"].append(f"催化剂匹配度低({match_ratio:.0f}%)")
        else:
            result["match_score"] = 30
            result["score_delta"] = -5
            result["risk_factors"].append("无法匹配历史催化剂")

        return result

    def _predict_pattern(self, vol_result: Dict, pos_result: Dict,
                         big_moves: List[Dict], catalyst: Dict,
                         close: pd.Series, change_pct: pd.Series) -> Dict:
        """基于历史模式预测当前走势"""
        result = {"confidence": 50, "score_delta": 0, "signals": [], "risk_factors": []}

        if not big_moves:
            result["signals"].append("无历史大涨模式可参考")
            return result

        # 综合评估
        confidence_factors = []

        # 1. 波动率是否匹配历史大涨前的波动率
        if vol_result["regime"] in ["low", "normal"]:
            confidence_factors.append(("波动率环境适宜", +10))
        elif vol_result["regime"] == "high":
            confidence_factors.append(("波动率偏高，类似大涨前状态", +5))
        else:
            confidence_factors.append(("波动率极高，类似大跌前状态", -10))

        # 2. 价格位置是否有利
        if pos_result["position"] == "bottom":
            confidence_factors.append(("处于底部区域，向上空间大", +12))
        elif pos_result["position"] == "mid":
            confidence_factors.append(("处于中部区域", +3))
        else:
            confidence_factors.append(("处于顶部区域，回调风险大", -10))

        # 3. 催化剂匹配度
        match_score = catalyst["match_score"]
        if match_score > 30:
            confidence_factors.append(("催化剂与历史大涨时相似", +10))
        elif match_score > 15:
            confidence_factors.append(("催化剂部分匹配", +5))
        else:
            confidence_factors.append(("催化剂不匹配", -5))

        # 4. 最近事件的后续走势
        last_move = big_moves[-1]
        if last_move["type"] == "up" and last_move["future_5d"] > 0:
            confidence_factors.append(("上次大涨后延续上涨", +8))
        elif last_move["type"] == "down" and last_move["future_5d"] < 0:
            confidence_factors.append(("上次大跌后延续下跌", -8))

        # 计算置信度
        base_confidence = 50
        for _, delta in confidence_factors:
            base_confidence += delta
        result["confidence"] = max(0, min(100, base_confidence))

        # 信号
        for factor_text, delta in confidence_factors:
            if delta > 0:
                result["signals"].append(factor_text)
            else:
                result["risk_factors"].append(factor_text)

        # 预测分
        if result["confidence"] >= 65:
            result["score_delta"] = 10
            result["signals"].append(f"预测置信度{result['confidence']}%，偏乐观")
        elif result["confidence"] <= 35:
            result["score_delta"] = -10
            result["risk_factors"].append(f"预测置信度{result['confidence']}%，偏悲观")
        else:
            result["score_delta"] = 0
            result["signals"].append(f"预测置信度{result['confidence']}%，中性")

        return result
