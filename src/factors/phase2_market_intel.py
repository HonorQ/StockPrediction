"""
第二阶段：板块联动+资金流向持续+机构调研+龙虎榜
"""
from src.utils.log import get_logger
logger = get_logger()

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List


class SectorLinkage:
    """板块联动分析（龙头效应+概念联动）"""

    _em_push2_down = False
    _em_push2_down_at = 0
    _warned = False

    @classmethod
    def _is_push2_down(cls) -> bool:
        """半衰期熔断：失败后5分钟内跳过，之后自动重试"""
        if cls._em_push2_down and time.time() - cls._em_push2_down_at < 300:
            return True
        if cls._em_push2_down:
            cls._em_push2_down = False  # 5分钟后自动恢复
        return False

    def __init__(self):
        self.concept_map = {}
        self.sector_cache = {}

    def get_stock_concepts(self, stock_code: str) -> List[str]:
        """获取个股所属概念板块"""
        if SectorLinkage._is_push2_down():
            return []
        code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")
        try:
            import akshare as ak
            board = ak.stock_board_concept_cons_em(symbol=code)
            if board is not None and not board.empty:
                concepts = board.iloc[:, 0].tolist() if board.shape[1] > 0 else []
                return [str(c) for c in concepts[:10]]
        except Exception:
            if not SectorLinkage._warned:
                logger.info('      ⚠️ 概念板块API(push2)不可达, 暂停5分钟后自动重试')
                SectorLinkage._warned = True
            SectorLinkage._em_push2_down = True
            SectorLinkage._em_push2_down_at = time.time()
        return []

    def get_sector_strength(self, concept_name: str) -> Dict:
        """获取板块强度"""
        if SectorLinkage._is_push2_down():
            return {"change_pct": 0, "up_count": 0, "down_count": 0}
        try:
            import akshare as ak
            spot = ak.stock_board_concept_spot_em()
            if spot is not None and not spot.empty:
                match = spot[spot["板块名称"] == concept_name]
                if not match.empty:
                    row = match.iloc[0]
                    return {
                        "change_pct": float(row.get("涨跌幅", 0)) if pd.notna(row.get("涨跌幅")) else 0,
                        "up_count": int(row.get("上涨家数", 0)) if pd.notna(row.get("上涨家数")) else 0,
                        "down_count": int(row.get("下跌家数", 0)) if pd.notna(row.get("下跌家数")) else 0,
                        "lead_stock": str(row.get("领涨股", "")) if pd.notna(row.get("领涨股")) else "",
                    }
        except Exception:
            if not SectorLinkage._warned:
                logger.info('      ⚠️ 概念板块API(push2)不可达, 暂停5分钟后自动重试')
                SectorLinkage._warned = True
            SectorLinkage._em_push2_down = True
            SectorLinkage._em_push2_down_at = time.time()
        return {"change_pct": 0, "up_count": 0, "down_count": 0}

    def analyze_sector_linkage(
        self, stock_code: str, all_stocks_data: List[Dict] = None
    ) -> Dict:
        """分析板块联动效应"""
        result = {
            "sector_linkage_score": 50,
            "sector_signals": [],
            "concept_count": 0,
            "hot_concepts": [],
        }

        concepts = self.get_stock_concepts(stock_code)
        if not concepts:
            return result

        result["concept_count"] = len(concepts)
        hot_concepts = []

        for concept in concepts[:5]:
            strength = self.get_sector_strength(concept)
            if strength["change_pct"] > 3:
                hot_concepts.append(concept)
                result["sector_signals"].append(
                    f"{concept}板块涨{strength['change_pct']:.1f}%（强势）"
                )
                result["sector_linkage_score"] += 10
            elif strength["change_pct"] < -3:
                result["sector_signals"].append(
                    f"{concept}板块跌{strength['change_pct']:.1f}%（弱势）"
                )
                result["sector_linkage_score"] -= 10

            # 同板块个股联动
            if strength["up_count"] > strength["down_count"] * 3:
                result["sector_linkage_score"] += 5

        result["hot_concepts"] = hot_concepts[:3]
        result["sector_linkage_score"] = max(0, min(100, result["sector_linkage_score"]))

        return result


class MoneyFlowContinuity:
    """资金流向连续跟踪（3/5日持续性）"""

    def __init__(self, money_fetcher=None):
        self._money_fetcher = money_fetcher

    def analyze(self, stock_code: str) -> Dict:
        """分析资金流向持续性"""
        result = {
            "flow_continuity_score": 50,
            "flow_continuity": "neutral",
            "flow_signals": [],
        }

        try:
            fetcher = self._money_fetcher
            if fetcher is None:
                from src.data.money_flow import MoneyFlowFetcher
                fetcher = MoneyFlowFetcher()

            flow_data = fetcher.get_stock_fund_flow(stock_code)
            if flow_data is None or flow_data.empty:
                return result

            # 检查字段
            main_net = flow_data.get("主力净流入", None)
            if main_net is None:
                main_net = flow_data.get("主力净流入-净额", None)
            if main_net is None:
                return result

            main_net = pd.to_numeric(main_net, errors="coerce")

            if len(main_net) < 5:
                return result

            recent_5 = main_net.tail(5)
            recent_3 = main_net.tail(3)

            # 连续5日净流入
            if all(x > 0 for x in recent_5):
                total = sum(recent_5)
                result["flow_continuity"] = "strong_inflow"
                result["flow_signals"].append(
                    f"主力连续5日净流入{total/1e8:.1f}亿（持续吸筹）"
                )
                result["flow_continuity_score"] = 85

            # 连续3日净流入
            elif all(x > 0 for x in recent_3):
                total = sum(recent_3)
                result["flow_continuity"] = "inflow"
                result["flow_signals"].append(
                    f"主力连续3日净流入{total/1e8:.1f}亿"
                )
                result["flow_continuity_score"] = 70

            # 连续5日净流出
            elif all(x < 0 for x in recent_5):
                total = abs(sum(recent_5))
                result["flow_continuity"] = "strong_outflow"
                result["flow_signals"].append(
                    f"主力连续5日净流出{total/1e8:.1f}亿（持续出货）"
                )
                result["flow_continuity_score"] = 20

            # 连续3日净流出
            elif all(x < 0 for x in recent_3):
                total = abs(sum(recent_3))
                result["flow_continuity"] = "outflow"
                result["flow_signals"].append(
                    f"主力连续3日净流出{total/1e8:.1f}亿"
                )
                result["flow_continuity_score"] = 30

            # 由流出转流入（逆转信号）
            else:
                is_reversal = (
                    recent_5.iloc[0] < 0 and recent_5.iloc[0] < recent_5.iloc[-1]
                )
                if is_reversal:
                    result["flow_continuity"] = "reversal"
                    result["flow_signals"].append("主力资金由流出转流入（逆转信号）")
                    result["flow_continuity_score"] = 65

        except Exception as e:
            pass

        return result


class InstitutionalResearch:
    """机构调研因子（公募/私募调研频率）"""

    def __init__(self):
        pass

    def analyze(self, stock_code: str) -> Dict:
        """分析机构调研情况"""
        result = {
            "research_score": 50,
            "research_signals": [],
            "research_count": 0,
        }

        code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")

        # 机构推荐（获取全量后按代码过滤）
        try:
            import akshare as ak
            research = ak.stock_institute_recommend(symbol="投资评级选股")
            if research is not None and not research.empty:
                # 按股票代码过滤
                code_col = "股票代码" if "股票代码" in research.columns else None
                if code_col:
                    stock_research = research[research[code_col].astype(str) == code]
                    total_recommend = len(stock_research)
                    result["research_count"] = total_recommend

                    if total_recommend >= 10:
                        result["research_signals"].append(
                            f"近{total_recommend}家机构调研（高度关注）"
                        )
                        result["research_score"] = 80
                    elif total_recommend >= 5:
                        result["research_signals"].append(
                            f"近{total_recommend}家机构调研"
                        )
                        result["research_score"] = 65
                    elif total_recommend >= 2:
                        result["research_score"] = 55
            else:
                # 数据源返回空数据，跳过
                pass

        except Exception as e:
            pass

        # 机构持仓（获取全量后按代码过滤）
        try:
            import akshare as ak
            # 获取最近季度的机构持仓
            for q in ["20051", "20251", "20241"]:
                try:
                    report = ak.stock_institute_hold(symbol=q)
                    if report is not None and not report.empty:
                        code_col = "证券代码" if "证券代码" in report.columns else None
                        if code_col:
                            stock_hold = report[report[code_col].astype(str) == code]
                            if not stock_hold.empty:
                                hold_info = stock_hold.iloc[0]
                                inst_count = hold_info.get("机构数", 0) or 0
                                hold_ratio = hold_info.get("持股比例", 0) or 0
                                if inst_count > 0:
                                    result["research_signals"].append(
                                        f"{inst_count}家机构持有，持股比例{hold_ratio}%"
                                    )
                                    result["research_score"] = min(result["research_score"] + 10, 90)
                                break
                except Exception:
                    continue

        except Exception as e:
            pass

        result["research_score"] = max(0, min(100, result["research_score"]))
        return result


class DragonTigerBoard:
    """龙虎榜游资席位分析（区分机构/一线游资/二线游资）"""

    def __init__(self):
        # 一线游资：市场公认的顶级游资席位（赵老哥、章盟主、作手新一等）
        # 这些席位买入的持续性远高于普通游资，加分权重高
        self.top_tier_hot_money = {
            "国泰君安证券股份有限公司南京太平南路证券营业部": "赵老哥(南京太平南路)",
            "国泰君安证券股份有限公司上海江苏路证券营业部": "章盟主(上海江苏路)",
            "华泰证券股份有限公司深圳益田路荣超商务中心证券营业部": "华泰荣超",
            "中国银河证券股份有限公司北京阜成路证券营业部": "银河阜成路(一线)",
            "东方财富证券股份有限公司拉萨团结路第二证券营业部": "拉萨团结路(顶级)",
            "东方财富证券股份有限公司拉萨东环路第二证券营业部": "拉萨东环路(顶级)",
            "中信证券股份有限公司上海溧阳路证券营业部": "中信溧阳路(一线)",
            "国元证券股份有限公司上海中山北路证券营业部": "国元中山北路(一线)",
            "财通证券股份有限公司杭州体育场路证券营业部": "财通体育场路(一线)",
            "华鑫证券有限责任公司上海分公司": "华鑫上海(量化一线)",
        }
        # 二线游资：知名但持续性略弱
        self.second_tier_hot_money = {
            "国泰君安证券股份有限公司上海分公司": "国君上海",
            "中信证券股份有限公司上海分公司": "中信上海",
            "兴业证券股份有限公司陕西分公司": "兴业陕西",
            "华泰证券股份有限公司总部": "华泰总部",
            "平安证券股份有限公司深圳深南东路罗湖商务中心证券营业部": "平安罗湖",
            "招商证券股份有限公司深圳蛇口三马路证券营业部": "招商蛇口",
            "申万宏源证券有限公司深圳分公司": "申万深圳",
            "中天国富证券有限公司上海分公司": "中天国富上海",
        }

    def _classify_seat(self, seat_name: str) -> str:
        """分类席位：institution / top_tier / second_tier / unknown"""
        if not seat_name:
            return "unknown"
        s = str(seat_name)
        # 机构席位
        if "机构专用" in s:
            return "institution"
        # 一线游资
        for name in self.top_tier_hot_money:
            if name in s:
                return "top_tier"
        # 二线游资
        for name in self.second_tier_hot_money:
            if name in s:
                return "second_tier"
        return "unknown"

    # 类级缓存：龙虎榜数据 T+1 公布，盘前查当天必然为空
    _lhb_cache: Dict[str, pd.DataFrame] = {}
    _lhb_cache_date: str = None

    def _get_lhb_data(self) -> pd.DataFrame:
        """获取最近交易日的龙虎榜明细（带类级缓存）。

        龙虎榜数据 T+1 才公布，盘前查当天必然为空。
        尝试过去 5 天，命中第一个有数据的日期后缓存。
        """
        import akshare as ak
        from datetime import datetime, timedelta

        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._lhb_cache_date == today_str and self._lhb_cache:
            latest = max(self._lhb_cache.keys())
            return self._lhb_cache[latest]

        for days_ago in range(1, 6):
            d = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
            if d in self._lhb_cache:
                continue
            try:
                df = ak.stock_lhb_detail_em(start_date=d, end_date=d)
                if df is not None and not df.empty:
                    self._lhb_cache[d] = df
                    self._lhb_cache_date = today_str
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def analyze(self, stock_code: str) -> Dict:
        """分析龙虎榜游资行为。
        主源：stock_lhb_detail_em（datacenter-web 端点，未被阻断）
        备源：stock_lhb_detail_daily_sina（新浪）
        基于净买额/上榜原因评分。
        """
        result = {
            "lh_score": 50,
            "lh_signals": [],
            "hot_money_count": 0,
            "institution_count": 0,
            "top_tier_count": 0,
            "second_tier_count": 0,
            "institution_buy_net": 0.0,
        }

        code = stock_code.replace("sh", "").replace("sz", "").replace("bj", "")
        import akshare as ak

        def _score_from_row(net_buy, buy_amt, sell_amt, reason):
            nonlocal result
            result["institution_buy_net"] = net_buy
            if net_buy > 0:
                result["lh_signals"].append(
                    f"龙虎榜净买入 {net_buy/1e4:.0f}万（买入{buy_amt/1e4:.0f}万/卖出{sell_amt/1e4:.0f}万）"
                )
                if net_buy / max(buy_amt + sell_amt, 1) > 0.5:
                    result["lh_score"] += 20
                    result["lh_signals"].append("净买额占成交额过半，资金优势明显")
                elif net_buy > 1000e4:
                    result["lh_score"] += 15
                else:
                    result["lh_score"] += 8
            elif net_buy < 0:
                result["lh_signals"].append(
                    f"龙虎榜净卖出 {abs(net_buy)/1e4:.0f}万（买入{buy_amt/1e4:.0f}万/卖出{sell_amt/1e4:.0f}万）"
                )
                if abs(net_buy) / max(buy_amt + sell_amt, 1) > 0.5:
                    result["lh_score"] -= 20
                    result["lh_signals"].append("净卖额占成交额过半，出货明显")
                elif abs(net_buy) > 1000e4:
                    result["lh_score"] -= 15
                else:
                    result["lh_score"] -= 8
            if reason:
                result["lh_signals"].append(f"上榜原因: {reason}")
            if "机构" in reason and net_buy > 0:
                result["institution_count"] = 1
                result["lh_score"] += 10
                result["lh_signals"].append("机构上榜买入")
            result["hot_money_count"] = 1 if net_buy > 0 else 0

        # 主源：使用缓存的最近交易日龙虎榜数据
        try:
            df = self._get_lhb_data()
            if df is not None and not df.empty:
                match = df[df["代码"].astype(str) == code]
                if not match.empty:
                    row = match.iloc[0]
                    _score_from_row(
                        net_buy=float(row.get("龙虎榜净买额", 0) or 0),
                        buy_amt=float(row.get("龙虎榜买入额", 0) or 0),
                        sell_amt=float(row.get("龙虎榜卖出额", 0) or 0),
                        reason=str(row.get("上榜原因", "")),
                    )
                    result["lh_score"] = max(0, min(100, result["lh_score"]))
                    return result
        except Exception:
            pass

        # 备源：sina（也用最近交易日）
        try:
            from datetime import datetime, timedelta
            for days_ago in range(1, 6):
                d = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
                df_sina = ak.stock_lhb_detail_daily_sina(date=d)
                if df_sina is not None and not df_sina.empty:
                    match = df_sina[df_sina["股票代码"].astype(str) == code]
                    if not match.empty:
                        row = match.iloc[0]
                        change_pct = float(row.get("对应值", 0) or 0)
                        volume = float(row.get("成交量", 0) or 0)
                        amount = float(row.get("成交额", 0) or 0) * 1e4
                        indicator = str(row.get("指标", ""))
                        _score_from_row(
                            net_buy=amount * (change_pct / 100) if abs(change_pct) > 1 else 0,
                            buy_amt=amount if change_pct > 0 else 0,
                            sell_amt=amount if change_pct < 0 else 0,
                            reason=indicator,
                        )
                    break
        except Exception:
            pass

        result["lh_score"] = max(0, min(100, result["lh_score"]))
        return result


class SectorLeader:
    """板块龙头检测"""

    def __init__(self):
        pass

    def detect_leader(self, df: pd.DataFrame, stock_code: str) -> Dict:
        """检测是否是板块龙头"""
        result = {
            "is_leader": False,
            "leader_score": 50,
            "leader_signals": [],
        }

        if df is None or df.empty or len(df) < 10:
            return result

        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df.get("pct_chg", pd.Series())
        close = df["收盘"] if "收盘" in df.columns else df.get("close", pd.Series())

        if len(change_pct) < 5 or len(close) < 5:
            return result

        # 龙头特征1：近期涨幅领先
        recent_5d_return = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        recent_20d_return = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0

        # 龙头特征2：率先涨停/大幅上涨
        recent_3d = change_pct.tail(3)
        leader_up = sum(1 for x in recent_3d if x > 7)

        # 龙头特征3：量价配合好
        volume = df["成交量"] if "成交量" in df.columns else df.get("volume", pd.Series())
        volume_ratio = 1
        if len(volume) >= 10:
            avg_vol = volume.tail(10).mean()
            today_vol = volume.iloc[-1]
            volume_ratio = today_vol / avg_vol if avg_vol > 0 else 1

        # 综合判断
        if leader_up >= 2 and volume_ratio > 1.5:
            result["is_leader"] = True
            result["leader_signals"].append(f"3日内{leader_up}次大涨（龙头启动）")
            result["leader_score"] = 85
        elif recent_5d_return > 15 and volume_ratio > 2:
            result["is_leader"] = True
            result["leader_signals"].append(
                f"5日涨幅{recent_5d_return:.1f}%（强势龙头）"
            )
            result["leader_score"] = 80
        elif recent_20d_return > 30:
            result["is_leader"] = True
            result["leader_signals"].append(
                f"20日涨幅{recent_20d_return:.1f}%（趋势龙头）"
            )
            result["leader_score"] = 75
        elif recent_5d_return > 10:
            result["leader_signals"].append(
                f"5日涨幅{recent_5d_return:.1f}%（跟涨品种）"
            )
            result["leader_score"] = 60

        # 高标股检测（涨幅过高风险）
        if recent_20d_return > 100:
            result["leader_signals"].append(
                f"20日涨{recent_20d_return:.0f}%（高标股，注意风险）"
            )
            result["leader_score"] = 30

        return result
