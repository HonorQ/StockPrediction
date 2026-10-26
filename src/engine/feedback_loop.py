"""
反馈循环编排器 — 每日自动归因 + IC衰减监控 + 自适应权重调整 + ML重训练

执行流程:
1. 加载历史推荐，对有T+N收益数据的做因子归因
2. 统计因子命中率/失败率，定位最弱的因子
3. 加载IC历史，检测IC衰减
4. 自动调整衰退因子的权重（全局 factor-level）
5. 调整每只股票的 per-stock 因子权重（Step 3 新增）
6. 定期 (每7天) 重训练ML模型

支持两种调用模式：
- 生产模式：today=None，使用 datetime.now()，写入 weight_history / model_versions
- 回测模式：today="YYYY-MM-DD"，所有时间引用使用 today，避免未来函数
"""
import os
import json
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from src.data.database import StockDatabase, SUB_SCORE_DIMS
from src.engine.factor_attribution import FactorAttribution
from src.engine.ml_selector import MLStockSelector, FactorOptimizer
from src.engine.scorer import StockScorer


RETRAIN_INTERVAL_DAYS = 7       # 每7天重训练一次
MIN_ATTRIBUTION_SAMPLES = 5     # 最少归因样本数才调权（低于此阈值统计噪声过大）
FAILURE_WEIGHT_PENALTY = 0.5    # 失败率>30%的因子权重乘数
MAX_WEIGHT_CHANGE = 0.05        # 单次最大权重调整幅度


class FeedbackLoop:
    def __init__(self, config: Dict = None,
                 stock_weight_manager=None):
        """Args:
            config: 配置字典（含 weights 等）
            stock_weight_manager: 可选的 StockWeightManager 实例。
                传入时启用 per-stock 权重调整；不传则只跑全局 factor-level 调权。
        """
        self.db = StockDatabase()
        self.config = config or {}
        self.stock_weight_manager = stock_weight_manager

    def run_all(self, today: str = None, skip_ml: bool = False,
                skip_weight_adjustment: bool = False) -> Dict:
        """运行完整反馈循环。
        Args:
            today: 当前回测日（YYYY-MM-DD）。None 时使用真实当前日期（生产模式）。
            skip_ml: True 时跳过 ML 重训练（回测通常跳过，避免污染生产模型）。
            skip_weight_adjustment: True 时跳过全局权重调整（回测时跳过，避免慢速API调用）。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        result = {
            "date": today,
            "attribution_done": False,
            "ic_analysis_done": False,
            "weight_adjusted": False,
            "stock_weight_adjusted": False,
            "model_retrained": False,
            "messages": [],
        }
        market_fetcher = None
        try:
            from src.data.market_data import MarketDataFetcher
            market_fetcher = MarketDataFetcher()
        except Exception as e:
            logger.warning("[feedback_loop.py:78] FeedbackLoop/run_all 失败(忽略): %s", e)

        # Step 1: 因子归因
        try:
            n_attr = self._run_attribution(market_fetcher, today=today)
            result["attribution_done"] = True
            result["n_attributions"] = n_attr
            result["messages"].append(f"归因完成: {n_attr} 条")
        except Exception as e:
            logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:87): %s", e)
            result["messages"].append(f"归因失败: {e}")

        # Step 2: IC分析
        try:
            ic_result = self._run_ic_analysis(today=today)
            result["ic_analysis_done"] = True
            result["ic_summary"] = ic_result
            result["messages"].append(f"IC分析完成: {len(ic_result.get('factors', {}))} 因子")
        except Exception as e:
            logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:96): %s", e)
            result["messages"].append(f"IC分析失败: {e}")

        # Step 3: 全局权重调整（基于归因+IC，factor-level）
        if not skip_weight_adjustment:
            try:
                weight_changes = self._adjust_weights(today=today)
                if weight_changes:
                    result["weight_adjusted"] = True
                    result["weight_changes"] = weight_changes
                    result["messages"].append(f"全局权重调整: {len(weight_changes)} 因子")
            except Exception as e:
                logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:107): %s", e)
                result["messages"].append(f"权重调整失败: {e}")

        # Step 4: 个股因子权重调整（per-stock，新增）
        if self.stock_weight_manager is not None:
            try:
                stock_changes = self._adjust_stock_weights(today)
                if stock_changes:
                    result["stock_weight_adjusted"] = True
                    result["stock_weight_changes"] = stock_changes
                    total = sum(len(v) for v in stock_changes.values())
                    result["messages"].append(
                        f"个股权重调整: {len(stock_changes)} 只股票 / {total} 因子"
                    )
            except Exception as e:
                logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:121): %s", e)
                result["messages"].append(f"个股权重调整失败: {e}")

        # Step 5: ML重训练（可跳过）
        if not skip_ml:
            try:
                retrained = self._maybe_retrain_model(today)
                if retrained:
                    result["model_retrained"] = not retrained.get("rejected")
                    result["model_metrics"] = retrained
                    if retrained.get("rejected"):
                        result["messages"].append(
                            f"模型重训练未上线({retrained.get('reject_reason')}), "
                            f"保留线上模型"
                        )
                    else:
                        result["messages"].append(
                            f"模型重训练: {retrained.get('model_type')} "
                            f"acc={retrained.get('accuracy')} "
                            f"auc={retrained.get('test_auc')}"
                        )
            except Exception as e:
                logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:135): %s", e)
                result["messages"].append(f"ML重训练失败: {e}")

        return result

    def _run_attribution(self, market_fetcher=None, today: str = None) -> int:
        """对有T+N收益数据的推荐做归因。
        today: 回测日。None 时使用真实当前日期。
        归因窗口自适应：仅处理 trade_date <= today 的推荐（避免未来函数）。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        # 取最近 60 天的推荐（覆盖 T+14 窗口），但只归因 trade_date <= today 的。
        # 仅归因 mode='premarket' 行：因子快照产自盘前全量计算，午后行与盘前行
        # 同票同日的因子几乎相同，混入会重复计数(午后表现通过 paper_trades 单独评估)。
        # (旧逻辑按 (code,trade_date) 去重会随机保留一条——mode 维度引入后必须显式过滤)
        all_recs = self.db.get_recent_recommendations(days=365)
        recs = [
            r for r in all_recs
            if r.get("trade_date", "") <= today
            and (r.get("mode") or "premarket") == "premarket"
        ]

        count = 0
        # 一次取出已存在的归因记录，避免 N+1 查询
        existing = self.db.get_attribution_history(days=90)
        existing_keys = {
            (r.get("code"), r.get("trade_date")) for r in existing
        }

        # 收益窗口字段与 horizon 对应；显式 is not None 判断——
        # 此前用 or 链，d14_return=0(有效值)会被误跳到下一窗口。
        # 窗口优先级 d3 最先：实盘持仓 hold_days=3，短窗口归因与调权目标一致；
        # 原顺序 d14 优先导致同一批归因记录持有窗口横跨 1~14 天且取决于
        # 回填任务的执行顺序而非业务规则
        horizon_fields = [
            ("d3_return", 3), ("d1_return", 1), ("d5_return", 5),
            ("d10_return", 10), ("d14_return", 14),
        ]

        for rec in recs:
            actual_return = None
            horizon = None
            for field, h in horizon_fields:
                v = rec.get(field)
                if v is not None:
                    actual_return = v
                    horizon = h
                    break
            if actual_return is None:
                continue
            trade_date = rec.get("trade_date", "")
            code = rec.get("code", "")
            if (code, trade_date) in existing_keys:
                continue
            sub_scores = self._get_sub_scores_for_recommendation(rec)
            if not sub_scores:
                continue
            rec["sub_scores"] = sub_scores
            attr = FactorAttribution(rec, actual_return, horizon=horizon)
            attr_dict = attr.to_dict()
            self.db.save_attribution(
                trade_date=attr_dict["trade_date"],
                code=attr_dict["code"],
                name=attr_dict["name"],
                score=attr_dict["score"],
                actual_return=attr_dict["actual_return"],
                factor_results=attr_dict["factor_results"],
                is_failure=attr_dict["is_failure"],
                return_horizon=horizon,
            )
            count += 1
        return count

    def _get_sub_scores_for_recommendation(self, rec: Dict) -> Dict:
        """获取推荐的子评分 — 优先从 factor_snapshots 精确查询，fallback 到 factors 字段"""
        trade_date = rec.get("trade_date", "")
        code = rec.get("code", "")
        # 优先: 按 trade_date+code 精确查询 factor_snapshots（避免 limit/排序漏查）
        sub = self.db.get_factor_snapshot_by_code_date(trade_date, code)
        if sub and isinstance(sub, dict):
            return sub
        # fallback: 从 recommendations.factors JSON 字段解析
        factors_raw = rec.get("factors")
        if factors_raw and isinstance(factors_raw, str):
            try:
                factors = json.loads(factors_raw)
                sub = {}
                for k in ["trend", "strategy", "chip", "fundamental", "institutional",
                          "sector_rotation", "oversold_rebound", "news", "relative_strength",
                          "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
                          "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
                          "policy_event", "national_team", "capital_behavior"]:
                    v = factors.get(k) or factors.get(f"score_{k}")
                    if v is not None:
                        sub[k] = float(v)
                return sub
            except Exception as e:
                logger.warning("[feedback_loop.py:232] FeedbackLoop/_get_sub_scores_for_recommendation 失败(忽略): %s", e)
        return {}

    @staticmethod
    def _load_dynamic_ortho_pairs() -> list:
        """读取持久化的动态正交化对(config/dynamic_ortho_pairs.json)"""
        import os
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "dynamic_ortho_pairs.json",
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [tuple(p) for p in json.load(f).get("pairs", [])]
        except Exception as e:
            logger.warning(f"加载动态正交化配对失败: {e}")
            return []

    def _run_ic_analysis(self, today: str = None) -> Dict:
        """计算最近14天因子IC，与30天IC对比检测衰减。
        注意：IC 基于 factor_snapshots（当日候选池，已过初筛），属于"候选池内 IC"，
        系统性高于全市场 IC。主要用于检测衰减趋势（差值），不作为绝对预测力指标。
        today: 回测日。None 时使用真实当前日期。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        from src.data.market_data import MarketDataFetcher
        MarketDataFetcher()

        factor_data = self._build_ic_dataset(days=30, today=today)
        if factor_data.empty:
            return {"status": "no_data"}

        optimizer = FactorOptimizer()
        recent_cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
        ic_recent = optimizer.calculate_ic(
            factor_data[factor_data["trade_date"] >= recent_cutoff]
        ) if not factor_data.empty else {}
        ic_all = optimizer.calculate_ic(factor_data)

        self.db.save_ic_record(today, ic_all, window_days=30, sample_count=len(factor_data))
        if ic_recent:
            self.db.save_ic_record(today + "_14d", ic_recent, window_days=14, sample_count=len(factor_data))

        decay_factors = {}
        for factor in ic_all:
            if factor in ic_recent:
                decay = ic_recent[factor] - ic_all[factor]
                if decay < -0.05:
                    decay_factors[factor] = {
                        "ic_all": round(ic_all[factor], 3),
                        "ic_recent": round(ic_recent[factor], 3),
                        "decay": round(decay, 3),
                    }

        # === 动态扩展正交化因子对 ===
        # 静态 ORTHOGONALIZE_PAIRS 来自代码级共线分析(硬编码4对)，此处基于
        # 近30天24维子评分截面补充检测：相关性>0.85 的新高相关对持久化到
        # config/dynamic_ortho_pairs.json，次日由调用方读入 batch_normalize。
        # 不再直接修改 StockScorer.ORTHOGONALIZE_PAIRS 类属性——那是进程级
        # 全局副作用，重启即丢、回测与生产行为漂移、且影响所有并发使用者。
        # 阈值比 analyze_factor_correlation 默认0.7 更保守，避免误杀独立信息；
        # 方向约定按因子名排序(target<-regressor)，保证可复现。
        try:
            corr_df = optimizer.analyze_factor_correlation(factor_data)
            if corr_df is not None and not corr_df.empty:
                existing_pairs = {frozenset(p) for p in self._load_dynamic_ortho_pairs()}
                existing_pairs.update(frozenset(p) for p in StockScorer.ORTHOGONALIZE_PAIRS)
                added = []
                for _, row in corr_df.iterrows():
                    if row["correlation"] < 0.85:
                        continue
                    pair = frozenset((row["factor1"], row["factor2"]))
                    if len(pair) < 2 or pair in existing_pairs:
                        continue
                    a, b = sorted(pair)
                    existing_pairs.add(pair)
                    added.append((a, b, row["correlation"]))
                if added:
                    current = [list(p) for p in sorted({frozenset(p) for p in self._load_dynamic_ortho_pairs()} | {frozenset((a, b)) for a, b, _ in added}, key=lambda s: sorted(s))]
                    import os as _os
                    cfg_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "config")
                    _os.makedirs(cfg_dir, exist_ok=True)
                    with open(_os.path.join(cfg_dir, "dynamic_ortho_pairs.json"), "w", encoding="utf-8") as f:
                        json.dump({"pairs": current, "updated_at": today}, f, ensure_ascii=False, indent=2)
                    logger.info("动态正交化新增 %d 对: %s", len(added),
                                ", ".join(f"{a}<-{b}({c:.2f})" for a, b, c in added))
        except Exception as e:
            logger.warning("动态正交化检测失败(不影响IC分析): %s", e)

        return {
            "status": "ok",
            "factors": {k: round(v, 3) for k, v in ic_all.items()},
            "decay_factors": decay_factors,
        }

    def _build_ic_dataset(self, days: int = 60, today: str = None) -> pd.DataFrame:
        """从factor_snapshots构建24维 sub_* IC数据集。

        直接 JOIN recommendations 取 T+N 真实收益（get_sub_score_panel），
        一次性批量返回，无 30 天窗口限制。

        注意：样本来自当日候选池（已过初筛），存在选择偏差，IC 系统性偏高。
        主要用于检测 IC 衰减趋势，不宜直接当作全市场预测力。
        today: 回测日。传入时仅使用 trade_date <= today 的快照（避免未来函数）。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        rows = self.db.get_sub_score_panel(days=days, today=today)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.rename(columns={"code": "stock_code"}, inplace=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        dims = [c for c in SUB_SCORE_DIMS if c in df.columns]
        for dim in dims:
            df[dim] = pd.to_numeric(df[dim], errors="coerce")
        df.dropna(subset=dims, how="all", inplace=True)
        for dim in dims:
            df[dim] = df[dim].fillna(50)
        return df

    def _get_future_return(self, code: str, date_str: str, days: int = 3) -> Optional[float]:
        """从数据库已存收益获取，不再重复拉取收盘价"""
        field_map = {1: "d1_return", 3: "d3_return", 5: "d5_return",
                     10: "d10_return", 14: "d14_return"}
        field = field_map.get(days)
        if not field:
            return None
        recs = self.db.get_recommendation_history(code, days=30)
        for r in recs:
            if r.get("trade_date") == date_str:
                return r.get(field)
        return None

    def _adjust_weights(self, today: str = None) -> List[Dict]:
        """基于归因统计+IC分析自动调整权重（全局 factor-level）。
        调整后自动写回 config.yaml（IC优化的权重/归因惩罚权重的合并结果）。
        today: 回测日。None 时使用真实当前日期。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        # 幂等: 同一天只调权一次。盘前补跑/重跑会重复惩罚同一因子
        # (2026-09-14 实测: 09:27 与 11:35 两次运行把 ashare_specific 连降两档 1.0→0.95→0.90)。
        # 仅真实当前日期生效, 回测(传入历史 today)不受影响。
        if today == datetime.now().strftime("%Y-%m-%d"):
            try:
                if self.db.has_weight_update(today):
                    logger.info("今日(%s)已完成调权, 跳过本轮重复调权", today)
                    return []
            except Exception as e:
                logger.warning("调权幂等检查失败, 继续执行: %s", e)
        all_attr = self.db.get_attribution_history(days=60)
        if not all_attr:
            return []

        # 调权基准与生产 scorer 完全一致：config.yaml 静态先验 + DB 最新调权结果。
        # 原实现直接读静态 config，DB 累积多次调权后 old_weight 失真、
        # MAX_WEIGHT_CHANGE 相对错误基准做 clip → weight_history 审计链不可信
        base_weights = dict(self.config.get("weights") or {})
        try:
            latest_db_weights = self.db.get_latest_weights()
            if latest_db_weights:
                for k, v in latest_db_weights.items():
                    # 清零因子免疫(与 premarket 合并口径一致): config 静态为 0
                    # 表示已被胜率校准否决, 历史调权结果不得作为调权基准复活它
                    if k in base_weights and float(base_weights[k]) != 0:
                        base_weights[k] = v
        except Exception as e:
            logger.warning("加载DB最新权重失败，退回config静态权重作为基准: %s", e)

        # 1)+2) 时序衰减归因统计 + 连续失败检测 + 惩罚调权
        changes = self._penalty_adjustments(all_attr, base_weights, today)
        # 3) IC优化调权。跳过惩罚调权已处理的因子: 两步同晚各自落库互不知情,
        # 曾致 strategy 同日先降后升来回振荡(2026-09-11 修复)
        penalized = {c["factor"] for c in changes}
        changes.extend(
            self._ic_adjustments(base_weights, today, skip_factors=penalized))

        # 4) 归一化（仅更新内存中的 config 对象，不写回 config.yaml）
        # 规则 §6：config.yaml 是静态先验，不被反馈循环改写。
        # 调权结果已在第 2、3 步通过 save_weight_update 落库到 weight_history 表。
        # scorer 次日启动时通过 db.get_latest_weights() 从库读取最新权重，
        # 叠加到 config.yaml 静态权重上生效，无需覆盖写回 config.yaml。
        if changes:
            # 归一化 changes（保证被调因子的权重之和不变）
            try:
                if self.config:
                    weights_dict = self.config.setdefault("weights", {})
                    for ch in changes:
                        weights_dict[ch["factor"]] = ch["new_weight"]
                    # 归一化确保所有权重之和 = 1
                    total = sum(weights_dict.values()) or 1
                    if abs(total - 1.0) > 0.01:
                        for k in weights_dict:
                            weights_dict[k] = round(max(weights_dict[k] / total, 0.001), 4)
            except Exception as e:
                logger.warning("内存权重归一化失败: %s", e)

        return changes

    def _penalty_adjustments(self, all_attr: List[Dict],
                             base_weights: Dict, today: str) -> List[Dict]:
        """时序衰减归因统计 + 连续失败检测 + 失败率惩罚调权(全局 factor-level)。"""
        from collections import defaultdict
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        changes = []

        factor_stats = {}
        factor_daily = defaultdict(list)
        for record in all_attr:
            days_ago = days_ago_of(record.get("trade_date", ""), today_dt)
            decay = recency_decay(days_ago)

            factor_results = record.get("factor_results", {})
            for factor, result in factor_results.items():
                res = result.get("result", "neutral")
                if factor not in factor_stats:
                    factor_stats[factor] = {"weighted_correct": 0.0, "weighted_wrong": 0.0, "total_weight": 0.0}
                if res == "correct":
                    factor_stats[factor]["weighted_correct"] += decay
                    factor_stats[factor]["total_weight"] += decay
                elif res == "wrong":
                    factor_stats[factor]["weighted_wrong"] += decay
                    factor_stats[factor]["total_weight"] += decay
                factor_daily[factor].append({"date": record["trade_date"], "result": res})

        # 连续失败检测（从最新日期往回数，连续 wrong 的交易日数）
        consecutive_failures = {}
        for factor, daily_list in factor_daily.items():
            daily_list.sort(key=lambda x: x["date"], reverse=True)
            count = 0
            for entry in daily_list:
                if entry["result"] == "wrong":
                    count += 1
                elif entry["result"] == "correct":
                    break
            if count >= 3:
                consecutive_failures[factor] = count

        for factor, stats in factor_stats.items():
            if stats["total_weight"] < MIN_ATTRIBUTION_SAMPLES:
                continue
            # 清零因子免疫: config 静态权重为 0 表示已被胜率校准否决,
            # 惩罚调权的 0.005 下限曾将其静默复活为非零权重(2026-09-11 修复)
            current_weight = float(base_weights.get(factor, 0))
            if current_weight <= 0:
                continue
            weighted_wrong_rate = stats["weighted_wrong"] / stats["total_weight"]

            if weighted_wrong_rate < 0.3:
                continue
            # 失败率 [0.3, 0.5) 中等惩罚砍半；>=0.5 严重失败砍 7 成
            penalty = FAILURE_WEIGHT_PENALTY if weighted_wrong_rate < 0.5 else 0.7
            if consecutive_failures.get(factor, 0) >= 3:
                penalty = min(penalty + 0.2, 0.7)

            new_weight = round(max(current_weight * (1 - penalty), 0.005), 4)
            new_weight = max(new_weight, current_weight - MAX_WEIGHT_CHANGE)
            new_weight = round(new_weight, 4)

            if abs(new_weight - current_weight) >= 0.005:
                changes.append({
                    "factor": factor,
                    "old_weight": current_weight,
                    "new_weight": new_weight,
                    "wrong_rate": round(weighted_wrong_rate, 2),
                    "reason": f"时序衰减失败率{weighted_wrong_rate:.0%}, 连续错误{consecutive_failures.get(factor, 0)}天",
                })
                self.db.save_weight_update(
                    today, factor, current_weight, new_weight,
                    reason=changes[-1]["reason"],
                )
        return changes

    def _ic_adjustments(self, base_weights: Dict, today: str,
                       skip_factors=None) -> List[Dict]:
        """IC优化调权：基于近期IC重新分配权重（24 维 sub_*，与 config.yaml.weights 一一对应）。

        失败不应中断归因惩罚调权的结果，但必须留痕便于排查。
        skip_factors: 本轮惩罚调权已处理的因子集合——IC调权跳过，
        避免同一天对同一因子先降后升落库互相矛盾(振荡根源)。
        """
        changes = []
        try:
            from src.engine.ml_selector import FactorOptimizer
            optimizer = FactorOptimizer()
            factor_data = self._build_ic_dataset(days=60, today=today)
            if factor_data is not None and not factor_data.empty and "future_return" in factor_data.columns:
                opt_weights = optimizer.optimize_weights(factor_data, return_col="future_return")
                if opt_weights:
                    # optimize_weights 按 |IC| softmax 分配权重，不区分信号方向；
                    # 负 IC 因子未做方向翻转前，给任何正权重都是反向加权
                    # (2026-09-11 修复: IC=-0.68 曾触发 strategy 上调)。
                    neg_factors = set(optimizer.negative_ic_factors or [])
                    if neg_factors:
                        logger.warning(
                            "IC优化跳过负IC因子(信号方向未翻转, 上调即反向加权): %s",
                            sorted(neg_factors))
                    skip = set(skip_factors or ())
                    for name, opt_w in opt_weights.items():
                        if name in skip or name in neg_factors:
                            continue
                        # 清零因子免疫: config 静态为 0 表示已被胜率校准否决
                        cur_w = float(base_weights.get(name, 0))
                        if cur_w <= 0:
                            continue
                        if abs(opt_w - cur_w) > MAX_WEIGHT_CHANGE:
                            clip_w = cur_w + (MAX_WEIGHT_CHANGE if opt_w > cur_w else -MAX_WEIGHT_CHANGE)
                            changes.append({
                                "factor": name,
                                "old_weight": cur_w,
                                "new_weight": round(clip_w, 4),
                                "wrong_rate": 0,
                                "reason": f"IC优化: IC={optimizer.factor_ic.get(name, 0):+.4f}",
                            })
                            self.db.save_weight_update(
                                today, name, cur_w, round(clip_w, 4),
                                reason=changes[-1]["reason"],
                            )
        except Exception as e:
            logger.warning("IC优化调权失败(跳过该步骤): %s", e)
        return changes

    def _adjust_stock_weights(self, today: str) -> Dict[str, List[Dict]]:
        """对当天有新归因数据的股票批量调整 per-stock 因子权重。
        返回 {code: [changes]}。
        """
        if self.stock_weight_manager is None:
            return {}

        # 收集所有归因记录，按 code 分组
        all_attr = self.db.get_attribution_history(days=365)
        if not all_attr:
            return {}

        attr_by_code: Dict[str, List[Dict]] = defaultdict(list)
        for rec in all_attr:
            code = rec.get("code", "")
            if code:
                attr_by_code[code].append(rec)

        if not attr_by_code:
            return {}

        # 批量调权
        codes = list(attr_by_code.keys())
        return self.stock_weight_manager.adjust_weights_for_codes(
            codes=codes,
            trade_date=today,
            attribution_by_code=attr_by_code,
        )

    def _maybe_retrain_model(self, today: str) -> Optional[Dict]:
        """每RETRAIN_INTERVAL_DAYS天重训练一次ML模型。
        today: 回测日。None 时使用真实当前日期。
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        latest = self.db.get_latest_model_version()
        if latest:
            last_train = latest.get("trained_at", "")
            try:
                last_date = datetime.strptime(last_train[:10], "%Y-%m-%d")
                today_date = datetime.strptime(today, "%Y-%m-%d")
                if (today_date - last_date).days < RETRAIN_INTERVAL_DAYS:
                    return None
            except Exception as e:
                logger.warning("[feedback_loop.py:565] FeedbackLoop/_maybe_retrain_model 失败(忽略): %s", e)

        selector = MLStockSelector()
        factor_data = self._build_ic_dataset(days=90, today=today)
        if factor_data.empty or len(factor_data) < 50:
            return None

        data_for_training = []
        for _, row in factor_data.iterrows():
            code = row["stock_code"]
            trade_date = row["trade_date"]
            date_str = trade_date.strftime("%Y-%m-%d") if hasattr(trade_date, "strftime") else str(trade_date)
            raw_factors = self.db.get_factor_snapshot_raw_factors(code, date_str)
            if not raw_factors:
                raw_factors = {dim: row[dim] for dim in factor_data.columns
                               if dim not in ["trade_date", "stock_code", "future_return"]}
            data_for_training.append({
                "date": date_str,
                "stock_code": code,
                "factors": raw_factors,
                "future_return": row["future_return"],
            })

        # min_return=1.0 即"涨幅>1%为正例"。future_return 来自 recommendations.dN_return
        # （百分数口径），原传 0.01 实际阈值是 0.01%，几乎所有微涨都成了 positive
        df = selector.prepare_training_data(data_for_training, min_return=1.0)
        if df.empty:
            return None

        dates = [d["date"] for d in data_for_training]
        metrics = selector.train(df, dates=dates)

        version = f"v{today.replace('-', '')}"
        self.db.save_model_version(
            version=version,
            trained_at=today,
            model_type=metrics.get("model_type", "unknown"),
            metrics=metrics,
            top_features=metrics.get("top_features", []),
            model_path=metrics.get("model_saved", ""),
        )
        return metrics


# ======================================================================
# 归因时序衰减(全局唯一口径, feedback_loop 与 stock_weight_manager 共用)
# ======================================================================
def recency_decay(days_ago: float) -> float:
    """归因记录时序衰减系数。

    档位: days_ago <=5 → 1.0, <=15 → 0.7, <=30 → 0.4, 其他 → 0.2。
    原两处内联实现合并于此(2026-08 技术债偿还), 修改档位须同步评审
    stock_weight_manager.adjust_weights 的行为一致性。
    """
    if days_ago <= 5:
        return 1.0
    if days_ago <= 15:
        return 0.7
    if days_ago <= 30:
        return 0.4
    return 0.2


def days_ago_of(date_str: str, today_dt) -> int:
    """解析 YYYY-MM-DD 并计算距 today_dt 的天数；解析失败返回 999(视为久远)。"""
    try:
        rec_date = datetime.strptime(date_str or "", "%Y-%m-%d")
        return (today_dt - rec_date).days
    except (ValueError, TypeError) as e:
        logger.warning("捕获异常已忽略(src/engine/feedback_loop.py:635): %s", e)
        return 999
