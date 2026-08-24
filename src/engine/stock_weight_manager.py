"""
个股因子权重管理器 — 每只股票独立维护 24 维因子权重

核心理念：
- 全局权重（config.yaml）是"市场平均水平"的先验
- 每只股票有自己的"因子有效性指纹"：某些因子对它预测得准，某些不准
- 基于该股票历史归因数据（命中/失败）滚动调整其专属权重
- 样本不足时 fallback 到全局权重

调整规则（参考 feedback_loop._adjust_weights，但作用在 per-stock 维度）：
- wrong_rate >= 0.5                 → 权重 × 0.5  (严重失效)
- wrong_rate >= 0.3 且 accuracy<0.4 → 权重 × 0.7  (明显偏差)
- accuracy > 0.7 且 samples >= 5    → 权重 × 1.1  (强化好因子)
- 单次调整后归一化，保证 24 维权重之和 = 1
- 样本数 < MIN_SAMPLES 时不调整（避免小样本过拟合）
"""
import os
import sys
from typing import Dict, List, Optional
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import StockDatabase
from src.data.market_data import MarketDataFetcher


MIN_SAMPLES = 5           # 单股单因子最少归因样本数才调权（低于此阈值统计噪声过大，与正向强化门槛一致）
MIN_WEIGHT_FLOOR = 0.005  # 单因子权重下限：防止连续失败×0.5 把权重压到趋近0后永久"死亡"
MAX_WEIGHT_STEP = 0.01    # 触底恢复单次最大回升幅度

# 24 维因子完整列表（与 StockScorer.get_sub_scores 一致）
ALL_FACTORS = [
    "trend", "strategy", "chip", "fundamental", "institutional",
    "sector_rotation", "oversold_rebound", "news", "relative_strength",
    "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
    "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
    "policy_event", "national_team", "capital_behavior", "moat",
    "holder_concentration", "circulate_holder",
]


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """归一化：保证 24 维权重之和 = 1"""
    total = sum(weights.values())
    if total <= 0:
        # 全 0 时退化为均匀分布
        n = len(weights) if weights else 1
        return {k: round(1.0 / n, 6) for k in weights}
    return {k: round(v / total, 6) for k, v in weights.items()}


class StockWeightManager:
    """每只股票独立的 24 维因子权重管理器"""

    def __init__(self, db: StockDatabase, global_weights: Dict[str, float]):
        self.db = db
        # 全局权重作为先验 + fallback。保证 24 维齐全
        self.global_weights = {k: float(global_weights.get(k, 0.0))
                               for k in ALL_FACTORS}
        # 归一化全局权重（防止 config.yaml 权重和≠1）
        self.global_weights = _normalize(self.global_weights)
        # 内存缓存：{code: {factor: weight}}，回测主循环预热一次后零查询
        self._cache: Optional[Dict[str, Dict[str, float]]] = None
        # 行业映射缓存
        self._industry_map: Optional[Dict[str, str]] = None
        # 行业权重缓存：{industry_name: {factor: weight}}
        self._sector_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._sector_init_done = False

    # ------------------------------------------------------------------
    # 行业映射
    # ------------------------------------------------------------------

    def _init_industry_map(self):
        if self._industry_map is not None:
            return
        try:
            fetcher = MarketDataFetcher()
            self._industry_map = fetcher.get_industry_map()
        except Exception:
            self._industry_map = {}
        if not self._industry_map:
            self._industry_map = {}

    def _get_sector_key(self, code: str) -> str:
        self._init_industry_map()
        return self._industry_map.get(code, "")

    def _load_sector_weights(self):
        if self._sector_cache is not None:
            return
        raw = self.db.get_all_sector_weights()
        self._sector_cache = {}
        for industry, weights in raw.items():
            merged = dict(self.global_weights)
            merged.update(weights)
            self._sector_cache[industry] = _normalize(merged)

    def _get_sector_weights(self, code: str) -> Optional[Dict[str, float]]:
        sector = self._get_sector_key(code)
        if not sector:
            return None
        self._load_sector_weights()
        sector_w = self._sector_cache.get(sector)
        if not sector_w:
            return None
        return dict(sector_w)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get_weights(self, code: str) -> Dict[str, float]:
        """获取该股票的因子权重。
        优先返回个股权重（有足够归因样本时）；
        不足时 fallback 到行业权重；
        行业也不足时回退全局权重。
        """
        if self._cache is not None:
            per_stock = self._cache.get(code)
        else:
            per_stock = self.db.get_stock_weights(code)
        if per_stock:
            result = dict(self.global_weights)
            result.update(per_stock)
            return _normalize(result)
        sector_w = self._get_sector_weights(code)
        if sector_w is not None:
            return sector_w
        return dict(self.global_weights)

    def get_weights_batch(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """批量获取多只股票的权重。
        优先用内存缓存；缓存为空时一次性加载全表。
        返回 {code: {factor: weight}}，每个 weight dict 都已归一化且 24 维齐全。
        """
        if self._cache is None:
            self._cache = self.db.get_all_stock_weights()

        result = {}
        for code in codes:
            per_stock = self._cache.get(code)
            if per_stock:
                merged = dict(self.global_weights)
                merged.update(per_stock)
                result[code] = _normalize(merged)
            else:
                sector_w = self._get_sector_weights(code)
                if sector_w is not None:
                    result[code] = sector_w
                else:
                    result[code] = dict(self.global_weights)
        return result

    def warmup_cache(self) -> None:
        """预热内存缓存（回测主循环开始时调用一次）"""
        self._cache = self.db.get_all_stock_weights()

    def invalidate_cache(self, code: str = None) -> None:
        """失效缓存。code=None 时清空整个缓存；否则只清该股票。"""
        if code is None:
            self._cache = None
            self._sector_cache = None
        elif self._cache is not None:
            self._cache.pop(code, None)

    # ------------------------------------------------------------------
    # 调权
    # ------------------------------------------------------------------
    def adjust_weights(self, code: str, trade_date: str,
                       attribution_records: List[Dict]) -> List[Dict]:
        """基于该股票的归因记录调整其 24 维因子权重。

        时序衰减与 feedback_loop._adjust_weights 一致：
        days_ago <=5 → 1.0, <=15 → 0.7, <=30 → 0.4, 其他 → 0.2

        Args:
            code: 股票代码
            trade_date: 当前回测日（用于写入 weight_history + 计算时序衰减）
            attribution_records: 该股票的归因记录列表，每条含 trade_date + factor_results 字段
                factor_results: {factor: {score, signal, result: "correct"/"wrong"/"neutral"}}

        Returns:
            变更列表 [{factor, old_weight, new_weight, reason, samples, accuracy}]
        """
        if not attribution_records:
            return []

        # 1. 聚合该股票每个因子的 correct/wrong 计数（带时序衰减）
        from datetime import datetime as _dt
        try:
            today_dt = _dt.strptime(trade_date, "%Y-%m-%d")
        except Exception:
            today_dt = _dt.now()

        factor_correct = defaultdict(float)
        factor_wrong = defaultdict(float)
        for rec in attribution_records:
            # 时序衰减权重
            try:
                rec_dt = _dt.strptime(rec.get("trade_date", ""), "%Y-%m-%d")
                days_ago = (today_dt - rec_dt).days
            except Exception:
                days_ago = 999
            if days_ago <= 5:
                decay = 1.0
            elif days_ago <= 15:
                decay = 0.7
            elif days_ago <= 30:
                decay = 0.4
            else:
                decay = 0.2

            factor_results = rec.get("factor_results") or {}
            for factor, info in factor_results.items():
                if factor not in ALL_FACTORS:
                    continue
                result = info.get("result", "neutral")
                if result == "correct":
                    factor_correct[factor] += decay
                elif result == "wrong":
                    factor_wrong[factor] += decay

        if not factor_correct and not factor_wrong:
            return []

        # 2. 当前权重（per-stock 优先，fallback 全局）
        current_weights = self.get_weights(code)
        changes = []

        # 3. 对每个有数据的因子应用调整规则
        new_weights = dict(current_weights)
        for factor in ALL_FACTORS:
            correct = factor_correct.get(factor, 0)
            wrong = factor_wrong.get(factor, 0)
            total_pred = correct + wrong
            if total_pred < MIN_SAMPLES:
                continue

            accuracy = correct / total_pred if total_pred > 0 else 0
            wrong_rate = wrong / total_pred if total_pred > 0 else 0
            old_w = current_weights[factor]
            new_w = old_w
            reason = ""

            # 调整规则（升权门槛比降权更严格，防止小样本虚高 accuracy 导致权重只升不降）
            # 降权带下限 MIN_WEIGHT_FLOOR：原实现连续 wrong≥0.5 每次×0.5 无下限，
            # 多次迭代后权重趋近0，而升权需 accuracy>75%+净胜≥3——低权重因子
            # 参与评分的变化极小很难积累净胜，事实上永久"死亡"且无恢复通道
            if wrong_rate >= 0.5:
                new_w = max(old_w * 0.5, MIN_WEIGHT_FLOOR)
                reason = f"wrong_rate={wrong_rate:.0%} (samples={total_pred})"
            elif wrong_rate >= 0.3 and accuracy < 0.4:
                new_w = max(old_w * 0.7, MIN_WEIGHT_FLOOR)
                reason = f"wrong_rate={wrong_rate:.0%} accuracy={accuracy:.0%}"
            elif accuracy > 0.75 and total_pred >= 10 and (correct - wrong) >= 3:
                # 升权要求：accuracy>75% + 样本≥10 + 净胜数≥3（防止 6对1错=86% 这种小样本虚高）
                # 恢复通道：已触底的因子只需略低的门槛即可回升，避免永久死亡
                if old_w <= MIN_WEIGHT_FLOOR * 1.01:
                    if accuracy > 0.65 and total_pred >= 8:
                        new_w = min(old_w * 1.5, old_w + MAX_WEIGHT_STEP)
                        reason = f"floor_recovery: accuracy={accuracy:.0%} (samples={total_pred})"
                else:
                    new_w = old_w * 1.1
                    reason = f"accuracy={accuracy:.0%} (samples={total_pred}, net_wins={int(correct-wrong)})"

            if abs(new_w - old_w) < 1e-6:
                continue

            new_weights[factor] = new_w
            changes.append({
                "factor": factor,
                "old_weight": round(old_w, 6),
                "new_weight": round(new_w, 6),
                "reason": reason,
                "samples": total_pred,
                "accuracy": round(accuracy, 4),
            })

        if not changes:
            return []

        # 4. 归一化（保证 24 维之和 = 1）
        new_weights = _normalize(new_weights)

        # 5. 持久化（每个因子 UPSERT + 写历史）
        # 重新计算归一化后的最终权重，覆盖 changes 里的 new_weight
        final_changes = []
        for ch in changes:
            factor = ch["factor"]
            final_w = new_weights[factor]
            if abs(final_w - ch["old_weight"]) < 1e-6:
                continue
            # 更新数据库
            correct = factor_correct.get(factor, 0)
            wrong = factor_wrong.get(factor, 0)
            total_pred = correct + wrong
            accuracy = correct / total_pred if total_pred > 0 else 0
            self.db.update_stock_weight(
                code=code,
                factor_name=factor,
                new_weight=final_w,
                samples=total_pred,
                accuracy=accuracy,
                trade_date=trade_date,
                old_weight=ch["old_weight"],
                reason=ch["reason"],
            )
            final_changes.append({
                "factor": factor,
                "old_weight": ch["old_weight"],
                "new_weight": final_w,
                "reason": ch["reason"],
                "samples": total_pred,
                "accuracy": round(accuracy, 4),
            })

        # 6. 更新内存缓存
        if self._cache is not None:
            self._cache[code] = dict(new_weights)

        return final_changes

    def adjust_weights_for_codes(self, codes: List[str], trade_date: str,
                                  attribution_by_code: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """批量调整多只股票的权重 + 行业聚簇归因。
        attribution_by_code: {code: [attribution_records]}
        返回 {code: [changes]}
        """
        result = {}
        for code in codes:
            recs = attribution_by_code.get(code, [])
            if not recs:
                continue
            changes = self.adjust_weights(code, trade_date, recs)
            if changes:
                result[code] = changes

        # 行业聚簇归因：将同行业股票的归因数据聚合，写入行业权重
        self._aggregate_sector_weights(attribution_by_code, trade_date)

        return result

    def _aggregate_sector_weights(self, attribution_by_code: Dict[str, List[Dict]],
                                   trade_date: str):
        """将个股归因数据按行业聚簇聚合，更新行业因子权重。"""
        self._init_industry_map()
        if not self._industry_map:
            return
        sector_attribution: Dict[str, List[Dict]] = defaultdict(list)
        for code, recs in attribution_by_code.items():
            sector = self._industry_map.get(code, "")
            if sector:
                sector_attribution[sector].extend(recs)

        for sector, recs in sector_attribution.items():
            factor_correct = defaultdict(int)
            factor_wrong = defaultdict(int)
            for rec in recs:
                factor_results = rec.get("factor_results") or {}
                for factor, info in factor_results.items():
                    if factor not in ALL_FACTORS:
                        continue
                    result = info.get("result", "neutral")
                    if result == "correct":
                        factor_correct[factor] += 1
                    elif result == "wrong":
                        factor_wrong[factor] += 1

            if not factor_correct and not factor_wrong:
                continue

            old_weights = self.db.get_sector_weights(sector)
            merged = dict(self.global_weights)
            merged.update(old_weights)
            current_weights = _normalize(merged)

            new_weights = dict(current_weights)
            for factor in ALL_FACTORS:
                correct = factor_correct.get(factor, 0)
                wrong = factor_wrong.get(factor, 0)
                total_pred = correct + wrong
                if total_pred < 1:
                    continue
                accuracy = correct / total_pred if total_pred > 0 else 0
                wrong_rate = wrong / total_pred if total_pred > 0 else 0
                old_w = current_weights[factor]
                new_w = old_w
                if wrong_rate >= 0.5:
                    new_w = old_w * 0.5
                elif wrong_rate >= 0.3 and accuracy < 0.4:
                    new_w = old_w * 0.7
                elif accuracy > 0.7 and total_pred >= 5:
                    new_w = old_w * 1.1
                if abs(new_w - old_w) < 1e-6:
                    continue
                new_weights[factor] = new_w

            new_weights = _normalize(new_weights)
            for factor in ALL_FACTORS:
                correct = factor_correct.get(factor, 0)
                wrong = factor_wrong.get(factor, 0)
                total_pred = correct + wrong
                if total_pred < 1:
                    continue
                self.db.update_sector_weight(
                    industry_name=sector,
                    factor_name=factor,
                    new_weight=new_weights[factor],
                    samples=total_pred,
                    accuracy=round(correct / total_pred, 4) if total_pred > 0 else 0,
                    trade_date=trade_date,
                    old_weight=current_weights.get(factor),
                    reason="sector aggregation",
                )

            if self._sector_cache is not None:
                self._sector_cache[sector] = dict(new_weights)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def get_global_weights(self) -> Dict[str, float]:
        """返回全局权重副本"""
        return dict(self.global_weights)

    def stats(self) -> Dict:
        """返回个股权重管理器的统计信息"""
        all_weights = self.db.get_all_stock_weights() if self._cache is None else self._cache
        return {
            "stocks_with_custom_weights": len(all_weights),
            "global_weights_sum": round(sum(self.global_weights.values()), 4),
            "global_factors": len(self.global_weights),
        }
