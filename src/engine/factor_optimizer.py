"""
因子组合优化器 + 影子ML模型

由 ml_selector.py 拆分(AGENTS.md 技术债偿还): FactorOptimizer/ShadowMLModel/train_shadow_ml。
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import os
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
class FactorOptimizer:
    """多因子组合优化"""

    def __init__(self):
        self.factor_ic = {}
        self.optimal_weights = {}
        self.factor_corr = None

    def calculate_ic(self, factor_data: pd.DataFrame, return_col: str = "future_return") -> Dict:
        """
        计算每个因子的IC（Information Coefficient）。
        严格做法：按 trade_date 分组计算每日截面 Rank IC，再求均值。
        全局合并计算会受样本分布扭曲（普涨日所有因子看起来都有效）。
        返回: {因子名: IC均值}
        """
        ic_results = {}

        # 排除非因子列 + 仅保留数值列（list 列如 quant_signals 无法算 IC，nunique 会抛 unhashable）
        exclude_cols = [return_col, "label", "stock_code", "date", "trade_date"]
        factor_cols = [
            c for c in factor_data.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(factor_data[c])
        ]

        # 决定分组列
        group_col = "trade_date" if "trade_date" in factor_data.columns else "date"
        has_group = group_col in factor_data.columns and factor_data[group_col].notna().any()

        if has_group:
            # 标准做法：按日分组计算每日IC，再求均值
            ic_by_factor = {col: [] for col in factor_cols}
            for date_val, group in factor_data.groupby(group_col):
                if len(group) < 5:
                    continue  # 截面样本太少不可靠
                for col in factor_cols:
                    if group[col].nunique() < 2 or group[return_col].nunique() < 2:
                        continue
                    ic = group[col].rank().corr(
                        group[return_col].rank(), method="spearman"
                    )
                    if not pd.isna(ic):
                        ic_by_factor[col].append(ic)

            for col in factor_cols:
                ic_list = ic_by_factor[col]
                ic_results[col] = round(sum(ic_list) / len(ic_list), 4) if ic_list else 0
        else:
            # 降级：无分组列时用全局IC
            for col in factor_cols:
                factor_rank = factor_data[col].rank()
                return_rank = factor_data[return_col].rank()
                ic = factor_rank.corr(return_rank, method="spearman")
                ic_results[col] = round(ic, 4) if not pd.isna(ic) else 0

        self.factor_ic = ic_results

        # 按IC绝对值排序
        sorted_ic = sorted(
            ic_results.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return dict(sorted_ic[:20])

    def optimize_weights(
        self, factor_data: pd.DataFrame, return_col: str = "future_return"
    ) -> Dict:
        """
        基于IC优化因子权重
        高IC因子获得更高权重，低IC因子降低权重
        """
        ic = self.calculate_ic(factor_data)
        if not ic:
            return {}

        # 负IC因子取绝对值参与权重分配(其预测力同样有效，方向相反)；
        # 原fallback把负IC因子与正IC同池softmax，负向因子获得正向权重
        # 却不做方向翻转 → 系统性反向加权
        abs_ic = {k: abs(v) for k, v in ic.items() if abs(v) > 0.02}
        if not abs_ic:
            abs_ic = {k: abs(v) for k, v in ic.items()}
            self._neg_ic_factors = {k for k, v in ic.items() if v < 0}

        # 对|IC|做softmax归一化
        ic_values = np.array(list(abs_ic.values()))
        ic_weights = np.exp(ic_values * 5) / np.sum(np.exp(ic_values * 5))

        self.optimal_weights = dict(
            zip(abs_ic.keys(), [round(w, 4) for w in ic_weights])
        )
        # 记录负IC因子清单供调用方决定是否取反该因子评分
        self.negative_ic_factors = sorted(k for k, v in ic.items() if v < 0)

        return self.optimal_weights

    def analyze_factor_correlation(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """分析因子间相关性，避免多重共线性"""
        exclude_cols = {"future_return", "label", "stock_code", "date", "trade_date"}
        factor_cols = [c for c in factor_data.columns if c not in exclude_cols
                       and pd.api.types.is_numeric_dtype(factor_data[c])]

        if len(factor_cols) < 2:
            return pd.DataFrame()

        corr = factor_data[factor_cols].corr()
        self.factor_corr = corr

        # 找出高度相关的因子对
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > 0.7:
                    high_corr_pairs.append(
                        {
                            "factor1": corr.columns[i],
                            "factor2": corr.columns[j],
                            "correlation": round(val, 3),
                        }
                    )

        return pd.DataFrame(high_corr_pairs)

    @staticmethod
    def load_from_database(lookback_days: int = 60,
                            forward_window: int = 3) -> pd.DataFrame:
        """
        从数据库加载因子数据并计算forward return
        lookback_days: 回溯天数
        forward_window: 未来窗口（天）
        """
        from src.data.database import StockDatabase
        from src.data.market_data import MarketDataFetcher

        db = StockDatabase()
        snapshots = db.get_factor_history(days=lookback_days)

        if not snapshots:
            return pd.DataFrame()

        fetcher = MarketDataFetcher()
        records = []

        for snap in snapshots:
            trade_date = snap["trade_date"]
            code = snap["code"]
            factors = snap.get("factors", {})
            if not factors or "z_scores" in factors:
                continue

            record = {"trade_date": trade_date, "code": code}
            record.update(factors)

            future_return = FactorOptimizer._calc_forward_return(
                fetcher, code, trade_date, forward_window
            )
            if future_return is not None:
                record["future_return"] = future_return
                records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def _calc_forward_return(fetcher, code: str, trade_date: str,
                              forward_window: int = 3) -> float:
        """计算 trade_date 起 T+N 的真实收益。

        此前实现取"最近N天K线"的 closes[0] 当入场价——对历史 trade_date 而言
        锚点完全错位(算的是最近几天的收益而非该日起的远期收益)。
        现按 trade_date 在K线中定位锚点行；拉取跨度按距今天数动态扩展。
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            span = max((datetime.now() - start).days, forward_window)
            history = fetcher.get_stock_history(code, days=span + forward_window + 10)
            if history is None or history.empty:
                return None
            close_col = "收盘" if "收盘" in history.columns else "close"
            date_col = "日期" if "日期" in history.columns else "date"
            if close_col not in history.columns or date_col not in history.columns:
                return None

            # 按 trade_date 定位锚点行，找不到精确匹配则放弃(避免用错误基准价)
            mask = history[date_col].astype(str).str.startswith(trade_date)
            if not mask.any():
                return None
            idx = int(mask.values.argmax())
            closes = history[close_col].values
            if idx + forward_window >= len(closes):
                return None
            entry = float(closes[idx])
            if entry <= 0:
                return None
            exit_val = float(closes[idx + forward_window])
            return (exit_val / entry - 1) * 100
        except Exception as e:
            logger.warning(f"计算未来收益失败: {e}")
            return None

    def calculate_icir(self, ic_series) -> float:
        """计算ICIR = mean(IC) / std(IC)。
        ic_series 可以是 pd.Series 或 list，表示该因子每日的IC序列。
        """
        if hasattr(ic_series, "empty") and ic_series.empty:
            return 0.0
        if hasattr(ic_series, "tolist"):
            ic_list = ic_series.tolist()
        else:
            ic_list = list(ic_series)
        ic_list = [v for v in ic_list if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if len(ic_list) < 2:
            return 0.0
        mean_ic = sum(ic_list) / len(ic_list)
        std_ic = (sum((v - mean_ic) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        if std_ic == 0:
            return 0.0
        return mean_ic / std_ic

    def calculate_ic_by_day(self, factor_data: pd.DataFrame,
                            return_col: str = "future_return") -> Dict[str, list]:
        """按日计算每个因子的 IC 序列，用于 ICIR 评估。
        返回: {因子名: [ic_day1, ic_day2, ...]}
        """
        result = {}
        exclude_cols = [return_col, "label", "stock_code", "date", "trade_date"]
        factor_cols = [c for c in factor_data.columns if c not in exclude_cols]
        group_col = "trade_date" if "trade_date" in factor_data.columns else "date"
        if group_col not in factor_data.columns:
            return {col: [] for col in factor_cols}

        for col in factor_cols:
            result[col] = []
        for date_val, group in factor_data.groupby(group_col):
            if len(group) < 5:
                continue
            for col in factor_cols:
                if group[col].nunique() < 2 or group[return_col].nunique() < 2:
                    continue
                ic = group[col].rank().corr(
                    group[return_col].rank(), method="spearman"
                )
                if not pd.isna(ic):
                    result[col].append(ic)
        return result

    def generate_weight_report(self) -> str:
        """生成权重优化报告"""
        lines = []
        lines.append("# 多因子优化报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if self.factor_ic:
            lines.append("## 因子IC排名\n")
            lines.append("| 排名 | 因子 | IC值 | 说明 |")
            lines.append("|------|------|------|------|")
            sorted_ic = sorted(
                self.factor_ic.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for i, (name, ic_val) in enumerate(sorted_ic[:15], 1):
                strength = "强" if abs(ic_val) > 0.05 else ("中" if abs(ic_val) > 0.03 else "弱")
                lines.append(f"| {i} | {name} | {ic_val:+.4f} | {strength} |")

        if self.optimal_weights:
            lines.append("\n## 优化权重\n")
            lines.append("| 因子 | 优化权重 |")
            lines.append("|------|----------|")
            for name, w in sorted(
                self.optimal_weights.items(), key=lambda x: x[1], reverse=True
            )[:15]:
                lines.append(f"| {name} | {w:.2%} |")

        if self.factor_corr is not None and not self.factor_corr.empty:
            # 高相关因子对
            pairs = []
            for i in range(len(self.factor_corr.columns)):
                for j in range(i + 1, len(self.factor_corr.columns)):
                    val = abs(self.factor_corr.iloc[i, j])
                    if val > 0.6:
                        pairs.append(
                            (self.factor_corr.columns[i], self.factor_corr.columns[j], val)
                        )
            if pairs:
                lines.append("\n## 高相关因子对（>0.6）\n")
                for f1, f2, v in pairs[:10]:
                    lines.append(f"- {f1} ↔ {f2}: {v:.3f}")

        return "\n".join(lines)


# ======================================================================
# P7A 影子模式：独立于主模型的 ML 训练 + 推理
# ======================================================================

class ShadowMLModel:
    """影子 ML 模型 — 与主模型完全隔离，不共享任何权重。"""

    def __init__(self, model, feature_names: List[str], trained_at: str):
        self.model = model
        self.feature_names = feature_names
        self.trained_at = trained_at

    def predict(self, sub_scores: Dict) -> Dict:
        features = [sub_scores.get(f, 50) for f in self.feature_names]
        try:
            proba = self.model.predict_proba([features])[0]
            up_prob = proba[1] if len(proba) > 1 else 0.5
        except Exception as e:
            logger.warning(f"ML预测失败，使用默认概率: {e}")
            up_prob = 0.5
        ml_score = int(up_prob * 100)
        if up_prob > 0.65:
            prediction = "bullish"
        elif up_prob > 0.55:
            prediction = "slightly_bullish"
        elif up_prob < 0.35:
            prediction = "bearish"
        elif up_prob < 0.45:
            prediction = "slightly_bearish"
        else:
            prediction = "neutral"
        return {
            "ml_score": ml_score,
            "ml_prediction": prediction,
            "ml_probability": round(up_prob, 3),
        }


def train_shadow_ml(today: str, db=None, lookback_days: int = 90,
                    horizon: int = 3, threshold: float = 1.0,
                    kc: Dict = None):
    """影子 ML 训练：从 factor_snapshots 加载数据，训练独立 LightGBM 模型。

    Args:
        today: 当前交易日
        db: StockDatabase 实例，None 时自动创建
        lookback_days: 回溯天数
        horizon: 预测窗口（天）
        threshold: 正例阈值（百分数口径：1.0 = 涨幅>1% 为正例，
                   与 recommendations.dN_return 单位一致）
        kc: K线缓存 dict {code: [(date, close, ...)]}，用于计算实际收益

    Returns:
        ShadowMLModel 实例，或 None（数据不足时）
    """
    if db is None:
        from src.data.database import StockDatabase
        db = StockDatabase()

    records = db.get_factor_history(days=lookback_days)
    if not records or len(records) < 100:
        return None

    rows = []
    for snap in records:
        factors = snap.get("factors", {})
        if not isinstance(factors, dict) or not factors or "z_scores" in factors:
            continue
        row = {"trade_date": snap["trade_date"], "stock_code": snap["code"]}
        row.update(factors)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if kc:
        future_returns = []
        found = 0
        for _, row in df.iterrows():
            code = row["stock_code"]
            trade_date = row["trade_date"]
            krows = kc.get(code)
            ret = _future_return_from_kline(krows, trade_date, horizon)
            if ret is not None:
                found += 1
            future_returns.append(ret)
        df["future_return"] = future_returns
        if found < 50:
            import logging
            logging.warning(f'train_shadow_ml: only {found}/{len(df)} future returns from kc, falling back to rec table')
            # fallback: use recommendations table
            field = {1: "d1_return", 3: "d3_return", 5: "d5_return"}.get(horizon)
            future_returns2 = []
            for _, row in df.iterrows():
                recs = db.get_recommendation_history(row["stock_code"], days=30)
                ret = None
                for r in recs:
                    if r.get("trade_date") == row["trade_date"]:
                        ret = r.get(field)
                        break
                future_returns2.append(ret)
            df["future_return"] = future_returns2
    else:
        field = {1: "d1_return", 3: "d3_return", 5: "d5_return"}.get(horizon)
        future_returns = []
        for _, row in df.iterrows():
            recs = db.get_recommendation_history(row["stock_code"], days=30)
            ret = None
            for r in recs:
                if r.get("trade_date") == row["trade_date"]:
                    ret = r.get(field)
                    break
            future_returns.append(ret)
        df["future_return"] = future_returns

    df.dropna(subset=["future_return"], inplace=True)
    if df.empty:
        return None

    dims = [c for c in df.columns if c not in ("trade_date", "stock_code", "future_return")]
    for dim in dims:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
    df.dropna(subset=dims, how="all", inplace=True)
    for dim in dims:
        df[dim].fillna(50, inplace=True)

    X = df[dims].values
    y = (df["future_return"].values > threshold).astype(int)

    if len(X) < 50 or y.sum() < 5 or (1 - y).sum() < 5:
        return None

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, num_leaves=15,
            learning_rate=0.08, min_child_samples=10,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        model.fit(X, y)
    except Exception as e:
        logger.warning(f"训练LightGBM模型失败: {e}")
        return None

    return ShadowMLModel(model, dims, today)


def _future_return_from_kline(krows, trade_date: str, horizon: int) -> float:
    """从 K 线行列表计算未来 N 天的收益。

    返回百分数口径 (如 2.5 表示 +2.5%)，与 recommendations.dN_return /
    _calc_forward_return 一致。原实现返回小数，与 rec 表 fallback 的百分数
    混用同一个 threshold=0.01 判断标签 → fallback 分支阈值实际是 0.01%。
    """
    if not krows:
        return None
    idx = None
    for i, row in enumerate(krows):
        if row[0] == trade_date:
            idx = i
            break
    if idx is None or idx + horizon >= len(krows):
        return None
    cur_close = float(krows[idx][1])
    fut_close = float(krows[idx + horizon][1])
    if cur_close <= 0:
        return None
    return (fut_close / cur_close - 1) * 100
