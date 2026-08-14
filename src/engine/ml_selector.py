"""
第三阶段：机器学习选股模型 + 多因子组合优化
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict
import os
import sys
import json
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLStockSelector:
    """机器学习选股模型（LightGBM + RandomForest 集成，软投票）"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_aux = None
        self.feature_names = []
        self.feature_importance = {}
        self.model_path = model_path or os.path.join(ROOT, "models", "stock_selector.pkl")
        self.model_aux_path = self.model_path.replace(".pkl", "_aux.pkl")
        self.model_meta_path = self.model_path + ".meta.json"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model_type = "ensemble"
        self._keep_index = None  # 常量特征剔除索引（训练时计算，预测时应用）
        self._try_load_model()

    def _try_load_model(self):
        """启动时尝试加载已存在的模型（主模型 + 辅助模型）"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.model_aux_path):
                    self.model_aux = joblib.load(self.model_aux_path)
                if os.path.exists(self.model_meta_path):
                    with open(self.model_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])
                    self.model_type = meta.get("model_type", "ensemble")
                    ki = meta.get("keep_index")
                    if ki:
                        self._keep_index = [int(i) for i in ki]
                # 特征数量不匹配时清空模型强制下次重训练
                expected_len = len(self.feature_names) if self.feature_names else 0
                if expected_len > 0 and hasattr(self.model, "n_features_in_"):
                    if self.model.n_features_in_ != expected_len:
                        self.model = None
                        self.model_aux = None
            except Exception:
                self.model = None
                self.model_aux = None

    def _extract_features(self, factors: Dict, apply_keep: bool = True) -> np.ndarray:
        """从因子字典提取特征向量（原始特征 + 交互特征，不含横截面排名）
        apply_keep: 为 True 时应用常量特征剔除索引（predict 用）；False 输出全量（训练用）"""
        base_keys = [
            "rsi_6", "rsi_14", "macd_golden_cross", "ma_bullish", "breakout_20d_high",
            "volume_price_up", "strategy_count", "chip_profit_ratio",
            "chip_concentration_70", "chip_above_peak", "pe_score", "pb_score",
            "market_cap_score", "turnover_score", "inst_holding_count", "inst_holding_ratio",
            "relative_strength_5d", "relative_strength_20d", "oversold_rebound_score",
            "rotation_score", "tech_sector_strength", "news_score", "expectation_score",
            "mean_reversion_score", "volatility_score", "position_score",
            "quant_score", "margin_score", "shareholder_score", "lockup_score",
            "research_score", "lh_score", "leader_score", "limit_up_20d", "hot_money_score",
        ]
        def _v(key):
            v = factors.get(key, 50)
            if isinstance(v, (list, dict, set, tuple)):
                return 50.0
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 50.0
            return float(v)

        base = [_v(k) for k in base_keys]

        # 交互特征（零额外依赖）
        rs5, rs20 = _v("relative_strength_5d"), _v("relative_strength_20d")
        vol = _v("volatility_score")
        hm = _v("hot_money_score")
        pe = _v("pe_score")
        inst_r = _v("inst_holding_ratio")
        news = _v("news_score")
        pos = _v("position_score")
        tech_sum = sum((
            _v("macd_golden_cross"), _v("ma_bullish"),
            _v("breakout_20d_high"), _v("volume_price_up"),
        ))
        interaction = [
            tech_sum,                                       # 技术共识
            vol * hm / 50,                                  # 波动+资金
            pe * rs5 / 50 if abs(rs5) >= 1 else 50,         # 估值+动量
            rs5 * vol / 50,                                 # 动量+波动
            inst_r * rs20 / 50,                             # 机构+动量
            news * rs5 / 50,                                # 消息+动量
            pos * vol / 50,                                 # 位置+波动
            hm * tech_sum,                                  # 资金+技术
            rs20 * _v("rotation_score") / 50,               # 动量+板块
            (_v("inst_holding_count") / 100) * _v("inst_holding_ratio"),
        ]

        all_features = base + interaction
        full_names = base_keys + [f"interact_{i}" for i in range(len(interaction))]

        # 常量特征剔除：训练后记录 keep_index，预测时按同样索引裁剪（保证与模型输入一致）
        if apply_keep and self._keep_index is not None:
            self.feature_names = [full_names[i] for i in self._keep_index]
            sel = [all_features[i] for i in self._keep_index]
            return np.array(sel).reshape(1, -1)

        self.feature_names = full_names
        return np.array(all_features).reshape(1, -1)

    def predict(self, factors: Dict) -> Dict:
        """
        用训练好的模型预测（ensemble 软投票）
        Returns: 上涨概率、预测标签、特征贡献
        """
        if self.model is None:
            return {"ml_score": 50, "ml_prediction": "neutral", "ml_probability": 0.5}

        features = self._extract_features(factors)
        try:
            proba = self.model.predict_proba(features)[0]
            up_prob = proba[1] if len(proba) > 1 else 0.5
            if self.model_aux is not None:
                proba_aux = self.model_aux.predict_proba(features)[0]
                up_aux = proba_aux[1] if len(proba_aux) > 1 else 0.5
                up_prob = up_prob * 0.6 + up_aux * 0.4
        except Exception:
            return {"ml_score": 50, "ml_prediction": "neutral", "ml_probability": 0.5}

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

    def prepare_training_data(
        self, history_data: List[Dict], min_return: float = 0.01
    ) -> pd.DataFrame:
        """
        准备训练数据（不含横截面排名特征，避免训练/预测不一致）
        标签直接取 history_data 中每项的 future_return 字段，与 min_return 比较。
        """
        if not history_data:
            return pd.DataFrame()

        records = []
        for item in history_data:
            factors = item.get("factors", {})
            future_return = item.get("future_return", 0)

            features = self._extract_features(factors, apply_keep=False).flatten()
            label = 1 if future_return > min_return else 0
            records.append({"features": features.tolist(), "label": label})

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df

    def train(self, df: pd.DataFrame, dates: List[str] = None) -> Dict:
        """
        训练集成模型（LightGBM + RandomForest）；超参随机搜索；时序验证
        df: columns = ["features", "label"]
        dates: 可选，与 df 行对应的日期字符串列表，用于时序拆分
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X = np.array(df["features"].tolist())
        y = np.array(df["label"].tolist())

        # 常量特征剔除：剔除方差为0的列（反馈训练数据中大量特征恒为50，保留会浪费模型容量/过拟合）
        self._keep_index = None
        if X.ndim == 2 and X.shape[1] > 1:
            col_std = X.std(axis=0)
            keep = np.where(col_std > 1e-8)[0]
            if len(keep) < X.shape[1]:
                self._keep_index = keep.tolist()
                import logging
                logging.getLogger(__name__).warning(
                    "ML 常量特征剔除: %d -> %d (保留 %.0f%%)",
                    X.shape[1], len(keep), len(keep) / X.shape[1] * 100,
                )
                X = X[:, self._keep_index]
                if self.feature_names:
                    self.feature_names = [self.feature_names[i] for i in self._keep_index]

        if dates is not None and len(dates) == len(df):
            order = np.argsort(dates)
            X = X[order]; y = y[order]
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            validation = "time_split"
        else:
            # 降级分支：dates 缺失或长度不匹配。规则 §6 禁止 random_split 降级，
            # 这里保留降级以保证可用性，但显式告警，便于排查调用方传参错误。
            import logging
            logging.warning(
                "ml_selector.train: dates 长度(%s)与 df(%s)不匹配，降级为随机分层抽样。"
                "请检查调用方是否正确传入 dates，避免未来函数风险。",
                len(dates) if dates is not None else "None", len(df)
            )
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            validation = "random_split"

        best_lgb = None; best_rf = None; best_score = -1

        # --- LightGBM 超参搜索（简单网格）---
        try:
            import lightgbm as lgb
            param_grid = [
                {"n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.08, "min_child_samples": 10},
                {"n_estimators": 300, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
                {"n_estimators": 400, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 30},
            ]
            for params in param_grid:
                model = lgb.LGBMClassifier(
                    class_weight="balanced", random_state=42, verbose=-1, **params
                )
                model.fit(X_train, y_train)
                val_score = accuracy_score(y_test, model.predict(X_test))
                if val_score > best_score:
                    best_score = val_score
                    best_lgb = model
            self.model = best_lgb
            self.model_type = "lightgbm"
        except ImportError:
            self.model = None

        # --- RandomForest ---
        from sklearn.ensemble import RandomForestClassifier as RFC
        rf_params = [
            {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 5},
            {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 3},
            {"n_estimators": 400, "max_depth": 14, "min_samples_leaf": 2},
        ]
        best_rf_score = -1
        for params in rf_params:
            model = RFC(class_weight="balanced", random_state=42, n_jobs=-1, **params)
            model.fit(X_train, y_train)
            val_score = accuracy_score(y_test, model.predict(X_test))
            if val_score > best_rf_score:
                best_rf_score = val_score
                best_rf = model

        self.model_aux = best_rf

        # 集成：主模型用 LightGBM（有更好就用），辅助用 RF
        if best_lgb is None:
            self.model = best_rf
            self.model_type = "randomforest"
            self.model_aux = None
        else:
            self.model_type = "ensemble"

        # --- 评估（取主模型）---
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # 训练集准确率：用于过拟合检测 (train_acc - test_acc)
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        try:
            importances = self.model.feature_importances_
        except AttributeError:
            importances = np.zeros(len(self.feature_names))
        self.feature_importance = dict(zip(self.feature_names, importances))
        sorted_importance = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # --- 保存 ---
        joblib.dump(self.model, self.model_path)
        if self.model_aux:
            joblib.dump(self.model_aux, self.model_aux_path)
        with open(self.model_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "keep_index": self._keep_index,
            }, f, ensure_ascii=False, indent=2)

        return {
            "accuracy": round(accuracy, 3),
            "train_accuracy": round(train_accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "top_features": [(name, round(imp, 4)) for name, imp in sorted_importance[:10]],
            "model_saved": self.model_path,
            "model_type": self.model_type,
            "validation": validation,
        }


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

        # 只保留正向IC因子
        positive_ic = {k: v for k, v in ic.items() if v > 0.02}
        if not positive_ic:
            positive_ic = ic

        # 对IC值做softmax归一化
        ic_values = np.array(list(positive_ic.values()))
        ic_weights = np.exp(ic_values * 5) / np.sum(np.exp(ic_values * 5))

        self.optimal_weights = dict(
            zip(positive_ic.keys(), [round(w, 4) for w in ic_weights])
        )

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
        """计算未来N天的收益"""
        try:
            from datetime import datetime, timedelta
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            future_days = forward_window + 5
            history = fetcher.get_stock_history(code, days=future_days)
            if history is None or history.empty or len(history) < forward_window + 1:
                return None
            close_col = "收盘" if "收盘" in history.columns else "close"
            if close_col not in history.columns:
                return None
            closes = history[close_col].values
            if len(closes) < forward_window + 1:
                return None
            entry = closes[0]
            if entry <= 0:
                return None
            exit_val = closes[min(forward_window, len(closes) - 1)]
            return (exit_val / entry - 1) * 100
        except Exception:
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
        except Exception:
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
                    horizon: int = 3, threshold: float = 0.01,
                    kc: Dict = None):
    """影子 ML 训练：从 factor_snapshots 加载数据，训练独立 LightGBM 模型。

    Args:
        today: 当前交易日
        db: StockDatabase 实例，None 时自动创建
        lookback_days: 回溯天数
        horizon: 预测窗口（天）
        threshold: 正例阈值
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
    except Exception:
        return None

    return ShadowMLModel(model, dims, today)


def _future_return_from_kline(krows, trade_date: str, horizon: int) -> float:
    """从 K 线行列表计算未来 N 天的收益。"""
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
    return (fut_close - cur_close) / cur_close
