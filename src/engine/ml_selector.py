"""
第三阶段：机器学习选股模型 + 多因子组合优化
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import os
import json
import joblib
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


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
        """启动时尝试加载已存在的模型（主模型 + 辅助模型）。

        加载前先查 meta 的质量门禁: 劣于随机的历史模型(门禁上线前训练并落盘的)
        直接按"无模型"处理, 否则它会持续以 5% 权重给推荐引入噪声。
        predict 在 model=None 时返回中性 50, 调用方(午后模式)会自动跳过 ML 微调。
        """
        if os.path.exists(self.model_path):
            try:
                meta = {}
                if os.path.exists(self.model_meta_path):
                    with open(self.model_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    auc = (meta.get("metrics") or {}).get("test_auc")
                    if auc is not None and float(auc) < self.MIN_TEST_AUC:
                        logger.warning(
                            "线上 ML 模型未达质量门禁(test_auc=%s < %.2f), 按无模型处理"
                            "(ML 分数取中性50, 不参与打分)", auc, self.MIN_TEST_AUC)
                        return
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.model_aux_path):
                    self.model_aux = joblib.load(self.model_aux_path)
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
            except Exception as e:
                logger.warning(f"加载模型失败: {e}")
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
        except Exception as e:
            logger.warning(f"ML预测失败: {e}")
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
        self, history_data: List[Dict], min_return="auto"
    ) -> pd.DataFrame:
        """
        准备训练数据（不含横截面排名特征，避免训练/预测不一致）

        标签 = future_return > 阈值。
        - min_return="auto": 按当期收益分布自适应阈值 = max(0.5, median|future_return|)（百分数口径），
          使正例标准随波动环境缩放（固定1%阈值在牛市几乎全是正例、熊市几乎全是负例，
          标签分布随 regime 漂移会损害跨期稳定性）
        - 传数值则按百分数口径使用（1.0 = 1%）
        future_return 缺失/非数值的样本直接丢弃（此前会被静默打成 label=0，污染负类）。
        """
        import logging

        if not history_data:
            return pd.DataFrame()

        def _ret_of(item):
            r = item.get("future_return")
            try:
                r = float(r)
            except (TypeError, ValueError) as e:
                logger.warning("捕获异常已忽略(src/engine/ml_selector.py:181): %s", e)
                return None
            return r if np.isfinite(r) else None

        if isinstance(min_return, str):
            all_rets = []
            for item in history_data:
                r = _ret_of(item)
                if r is not None:
                    all_rets.append(r)
            if not all_rets:
                return pd.DataFrame()
            # 下限 0.5 个百分点（百分数口径），避免低波动期阈值过小
            min_return = round(max(0.5, float(np.median(np.abs(all_rets)))), 4)
            logging.getLogger(__name__).info(
                "prepare_training_data: 自适应标签阈值 %.4f (有效收益样本 %d)",
                min_return, len(all_rets),
            )

        records = []
        for item in history_data:
            factors = item.get("factors", {})
            r = _ret_of(item)
            if r is None:
                continue
            features = self._extract_features(factors, apply_keep=False).flatten()
            label = 1 if r > min_return else 0
            records.append({"features": features.tolist(), "label": label})

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df

    def train(self, df: pd.DataFrame, dates: List[str] = None,
              label_horizon: int = 3) -> Dict:
        """
        训练集成模型（LightGBM + RandomForest）；超参随机搜索；时序验证
        df: columns = ["features", "label"]
        dates: 可选，与 df 行对应的日期字符串列表，用于时序拆分
        label_horizon: 标签收益窗口(天)，用于切分点 embargo——T日标签用到T+N价格，
            train/val、val/test 边界需跳过 N 个样本，防止训练段尾部标签偷看验证段价格

        内部拆为三段: _train_prepare(时序切分/embargo/特征筛选)
        → _train_search(LightGBM+RF 超参搜索) → _train_evaluate(终评+版本保存)；
        中间状态经 self._train_state dict 传递。
        """
        self._train_state = {}
        self._train_prepare(df, dates, label_horizon)
        self._train_search()
        return self._train_evaluate()

    def _train_prepare(self, df: pd.DataFrame, dates: List[str],
                       label_horizon: int):
        """阶段1: 特征矩阵构建 + 时序切分(embargo 防标签泄漏) + 常量特征剔除。

        切分结果写入 self._train_state:
        X_train/X_val/X_test/y_train/y_val/y_test/validation。
        """
        X = np.array(df["features"].tolist())
        y = np.array(df["label"].tolist())
        n = len(X)
        gap = max(int(label_horizon), 0)

        if dates is not None and len(dates) == len(df):
            order = np.argsort(dates)
            X = X[order]; y = y[order]
            if n >= 60:
                # 三段时序切分: 前60%训练 / 中20%验证(超参选择) / 后20%测试(仅终评)
                # 切分点跳过 label_horizon 个样本(embargo)，消除标签窗口重叠泄漏；
                # 测试集不参与模型选择，metrics 中的 test 指标才是无偏估计
                t_end = int(n * 0.6)
                v_end = int(n * 0.8)
                v_start, t_start = min(t_end + gap, v_end), min(v_end + gap, n)
                X_train, X_val, X_test = X[:t_end], X[v_start:v_end], X[t_start:]
                y_train, y_val, y_test = y[:t_end], y[v_start:v_end], y[t_start:]
                if len(X_val) == 0 or len(X_test) == 0:
                    # embargo 后验证/测试段为空(样本过少) → 放弃 embargo 保证可用性
                    X_val, X_test = X[t_end:v_end], X[v_end:]
                    y_val, y_test = y[t_end:v_end], y[v_end:]
                validation = "time_split_3way"
            else:
                # 样本不足以三段: 退化为二段, 验证集与测试集共用
                # (超参选择接触测试集, test 指标会轻微乐观, 已通过 validation 字段标注)
                split_idx = int(n * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                X_val, y_val = X_test, y_test
                validation = "time_split_2way"
        else:
            # 降级分支：dates 缺失或长度不匹配。
            # 按行序切分(数据通常已按时间升序)替代原随机分层抽样——
            # 时序样本相邻高度自相关，随机切分会让测试指标严重虚高；
            # 同样加 label_horizon 的 embargo 防标签窗口重叠
            import logging
            logging.warning(
                "ml_selector.train: dates 缺失/长度不匹配(%s vs %s)，按行序切分降级。"
                "请检查调用方是否正确传入 dates。",
                len(dates) if dates is not None else "None", len(df)
            )
            t_end = int(n * 0.6)
            v_end = int(n * 0.8)
            v_start, t_start = min(t_end + gap, v_end), min(v_end + gap, n)
            X_train, y_train = X[:t_end], y[:t_end]
            X_val, y_val = X[v_start:v_end], y[v_start:v_end]
            X_test, y_test = X[t_start:], y[t_start:]
            if len(X_val) == 0 or len(X_test) == 0:
                X_val, X_test = X[t_end:v_end], X[v_end:]
                y_val, y_test = y[t_end:v_end], y[v_end:]
            validation = "row_order_split_3way"

        # 常量特征剔除：基于【训练段】方差决定保留哪些列。
        # 原实现在切分前用全量X(含测试段)方差——测试段信息轻微泄漏进特征选择；
        # 反馈训练数据中大量特征恒为50，保留会浪费模型容量/过拟合
        self._keep_index = None
        if X_train.ndim == 2 and X_train.shape[1] > 1:
            col_std = X_train.std(axis=0)
            keep = np.where(col_std > 1e-8)[0]
            if len(keep) < X.shape[1]:
                self._keep_index = keep.tolist()
                import logging
                logging.getLogger(__name__).warning(
                    "ML 常量特征剔除(仅按训练段): %d -> %d (保留 %.0f%%)",
                    X.shape[1], len(keep), len(keep) / X.shape[1] * 100,
                )
                X_train = X_train[:, self._keep_index]
                X_val = X_val[:, self._keep_index]
                X_test = X_test[:, self._keep_index]
                if self.feature_names:
                    self.feature_names = [self.feature_names[i] for i in self._keep_index]

        self._train_state.update(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            validation=validation,
        )

    def _train_search(self):
        """阶段2: LightGBM + RandomForest 超参随机搜索（验证集 AUC 选模）。"""
        from sklearn.metrics import accuracy_score
        st = self._train_state
        X_train, y_train = st["X_train"], st["y_train"]
        X_val, y_val = st["X_val"], st["y_val"]

        def _val_auc(model):
            """验证集 AUC 选模：class_weight=balanced 说明类别不平衡，
            accuracy 在多数类上虚高；AUC 对阈值不敏感更能反映排序能力"""
            from sklearn.metrics import roc_auc_score
            proba = model.predict_proba(X_val)[:, 1]
            if len(set(y_val)) < 2:
                return accuracy_score(y_val, model.predict(X_val))
            return roc_auc_score(y_val, proba)

        best_lgb = None; best_rf = None; best_score = -1

        # --- LightGBM 超参搜索（带正则化防过拟合）---
        try:
            import lightgbm as lgb
            param_grid = [
                {"n_estimators": 100, "max_depth": 3, "num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 20,
                 "reg_alpha": 0.1, "reg_lambda": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
                {"n_estimators": 150, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 30,
                 "reg_alpha": 0.5, "reg_lambda": 0.5, "subsample": 0.7, "colsample_bytree": 0.7},
                {"n_estimators": 200, "max_depth": 5, "num_leaves": 20, "learning_rate": 0.02, "min_child_samples": 40,
                 "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.6, "colsample_bytree": 0.6},
            ]
            for params in param_grid:
                model = lgb.LGBMClassifier(
                    class_weight="balanced", random_state=42, verbose=-1, **params
                )
                model.fit(X_train, y_train)
                val_score = _val_auc(model)
                if val_score > best_score:
                    best_score = val_score
                    best_lgb = model
            self.model = best_lgb
            self.model_type = "lightgbm"
        except ImportError as e:
            logger.warning("捕获异常已忽略(src/engine/ml_selector.py:359): %s", e)
            self.model = None

        # --- RandomForest（限制复杂度）---
        from sklearn.ensemble import RandomForestClassifier as RFC
        rf_params = [
            {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 10},
            {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 15},
            {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20},
        ]
        best_rf_score = -1
        for params in rf_params:
            model = RFC(class_weight="balanced", random_state=42, n_jobs=-1, **params)
            model.fit(X_train, y_train)
            val_score = _val_auc(model)
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

    def _train_evaluate(self) -> Dict:
        """阶段3: 测试集评估 + feature_importance + 版本化保存，返回 metrics。"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        st = self._train_state
        X_train, X_val, X_test = st["X_train"], st["X_val"], st["X_test"]
        y_train, y_test = st["y_train"], st["y_test"]

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
        except AttributeError as e:
            logger.warning("捕获异常已忽略(src/engine/ml_selector.py:407): %s", e)
            importances = np.zeros(len(self.feature_names))
        self.feature_importance = dict(zip(self.feature_names, importances))
        sorted_importance = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # 测试集 AUC: accuracy 在类别不平衡下不可比(选模阶段即用 AUC)，
        # 上线门禁与劣化诊断都以 AUC 为准；无法计算时置 None
        test_auc = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(y_test)) > 1 and hasattr(self.model, "predict_proba"):
                test_auc = round(float(roc_auc_score(
                    y_test, self.model.predict_proba(X_test)[:, 1])), 4)
        except Exception as e:
            logger.warning("测试集 AUC 计算失败(忽略): %s", e)

        # --- 保存（版本化：保留最近 N 版，模型劣化时可回滚对比）---
        metrics = {
            "accuracy": round(accuracy, 3),
            "train_accuracy": round(train_accuracy, 3),
            "test_auc": test_auc,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "top_features": [(name, round(imp, 4)) for name, imp in sorted_importance[:10]],
            "model_saved": self.model_path,
            "model_type": self.model_type,
            "validation": st["validation"],
        }
        self._save_model_versioned(metrics)
        return metrics

    MODEL_HISTORY_KEEP = 5
    # 上线门禁: 测试集 AUC 不得低于随机基线。劣于随机的模型在午后模式按 5% 权重
    # 参与打分, 只会给推荐引入噪声(2026-09-14 实测 train_acc=0.81 / test_acc=0.413)。
    MIN_TEST_AUC = 0.5

    def _save_model_versioned(self, metrics: Dict):
        """保存当前模型并归档历史版本(models/history/)，
        meta.json 记录训练指标供劣化诊断与回滚对比。

        质量门禁不通过时不覆盖线上模型: 只把本次指标存到 history/rejected_*.json
        供诊断, 线上继续用旧模型(无旧模型时 ML 分数按中性 50 处理)。
        """
        import glob
        import shutil
        auc = metrics.get("test_auc")
        if auc is not None and auc < self.MIN_TEST_AUC:
            metrics["rejected"] = True
            metrics["reject_reason"] = f"test_auc={auc} < {self.MIN_TEST_AUC}"
            logger.warning(
                "ML 模型未通过质量门禁(test_auc=%s < %.2f), 保留线上模型不覆盖: %s",
                auc, self.MIN_TEST_AUC, self.model_path)
            self._record_rejection(metrics)
            return
        # 归档上一版
        if os.path.exists(self.model_path):
            hist_dir = os.path.join(os.path.dirname(self.model_path), "history")
            os.makedirs(hist_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(self.model_path, os.path.join(hist_dir, f"stock_selector_{stamp}.pkl"))
            old_meta = self.model_meta_path
            meta_dst = os.path.join(hist_dir, f"stock_selector_{stamp}.meta.json")
            if os.path.exists(old_meta):
                shutil.copy2(old_meta, meta_dst)
            else:
                json.dump(metrics, open(meta_dst, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2, default=str)
            # 只保留最近 N 版
            archives = sorted(glob.glob(os.path.join(hist_dir, "stock_selector_*.pkl")))
            for old in archives[:max(0, len(archives) - self.MODEL_HISTORY_KEEP)]:
                try:
                    os.remove(old)
                    m = old.replace(".pkl", ".meta.json")
                    if os.path.exists(m):
                        os.remove(m)
                except OSError as e:
                    logger.warning("捕获异常已忽略(src/engine/ml_selector.py:460): %s", e)
                    pass

        joblib.dump(self.model, self.model_path)
        if self.model_aux:
            joblib.dump(self.model_aux, self.model_aux_path)
        with open(self.model_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "keep_index": self._keep_index,
                "metrics": metrics,
                # metrics 含 numpy int32/float32, 无 default 会抛
                # "Object of type int32 is not JSON serializable" 导致整轮重训练结果丢失
            }, f, ensure_ascii=False, indent=2, default=str)

    def _record_rejection(self, metrics: Dict):
        """记录被门禁拒绝的训练指标(models/history/rejected_*.json), 供劣化诊断。"""
        try:
            hist_dir = os.path.join(os.path.dirname(self.model_path), "history")
            os.makedirs(hist_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(hist_dir, f"rejected_{stamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        except OSError as e:
            logger.warning("门禁拒绝记录写入失败(忽略): %s", e)



# 兼容 re-export: FactorOptimizer/ShadowMLModel/train_shadow_ml/_future_return_from_kline 已拆至 factor_optimizer.py
from src.engine.factor_optimizer import (  # noqa: F401,E402
    FactorOptimizer, ShadowMLModel, train_shadow_ml, _future_return_from_kline,
)
