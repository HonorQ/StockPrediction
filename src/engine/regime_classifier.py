"""
市场状态ML分类器（长期方案）

使用机器学习预测市场状态（上涨/下跌/震荡），动态切换策略模式。

注意：当前数据量不足（仅8个月），此模块为框架实现，待数据积累后完善。
"""
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """市场状态ML分类器"""

    # 状态定义
    REGIMES = ["bull", "neutral", "bear", "crash"]

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(ROOT, "models", "regime_classifier.pkl")
        self.model = None
        self.feature_names = []
        self._try_load_model()

    def _try_load_model(self):
        """尝试加载已训练的模型"""
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                meta_path = self.model_path + ".meta.json"
                if os.path.exists(meta_path):
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])
            except Exception as e:
                logger.warning(f"加载模型失败: {e}")
                self.model = None

    def extract_features(self, market_data: Dict) -> np.ndarray:
        """
        从市场数据提取特征
        
        Args:
            market_data: 包含以下字段的字典
                - index_change: 主指数涨跌幅
                - limit_up: 涨停家数
                - limit_down: 跌停家数
                - broken: 炸板家数
                - volume_ratio: 成交量比率
                - volatility: 波动率
                - northbound_flow: 北向资金流入
                - ma20_position: 相对MA20位置
        """
        features = [
            market_data.get("index_change", 0),
            market_data.get("limit_up", 0),
            market_data.get("limit_down", 0),
            market_data.get("broken", 0),
            market_data.get("volume_ratio", 1),
            market_data.get("volatility", 0),
            market_data.get("northbound_flow", 0),
            market_data.get("ma20_position", 0),
            # 交互特征
            market_data.get("limit_up", 0) - market_data.get("limit_down", 0),
            market_data.get("volume_ratio", 1) * market_data.get("volatility", 0),
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, market_data: Dict) -> Dict:
        """
        预测市场状态
        
        Returns:
            {
                "regime": "bull|neutral|bear|crash",
                "confidence": float,
                "probabilities": Dict[str, float]
            }
        """
        if self.model is None:
            return self._fallback_predict(market_data)

        features = self.extract_features(market_data)
        try:
            proba = self.model.predict_proba(features)[0]
            # 用 model.classes_ 映射类别：训练集缺某类样本(如 crash)时
            # 模型只有部分类，argmax 直索引 REGIMES 会张冠李戴
            classes = list(getattr(self.model, "classes_", self.REGIMES))
            regime_idx = int(np.argmax(proba))
            regime = classes[regime_idx] if regime_idx < len(classes) else "neutral"
            confidence = proba[regime_idx]
            probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
            return {
                "regime": regime,
                "confidence": float(confidence),
                "probabilities": probabilities,
            }
        except Exception as e:
            logger.warning(f"ML预测失败，使用规则预测: {e}")
            return self._fallback_predict(market_data)

    def _fallback_predict(self, market_data: Dict) -> Dict:
        """备用预测（基于规则）"""
        index_change = market_data.get("index_change", 0)
        limit_down = market_data.get("limit_down", 0)

        if index_change <= -2 or limit_down >= 30:
            regime = "crash"
        elif index_change <= -1 or limit_down >= 15:
            regime = "bear"
        elif index_change >= 1:
            regime = "bull"
        else:
            regime = "neutral"

        return {
            "regime": regime,
            "confidence": 0.5,
            "probabilities": {r: 0.25 for r in self.REGIMES},
        }

    def train(self, historical_data: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        训练模型
        
        Args:
            historical_data: 历史市场数据 DataFrame
            labels: 对应的市场状态标签 (0=bull, 1=neutral, 2=bear, 3=crash)
        """
        if len(historical_data) < 100:
            return {"status": "pending", "reason": "insufficient_data", "samples": len(historical_data)}
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            X = historical_data.values
            y = labels.values

            # 时序切分（市场状态强自相关，随机切分会让 test_accuracy 严重虚高）：
            # 前80%训练 / 后20%测试，并在边界跳过5个样本(embargo)防止标签窗口重叠泄漏
            n = len(X)
            split = int(n * 0.8)
            embargo = 5
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split + embargo:], y[split + embargo:]
            if len(X_test) == 0:
                return {"status": "pending", "reason": "test_split_empty_after_embargo", "samples": n}

            # 训练模型
            model = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                class_weight="balanced", random_state=42
            )
            model.fit(X_train, y_train)

            # 评估
            from sklearn.metrics import accuracy_score
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            # 保存模型
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)
            self.model = model
            
            # 保存元数据
            import json
            meta_path = self.model_path + ".meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "feature_names": self.feature_names,
                    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                }, f, ensure_ascii=False, indent=2)
            
            return {
                "status": "success",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        except Exception as e:
            logger.warning("捕获异常已忽略(src/engine/regime_classifier.py:196): %s", e)
            return {"status": "error", "error": str(e)}
