"""MLStockSelector 单元测试（离线、纯函数、无网络）"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.ml_selector import MLStockSelector


def _tmp_model_path(tag):
    return os.path.join(tempfile.gettempdir(), "test_ml_model_%s.pkl" % tag)


def _make_df(n=200, seed=42):
    import pandas as pd
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        feats = (rng.rand(45) * 100).tolist()
        label = int(feats[0] + rng.randn() * 10 > 50)
        rows.append({"features": feats, "label": label})
    return pd.DataFrame(rows)


class TestPrepareTrainingData(unittest.TestCase):
    def setUp(self):
        self.sel = MLStockSelector(model_path=_tmp_model_path("prep"))

    def test_auto_threshold_scales_with_distribution(self):
        data = [
            {"factors": {}, "future_return": r}
            for r in [3.0, -4.0, 5.5, -2.5, 6.0, -7.0, 4.2, 1.9, -3.3, 8.1]
        ]
        df = self.sel.prepare_training_data(data, min_return="auto")
        self.assertFalse(df.empty)
        # 中位|ret|=3.65 → 仅 5.5/6.0/4.2/8.1 为正例
        pos = int(df["label"].sum())
        self.assertEqual(pos, 4)

    def test_numeric_threshold_keeps_legacy_behavior(self):
        data = [{"factors": {}, "future_return": 2.0},
                {"factors": {}, "future_return": -0.5}]
        # 百分数口径: 1.0 = 1% (与 production feedback_loop/run_batch_training 一致)
        df = self.sel.prepare_training_data(data, min_return=1.0)
        self.assertEqual(list(df["label"]), [1, 0])

    def test_missing_returns_dropped_not_labeled_zero(self):
        data = [{"factors": {}, "future_return": 2.0},
                {"factors": {}, "future_return": None},
                {"factors": {}}]
        # 百分数口径: 1.0 = 1%
        df = self.sel.prepare_training_data(data, min_return=1.0)
        self.assertEqual(len(df), 1)


class TestTrainSplit(unittest.TestCase):
    def test_time_split_3way_no_overlap(self):
        sel = MLStockSelector(model_path=_tmp_model_path("split"))
        n = 200
        dates = ["2026-%02d-%02d" % (1 + i // 28, 1 + i % 28) for i in range(n)]
        m = sel.train(_make_df(n), dates=dates, label_horizon=3)
        self.assertEqual(m["validation"], "time_split_3way")
        # t_end=120, v_start=123(embargo 3), v_end=160, t_start=163
        self.assertEqual(m["train_samples"], 120)
        self.assertEqual(m["val_samples"], 37)
        self.assertEqual(m["test_samples"], 37)

    def test_small_sample_falls_back_to_2way(self):
        sel = MLStockSelector(model_path=_tmp_model_path("small"))
        n = 40
        dates = ["2026-01-%02d" % (1 + i % 28) for i in range(n)]
        m = sel.train(_make_df(n), dates=dates, label_horizon=3)
        self.assertEqual(m["validation"], "time_split_2way")

    def test_horizon_changes_embargo_size(self):
        sel = MLStockSelector(model_path=_tmp_model_path("horizon"))
        n = 200
        dates = ["2026-%02d-%02d" % (1 + i // 28, 1 + i % 28) for i in range(n)]
        m = sel.train(_make_df(n), dates=dates, label_horizon=10)
        # t_end=120, v_start=130 → val=30; v_end=160, t_start=170 → test=30
        self.assertEqual(m["val_samples"], 30)
        self.assertEqual(m["test_samples"], 30)


if __name__ == "__main__":
    unittest.main()
