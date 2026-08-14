"""StockScorer 单元测试（离线、纯函数、无网络）"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.scorer import StockScorer


class TestStockScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = StockScorer()

    def test_sub_scores_count(self):
        subs = self.scorer.get_sub_scores({})
        self.assertEqual(len(subs), 24)
        self.assertIn("moat", subs)

    def test_score_moat_reads_moat_score(self):
        self.assertEqual(self.scorer._score_moat({"moat_score": 87.7}), 87.7)
        self.assertEqual(self.scorer._score_moat({}), 50.0)

    def test_calculate_score_empty_is_neutral(self):
        score = self.scorer.calculate_score({})
        self.assertIsInstance(score, (int, float))
        self.assertTrue(0.0 <= score <= 100.0)
        # 单股版(兼容旧调用)按权重和中心化，空因子接近中性(权重和略>1 → ~52)
        self.assertTrue(48.0 <= score <= 56.0)

    def test_calculate_score_in_range(self):
        for mv in [0, 100, 1000, 10000]:
            score = self.scorer.calculate_score({
                "concept_count": 1, "is_leader": True, "institutional_score": 90,
                "total_mv_yi": mv, "moat_score": 90,
            })
            self.assertTrue(0.0 <= score <= 100.0)

    def test_batch_normalize(self):
        # 12 只等权候选 -> 分数应归一为 ~50
        cands = []
        for _ in range(12):
            cands.append({
                "sub_scores": {k: 50 for k in self.scorer.weights},
                "expectation_penalty": 0,
            })
        StockScorer.batch_normalize(cands, self.scorer.weights)
        for c in cands:
            self.assertIn("score", c)
            self.assertIn("z_scores", c)
            self.assertTrue(0.0 <= c["score"] <= 100.0)
            self.assertAlmostEqual(c["score"], 50.0, delta=0.01)

    def test_batch_normalize_differentiates(self):
        cands = []
        for i in range(12):
            subs = {k: 50 for k in self.scorer.weights}
            if i == 0:
                subs["moat"] = 100  # 让第一只凸显
            cands.append({"sub_scores": subs, "expectation_penalty": 0})
        StockScorer.batch_normalize(cands, self.scorer.weights)
        scores = [c["score"] for c in cands]
        self.assertGreater(scores[0], scores[1])  # 第一只应更高


if __name__ == "__main__":
    unittest.main()
