"""MoatFactors 单元测试（离线、纯函数、无网络）"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.factors.moat import MoatFactors


class TestMoatFactors(unittest.TestCase):
    def setUp(self):
        self.m = MoatFactors()

    def test_return_keys(self):
        out = self.m.calculate_all({"concept_count": 2})
        for k in ["moat_score", "moat_concept_rarity", "moat_leader",
                  "moat_institution", "moat_market_position"]:
            self.assertIn(k, out)

    def test_concept_rarity_table(self):
        f = self.m._concept_rarity
        self.assertEqual(f(0), 50.0)   # 无标签信息缺失 -> 中性
        self.assertEqual(f(1), 90.0)
        self.assertEqual(f(2), 85.0)
        self.assertEqual(f(3), 75.0)
        self.assertEqual(f(4), 60.0)
        self.assertEqual(f(6), 45.0)   # >=5 题材杂毛

    def test_market_position_table(self):
        f = self.m._market_position
        self.assertEqual(f(0), 50.0)
        self.assertEqual(f(10), 55.0)
        self.assertEqual(f(50), 65.0)
        self.assertEqual(f(200), 72.0)
        self.assertEqual(f(1000), 80.0)
        self.assertEqual(f(5000), 82.0)

    def test_strong_moat(self):
        # 龙头 + 单一概念 + 机构重仓 + 大市值
        out = self.m.calculate_all({
            "concept_count": 1, "leader_score": 85, "is_leader": True,
            "institutional_score": 82, "total_mv_yi": 19000,
        })
        # 0.30*90 + 0.30*93 + 0.25*82 + 0.15*82 = 87.7
        self.assertAlmostEqual(out["moat_score"], 87.7, delta=0.5)
        self.assertGreaterEqual(out["moat_score"], 80)

    def test_weak_moat(self):
        out = self.m.calculate_all({
            "concept_count": 6, "leader_score": 50, "is_leader": False,
            "institutional_score": 50, "total_mv_yi": 10,
        })
        # 0.30*45 + 0.30*50 + 0.25*50 + 0.15*55 = 49.25
        self.assertAlmostEqual(out["moat_score"], 49.25, delta=0.5)

    def test_neutral(self):
        # 全缺省 -> 50
        out = self.m.calculate_all({})
        self.assertAlmostEqual(out["moat_score"], 50.0, delta=0.01)

    def test_score_in_range(self):
        for cc in range(0, 12):
            out = self.m.calculate_all({
                "concept_count": cc, "leader_score": 100, "is_leader": True,
                "institutional_score": 100, "total_mv_yi": 5000,
            })
            self.assertTrue(0.0 <= out["moat_score"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
