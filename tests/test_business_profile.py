"""BusinessProfileFactor 单元测试（离线、规则抽取、无网络）"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.factors.business_profile import BusinessProfileFactor


class TestBusinessProfile(unittest.TestCase):
    def setUp(self):
        self.bp = BusinessProfileFactor()

    def test_maotai(self):
        out = self.bp.build("600519", "贵州茅台", {
            "industry": "白酒", "hot_concepts": ["白酒", "消费"],
            "concept_count": 2, "is_leader": True, "institutional_score": 82,
            "total_mv_yi": 19000, "moat_score": 85,
        })
        self.assertEqual(out["industry"], "白酒")
        self.assertIn("消费", out["themes"])
        self.assertIn("行业龙头", out["moat_labels"])
        self.assertIn("机构重仓", out["moat_labels"])
        self.assertIn("业务聚焦", out["moat_labels"])      # 2 个概念
        self.assertIn("大市值权重", out["moat_labels"])
        self.assertEqual(out["moat_level"], "强护城河")
        self.assertAlmostEqual(out["moat_score"], 85.0, delta=0.01)

    def test_ningde_no_focus(self):
        out = self.bp.build("300750", "宁德时代", {
            "industry": "电池", "hot_concepts": ["新能源车", "锂电池", "储能"],
            "concept_count": 6, "is_leader": True, "institutional_score": 75,
            "total_mv_yi": 11000, "moat_score": 72,
        })
        self.assertIn("新能源", out["themes"])
        self.assertIn("新能源车", out["themes"])            # 细分子主题
        self.assertIn("行业龙头", out["moat_labels"])
        self.assertNotIn("业务聚焦", out["moat_labels"])   # 6 个概念不应标聚焦
        self.assertEqual(out["moat_level"], "强护城河")

    def test_zero_concept_no_focus(self):
        # 概念数为 0 时不应误标"业务聚焦"
        out = self.bp.build("999999", "无名氏", {
            "industry": "未知", "hot_concepts": [], "concept_count": 0,
            "is_leader": False, "institutional_score": 50, "total_mv_yi": 0,
            "moat_score": 50,
        })
        self.assertEqual(out["industry"], "未知")
        self.assertEqual(out["themes"], [])
        self.assertNotIn("业务聚焦", out["moat_labels"])
        self.assertEqual(out["moat_level"], "护城河一般")
        self.assertIn("未知", out["business_summary"])

    def test_no_crash_on_missing_fields(self):
        out = self.bp.build("000001", "测试", {})
        self.assertIsInstance(out, dict)
        self.assertIn("moat_labels", out)
        self.assertIn("themes", out)


if __name__ == "__main__":
    unittest.main()
