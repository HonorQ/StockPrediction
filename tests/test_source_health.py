"""source_health 单元测试（离线，mock requests）"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.data.source_health as sh


class FakeResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class TestSourceHealth(unittest.TestCase):
    def test_check_eastmoney_ok(self):
        with mock.patch.object(sh.requests, "get", return_value=FakeResp('{"data":1}')):
            self.assertTrue(sh.check_eastmoney())

    def test_check_tencent_ok(self):
        with mock.patch.object(sh.requests, "get", return_value=FakeResp("贵州茅台")):
            self.assertTrue(sh.check_tencent())

    def test_check_sina_ok(self):
        with mock.patch.object(sh.requests, "get", return_value=FakeResp("600519")):
            self.assertTrue(sh.check_sina())

    def test_all_down_returns_false(self):
        with mock.patch.object(sh.requests, "get", side_effect=RuntimeError("网络不通")):
            self.assertFalse(sh.check_eastmoney())
            self.assertFalse(sh.check_tencent())
            self.assertFalse(sh.check_sina())

    def test_health_check_keys(self):
        with mock.patch.object(sh.requests, "get", return_value=FakeResp("ok")):
            status = sh.health_check()
        # 2026-09-11 新增 fomc_calendar(事件日历自动维护源)
        self.assertEqual(
            set(status.keys()), {"eastmoney", "tencent", "sina", "fomc_calendar"})

    def test_health_check_no_raise_on_network_error(self):
        with mock.patch.object(sh.requests, "get", side_effect=RuntimeError):
            status = sh.health_check()
        self.assertEqual(status, {
            "eastmoney": False, "tencent": False, "sina": False,
            "fomc_calendar": False})

    def test_summarize_marks(self):
        s = sh.summarize({"eastmoney": True, "tencent": False, "sina": True})
        self.assertIn("数据源健康", s)
        self.assertIn("✅", s)
        self.assertIn("❌", s)


if __name__ == "__main__":
    unittest.main()
