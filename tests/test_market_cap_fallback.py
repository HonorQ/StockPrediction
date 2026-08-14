"""腾讯总市值 fallback 单元测试（离线，mock requests / akshare）"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import src.data.market_data as md


def _fake_resp(text, status_code=200):
    class R:
        pass
    r = R()
    r.status_code = status_code
    r.text = text
    return r


def _gtimg_payload(rows):
    """rows: [(sym, code, name, mv_yi), ...] -> qt.gtimg.cn 风格响应"""
    parts = []
    for sym, code, name, mv_yi in rows:
        p = [""] * 46
        p[1] = name
        p[3] = "100.00"
        p[45] = str(mv_yi)
        parts.append(f'v_{sym}="{"~".join(p)}"')
    return ";".join(parts) + ";"


class TestMarketCapFallback(unittest.TestCase):
    def _fetcher(self):
        f = md.MarketDataFetcher.__new__(md.MarketDataFetcher)
        f._cache = {}
        return f

    def test_parse_market_cap(self):
        f = self._fetcher()
        payload = _gtimg_payload([
            ("sh600519", "600519", "贵州茅台", 16206.56),
            ("sz300750", "300750", "宁德时代", 17807.52),
        ])
        with mock.patch.object(md.requests, "get", return_value=_fake_resp(payload)):
            cap = f._fetch_market_cap_tencent(["600519", "300750"])
        self.assertAlmostEqual(cap["600519"], 16206.56e8, delta=1)
        self.assertAlmostEqual(cap["300750"], 17807.52e8, delta=1)

    def test_sanity_filter_skips_garbage(self):
        f = self._fetcher()
        payload = _gtimg_payload([
            ("sh600519", "600519", "茅台", 0),          # 0 被过滤
            ("sz300750", "300750", "宁德", 999999),      # 超上限被过滤
        ])
        with mock.patch.object(md.requests, "get", return_value=_fake_resp(payload)):
            cap = f._fetch_market_cap_tencent(["600519", "300750"])
        self.assertEqual(cap, {})

    def test_batch_fallback_when_eastmoney_down(self):
        f = self._fetcher()
        df = pd.DataFrame({"代码": ["600519", "300750"], "总市值": [0.0, 0.0]})
        payload = _gtimg_payload([
            ("sh600519", "600519", "贵州茅台", 16206.56),
            ("sz300750", "300750", "宁德时代", 17807.52),
        ])
        # 东财探针与拉取均失败 -> 应走腾讯 fallback
        with mock.patch.object(md.ak, "stock_individual_info_em", side_effect=RuntimeError("东财不可达")), \
             mock.patch.object(md.requests, "get", return_value=_fake_resp(payload)):
            out = f._batch_supplement_market_cap(df)
        self.assertGreater(out.loc[out["代码"] == "600519", "总市值"].iloc[0], 0)
        self.assertGreater(out.loc[out["代码"] == "300750", "总市值"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
