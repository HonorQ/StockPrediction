# -*- coding: utf-8 -*-
"""集合竞价数据源可得性探测（一次性研究工具）

目标：验证 9:25 集合竞价快照的可获取性，为"竞价强度因子"铺路。
探测的候选接口：
  1. 东财 push2 实时行情(含集合竞价价格/量)——9:15-9:25 竞价期间即可见
  2. akshare 分时成交/竞价相关接口
  3. 腾讯 qt.gtimg.cn 实时串(字段含开盘价/开盘量)

用法: python scripts/probe_auction_data.py [code]
示例: python scripts/probe_auction_data.py sh600519
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import requests


def probe_eastmoney(code_clean: str):
    """东财 push2 单票实时：字段 f46(开盘)/f47(成交量)/f50(量比)/f168(换手)"""
    market = "1" if code_clean.startswith(("6", "5", "9")) else "0"
    secid = f"{market}.{code_clean}"
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        # f46 开盘价, f47 成交量(手), f48 成交额, f50 量比, f60 昨收,
        # f168 换手率, f169 涨跌, f170 涨跌幅
        "fields": "f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f107,f168,f169,f170",
        "fltt": "2",
        "invt": "2",
    }
    r = requests.get(url, params=params, timeout=8,
                     headers={"User-Agent": "Mozilla/5.0"})
    d = r.json().get("data") or {}
    if not d:
        return None
    return {
        "source": "eastmoney_push2",
        "name": d.get("f58"),
        "open": d.get("f46"),
        "volume_hands": d.get("f47"),
        "amount": d.get("f48"),
        "volume_ratio": d.get("f50"),      # 量比: 竞价强度的核心字段
        "pre_close": d.get("f60"),
        "turnover": d.get("f168"),
        "change_pct": d.get("f170"),
        "note": "9:20-9:25 间轮询可记录竞价撮合演变; 9:26 后为最终竞价结果",
    }


def probe_tencent(code_clean: str):
    """腾讯实时串：v_sz000001=... 字段含开盘价/指数等"""
    prefix = ("sh" if code_clean.startswith(("6", "5", "9"))
              else ("bj" if code_clean.startswith(("4", "8")) else "sz"))
    url = f"https://qt.gtimg.cn/q={prefix}{code_clean}"
    r = requests.get(url, timeout=8)
    txt = r.text
    if "=" not in txt:
        return None
    parts = txt.split("~")
    if len(parts) < 40:
        return None
    return {
        "source": "tencent_qt",
        "name": parts[1],
        "open": float(parts[5] or 0),
        "pre_close": float(parts[4] or 0),
        "volume_hands": float(parts[6] or 0),
        "note": "字段5=开盘 字段4=昨收; 竞价阶段 open 为当时虚拟撮合价",
    }


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    print(f"=== 集合竞价数据源探测: {code} ===\n")
    results = []
    for fn in (probe_eastmoney, probe_tencent):
        try:
            r = fn(code)
            status = "OK" if r else "EMPTY"
        except Exception as e:
            r = {"error": str(e)}
            status = "FAIL"
        results.append(r)
        print(f"[{status}] {fn.__name__}")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print()

    ok = [r for r in results if r and not r.get("error")]
    print("=== 结论 ===")
    print(f"可用源: {len(ok)}/2")
    print("因子化建议: 盘中模式(09:25)抓取 {量比, 开盘涨幅, 开盘量/5日均量}")
    print("  → 竞价抢筹(量比>3 且 高开0~3%) 历史上是短线强势信号;")
    print("  → 高开>5% 则是诱多高发区。需积累样本后用IC验证方向。")

    out = os.path.join(ROOT, "reports", "auction_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"code": code, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
