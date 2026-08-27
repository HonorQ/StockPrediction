# -*- coding: utf-8 -*-
"""美股板块数据采集 + 板块→A股映射表

每天收盘后运行一次，采集美股板块个股快照，存入 cache/us_daily/
"""
import os, sys, json, time
from datetime import datetime
import requests

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"
US_CACHE = os.path.join(ROOT, "cache", "us_daily")
os.makedirs(US_CACHE, exist_ok=True)

# ── 美股板块映射表 ──
# 格式: {板块: [(腾讯代码, 名称, A股对应行业关键词)]}
# 注意: 腾讯代码格式为 s_usXXXX (如 s_usNVDA)
SECTOR_MAP = {
    "半导体": [
        ("s_usNVDA", "英伟达", ["芯片", "半导体", "GPU"]),
        ("s_usAMD", "AMD", ["芯片", "半导体", "CPU"]),
        ("s_usMU", "美光", ["存储", "芯片"]),
        ("s_usAVGO", "博通", ["芯片", "半导体", "网络"]),
        ("s_usINTC", "英特尔", ["芯片", "半导体", "CPU"]),
    ],
    "液冷服务器": [
        ("s_usSMCI", "超微电脑", ["液冷", "服务器", "算力"]),
        ("s_usDELL", "戴尔", ["服务器", "算力"]),
        ("s_usHPE", "慧与", ["服务器", "算力"]),
    ],
    "光通信/CPO": [
        ("s_usCOHR", "Coherent", ["光模块", "CPO", "光通信"]),
    ],
    "电源/电力": [
        ("s_usVRT", "Vertiv", ["电源", "电力设备"]),
        ("s_usETN", "伊顿", ["电源", "电力设备"]),
    ],
    "新能源": [
        ("s_usTSLA", "特斯拉", ["新能源车", "锂电池", "储能"]),
        ("s_usENPH", "Enphase", ["光伏", "储能"]),
        ("s_usFSLR", "First Solar", ["光伏"]),
        ("s_usRIVN", "Rivian", ["新能源车"]),
    ],
    "AI应用": [
        ("s_usMSFT", "微软", ["AI", "软件", "算力"]),
        ("s_usGOOGL", "谷歌", ["AI", "软件"]),
        ("s_usMETA", "Meta", ["AI", "软件"]),
    ],
    "医药": [
        ("s_usABBV", "艾伯维", ["医药", "创新药"]),
        ("s_usLLY", "礼来", ["医药", "创新药"]),
    ],
}

# 腾讯代码→名称映射
CODE_NAME = {}
ALL_CODES = []
for sector, stocks in SECTOR_MAP.items():
    for code, name, _ in stocks:
        CODE_NAME[code] = name
        ALL_CODES.append(code)


def fetch_us_batch(codes):
    """批量拉取美股实时行情"""
    result = {}
    url = "https://qt.gtimg.cn/q=" + ",".join(codes)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    for line in resp.text.strip().split(";"):
        if "=" not in line or "none_match" in line:
            continue
        key = line.split("=")[0].replace("v_", "").strip()
        seg = line.split('"')
        raw = seg[1] if len(seg) > 1 else ""
        if not raw:
            continue
        try:
            if key.startswith("s_us"):
                p = raw.split("~")
                if len(p) >= 6:
                    result[key] = {
                        "price": float(p[3]),
                        "change_pct": float(p[5]),
                        "name": CODE_NAME.get(key, key),
                    }
        except (ValueError, IndexError):
            continue
    return result


def build_sector_returns(data):
    """按板块聚合涨跌幅（等权平均）"""
    sector_returns = {}
    for sector, stocks in SECTOR_MAP.items():
        pcts = []
        for code, name, _ in stocks:
            if code in data:
                pcts.append(data[code]["change_pct"])
        if pcts:
            sector_returns[sector] = round(sum(pcts) / len(pcts), 2)
    return sector_returns


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    all_codes = list(CODE_NAME.keys())

    print(f"=== 美股板块数据采集 {today} ===\n")

    # 拉取数据
    data = fetch_us_batch(ALL_CODES)
    print(f"获取: {len(data)}/{len(ALL_CODES)} 只\n")

    # 个股明细
    print(f"{'名称':<12}{'板块':<10}{'最新价':>10}{'涨跌幅':>10}")
    print("─" * 42)
    for sector, stocks in SECTOR_MAP.items():
        for code, name, _ in stocks:
            if code in data:
                d = data[code]
                print(f"  {d['name']:<10}{sector:<8}{d['price']:>10.2f}{d['change_pct']:>+9.2f}%")

    # 板块聚合
    sector_ret = build_sector_returns(data)
    print(f"\n=== 板块等权平均涨跌幅 ===")
    for sector, ret in sorted(sector_ret.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(abs(ret)))
        sign = "+" if ret > 0 else ""
        print(f"  {sector:<12}{sign}{ret:.2f}%  {'🟢' if ret > 0 else '🔴'} {bar}")

    # 保存
    out = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "stocks": {code: data[code] for code in data if code in data},
        "sector_returns": sector_ret,
    }
    out_path = os.path.join(US_CACHE, f"{today}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    # 统计已有历史
    history = sorted([f for f in os.listdir(US_CACHE) if f.endswith(".json")])
    print(f"历史记录: {len(history)} 天 (最早: {history[0] if history else '无'})")


if __name__ == "__main__":
    main()
