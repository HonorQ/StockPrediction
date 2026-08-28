# -*- coding: utf-8 -*-
"""美股→A股供应链映射表

基于公开信息整理: 互动平台回复、券商研报、公司公告。
部分关系有保密协议无法公开确认, 标注为"潜在"。
"""
import json, os

ROOT = r"C:\Users\EDY\Desktop\StockPrediction"

# 美股公司 → A股供应商
# 关联度: confirmed(已确认合作) / potential(潜在/送样中) / rumor(传闻)
SUPPLY_CHAIN = {
    "NVDA": {
        "name": "英伟达",
        "a_share_suppliers": [
            {"code": "sh603186", "name": "华正新材", "link": "PCB覆铜板", "degree": "potential"},
            {"code": "sz002463", "name": "沪电股份", "link": "高端PCB(算力板)", "degree": "confirmed"},
            {"code": "sz300476", "name": "胜宏科技", "link": "GPU显卡板/服务器板", "degree": "confirmed"},
            {"code": "sz002916", "name": "深南电路", "link": "PCB/IC载板", "degree": "confirmed"},
            {"code": "sz002036", "name": "联创电子", "link": "光学镜头(机器人)", "degree": "potential"},
            {"code": "sh600584", "name": "长电科技", "link": "芯片封测(CoWoS)", "degree": "confirmed"},
            {"code": "sz002185", "name": "华天科技", "link": "芯片封测", "degree": "confirmed"},
            {"code": "sz002049", "name": "紫光国微", "link": "特种芯片", "degree": "potential"},
            {"code": "sz300458", "name": "全志科技", "link": "AI芯片设计", "degree": "potential"},
            {"code": "sh688256", "name": "寒武纪", "link": "AI芯片(国产替代)", "degree": "potential"},
            {"code": "sh688041", "name": "海光信息", "link": "GPU(国产替代)", "degree": "potential"},
            {"code": "sz002241", "name": "歌尔股份", "link": "XR设备代工", "degree": "confirmed"},
            {"code": "sz300433", "name": "蓝思科技", "link": "AI眼镜结构件", "degree": "confirmed"},
            {"code": "sz002475", "name": "立讯精密", "link": "AI服务器连接器/光模块", "degree": "confirmed"},
            {"code": "sz300308", "name": "中际旭创", "link": "800G光模块", "degree": "confirmed"},
            {"code": "sh600487", "name": "亨通光电", "link": "光纤光缆", "degree": "confirmed"},
        ],
    },
    "AAPL": {
        "name": "苹果",
        "a_share_suppliers": [
            {"code": "sz002475", "name": "立讯精密", "link": "AirPods/iPhone/Vision Pro", "degree": "confirmed"},
            {"code": "sz002241", "name": "歌尔股份", "link": "AirPods/声学器件", "degree": "confirmed"},
            {"code": "sz300433", "name": "蓝思科技", "link": "玻璃/金属结构件", "degree": "confirmed"},
            {"code": "sz002600", "name": "领益智造", "link": "精密功能件", "degree": "confirmed"},
            {"code": "sz300136", "name": "信维通信", "link": "天线/无线充电", "degree": "confirmed"},
            {"code": "sz002456", "name": "欧菲光", "link": "触控/光学", "degree": "confirmed"},
            {"code": "sz002920", "name": "德赛西威", "link": "智能座舱", "degree": "potential"},
            {"code": "sh603501", "name": "韦尔股份", "link": "CIS芯片", "degree": "confirmed"},
            {"code": "sz300750", "name": "宁德时代", "link": "电池(苹果汽车传闻)", "degree": "rumor"},
            {"code": "sz002415", "name": "海康威视", "link": "AI视觉", "degree": "potential"},
        ],
    },
    "TSLA": {
        "name": "特斯拉",
        "a_share_suppliers": [
            {"code": "sz002050", "name": "三花智控", "link": "旋转关节总成", "degree": "confirmed"},
            {"code": "sh601689", "name": "拓普集团", "link": "线性/旋转执行器", "degree": "confirmed"},
            {"code": "sh688017", "name": "绿的谐波", "link": "谐波减速器", "degree": "confirmed"},
            {"code": "sh603667", "name": "五洲新春", "link": "行星滚柱丝杠", "degree": "confirmed"},
            {"code": "sh603305", "name": "旭升集团", "link": "关节镁合金壳体", "degree": "confirmed"},
            {"code": "sz300115", "name": "长盈精密", "link": "关节齿轮/轴承/连接器", "degree": "confirmed"},
            {"code": "sz300750", "name": "宁德时代", "link": "动力电池", "degree": "confirmed"},
            {"code": "sz300124", "name": "汇川技术", "link": "电驱系统", "degree": "confirmed"},
            {"code": "sz002747", "name": "埃斯顿", "link": "伺服系统", "degree": "confirmed"},
            {"code": "sz300024", "name": "机器人", "link": "工业机器人", "degree": "potential"},
        ],
    },
    "AMD": {
        "name": "AMD",
        "a_share_suppliers": [
            {"code": "sz002185", "name": "华天科技", "link": "芯片封测", "degree": "confirmed"},
            {"code": "sh600584", "name": "长电科技", "link": "芯片封测", "degree": "confirmed"},
            {"code": "sz002049", "name": "紫光国微", "link": "特种芯片", "degree": "potential"},
            {"code": "sz300458", "name": "全志科技", "link": "芯片设计", "degree": "potential"},
        ],
    },
    "MSFT": {
        "name": "微软",
        "a_share_suppliers": [
            {"code": "sz002475", "name": "立讯精密", "link": "Surface代工", "degree": "confirmed"},
            {"code": "sh603019", "name": "中科曙光", "link": "服务器/算力", "degree": "confirmed"},
            {"code": "sz000977", "name": "浪潮信息", "link": "AI服务器", "degree": "confirmed"},
            {"code": "sz002230", "name": "科大讯飞", "link": "AI应用", "degree": "potential"},
        ],
    },
    "AMZN": {
        "name": "亚马逊",
        "a_share_suppliers": [
            {"code": "sz002475", "name": "立讯精密", "link": "连接器/服务器", "degree": "confirmed"},
            {"code": "sh603019", "name": "中科曙光", "link": "服务器", "degree": "confirmed"},
            {"code": "sz000977", "name": "浪潮信息", "link": "AI服务器", "degree": "confirmed"},
        ],
    },
    "META": {
        "name": "Meta",
        "a_share_suppliers": [
            {"code": "sz002241", "name": "歌尔股份", "link": "Quest代工", "degree": "confirmed"},
            {"code": "sz300433", "name": "蓝思科技", "link": "AI眼镜镜片", "degree": "confirmed"},
            {"code": "sz002475", "name": "立讯精密", "link": "AI眼镜/PVG光波导", "degree": "confirmed"},
        ],
    },
    "GOOGL": {
        "name": "谷歌",
        "a_share_suppliers": [
            {"code": "sz002475", "name": "立讯精密", "link": "Pixel代工", "degree": "confirmed"},
            {"code": "sz300433", "name": "蓝思科技", "link": "手机玻璃", "degree": "confirmed"},
        ],
    },
    "LLY": {
        "name": "礼来",
        "a_share_suppliers": [
            {"code": "sz300347", "name": "泰格医药", "link": "CRO", "degree": "confirmed"},
            {"code": "sz300759", "name": "康龙化成", "link": "CRO", "degree": "confirmed"},
            {"code": "sh603259", "name": "药明康德", "link": "CRO/CDMO", "degree": "confirmed"},
        ],
    },
    "ABBV": {
        "name": "艾伯维",
        "a_share_suppliers": [
            {"code": "sz300347", "name": "泰格医药", "link": "CRO", "degree": "confirmed"},
            {"code": "sh603259", "name": "药明康德", "link": "CRO/CDMO", "degree": "confirmed"},
        ],
    },
}

# A股代码 → 供应商信息(反向索引)
A_SHARE_SUPPLIERS = {}
for us_code, info in SUPPLY_CHAIN.items():
    for s in info["a_share_suppliers"]:
        code = s["code"]
        if code not in A_SHARE_SUPPLIERS:
            A_SHARE_SUPPLIERS[code] = []
        A_SHARE_SUPPLIERS[code].append({
            "us_code": us_code,
            "us_name": info["name"],
            "link": s["link"],
            "degree": s["degree"],
        })

if __name__ == "__main__":
    print("=== 美股→A股供应链映射 ===\n")
    for us_code, info in SUPPLY_CHAIN.items():
        suppliers = info["a_share_suppliers"]
        confirmed = [s for s in suppliers if s["degree"] == "confirmed"]
        potential = [s for s in suppliers if s["degree"] == "potential"]
        print(f"{info['name']}({us_code}): {len(confirmed)}家确认 + {len(potential)}家潜在")
        for s in confirmed:
            print(f"  ✅ {s['name']}: {s['link']}")
        for s in potential:
            print(f"  ⚠️ {s['name']}: {s['link']}")
        print()

    print(f"\n总计: {len(A_SHARE_SUPPLIERS)}只A股公司有美股供应链关系")

    # 保存
    out = {
        "supply_chain": SUPPLY_CHAIN,
        "a_share_index": A_SHARE_SUPPLIERS,
        "note": "基于公开信息整理(互动平台/券商研报/公司公告), 部分有保密协议",
    }
    outp = os.path.join(ROOT, "data", "us_a_supply_chain.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {outp}")
