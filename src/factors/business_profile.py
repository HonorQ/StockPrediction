"""
业务画像因子（规则抽取，不依赖 LLM）

设计原则（贴合 MoatFactors 的「零成本派生」风格）：
  - 不引入任何新数据源 / 不调用 LLM，仅从 run_daily 已计算的因子字段做关键词规则抽取；
  - 输入：industry（行业）、hot_concepts（概念列表）、concept_count（概念数）、
          MoatFactors 信号（is_leader / institutional_score / total_mv_yi / moat_score）；
  - 输出：结构化业务标签（赛道 / 产业链环节 / 壁垒关键词 / 龙头·机构重仓等画像标签）。

说明：本项目环境对东财 `stock_individual_info_em` 等个股资料接口不可达，
因此不抓取「主营业务」原文，改以「行业 + 概念 + 护城河信号」规则合成画像，
稳定、离线、确定性可复现。若日后接入本地 Ollama(Qwen) 等免费模型，
可在此模块后挂一个可选的文本摘要增强（默认关闭），不影响主流程。
"""
from typing import Dict, List


# 赛道 / 主题关键词 -> 标签
THEME_KEYWORDS = {
    "新能源": ["新能源", "光伏", "锂电", "储能", "氢能", "风电", "充电桩", "电池"],
    "半导体": ["半导体", "芯片", "集成电路", "晶圆", "光刻", "封测", "MCU", "IGBT"],
    "AI/算力": ["人工智能", "AI", "算力", "大模型", "机器人", "自动驾驶", "机器视觉", "CPO"],
    "医药": ["医药", "创新药", "生物", "疫苗", "CRO", "医疗器械", "中药", "医疗"],
    "军工": ["军工", "航天", "航空", "国防", "卫星", "兵器", "船舶"],
    "消费": ["白酒", "食品", "饮料", "家电", "零售", "消费", "美妆", "乳业"],
    "周期": ["钢铁", "有色", "煤炭", "化工", "水泥", "石化", "建材", "电力"],
    "金融": ["银行", "证券", "保险", "信托", "期货", "金控"],
    "TMT": ["通信", "5G", "传媒", "游戏", "互联网", "软件", "数字经济"],
    "高端制造": ["高端装备", "工业母机", "机械", "专精特新", "智能制造", "无人机"],
}

# 二三级赛道细分（用于更精准的赛道标签）
SUBTHEME_KEYWORDS = {
    "新能源车": ["新能源车", "电动汽车", "整车", "比亚迪", "蔚来", "理想"],
    "光伏设备": ["光伏设备", "硅片", "组件", "逆变器", "HJT", "TOPCon"],
    "半导体设备": ["半导体设备", "刻蚀", "薄膜沉积", "清洗设备"],
    "CXO": ["CXO", "CRO", "CDMO", "临床"],
}


def _match_any(text: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text]


class BusinessProfileFactor:
    """业务画像因子（规则抽取，无状态、可复用于单股与回测）"""

    def build(self, code: str, name: str, factors: Dict) -> Dict:
        industry = factors.get("industry", "未知") or "未知"
        hot = factors.get("hot_concepts", []) or []
        if isinstance(hot, str):
            hot = [hot]
        concepts: List[str] = list(hot)

        # 合并文本用于关键词匹配（行业 + 名称 + 概念）
        text = " ".join([str(industry)] + [str(name)] + [str(c) for c in concepts])

        # 赛道 / 主题标签
        themes = []
        for tag, kws in THEME_KEYWORDS.items():
            if _match_any(text, kws):
                themes.append(tag)
        for tag, kws in SUBTHEME_KEYWORDS.items():
            if _match_any(text, kws):
                themes.append(tag)
        # 去重并保持顺序
        seen = set()
        themes = [t for t in themes if not (t in seen or seen.add(t))]

        # 复用的护城河信号（来自 MoatFactors，已在 all_factors）
        is_leader = bool(factors.get("is_leader", False))
        inst_score = float(factors.get("institutional_score", 50) or 50)
        concept_count = int(factors.get("concept_count", 0) or 0)
        moat_score = float(factors.get("moat_score", 50) or 50)
        total_mv_yi = float(factors.get("total_mv_yi", 0) or 0)

        # 不可替代性画像标签
        moat_labels: List[str] = []
        if is_leader:
            moat_labels.append("行业龙头")
        if inst_score >= 70:
            moat_labels.append("机构重仓")
        if 0 < concept_count <= 3:
            moat_labels.append("业务聚焦")
        if total_mv_yi >= 1000:
            moat_labels.append("大市值权重")

        # 不可替代性等级
        if moat_score >= 70:
            moat_level = "强护城河"
        elif moat_score >= 55:
            moat_level = "中等护城河"
        else:
            moat_level = "护城河一般"

        # 业务画像摘要（一句话）
        summary_parts = [industry]
        if themes:
            summary_parts.append("·".join(themes))
        business_summary = " ".join(summary_parts)

        return {
            "industry": industry,
            "concepts": concepts[:10],
            "themes": themes,
            "moat_labels": moat_labels,
            "moat_score": round(moat_score, 1),
            "moat_level": moat_level,
            "business_summary": business_summary,
            # 透传 moat 子维度，供报告展示
            "moat_concept_rarity": factors.get("moat_concept_rarity"),
            "moat_leader": factors.get("moat_leader"),
            "moat_institution": factors.get("moat_institution"),
            "moat_market_position": factors.get("moat_market_position"),
        }
