"""
护城河 / 不可替代性因子（A层：零成本派生）

不引入任何新数据源，仅从 run_daily 已计算的因子字段派生「稀缺度 / 不可替代性」评分：
  - concept_count         : 所属概念数量（来自 SectorLinkage，概念越少越聚焦独特）
  - leader_score/is_leader: 龙头地位（来自 SectorLeader）
  - institutional_score   : 机构重仓度（来自 InstitutionalResearch）
  - total_mv_yi           : 总市值（亿元，来自 Fundamental）

四个子维度各算 0-100，再加权合成 moat_score (0-100)。

设计原则（贴合现有因子风格，纯函数式、无状态、可复用于单股与回测）：
  - 概念稀缺度：概念越多越同质化（题材杂毛），越少越聚焦独特；
  - 龙头护城河：行业龙头难以被替代，is_leader 加成；
  - 机构护城河：机构深度研究后重仓，代表壁垒；
  - 市值地位：大市值通常行业地位稳固，边际递减并封顶，避免巨无霸虚高。
"""
from typing import Dict


class MoatFactors:
    """护城河 / 不可替代性因子（零成本派生，无状态）"""

    def calculate_all(self, factors: Dict) -> Dict:
        concept_count = int(factors.get("concept_count", 0) or 0)
        leader_score = float(factors.get("leader_score", 50) or 50)
        is_leader = bool(factors.get("is_leader", False))
        institutional_score = float(factors.get("institutional_score", 50) or 50)
        total_mv = float(factors.get("total_mv_yi", 0) or 0)

        concept_rarity = self._concept_rarity(concept_count)
        # 龙头护城河：is_leader 给额外加成（龙头=难以替代）
        leader_moat = min(100.0, leader_score + (8.0 if is_leader else 0.0))
        institution_moat = institutional_score
        market_position = self._market_position(total_mv)

        moat_score = (
            0.30 * concept_rarity
            + 0.30 * leader_moat
            + 0.25 * institution_moat
            + 0.15 * market_position
        )
        moat_score = max(0.0, min(100.0, moat_score))

        return {
            "moat_score": round(moat_score, 2),
            "moat_concept_rarity": round(concept_rarity, 2),
            "moat_leader": round(leader_moat, 2),
            "moat_institution": round(institution_moat, 2),
            "moat_market_position": round(market_position, 2),
        }

    @staticmethod
    def _concept_rarity(concept_count: int) -> float:
        """概念越少越聚焦独特 -> 稀缺度越高。
        0 个（无标签/僵尸股）信息缺失，不给臆断，中性 50。
        """
        if concept_count <= 0:
            return 50.0
        if concept_count == 1:
            return 90.0
        if concept_count == 2:
            return 85.0
        if concept_count == 3:
            return 75.0
        if concept_count == 4:
            return 60.0
        # >=5 题材杂毛，同质化，稀缺度低
        return 45.0

    @staticmethod
    def _market_position(total_mv_yi: float) -> float:
        """市值越大通常行业地位越稳固（护城河越深），边际递减并封顶。"""
        if total_mv_yi <= 0:
            return 50.0
        if total_mv_yi < 30:
            return 55.0
        if total_mv_yi < 100:
            return 65.0
        if total_mv_yi < 500:
            return 72.0
        if total_mv_yi < 2000:
            return 80.0
        return 82.0
