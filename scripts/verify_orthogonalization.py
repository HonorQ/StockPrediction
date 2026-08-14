"""
正交化审查脚本 — 验证 scorer.py 中 4 对正交化的实际效果
用法: python scripts/verify_orthogonalization.py [--days 60]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import StockDatabase
from src.engine.scorer import StockScorer

ALL_FACTORS = [
    "trend", "strategy", "chip", "fundamental", "institutional",
    "sector_rotation", "oversold_rebound", "news", "relative_strength",
    "ashare_specific", "quant_behavior", "margin_trading", "shareholder",
    "lockup_risk", "research", "dragon_tiger", "leader", "deep_research",
    "policy_event", "national_team", "capital_behavior", "moat",
]

ORTHO_PAIRS = StockScorer.ORTHOGONALIZE_PAIRS


def load_sub_scores(db, days=60):
    """从 factor_snapshots 加载 sub_scores 数据。"""
    rows = db.execute(
        "SELECT trade_date, code, sub_scores FROM factor_snapshots ORDER BY trade_date DESC LIMIT ?",
        (100000,)
    ).fetchall()
    records = []
    seen = set()
    for r in rows:
        key = (r["trade_date"], r["code"])
        if key in seen:
            continue
        seen.add(key)
        try:
            subs = json.loads(r["sub_scores"]) if isinstance(r["sub_scores"], str) else r["sub_scores"]
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(subs, dict):
            continue
        records.append(subs)

    cutoff = max(0, len(records) - days * 5000)
    records = records[cutoff:] if cutoff > 0 else records
    return records


def compute_corr_matrix(records):
    """计算 22 维 sub_score 相关系数矩阵。"""
    df = pd.DataFrame(records)
    for col in ALL_FACTORS:
        if col not in df.columns:
            df[col] = 50.0
    df = df[ALL_FACTORS]
    return df.corr()


def check_pairs(corr_matrix):
    """逐一检查 4 对正交化的相关系数。"""
    results = []
    for target, regressor in ORTHO_PAIRS:
        corr = corr_matrix.loc[target, regressor]
        if abs(corr) < 0.5:
            action = "移除正交化（不相关）"
        elif abs(corr) < 0.7:
            action = "保留（弱相关）"
        else:
            action = "保留（高相关，正交化有效）"
        results.append({
            "pair": f"{target} vs {regressor}",
            "correlation": round(corr, 4),
            "action": action,
        })
    return results


def find_missing_pairs(corr_matrix, threshold=0.7):
    """查找 >0.7 但不在正交化列表中的因子对。"""
    existing = {(a, b) for a, b in ORTHO_PAIRS} | {(b, a) for a, b in ORTHO_PAIRS}
    missing = []
    for i, f1 in enumerate(ALL_FACTORS):
        for f2 in ALL_FACTORS[i + 1:]:
            corr = corr_matrix.loc[f1, f2]
            if abs(corr) > threshold and (f1, f2) not in existing:
                missing.append({
                    "pair": f"{f1} vs {f2}",
                    "correlation": round(corr, 4),
                })
    missing.sort(key=lambda x: -abs(x["correlation"]))
    return missing


def main():
    parser = argparse.ArgumentParser(description="正交化审查")
    parser.add_argument("--days", type=int, default=60, help="分析天数")
    args = parser.parse_args()

    db = StockDatabase()
    records = load_sub_scores(db, args.days)
    if not records:
        print("factor_snapshots 无数据。请先运行 run_feedback_training.py 生成数据。")
        return

    print(f"  加载 {len(records)} 条 sub_score 记录\n")

    corr_matrix = compute_corr_matrix(records)

    print("=== 现有正交化对检查 ===\n")
    pair_results = check_pairs(corr_matrix)
    for pr in pair_results:
        print(f"  {pr['pair']}: r={pr['correlation']} → {pr['action']}")

    print("\n=== 未处理的高相关因子对 (>0.7) ===\n")
    missing = find_missing_pairs(corr_matrix)
    if missing:
        for mp in missing[:10]:
            print(f"  {mp['pair']}: r={mp['correlation']}")
    else:
        print("  无")

    print(f"\n=== 完整相关系数矩阵 ===\n")
    pd.set_option('display.max_columns', 22)
    pd.set_option('display.width', 200)
    print(corr_matrix.round(3))

    output_dir = os.path.join(ROOT, "reports", "research")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "orthogonalization_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 正交化审查报告\n\n")
        f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"样本数: {len(records)}\n\n")
        f.write("## 现有正交化对\n\n")
        f.write("| 因子对 | 相关系数 | 建议 |\n")
        f.write("|--------|----------|------|\n")
        for pr in pair_results:
            f.write(f"| {pr['pair']} | {pr['correlation']} | {pr['action']} |\n")
        f.write("\n## 未处理的高相关因子对 (>0.7)\n\n")
        if missing:
            f.write("| 因子对 | 相关系数 |\n")
            f.write("|--------|----------|\n")
            for mp in missing:
                f.write(f"| {mp['pair']} | {mp['correlation']} |\n")
        else:
            f.write("无\n")
        f.write("\n## 完整相关系数矩阵\n\n")
        f.write(corr_matrix.to_markdown() if hasattr(pd, 'to_markdown') else str(corr_matrix.round(3)))
        f.write("\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
