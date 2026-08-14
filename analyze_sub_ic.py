"""
24 维子评分 (sub_*) 日频 Rank IC/ICIR 分析脚本（ML_TRAINING_PLAN.md 阶段 1）。

口径：
  - 因子对象：factor_snapshots 表的 24 个 sub_* 独立列（SUB_SCORE_DIMS），
    与 config.yaml.weights 一一对应，直接指导权重调整。
  - 收益口径：路径 A —— factor_snapshots JOIN recommendations.d3_return
    （候选池 50 只/天，T+3 真实收盘价回看，未推荐但入 snapshots 的票 d3_return 为
    NULL 时直接剔除，在报告中标注有效样本量）。
  - IC 类型：日频截面 Rank IC（spearman），按 trade_date 分组计算后取均值；
    ICIR = mean(IC_daily) / std(IC_daily)。
  - 原始因子下钻：对全部原始因子（factor_snapshots.factors JSON）同步算 IC，
    作为附录诊断；当 sub_* IC 异常（显著为负或 |IC|>0.05）时，在主表中标注。

阈值（按绝对值判断）：
  |IC|>0.03 注意，>0.05 有效，>0.1 强；
  |ICIR|>0.5 可用，>1.0 优秀。

参考：ML_TRAINING_PLAN.md §4 阶段 1；ml_selector.calculate_ic_by_day / calculate_icir。
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import SUB_SCORE_DIMS, DB_PATH  # noqa: E402

# IC 阈值
IC_ATTENTION = 0.03
IC_EFFECTIVE = 0.05
IC_STRONG = 0.10
ICIR_USABLE = 0.5
ICIR_EXCELLENT = 1.0

# 每日截面最少样本数（与 ml_selector.calculate_ic_by_day 一致）
MIN_CROSS_SECTION = 5


def load_sub_score_panel(lookback_days: int = 365, mode: str = None) -> pd.DataFrame:
    """路径 A：factor_snapshots JOIN recommendations.d3_return。

    SELECT fs.trade_date, fs.code, fs.sub_*, r.d3_return
    FROM factor_snapshots fs
    INNER JOIN recommendations r
      ON fs.trade_date = r.trade_date AND fs.code = r.code
    WHERE r.d3_return IS NOT NULL
      AND fs.trade_date >= cutoff
      [AND r.mode = ?]   # 可选：按 mode 过滤（premarket=实盘 / feedback_training=模拟）

    返回 DataFrame：trade_date, code, 24 个 sub_* 列, d3_return
    """
    cutoff = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    sub_cols = [f"sub_{d}" for d in SUB_SCORE_DIMS]
    select_cols = ["fs.trade_date", "fs.code"] + [f"fs.{c}" for c in sub_cols] + ["r.d3_return"]
    sql = (
        "SELECT " + ", ".join(select_cols) +
        " FROM factor_snapshots fs"
        " INNER JOIN recommendations r"
        "   ON fs.trade_date = r.trade_date AND fs.code = r.code"
        " WHERE r.d3_return IS NOT NULL AND fs.trade_date >= ?"
    )
    params: list = [cutoff]
    if mode:
        sql += " AND r.mode = ?"
        params.append(mode)
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def compute_daily_rank_ic(panel: pd.DataFrame, factor_col: str,
                          return_col: str = "d3_return") -> List[float]:
    """按 trade_date 分组计算日频 Rank IC（spearman），返回每日 IC 序列。

    与 ml_selector.calculate_ic_by_day 等价：
      - 截面样本数 < MIN_CROSS_SECTION 跳过
      - factor 或 return 退化（nunique<2）跳过
    """
    ics: List[float] = []
    for _, group in panel.groupby("trade_date"):
        if len(group) < MIN_CROSS_SECTION:
            continue
        if group[factor_col].nunique() < 2 or group[return_col].nunique() < 2:
            continue
        ic = group[factor_col].rank().corr(
            group[return_col].rank(), method="spearman"
        )
        if not pd.isna(ic):
            ics.append(float(ic))
    return ics


def compute_stats(ic_series: List[float]) -> Dict[str, float]:
    """计算 IC 均值、std、ICIR、样本日数、t 统计量、95% CI、正向占比。"""
    ic_arr = np.array(ic_series, dtype=float)
    ic_arr = ic_arr[~np.isnan(ic_arr)]
    n = len(ic_arr)
    if n == 0:
        return {
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "n_days": 0,
            "t_stat": 0.0, "ci95_half": 0.0, "positive_ratio": 0.0,
        }
    mean_ic = float(ic_arr.mean())
    std_ic = float(ic_arr.std(ddof=1)) if n > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    # t 统计量：mean / (std / sqrt(n))；n-1 自由度近似用正态临界值 1.96（n>=30 时足够）
    se = std_ic / (n ** 0.5) if n > 0 else 0.0
    t_stat = mean_ic / se if se > 0 else 0.0
    # 95% 置信区间半宽（正态近似，n 大时与 t 分布接近）
    ci95_half = 1.96 * se if se > 0 else 0.0
    positive_ratio = float((ic_arr > 0).mean())
    return {
        "ic_mean": mean_ic, "ic_std": std_ic, "icir": icir, "n_days": n,
        "t_stat": t_stat, "ci95_half": ci95_half, "positive_ratio": positive_ratio,
    }


def compute_data_quality(panel: pd.DataFrame, factor_col: str) -> Dict[str, float]:
    """数据质量诊断：独立列取值的去重数、非空率、是否全默认值。"""
    if factor_col not in panel.columns:
        return {"n_non_null": 0, "n_unique": 0, "non_null_ratio": 0.0, "is_constant": True}
    series = panel[factor_col].dropna()
    n_non_null = int(len(series))
    n_unique = int(series.nunique())
    n_total = int(len(panel))
    return {
        "n_non_null": n_non_null,
        "n_unique": n_unique,
        "non_null_ratio": float(n_non_null / n_total) if n_total > 0 else 0.0,
        "is_constant": n_unique <= 1,
    }


def classify_factor(stats: Dict[str, float],
                    data_quality: Optional[Dict] = None) -> Tuple[str, str]:
    """按 ML_TRAINING_PLAN.md §4 阶段 1 阈值分类。

    返回 (类别, 强度)
      类别 ∈ {有效, 噪声, 反向, 数据缺失, 样本不足}
      强度 ∈ {强, 中, 弱, 极弱, —}
    """
    ic = stats["ic_mean"]
    icir = stats["icir"]
    n = stats["n_days"]

    # 优先识别"数据缺失"——独立列取值全相同（nunique<=1）说明历史 premarket
    # 根本没真正计算该因子，全部为默认中性分（50 或 60），IC 无意义
    if data_quality is not None and data_quality.get("is_constant", False):
        return ("数据缺失", "—")

    # 样本日数太少（<10）不轻易下结论
    if n < 10:
        return ("样本不足", "—")
    abs_ic = abs(ic)
    if abs_ic >= IC_STRONG:
        strength = "强"
    elif abs_ic >= IC_EFFECTIVE:
        strength = "中"
    elif abs_ic >= IC_ATTENTION:
        strength = "弱"
    else:
        strength = "极弱"

    # 分类：以 IC 符号 + 显著性为准
    # t_stat > 1.96 视为显著异于 0
    significant = abs(stats["t_stat"]) >= 1.96
    if ic < 0 and (abs_ic >= IC_ATTENTION or significant):
        return ("反向", strength)
    if abs_ic < IC_ATTENTION:
        return ("噪声", strength)
    return ("有效", strength)


def load_raw_factor_panel(lookback_days: int = 365) -> pd.DataFrame:
    """加载原始因子面板（factor_snapshots.factors JSON × d3_return）。
    用于附录的下钻诊断（异常 sub_* 时查看具体原始因子 IC）。
    """
    cutoff = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    sql = (
        "SELECT fs.trade_date, fs.code, fs.factors, r.d3_return"
        " FROM factor_snapshots fs"
        " INNER JOIN recommendations r"
        "   ON fs.trade_date = r.trade_date AND fs.code = r.code"
        " WHERE r.d3_return IS NOT NULL AND fs.trade_date >= ?"
        "   AND fs.factors IS NOT NULL"
    )
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=(cutoff,))
    if df.empty:
        return df
    # 展开 factors JSON 为列
    records = []
    for _, row in df.iterrows():
        try:
            factors = json.loads(row["factors"]) if isinstance(row["factors"], str) else row["factors"]
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(factors, dict) or "z_scores" in factors:
            continue
        rec = {"trade_date": row["trade_date"], "code": row["code"], "d3_return": row["d3_return"]}
        rec.update(factors)
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def compute_raw_factor_ic(panel: pd.DataFrame) -> pd.DataFrame:
    """对原始因子面板计算 IC/ICIR，返回 DataFrame（按 |IC| 降序）。"""
    if panel.empty or "d3_return" not in panel.columns:
        return pd.DataFrame()
    exclude = {"d3_return", "label", "stock_code", "code", "date", "trade_date"}
    factor_cols = [
        c for c in panel.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])
    ]
    rows = []
    for col in factor_cols:
        sub = panel[["trade_date", col, "d3_return"]].dropna()
        if sub.empty:
            continue
        ics = compute_daily_rank_ic(sub, col, "d3_return")
        if not ics:
            continue
        stats = compute_stats(ics)
        rows.append({"factor": col, **stats})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["abs_ic"] = df["ic_mean"].abs()
    df = df.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    return df


def generate_report(panel: pd.DataFrame,
                    stats_by_dim: Dict[str, Dict[str, float]],
                    quality_by_dim: Dict[str, Dict],
                    raw_ic_df: Optional[pd.DataFrame],
                    weights: Dict[str, float],
                    forward_window: int) -> str:
    """生成 markdown 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_days = int(panel["trade_date"].nunique()) if not panel.empty else 0
    n_samples = len(panel)
    date_min = panel["trade_date"].min() if not panel.empty else "—"
    date_max = panel["trade_date"].max() if not panel.empty else "—"

    # 预先计算分类
    classified = {dim: classify_factor(stats, quality_by_dim.get(dim))
                  for dim, stats in stats_by_dim.items()}

    lines = []
    lines.append("# 24 维子评分 (sub_*) 日频 Rank IC 分析报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> 数据范围：{date_min} ~ {date_max}（{n_days} 个交易日，{n_samples} 条样本）")
    lines.append(f"> 收益口径：路径 A —— factor_snapshots INNER JOIN recommendations.d3_return（T+{forward_window} 真实收盘价）")
    lines.append(f"> IC 类型：日频截面 Rank IC（spearman），按 trade_date 分组后取均值；ICIR = mean(IC)/std(IC)")
    lines.append(f"> 阈值：|IC|>{IC_ATTENTION} 注意，>{IC_EFFECTIVE} 有效，>{IC_STRONG} 强；|ICIR|>{ICIR_USABLE} 可用，>{ICIR_EXCELLENT} 优秀")
    lines.append(f"> 每日截面最少样本数：{MIN_CROSS_SECTION}")
    lines.append("")
    lines.append("> 遵循项目规则 §6：本报告只输出客观 IC 指标与调权建议，**不自动改写 config.yaml**；")
    lines.append("> 权重调整需用户确认（规则 §2）。阶段 1 的调权属轻度 data snooping，须在阶段 3b 滚动验证中复核。")
    lines.append("")

    # 数据质量摘要（关键，避免误导用户）
    n_data_missing = sum(1 for dim in stats_by_dim if classified[dim][0] == "数据缺失")
    n_insufficient = sum(1 for dim in stats_by_dim if classified[dim][0] == "样本不足")
    n_evaluable = len(stats_by_dim) - n_data_missing - n_insufficient
    lines.append("## 〇、数据质量摘要（必读）")
    lines.append("")
    lines.append(f"- 24 维 sub_* 中，**仅 {n_evaluable} 个**有真实因子值波动（独立列取值去重数 ≥2），可计算 IC；")
    lines.append(f"- **{n_data_missing} 个**子评分独立列取值全为默认中性分（50 或 60），说明历史 premarket **未真正计算**这些因子（外部数据源未接入或降级），IC 无意义；")
    lines.append(f"- **{n_insufficient} 个**子评分有效截面日数 < 10，样本不足，不下结论。")
    lines.append("")
    lines.append("> **结论含义**：若大量因子为「数据缺失」，阶段 1 的 IC 分析覆盖范围有限，")
    lines.append("> 真正能指导权重调整的只有「可评估」的几个因子。需先修复 premarket 让全部 24 维真正计算，")
    lines.append("> 否则阶段 3b 滚动验证无法对全因子谱系做权重优化。")
    lines.append("")

    # 主表：24 维 IC 排名
    lines.append("## 一、24 维子评分 IC 排名")
    lines.append("")
    lines.append("| 排名 | 子评分 | 当前权重 | IC 均值 | IC std | ICIR | 样本日 | t 统计 | 正向占比 | 95% CI 半宽 | 去重值数 | 类别 | 强度 |")
    lines.append("|------|--------|---------|---------|--------|------|--------|---------|----------|-------------|---------|------|------|")

    sorted_dims = sorted(stats_by_dim.items(), key=lambda x: abs(x[1]["ic_mean"]), reverse=True)
    for i, (dim, stats) in enumerate(sorted_dims, 1):
        category, strength = classified[dim]
        w = weights.get(dim, 0.0)
        q = quality_by_dim.get(dim, {})
        n_unique = q.get("n_unique", 0)
        lines.append(
            f"| {i} | {dim} | {w:.4f} | {stats['ic_mean']:+.4f} | {stats['ic_std']:.4f} | "
            f"{stats['icir']:+.2f} | {int(stats['n_days'])} | {stats['t_stat']:+.2f} | "
            f"{stats['positive_ratio']:.0%} | ±{stats['ci95_half']:.4f} | {n_unique} | {category} | {strength} |"
        )
    lines.append("")

    # 分类汇总
    lines.append("## 二、因子分类汇总")
    lines.append("")
    effective = [d for d in stats_by_dim if classified[d][0] == "有效"]
    reverse_ = [d for d in stats_by_dim if classified[d][0] == "反向"]
    noise = [d for d in stats_by_dim if classified[d][0] == "噪声"]
    missing = [d for d in stats_by_dim if classified[d][0] == "数据缺失"]
    insufficient = [d for d in stats_by_dim if classified[d][0] == "样本不足"]
    lines.append(f"- **有效因子（{len(effective)} 个）**：" + (", ".join(effective) if effective else "无"))
    lines.append(f"- **反向因子（{len(reverse_)} 个）**：" + (", ".join(reverse_) if reverse_ else "无"))
    lines.append(f"- **噪声因子（{len(noise)} 个）**：" + (", ".join(noise) if noise else "无"))
    lines.append(f"- **数据缺失（{len(missing)} 个）**：" + (", ".join(missing) if missing else "无"))
    if insufficient:
        lines.append(f"- **样本不足（{len(insufficient)} 个）**：" + (", ".join(insufficient) if insufficient else "无"))
    lines.append("")

    # 调权建议
    lines.append("## 三、权重调整建议（待用户确认）")
    lines.append("")
    lines.append("> 处理原则（ML_TRAINING_PLAN.md §4 阶段 1）：")
    lines.append("> - 噪声因子降权至 ≤0.01")
    lines.append("> - 反向因子改为负向权重或从组合中剔除")
    lines.append("> - 有效因子保持/小幅提升权重，**单次调整幅度 ≤3pp**")
    lines.append("> - 数据缺失因子：**不调权**，等 premarket 修复数据后重新评估")
    lines.append("> - 调整结果必须在阶段 3b 滚动验证中复核，若验证集 α 显著为负则回退")
    lines.append("")
    lines.append("| 子评分 | 当前权重 | 类别 | IC | ICIR | 建议动作 |")
    lines.append("|--------|---------|------|-----|------|----------|")
    for dim, stats in sorted_dims:
        category, strength = classified[dim]
        w = weights.get(dim, 0.0)
        if category == "数据缺失":
            suggestion = "不调权，先修复 premarket 让因子真正计算"
        elif category == "样本不足":
            suggestion = "保持，等样本积累"
        elif category == "反向":
            if abs(stats["ic_mean"]) >= IC_EFFECTIVE:
                suggestion = f"建议改为负向权重或剔除（IC={stats['ic_mean']:+.4f} 显著为负）"
            else:
                suggestion = "建议剔除或置 0"
        elif category == "噪声":
            suggestion = "降权至 ≤0.01"
        else:  # 有效
            if abs(stats["ic_mean"]) >= IC_STRONG:
                suggestion = f"小幅提升（≤+3pp），IC={stats['ic_mean']:+.4f} 强信号"
            elif abs(stats["ic_mean"]) >= IC_EFFECTIVE:
                suggestion = "保持或小幅提升（≤+2pp）"
            else:
                suggestion = "保持"
        lines.append(f"| {dim} | {w:.4f} | {category} | {stats['ic_mean']:+.4f} | {stats['icir']:+.2f} | {suggestion} |")
    lines.append("")

    # 原始因子下钻（附录）
    if raw_ic_df is not None and not raw_ic_df.empty:
        lines.append("## 四、原始因子下钻诊断（附录）")
        lines.append("")
        lines.append("> 当 sub_* IC 异常时，下钻到具体原始因子查看根因。下表为全部原始因子 IC top 15。")
        lines.append("> 此表仅用于异常诊断，不直接指导权重调整（权重调整以 24 维 sub_* 为主）。")
        lines.append("")
        lines.append("| 排名 | 原始因子 | IC 均值 | ICIR | 样本日 | t 统计 | 正向占比 |")
        lines.append("|------|----------|---------|------|--------|---------|----------|")
        for i, row in raw_ic_df.head(15).iterrows():
            lines.append(
                f"| {i+1} | {row['factor']} | {row['ic_mean']:+.4f} | {row['icir']:+.2f} | "
                f"{int(row['n_days'])} | {row['t_stat']:+.2f} | {row['positive_ratio']:.0%} |"
            )
        lines.append("")

    # 风险与限制
    lines.append("## 五、风险与限制")
    lines.append("")
    lines.append("- **样本量**：6 个月数据，部分子评分样本日数偏少，统计显著性有限；t 统计量仅供参考。")
    lines.append("- **数据缺失**：大量外部数据因子在历史 premarket 中未真正计算（全为默认值），阶段 1 仅能评估少数 K 线类因子。")
    lines.append("- **轻度 data snooping**：用全样本 IC 决策再用于未来属轻度过拟合，必须满足：① 单次调整幅度 ≤3pp；② 阶段 3b 滚动验证复核。")
    lines.append("- **选择偏差**：基于已推荐/入候选池的票池，不能直接外推到全市场。")
    lines.append("- **T+N ≠ 实盘**：T+3 固定持仓收益与实盘止损/止盈/到期平仓不同，IC 高不代表实盘必赚。")
    lines.append("- **路径 A 缺口**：7 月数据 d3_return 未回填，需 premarket 持续运行补齐。")
    lines.append("")
    lines.append("---")
    lines.append("*本报告遵循项目规则 §6（回测与 ML 的诚实性），禁止 mock/假数据。权重调整需用户确认（规则 §2）。*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="24 维子评分日频 Rank IC/ICIR 分析（阶段 1）")
    parser.add_argument("--lookback", type=int, default=365, help="回溯天数（默认 365）")
    parser.add_argument("--forward", type=int, default=3, help="收益窗口（默认 3，对应 d3_return）")
    parser.add_argument("--no-raw", action="store_true", help="跳过原始因子下钻（加速）")
    parser.add_argument("--mode", default=None,
                        help="按 recommendations.mode 过滤：premarket=实盘 / feedback_training=反馈训练模拟（默认不过滤）")
    args = parser.parse_args()

    # 加载 config.yaml 中的 weights（只读，不修改）
    import yaml
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    weights = config.get("weights", {})

    print("=" * 60)
    print("  24 维子评分 (sub_*) 日频 Rank IC 分析")
    print(f"  回溯 {args.lookback} 天 | 收益窗口 T+{args.forward}")
    print("=" * 60)
    print()

    # 1. 加载面板
    panel = load_sub_score_panel(lookback_days=args.lookback, mode=args.mode)
    if panel.empty:
        mode_desc = f"（mode={args.mode}）" if args.mode else ""
        print(f"  ⚠️ 数据库无可用数据{mode_desc}：factor_snapshots JOIN recommendations.d3_return 为空")
        print("  💡 提示：需要 premarket 持续运行并回填 d3_return（截至 2026-06-30 已回填）")
        return
    print(f"  📊 加载样本：{len(panel)} 条 / {panel['trade_date'].nunique()} 个交易日"
          + (f"（mode={args.mode}）" if args.mode else "（全部 mode）"))
    print(f"     时间范围：{panel['trade_date'].min()} ~ {panel['trade_date'].max()}")
    print(f"     日均样本：{len(panel) / panel['trade_date'].nunique():.1f}")
    print()

    # 2. 对 24 维 sub_* 计算 IC/ICIR + 数据质量
    stats_by_dim: Dict[str, Dict[str, float]] = {}
    quality_by_dim: Dict[str, Dict] = {}
    for dim in SUB_SCORE_DIMS:
        col = f"sub_{dim}"
        quality_by_dim[dim] = compute_data_quality(panel, col)
        if col not in panel.columns:
            stats_by_dim[dim] = {
                "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "n_days": 0,
                "t_stat": 0.0, "ci95_half": 0.0, "positive_ratio": 0.0,
            }
            continue
        ics = compute_daily_rank_ic(panel, col, "d3_return")
        stats_by_dim[dim] = compute_stats(ics)

    # 3. 打印简表
    n_evaluable = sum(1 for d in SUB_SCORE_DIMS
                      if classify_factor(stats_by_dim[d], quality_by_dim[d])[0] not in ("数据缺失", "样本不足"))
    n_missing = sum(1 for d in SUB_SCORE_DIMS
                    if classify_factor(stats_by_dim[d], quality_by_dim[d])[0] == "数据缺失")
    print(f"  📊 数据质量：可评估 {n_evaluable} 个 | 数据缺失 {n_missing} 个 | 总 {len(SUB_SCORE_DIMS)} 维")
    print()
    print("  📈 24 维子评分 IC 排名：")
    print("  " + "-" * 110)
    print(f"  {'排名':>4} {'子评分':<22} {'权重':>8} {'IC':>9} {'ICIR':>7} {'日数':>6} {'t':>7} {'正向%':>7} {'去重值':>7} {'类别':<8} {'强度':<6}")
    print("  " + "-" * 110)
    sorted_dims = sorted(stats_by_dim.items(), key=lambda x: abs(x[1]["ic_mean"]), reverse=True)
    for i, (dim, stats) in enumerate(sorted_dims, 1):
        category, strength = classify_factor(stats, quality_by_dim.get(dim))
        w = weights.get(dim, 0.0)
        n_unique = quality_by_dim.get(dim, {}).get("n_unique", 0)
        print(
            f"  {i:>4} {dim:<22} {w:>8.4f} {stats['ic_mean']:>+9.4f} {stats['icir']:>+7.2f} "
            f"{int(stats['n_days']):>6} {stats['t_stat']:>+7.2f} {stats['positive_ratio']:>6.0%} "
            f"{n_unique:>7} {category:<8} {strength:<6}"
        )
    print()

    # 4. 原始因子下钻
    raw_ic_df = None
    if not args.no_raw:
        print("  🔍 原始因子下钻诊断中...")
        raw_panel = load_raw_factor_panel(lookback_days=args.lookback)
        if not raw_panel.empty:
            raw_ic_df = compute_raw_factor_ic(raw_panel)
            print(f"     原始因子数：{len(raw_ic_df)}，top 5 IC：")
            for _, row in raw_ic_df.head(5).iterrows():
                print(f"       {row['factor']:<25} IC={row['ic_mean']:+.4f}  ICIR={row['icir']:+.2f}  日数={int(row['n_days'])}")
        else:
            print("     ⚠️ 原始因子面板为空，跳过下钻")
        print()

    # 5. 生成报告
    report = generate_report(panel, stats_by_dim, quality_by_dim, raw_ic_df, weights, args.forward)
    report_dir = os.path.join(ROOT, "reports", "factor_ic")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(
        report_dir,
        f"factor_ic_sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ 报告已保存：{report_path}")
    print()
    print("  ⚠️ 注意：本报告仅输出 IC 指标与调权建议，**未修改 config.yaml**。")
    print("         权重调整需用户确认（项目规则 §2），建议在阶段 3b 滚动验证中复核。")


if __name__ == "__main__":
    main()
