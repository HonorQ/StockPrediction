"""
滚动 IC 加权稳健性检验（ML_TRAINING_PLAN.md 阶段 3b 方案 A）。

目标：检验"基于 IC 生成的权重"在样本外（测试窗口）的稳定性，防止过拟合。

方法：
  [train 4个月] → [test 1个月] → 滑动
   train 窗口内计算各因子日频 Rank IC/ICIR → 生成 IC 加权 weights
   test 窗口内用该 weights 对已推荐票池回测（期望收益 α / 胜率 / 最大回撤）

数据源：factor_snapshots 24 维 sub_* 独立列 JOIN recommendations.d3_return
  （get_sub_score_panel），全部为真实历史快照 + 真实 T+3 收益，不重算因子。

权重生成规则：
  w_i ∝ prior_i × max(0, ICIR_i)    （ICIR 调整 × config 先验权重）
  - 仅对 IC>0 且 ICIR>0 的因子赋权；IC 为负的因子不参与（不反向，保守）
  - 若某窗口所有因子 ICIR≤0（IC 信号失效），该窗口沿用 config 默认权重，标记"IC 信号失效窗口"
  - 连续 2 个窗口失效 → 触发决策点（建议回滚/暂停调整）

输出：reports/rolling_ic_weight_result.md
参考：ML_TRAINING_PLAN.md §4 阶段 3b。
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import SUB_SCORE_DIMS, StockDatabase  # noqa: E402

# 窗口参数（交易日）
TRAIN_MONTHS = 4
TEST_MONTHS = 1
MIN_CROSS_SECTION = 5       # 每日截面最少样本
MIN_IC_DAYS = 20            # 训练窗口最少有效 IC 天数
MAX_WEIGHT_CHANGE = 0.05    # 单次权重调整幅度上限


def load_panel(today: str = None) -> pd.DataFrame:
    """加载 24 维 sub_* + d3_return 面板（get_sub_score_panel）。"""
    db = StockDatabase()
    rows = db.get_sub_score_panel(days=400, today=today)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # get_sub_score_panel 返回的收益列名为 future_return（= d3_return 值）
    if "future_return" in df.columns:
        df.rename(columns={"future_return": "d3_return"}, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def compute_factor_icir(panel: pd.DataFrame, return_col: str = "d3_return") -> Dict[str, float]:
    """训练窗口内计算各因子日频 Rank IC 的均值与 ICIR。"""
    dims = [d for d in SUB_SCORE_DIMS if d in panel.columns]
    ic_by_factor: Dict[str, List[float]] = {d: [] for d in dims}
    for _, group in panel.groupby("trade_date"):
        if len(group) < MIN_CROSS_SECTION:
            continue
        if group[return_col].nunique() < 2:
            continue
        for dim in dims:
            if group[dim].nunique() < 2:
                continue
            ic = group[dim].rank().corr(group[return_col].rank(), method="spearman")
            if not pd.isna(ic):
                ic_by_factor[dim].append(float(ic))

    result = {}
    for dim, ics in ic_by_factor.items():
        if len(ics) < MIN_IC_DAYS:
            continue
        arr = np.array(ics)
        mean_ic = arr.mean()
        std_ic = arr.std(ddof=1) if len(arr) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        result[dim] = {"ic": mean_ic, "icir": icir, "n_days": len(ics)}
    return result


def build_ic_weights(icir_info: Dict[str, Dict], prior: Dict[str, float],
                     signal_dead: List[str]) -> Tuple[Dict[str, float], bool]:
    """ICIR 调整 × config 先验 → 权重。

    只对"有效 IC 因子"（|IC|>0.03 且 ICIR>0 且样本日数足够）调权，
    其余因子保持 config 先验不变（避免噪声因子被无谓缩放）。
    归一化保证权重总和 ≈ config 先验总和（保持大类平衡）。

    返回 (weights, signal_failed)。signal_failed = 该窗口 IC 信号全部失效。
    """
    IC_ATTENTION = 0.03  # |IC|>0.03 才算值得注意的信号

    if not icir_info:
        return dict(prior), True

    # 挑选有效 IC 因子
    effective = {}
    for dim in SUB_SCORE_DIMS:
        info = icir_info.get(dim)
        if not info:
            continue
        if abs(info["ic"]) < IC_ATTENTION or info["icir"] <= 0:
            continue
        if info["n_days"] < MIN_IC_DAYS:
            continue
        effective[dim] = max(0.0, info["icir"])

    if not effective:
        return dict(prior), True

    # 只对有效因子做 ICIR 调整（先验 × ICIR），其它保持先验
    weights = dict(prior)
    sum(prior.values()) or 1.0
    # 计算有效因子调整后总和相对先验总和的缩放
    adjusted = {dim: prior.get(dim, 0.01) * effective[dim] for dim in effective}
    adj_sum = sum(adjusted.values())
    if adj_sum <= 0:
        return dict(prior), True
    # 将有效因子的权重增量按比例缩放，保持总和不变：
    # 先计算调整比例，确保不因归一化改变大类结构
    prior_effective_sum = sum(prior.get(dim, 0.01) for dim in effective)
    # 有效因子总权重保持为 prior_effective_sum，但按 ICIR 重新分配
    scale = prior_effective_sum / adj_sum if adj_sum > 0 else 1.0
    new_effective = {dim: adj * scale for dim, adj in adjusted.items()}
    for dim in effective:
        weights[dim] = new_effective[dim]

    # 限制单次调整幅度（相对 config 先验）
    final = {}
    for dim in SUB_SCORE_DIMS:
        cur = prior.get(dim, 0.01)
        new = weights.get(dim, cur)
        if abs(new - cur) > MAX_WEIGHT_CHANGE:
            new = cur + (MAX_WEIGHT_CHANGE if new > cur else -MAX_WEIGHT_CHANGE)
        final[dim] = round(new, 4)
    return final, False


def score_panel_cross_sectional(panel: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """对面板按日做截面 Z-Score 加权评分（与 batch_normalize 同口径，简化版）。

    返回带 'score' 列的面板。
    """
    dims = [d for d in SUB_SCORE_DIMS if d in panel.columns and weights.get(d, 0) > 0]
    if not dims:
        panel = panel.copy()
        panel["score"] = 50.0
        return panel

    groups = []
    for date, g in panel.groupby("trade_date"):
        g = g.copy()
        raw = {d: g[d].astype(float).clip(
            g[d].quantile(0.05), g[d].quantile(0.95)).values for d in dims}
        z_total = np.zeros(len(g))
        for d in dims:
            vals = raw[d]
            mu = vals.mean()
            sigma = vals.std()
            z = np.zeros(len(g)) if sigma < 1e-6 else np.clip((vals - mu) / sigma, -3, 3)
            z_total += z * weights.get(d, 0)
        g["score"] = 50 + z_total * (50 / 3)
        groups.append(g)
    if not groups:
        return panel
    return pd.concat(groups)


def eval_window(panel: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """测试窗口内用给定 weights 评估：期望收益 α / 胜率 / 最大回撤。"""
    if panel.empty or "d3_return" not in panel.columns:
        return {"n": 0, "alpha": 0, "win_rate": 0, "max_drawdown": 0}

    scored = score_panel_cross_sectional(panel, weights)
    if scored.empty:
        return {"n": 0, "alpha": 0, "win_rate": 0, "max_drawdown": 0}

    rets = scored["d3_return"].dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0, "alpha": 0, "win_rate": 0, "max_drawdown": 0}

    alpha = float(rets.mean())
    win_rate = float((rets > 0).mean())
    # 最大回撤（用按日期排序的累计收益）
    scored = scored.sort_values("trade_date")
    cum = scored["d3_return"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    return {"n": n, "alpha": alpha, "win_rate": win_rate, "max_drawdown": max_dd}


def generate_report(windows: List[Dict], prior: Dict[str, float], signal_fail_windows: List[int]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# 滚动 IC 加权稳健性检验报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> 方法：[{TRAIN_MONTHS}个月 train → {TEST_MONTHS}个月 test] 滑动窗口；train 算 ICIR 加权权重，test 用该权重回测")
    lines.append("> 数据源：factor_snapshots 24 维 sub_* JOIN recommendations.d3_return（真实历史快照 + T+3 收益）")
    lines.append(f"> 权重规则：w ∝ prior_i × max(0, ICIR_i)，仅 IC>0 且 ICIR>0 的因子参与；单次调整 ≤ {MAX_WEIGHT_CHANGE}")
    lines.append("")
    lines.append("## 一、各窗口结果")
    lines.append("")
    lines.append("| 窗口 | train 起止 | test 起止 | test 样本 | 期望收益 α | 胜率 | 最大回撤 | IC 信号状态 |")
    lines.append("|------|-----------|-----------|-----------|-----------|------|----------|------------|")

    total_test_n = 0
    total_alpha = []
    for w in windows:
        signal_state = "⚠️ 失效" if w["idx"] in signal_fail_windows else "正常"
        lines.append(
            f"| {w['idx']} | {w['train_start']} ~ {w['train_end']} | "
            f"{w['test_start']} ~ {w['test_end']} | {w['eval']['n']} | "
            f"{w['eval']['alpha']:+.2f}% | {w['eval']['win_rate']:.0%} | "
            f"{w['eval']['max_drawdown']:.2f} | {signal_state} |"
        )
        if w["eval"]["n"] > 0:
            total_test_n += w["eval"]["n"]
            total_alpha.append(w["eval"]["alpha"])
    lines.append("")

    # 汇总
    lines.append("## 二、汇总")
    lines.append("")
    lines.append(f"- 窗口数：{len(windows)} | test 总样本：{total_test_n}")
    if total_alpha:
        arr = np.array(total_alpha)
        mean_alpha = arr.mean()
        std_alpha = arr.std(ddof=1) if len(arr) > 1 else 0
        # 95% CI（正态近似）
        se = std_alpha / (len(arr) ** 0.5)
        ci_low = mean_alpha - 1.96 * se
        lines.append(f"- 各窗口 α 均值：{mean_alpha:+.2f}% | 95% CI 下界：{ci_low:+.2f}%")
        lines.append(f"- **验收①：α 95% CI 下界不显著为负** → {'✅ 通过' if ci_low > 0 else '❌ 未通过'}")
    else:
        lines.append("- 无有效 test 样本，无法评估")
    lines.append(f"- IC 信号失效窗口数：{len(signal_fail_windows)}"
                 f"（连续 ≥2 个失效需暂停调整，触发决策点）")
    lines.append("")

    # 权重稳定性
    lines.append("## 三、窗口间权重稳定性")
    lines.append("")
    lines.append("| 因子 | config 先验 | 各窗口权重 | 变异系数(CV) |")
    lines.append("|------|-------------|-----------|--------------|")
    for dim in SUB_SCORE_DIMS:
        vals = [w.get("weights", {}).get(dim, prior.get(dim, 0)) for w in windows if "weights" in w]
        if not vals:
            continue
        arr = np.array(vals)
        cv = arr.std(ddof=1) / (arr.mean() + 1e-9) if arr.mean() > 0 else 0
        flags = "".join(f"{v:.3f}" for v in arr)
        lines.append(f"| {dim} | {prior.get(dim, 0):.4f} | {flags} | {cv:.2f} |")
    lines.append("")
    lines.append("> CV < 0.5 表示权重跨窗口稳定；CV 过高说明 IC 信号不稳定，需谨慎采用动态权重。")
    lines.append("")
    lines.append("---")
    lines.append("*本报告遵循项目规则 §6（回测与 ML 的诚实性），数据全部来自真实历史快照，权重调整需用户确认（规则 §2）。*")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="滚动 IC 加权稳健性检验（阶段 3b 方案 A）")
    parser.add_argument("--today", default=None, help="截止日期 (YYYY-MM-DD)，默认真实当前日期")
    parser.add_argument("--no-report", action="store_true", help="不写报告")
    args = parser.parse_args()

    print("=" * 60)
    print("  滚动 IC 加权稳健性检验（阶段 3b 方案 A）")
    print("=" * 60)

    import yaml
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prior = config.get("weights", {})

    df = load_panel(today=args.today)
    if df.empty:
        print("  ⚠️ 无可用面板数据")
        return
    print(f"  面板：{len(df)} 条 / {df['trade_date'].nunique()} 个交易日 "
          f"({df['trade_date'].min().date()} ~ {df['trade_date'].max().date()})")

    # 构造滚动窗口（按交易日顺序）
    dates = sorted(df["trade_date"].unique())
    print(f"  交易日数：{len(dates)}（window 参数以自然月近似）")

    # 用自然月切分：train=120 交易日 ≈ 4 个月，test=20 ≈ 1 个月（简化用自然月）
    windows = []
    idx = 0
    # 滑动：从最早日期开始，test 窗口 1 个月，train 取 test 起点前 4 个月
    cursor = dates[0] + pd.DateOffset(months=TRAIN_MONTHS)
    end_limit = dates[-1]
    while cursor <= end_limit:
        test_start_dt = cursor
        test_end_dt = cursor + pd.DateOffset(months=TEST_MONTHS)
        train_start_dt = test_start_dt - pd.DateOffset(months=TRAIN_MONTHS)

        train = df[(df["trade_date"] >= train_start_dt) & (df["trade_date"] < test_start_dt)]
        test = df[(df["trade_date"] >= test_start_dt) & (df["trade_date"] < test_end_dt)]
        if test.empty:
            break

        icir_info = compute_factor_icir(train)
        signal_failed = False
        weights, signal_failed = build_ic_weights(icir_info, prior, [])
        eval_res = eval_window(test, weights)

        windows.append({
            "idx": idx + 1,
            "train_start": train_start_dt.strftime("%Y-%m-%d"),
            "train_end": (test_start_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "test_start": test_start_dt.strftime("%Y-%m-%d"),
            "test_end": (test_end_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "n_train": len(train),
            "weights": weights,
            "eval": eval_res,
            "signal_failed": signal_failed,
        })
        print(f"  窗口 {idx+1}: train={len(train)}条 | test={len(test)}条 "
              f"α={eval_res['alpha']:+.2f}% 胜率={eval_res['win_rate']:.0%}"
              f"{' [IC失效]' if signal_failed else ''}")
        idx += 1
        cursor = test_end_dt

    if not windows:
        print("  ⚠️ 无有效窗口（数据不足）")
        return

    signal_fail_windows = [w["idx"] for w in windows if w["signal_failed"]]
    print(f"\n  ✅ 共 {len(windows)} 个窗口，IC 失效 {len(signal_fail_windows)} 个")

    if args.no_report:
        return

    report = generate_report(windows, prior, signal_fail_windows)
    report_dir = os.path.join(ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "rolling_ic_weight_result.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ 报告已保存：{path}")


if __name__ == "__main__":
    main()
