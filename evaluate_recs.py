"""
Regime 分层归因分析（ML_TRAINING_PLAN.md 阶段 2）。

目标：看 24 维子评分在不同市场环境（bull / neutral / bear / crash）下的表现差异，
识别"全 regime 有效"vs"仅特定 regime 有效"的因子，为动态加权提供依据。

口径：
  - 因子对象：factor_snapshots 表的 24 个 sub_* 独立列（SUB_SCORE_DIMS）
  - 收益：recommendations.d3_return（T+3 真实收盘价回看，避免未来函数）
  - regime：
      * 优先用 recommendations.regime（新数据，premarket 已落库）
      * 旧数据（regime IS NULL）用 T-1 日收盘后可得数据回算：
          上证指数当日涨跌 + 上证 vs MA20 趋势（指数历史日线可得）
          涨跌停数(sentiment)历史无法拉取 → 该信号退化（与 ML_TRAINING_PLAN 阶段2 风险说明一致）
      * 注意：回算只用 T-1 日收盘数据，禁止用 T 日盘中/收盘数据（规则 §1）
  - 归因：因子高分(>60)视为看多信号，实际 d3_return > 0 记为命中；收益为负记为失败。

输出：reports/regime_factor_attribution.md
参考：ML_TRAINING_PLAN.md §4 阶段 2；AttributionAggregator。
"""
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.data.database import SUB_SCORE_DIMS, DB_PATH  # noqa: E402

BULLISH_THRESHOLD = 60   # sub_* > 60 视为看多信号（与 FactorAttribution 一致）
BEARISH_THRESHOLD = 40   # sub_* < 40 视为看空信号
RETURN_THRESHOLD = 0.0   # d3_return > 0 记为命中
MIN_SAMPLES = 5          # 最少样本数才统计


def load_recommendations_with_sub(today: str = None) -> pd.DataFrame:
    """加载 recommendations JOIN factor_snapshots 的 24 维 sub_* 独立列。

    保留 regime 字段（可能为 NULL，后续回算）。
    """
    sub_cols = ", ".join(f"fs.sub_{dim} AS sub_{dim}" for dim in SUB_SCORE_DIMS)
    sql = (
        "SELECT r.trade_date, r.code, r.name, r.score, r.mode, r.regime, "
        "r.d3_return, " + sub_cols +
        " FROM recommendations r"
        " INNER JOIN factor_snapshots fs"
        "   ON r.trade_date = fs.trade_date AND r.code = fs.code"
        " WHERE r.d3_return IS NOT NULL"
    )
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        return df
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def compute_regime_ts(trade_date: str, market_fetcher) -> str:
    """用 T-1 日（或更早）收盘后可得数据回算当日 regime。

    只用 trade_date 之前的指数数据，避免未来函数（规则 §1）。
    信号集：上证指数当日涨跌 + 上证 vs MA20 趋势（sentiment 历史不可得，退化）。
    """
    try:
        from src.data.market_data import MarketDataFetcher
        fetcher = market_fetcher or MarketDataFetcher()
        d = datetime.strptime(trade_date, "%Y-%m-%d")
        end = (d - timedelta(days=1)).strftime("%Y-%m-%d")
        history = fetcher.get_stock_history("sh000001", days=60)
        if history is None or history.empty:
            return "neutral"
        df = history.copy()
        date_col = "日期" if "日期" in df.columns else "date"
        close_col = "收盘" if "收盘" in df.columns else "close"
        if date_col not in df.columns or close_col not in df.columns:
            return "neutral"
        df[date_col] = df[date_col].astype(str)
        df = df[(df[date_col] <= end)].copy()
        if len(df) < 25:
            return "neutral"
        closes = df[close_col].astype(float).reset_index(drop=True)

        # T-1 涨跌
        chg_prev = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
        # MA20 趋势
        ma20 = closes.rolling(20).mean().iloc[-1]
        current = closes.iloc[-1]
        prev = closes.iloc[-2]
        prev_ma20 = closes.iloc[-2:].rolling(20).mean().iloc[-1] if len(closes) >= 21 else ma20

        score = 50.0
        bear, bull, crash = 0, 0, 0
        if chg_prev <= -2.0:
            crash += 1
            score -= 25
        elif chg_prev <= -1.0:
            bear += 1
            score -= 12
        elif chg_prev >= 1.0:
            bull += 1
            score += 12

        if current > ma20 and ma20 > prev_ma20:
            bull += 1
            score += 5
        elif current < ma20 and ma20 < prev_ma20:
            bear += 1
            score -= 10
        elif current < ma20:
            bear += 1
            score -= 5

        score = max(0, min(100, score))
        if crash >= 2 or score <= 20:
            return "crash"
        if bear >= 2 or score <= 35:
            return "bear"
        if bull >= 2 or score >= 65:
            return "bull"
        return "neutral"
    except Exception:
        return "neutral"


def compute_regime_map(dates: List[str]) -> Dict[str, str]:
    """批量回算 regime：一次拉取指数历史，本地按各交易日 T-1 切片计算。

    dates: 需要回算 regime 的交易日列表（按日期升序）。
    返回 {trade_date: regime}。
    """
    result = {}
    if not dates:
        return result
    try:
        import akshare as ak
        # akshare 标准 A 股指数日线（上证指数 sh000001），历史数据完整
        history = ak.stock_zh_index_daily(symbol="sh000001")
        if history is None or history.empty:
            return {d: "neutral" for d in dates}
        df = history.copy()
        date_col = "date"
        close_col = "close"
        df[date_col] = df[date_col].astype(str)
        df[close_col] = df[close_col].astype(float)
        df = df.sort_values(date_col).reset_index(drop=True)

        for td in dates:
            end = (datetime.strptime(td, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            sub = df[df[date_col] <= end]
            if len(sub) < 25:
                result[td] = "neutral"
                continue
            closes = sub[close_col].reset_index(drop=True)
            chg_prev = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
            ma20 = closes.rolling(20).mean().iloc[-1]
            current = closes.iloc[-1]
            prev = closes.iloc[-2]
            prev_ma20 = closes.iloc[-2:].rolling(20).mean().iloc[-1] if len(closes) >= 21 else ma20

            score = 50.0
            bear, bull, crash = 0, 0, 0
            if chg_prev <= -2.0:
                crash += 1
                score -= 25
            elif chg_prev <= -1.0:
                bear += 1
                score -= 12
            elif chg_prev >= 1.0:
                bull += 1
                score += 12
            if current > ma20 and ma20 > prev_ma20:
                bull += 1
                score += 5
            elif current < ma20 and ma20 < prev_ma20:
                bear += 1
                score -= 10
            elif current < ma20:
                bear += 1
                score -= 5

            score = max(0, min(100, score))
            if crash >= 2 or score <= 20:
                result[td] = "crash"
            elif bear >= 2 or score <= 35:
                result[td] = "bear"
            elif bull >= 2 or score >= 65:
                result[td] = "bull"
            else:
                result[td] = "neutral"
        return result
    except Exception as e:
        print(f"  ⚠️ 回算失败: {e}")
        return {d: "neutral" for d in dates}


def build_regime_factor_table(df: pd.DataFrame) -> pd.DataFrame:
    """构建 因子 × regime 二维归因表。

    对每个 (regime, factor)，统计高分信号样本的胜率与平均收益。
    """
    rows = []
    for regime in ["bull", "neutral", "bear", "crash"]:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        for dim in SUB_SCORE_DIMS:
            col = f"sub_{dim}"
            if col not in sub.columns:
                continue
            sig = sub[sub[col] > BULLISH_THRESHOLD]
            n = len(sig)
            if n < MIN_SAMPLES:
                continue
            wins = int((sig["d3_return"] > RETURN_THRESHOLD).sum())
            avg_ret = float(sig["d3_return"].mean())
            rows.append({
                "regime": regime,
                "factor": dim,
                "n": n,
                "win_rate": wins / n,
                "avg_return": avg_ret,
                "exp_ret": float(sig["d3_return"].sum() / n),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame, table: pd.DataFrame,
                    backfilled: Dict[str, int]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Regime 分层归因报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> 数据范围：{df['trade_date'].min()} ~ {df['trade_date'].max()}（{df['trade_date'].nunique()} 个交易日，{len(df)} 条样本）")
    lines.append("> 收益口径：recommendations.d3_return（T+3 真实收盘价回看）")
    lines.append("> 因子对象：factor_snapshots 24 维 sub_* 独立列")
    lines.append(f"> 信号口径：sub_* > {BULLISH_THRESHOLD} 视为看多；d3_return > 0 记为命中")
    lines.append(f"> regime 来源：新数据用 premarket 落库值；旧数据回算数 {sum(backfilled.values())} 条"
                 f"（回算仅用 T-1 日指数数据 + MA20 趋势，sentiment 历史不可得已退化，见规则 §1）")
    lines.append("")
    lines.append("> 结论含义：")
    lines.append("> - **全 regime 有效**：胜率在 bull/bear/neutral 均高于基准 → 权重保持/提升")
    lines.append("> - **特定 regime 有效**：仅在部分 regime 胜率高 → 考虑动态加权（如 regime=bear 时侧重防御因子）")
    lines.append("> - **全 regime 无效**：胜率普遍低 → 考虑降权")
    lines.append("> 注意：本报告只输出客观统计，**不自动改写 config.yaml**（规则 §2）。")
    lines.append("")

    # 各 regime 的样本分布
    lines.append("## 〇、样本分布")
    lines.append("")
    lines.append("| regime | 样本数 | 高分信号样本数 | 整体胜率 | 整体平均收益 |")
    lines.append("|--------|--------|----------------|----------|--------------|")
    for regime in ["bull", "neutral", "bear", "crash"]:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        n = len(sub)
        sig = sub[sub["d3_return"].notna()]
        win_rate = (sig["d3_return"] > 0).mean() if len(sig) else 0
        avg_ret = sig["d3_return"].mean() if len(sig) else 0
        lines.append(f"| {regime} | {n} | {len(table[table['regime']==regime] if not table.empty else table)} | {win_rate:.0%} | {avg_ret:+.2f}% |")
    lines.append("")

    # 因子 × regime 二维表
    if table.empty:
        lines.append("## 一、因子 × regime 二维归因")
        lines.append("")
        lines.append("无足够样本（高分信号样本 < {MIN_SAMPLES}）")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 一、因子 × regime 二维归因（胜率）")
    lines.append("")
    lines.append("| 因子 | " + " | ".join(f"{r}" for r in ["bull", "neutral", "bear", "crash"]) + " |")
    lines.append("|------|" + "|".join(["---"] * 4) + "|")
    for dim in SUB_SCORE_DIMS:
        cells = []
        for regime in ["bull", "neutral", "bear", "crash"]:
            row = table[(table["factor"] == dim) & (table["regime"] == regime)]
            if row.empty:
                cells.append("—")
            else:
                cells.append(f"{row.iloc[0]['win_rate']:.0%}(n={int(row.iloc[0]['n'])})")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 二、因子 × regime 二维归因（平均收益 %）")
    lines.append("")
    lines.append("| 因子 | " + " | ".join(f"{r}" for r in ["bull", "neutral", "bear", "crash"]) + " |")
    lines.append("|------|" + "|".join(["---"] * 4) + "|")
    for dim in SUB_SCORE_DIMS:
        cells = []
        for regime in ["bull", "neutral", "bear", "crash"]:
            row = table[(table["factor"] == dim) & (table["regime"] == regime)]
            if row.empty:
                cells.append("—")
            else:
                cells.append(f"{row.iloc[0]['avg_return']:+.2f}")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    lines.append("")

    # 摘要：全 regime 有效 / 特定 regime 有效
    lines.append("## 三、因子分类摘要")
    lines.append("")
    all_regimes_ok, specific_ok, all_bad = [], [], []
    for dim in SUB_SCORE_DIMS:
        rows_dim = table[table["factor"] == dim]
        if rows_dim.empty:
            continue
        base_win = df[df["d3_return"].notna()]["d3_return"].gt(0).mean()
        wins = {r["regime"]: r["win_rate"] for _, r in rows_dim.iterrows()}
        n_regimes = len(rows_dim)
        n_good = sum(1 for w in wins.values() if w >= base_win)
        if n_regimes >= 3 and n_good >= n_regimes * 0.7:
            all_regimes_ok.append(dim)
        elif n_good == 0:
            all_bad.append(dim)
        else:
            specific_ok.append(dim)
    lines.append(f"- **全 regime 有效（{len(all_regimes_ok)}）**：" + (", ".join(all_regimes_ok) if all_regimes_ok else "无"))
    lines.append(f"- **特定 regime 有效（{len(specific_ok)}）**：" + (", ".join(specific_ok) if specific_ok else "无"))
    lines.append(f"- **全 regime 无效（{len(all_bad)}）**：" + (", ".join(all_bad) if all_bad else "无"))
    lines.append("")
    lines.append("> 基准：整体胜率 = {:.0%}。分类阈值：覆盖 ≥3 个 regime 且其中 ≥70% 胜率不低于整体胜率。".format(
        df[df["d3_return"].notna()]["d3_return"].gt(0).mean() if not df.empty else 0))
    lines.append("")
    lines.append("---")
    lines.append("*本报告遵循项目规则 §6（回测与 ML 的诚实性），regime 回算仅用 T-1 日可得数据（规则 §1），权重调整需用户确认（规则 §2）。*")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regime 分层归因分析（阶段 2）")
    parser.add_argument("--backfill", action="store_true",
                        help="对 regime IS NULL 的旧数据回算 regime（需联网拉指数，较慢）")
    parser.add_argument("--no-report", action="store_true", help="不写报告，仅打印")
    args = parser.parse_args()

    print("=" * 60)
    print("  Regime 分层归因分析")
    print("=" * 60)

    df = load_recommendations_with_sub()
    if df.empty:
        print("  ⚠️ 数据库无可用数据（recommendations JOIN factor_snapshots 为空）")
        return
    print(f"  加载样本：{len(df)} 条 / {df['trade_date'].nunique()} 个交易日")

    # regime 填充/回算
    backfilled = defaultdict(int)
    n_missing = int(df["regime"].isna().sum())
    print(f"  已落库 regime：{len(df) - n_missing} 条 | 缺失：{n_missing} 条")
    if n_missing > 0:
        if args.backfill:
            # regime 按交易日唯一，只对缺失的交易日各回算一次（避免 N 次联网）
            missing_dates = sorted(df[df["regime"].isna()]["trade_date"].unique())
            print(f"  🔄 回算缺失 regime（{len(missing_dates)} 个交易日，一次拉指数历史 + 本地切片，仅 T-1 数据，sentiment 退化）...")
            date_to_regime = compute_regime_map(missing_dates)
            for r in date_to_regime.values():
                backfilled[r] += 1
            df["regime"] = df.apply(
                lambda row: date_to_regime.get(row["trade_date"], row["regime"])
                if pd.isna(row["regime"]) else row["regime"], axis=1
            )
            print(f"     回算完成：{dict(backfilled)}")
        else:
            # 未传 --backfill 时，缺失 regime 标记为 neutral（不影响有 regime 的部分）
            df["regime"] = df["regime"].fillna("neutral")
            print("  ⚠️ 未传 --backfill，缺失 regime 暂按 neutral 处理（建议联网回算：python evaluate_recs.py --backfill）")

    print("  regime 分布：" + str(df["regime"].value_counts().to_dict()))

    table = build_regime_factor_table(df)
    print(f"  二维归因表：{len(table)} 行")

    if args.no_report:
        return

    report = generate_report(df, table, dict(backfilled))
    report_dir = os.path.join(ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "regime_factor_attribution.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ 报告已保存：{path}")


if __name__ == "__main__":
    main()
