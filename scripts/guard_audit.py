# -*- coding: utf-8 -*-
"""guard 剔除审计 — 用 T+N 真实收益回查各 guard 是避雷还是误杀。

背景(2026-09-11): guard 上线依据是一次性 IC/lift 分析, 此前无闭环验证。
本脚本读取 guard_exclusions(剔除清单) + factor_snapshots(存活池基线) +
kline_daily(T+N 收盘价), 对比被剔除票与存活票的真实收益:
    excluded_avg < survived_avg → guard 有效(避雷)
    excluded_avg > survived_avg → guard 误杀(剔除的反而涨), 需复核阈值
policy_shadow 是 policy_guard 影子模式的专项验证: 若其 excluded_avg 显著
低于基线, 说明硬利空点名确实避雷, 可将 config.yaml policy_guard.shadow
置 false 转正。

数据全部来自本地 DB, 离线可跑; 当日/近几日 K 线尚未落库的记录自动跳过。
收益口径: 百分数(1.0=1%), 与项目唯一口径一致。

用法:
    python scripts/guard_audit.py                     # 近30天, T+3
    python scripts/guard_audit.py --days 60 --horizon 5
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "stock_history.db")
REPORTS_DIR = os.path.join(ROOT, "reports")


def _load_closes(conn):
    """code → {trade_date: close} 字典, 一次全量读取供回查。"""
    closes = {}
    for code, d, c in conn.execute(
            "SELECT code, trade_date, close FROM kline_daily"):
        closes.setdefault(code, {})[str(d)[:10]] = float(c)
    return closes


def _t_plus_n_return(closes_by_code, code, entry_date, horizon):
    """T+N 真实收益(百分数)。entry_date 或 T+N 收盘未落库时返回 None。

    entry 基准取 entry_date 当日收盘(premarket 在 T 日 07:00 用 T-1 数据决策、
    T 日入场, 用 T 日收盘近似入场价; 全体票同一口径, 对比语义不受影响)。
    """
    dates = closes_by_code.get(code)
    if not dates or entry_date not in dates:
        return None
    ordered = sorted(dates)
    i = ordered.index(entry_date)
    if i + horizon >= len(ordered):
        return None  # T+N 尚未落库
    entry = dates[entry_date]
    exit_ = dates[ordered[i + horizon]]
    if entry <= 0:
        return None
    return (exit_ / entry - 1) * 100


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description="guard 剔除审计(T+N真实收益回查)")
    parser.add_argument("--days", type=int, default=30, help="回查天数(默认30)")
    parser.add_argument("--horizon", type=int, default=3,
                        help="收益窗口N(默认3, 匹配max_hold_days)")
    args = parser.parse_args()

    since = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    horizon = args.horizon

    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库不存在: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. 剔除清单(表由 premarket 首次运行时自动创建, 兼容尚无表的旧库)
    try:
        excl_rows = conn.execute(
            """SELECT trade_date, code, name, guard_name, guard_reason
               FROM guard_exclusions WHERE trade_date >= ? ORDER BY trade_date""",
            (since,)).fetchall()
    except sqlite3.OperationalError:
        excl_rows = []
    # 2. 存活池基线(factor_snapshots 即过审票)
    try:
        surv_rows = conn.execute(
            """SELECT DISTINCT trade_date, code FROM factor_snapshots
               WHERE trade_date >= ?""", (since,)).fetchall()
    except sqlite3.OperationalError:
        surv_rows = []
    # 3. K线收盘价
    closes = _load_closes(conn)

    print("=" * 72)
    print(f"guard 剔除审计 | 窗口: {since} ~ 今天 | T+{horizon} 真实收益")
    print("=" * 72)

    if not excl_rows:
        print("guard_exclusions 无记录(2026-09-11 起累计, 新表无历史属正常)。")
        print("持续运行 premarket 数日后重跑本脚本即可获得首批审计数据。")
        return

    # 基线: 存活票 T+N 平均收益
    surv_rets = [
        _t_plus_n_return(closes, r["code"], str(r["trade_date"])[:10], horizon)
        for r in surv_rows
    ]
    baseline = _avg(surv_rets)
    n_surv_measured = sum(1 for v in surv_rets if v is not None)
    if baseline is None:
        print(f"\n存活池: {len(surv_rows)} 只, 0 只可测(近{horizon}日K线未落库), 稍后重跑")
    else:
        print(f"\n存活池基线: {n_surv_measured}/{len(surv_rows)} 只可测, "
              f"T+{horizon} 平均收益 {baseline:+.2f}%")

    # 分 guard 统计
    by_guard = {}
    for r in excl_rows:
        by_guard.setdefault(r["guard_name"], []).append(r)

    print(f"\n{'guard':<18}{'剔除数':>6}{'可测':>6}{'剔除票均值':>12}{'对比基线':>10}  判读")
    print("-" * 72)
    report_lines = []
    for gname, rows in sorted(by_guard.items(), key=lambda kv: -len(kv[1])):
        rets = [_t_plus_n_return(closes, r["code"], str(r["trade_date"])[:10], horizon)
                for r in rows]
        measured = [v for v in rets if v is not None]
        gavg = _avg(measured)
        if gavg is None:
            print(f"{gname:<18}{len(rows):>6}{0:>6}{'N/A':>12}{'N/A':>10}  K线未落库")
            continue
        if baseline is None:
            verdict, diff_str = "基线缺失", "N/A"
        else:
            diff = baseline - gavg
            diff_str = f"{diff:+.2f}%"
            if diff > 0.5:
                verdict = "✅ 有效避雷"
            elif diff < -0.5:
                verdict = "⚠️ 疑似误杀, 复核阈值"
            else:
                verdict = "➖ 中性"
        print(f"{gname:<18}{len(rows):>6}{len(measured):>6}"
              f"{gavg:>+11.2f}%{diff_str:>10}  {verdict}")
        report_lines.append(
            f"{gname}: n={len(rows)} 可测={len(measured)} "
            f"平均T+{horizon}={gavg:+.2f}% 对比基线={diff_str} {verdict}")
        # 误杀个案 Top3(被剔除但涨幅最大的票)
        pairs = sorted([(v, r) for v, r in zip(rets, rows) if v is not None],
                       key=lambda x: -x[0])[:3]
        for v, r in pairs:
            if v > 0:
                print(f"    ↳ 误杀个案: {r['name']}({r['code']}) "
                      f"{r['trade_date']} T+{horizon} {v:+.2f}% "
                      f"[{str(r['guard_reason'])[:30]}]")

    # 落盘报告
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(
        REPORTS_DIR, f"guard_audit_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"guard 剔除审计 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"窗口: {since}~今天 | T+{horizon} | "
                f"存活基线: {f'{baseline:+.2f}%' if baseline is not None else 'N/A'}\n\n")
        f.write("\n".join(report_lines) + "\n")
    print(f"\n📄 报告已保存: {out}")
    print("注: 全部数据来自本地 kline_daily, 近几日K线未落库的记录已自动跳过。")


if __name__ == "__main__":
    main()
