# -*- coding: utf-8 -*-
"""跨窗口基线回填与对比分析（一次性工具，可重复运行）

从各窗口K线缓存回填 backtest 模式推荐的 T+N 收益，
输出 bear/bull/neutral 三窗口的超额收益对比。
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.data.database import StockDatabase

WINDOWS = [
    ("bear(06末-07)", "2026-06-29", "2026-07-22"),
    ("bull(04反弹)", "2026-04-07", "2026-04-30"),
    ("neutral(02)", "2026-02-05", "2026-02-26"),
]


def load_kc(win_start, win_end):
    path = os.path.join("cache", f"bt_kline_{win_start}_{win_end}.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def to_float(v, default=0.0):
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def main():
    db = StockDatabase()
    conn = db._connection().__enter__() and None  # noqa - 直接用 sqlite3 简化
    import sqlite3
    conn = sqlite3.connect("data/stock_history.db")
    conn.row_factory = sqlite3.Row

    for label, ws, we in WINDOWS:
        kc = load_kc(ws, we)
        if not kc:
            print(f"[{label}] 无K线缓存，跳过")
            continue
        rows = conn.execute(
            "SELECT trade_date, code, current_price FROM recommendations "
            "WHERE mode='backtest' AND trade_date BETWEEN ? AND ?",
            (ws, we)).fetchall()
        filled = 0
        for r in rows:
            date, code = r["trade_date"], r["code"]
            entry = r["current_price"] or 0
            if entry <= 0:
                continue
            baos_code = code.replace("sh", "sh.").replace("sz", "sz.")
            krows = kc.get(code) or kc.get(baos_code)
            if not krows:
                continue
            future = [x for x in krows if x[0] > date]
            updates = {}
            for off, kw in [(3, "d3"), (5, "d5"), (10, "d10")]:
                if len(future) >= off and r["current_price"]:
                    ep = to_float(future[off - 1][1])
                    if ep > 0:
                        updates[kw] = round((ep / entry - 1) * 100, 2)
            if updates:
                db.update_performance(date, code, **updates)
                filled += 1
        print(f"[{label}] 回填 {filled}/{len(rows)} 条")

        # 统计该窗口
        rows2 = conn.execute(
            """SELECT score, d3_return, d5_return, d10_return
               FROM recommendations
               WHERE mode='backtest' AND trade_date BETWEEN ? AND ?""",
            (ws, we)).fetchall()

        def stat(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            wins = sum(1 for v in vals if v > 0)
            return len(vals), wins / len(vals) * 100, sum(vals) / len(vals)

        line = [f"[{label}] n={len(rows2)}"]
        bench = {3: None, 5: None, 10: None}
        # 市场基准：全池滚动入场等权
        dates_all = sorted({x[0] for rs in kc.values() for x in rs if x[0]})
        wds = [d for d in dates_all if ws <= d <= we][:8]
        for h in (3, 5, 10):
            allv = []
            for ed in wds:
                for code, rs in kc.items():
                    seg = [x for x in rs if x[0] >= ed]
                    if len(seg) < h + 1:
                        continue
                    e, x2 = to_float(seg[0][1]), to_float(seg[h][1])
                    if e > 0 and x2 > 0:
                        allv.append((x2 / e - 1) * 100)
            if allv:
                bench[h] = sum(allv) / len(allv)

        for col, h in [("d3_return", 3), ("d5_return", 5), ("d10_return", 10)]:
            s = stat([r[col] for r in rows2])
            if s:
                excess = s[2] - (bench[h] or 0)
                line.append(f"{col}: 胜率{s[1]:.0f}% avg={s[2]:+.2f}% (基准{bench[h]:+.2f}% 超额{excess:+.2f}%)")
        # 高分组
        scored = sorted([(r["score"], r["d5_return"]) for r in rows2
                         if r["d5_return"] is not None], key=lambda x: -x[0])
        if len(scored) >= 40:
            top = [v for _, v in scored[:len(scored) // 5]]
            rest = [v for _, v in scored[len(scored) // 5:]]
            line.append(f"Top20%分数组 d5={sum(top)/len(top):+.2f}% | 其余={sum(rest)/len(rest):+.2f}%")
        print("\n".join(line))
        print()


if __name__ == "__main__":
    main()
