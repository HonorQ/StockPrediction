# -*- coding: utf-8 -*-
"""超跌反弹 72.5 档影子追踪 —— 独立于生产推荐链路, 只记录不下单。

背景(2026-10-16 面板终审):
  oversold_rebound=72.5 档是唯一经单边模拟检验的正超额档位
  (语义: 5日跌5-10% + 10日跌8-15% 温和双跌, 无其它信号)。
  但实盘推荐 58 条中 0 条命中该档 —— 现有候选漏斗根本选不出它们,
  因此需要独立影子追踪累积"向前看"的实盘外推样本。

机制:
  - bars_recent: 每只股票最近约 45 根日K。种子来自 cache/klines 缓存;
    每日运行时由东财全市场快照(一次请求)追加当日收盘 bar。
    72.5 档条件最长依赖 21 根K线, 45 根绰绰有余, 且与全历史口径等价。
  - picks: 每日 72.5 档清单 + T+1/3/5/10 前向收益
    (口径: (T+N收盘/T收盘-1)*100, 与 rebuild/回填链路一致), 后续自动补齐。
  - 每日流程: 快照 → 落库 → 重算当日档位 → 补齐历史 picks 前向收益 → 写摘要。
  - 与生产完全隔离: 独立库 data/shadow_rebound.db, 不触碰推荐/评分链路。

用法:
  python -u scripts/shadow_rebound_track.py --seed    # 首次: 近75交易日种子+面板交叉校验
  python -u scripts/shadow_rebound_track.py           # 每日(收盘后, 计划任务16:40)
  python -u scripts/shadow_rebound_track.py --report  # 只读摘要
"""
import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime

# 绕过系统残留代理(AGENTS 运维项): 东财会因不存在代理而连接失败
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

DB_PATH = os.path.join(ROOT, "data", "shadow_rebound.db")
CACHE = os.path.join(ROOT, "cache", "klines")
INDEX_PATH = os.path.join(CACHE, "_code_index.json")
PANEL_PATH = os.path.join(ROOT, "reports", "market_panel_daily.csv")
LATEST_JSON = os.path.join(ROOT, "reports", "shadow_rebound_latest.json")

ALLOW_PREFIXES = ("60", "00", "30", "68")
OB_TARGET = 72.5
KEEP_SEED_BARS = 45      # 每只股票种入 bars_recent 的K根数(条件最长依赖21根)
PRUNE_KEEP = 60          # 每日修剪后每只保留的K根数
SEED_DAYS = 75           # 种子回填重算的交易日数
SPOT_MIN_ROWS = 3000     # 快照有效行数下限(低于此视为数据异常)
FWD = (1, 3, 5, 10)


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS bars_recent(
            code TEXT, date TEXT, close REAL, volume REAL, turn REAL,
            pctchg REAL, open REAL, high REAL, low REAL,
            PRIMARY KEY(code, date));
        CREATE TABLE IF NOT EXISTS picks(
            date TEXT, code TEXT, entry_close REAL, signals TEXT,
            d1 REAL, d3 REAL, d5 REAL, d10 REAL, updated_at TEXT,
            PRIMARY KEY(date, code));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(date);
        """
    )
    conn.commit()


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))


def get_meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def market_code(num6):
    return ("sh." if num6.startswith("6") else "sz.") + num6


def cache_path(code):
    key = hashlib.md5(code.encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"1year_{key}.pkl")


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _load_cache_rows(code):
    p = cache_path(code)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            rows = pickle.load(f)
        return rows if rows and len(rows) >= 60 else None
    except Exception:
        return None


# ---------------- 生产公式复刻(避免口径漂移) ----------------
_WORKER_S = None


def _worker_init():
    """Pool 初始化: 每进程实例化一次策略/评分器"""
    global _WORKER_S
    from src.engine.scorer import StockScorer
    from src.factors.sector_rotation import OversoldReboundStrategy
    _WORKER_S = (OversoldReboundStrategy(), StockScorer({}))


def _ok(rows):
    """K线行是否可用于因子计算(close>0)"""
    return _f(rows[1]) is not None and _f(rows[1]) > 0


def ob_level(strat, scorer, df):
    """复刻 backtest_factors 链路: check_oversold_rebound → _score_oversold_rebound

    返回 (档位分, 命中信号列表)
    """
    r = strat.check_oversold_rebound(df)
    level = scorer._score_oversold_rebound({
        "oversold_rebound_detected": r.get("detected", False),
        "oversold_rebound_score": r.get("score", 0),
        "oversold_rebound_signals": r.get("signals", []),
    })
    return level, r.get("signals", [])


def rows_to_kline(code, bars):
    """bars_recent 行 → baostock 行格式 [date, close, volume, turn, pctChg, open, high, low]"""
    out = []
    for d, c, v, t, pc, o, h, l in bars:
        out.append([
            d,
            "" if c is None else f"{c:.4f}",
            "" if v is None else str(int(v)),
            "" if t is None else f"{t:.6f}",
            "" if pc is None else f"{pc:.6f}",
            "" if o is None else f"{o:.4f}",
            "" if h is None else f"{h:.4f}",
            "" if l is None else f"{l:.4f}",
        ])
    return out


# ---------------- 种子 ----------------
def seed_chunk(codes):
    """worker: 单批股票 → (bars, picks)"""
    from src.engine.backtest_factors import baostock_rows_to_dataframe
    strat, scorer = _WORKER_S
    bars, picks = [], []
    for code in codes:
        rows = _load_cache_rows(code)
        if not rows:
            continue
        for r in rows[-KEEP_SEED_BARS:]:
            if not _ok(r):
                continue
            bars.append((code, r[0], _f(r[1]), _f(r[2]), _f(r[3]),
                         _f(r[4]), _f(r[5]), _f(r[6]), _f(r[7])))
        start = max(60, len(rows) - SEED_DAYS)
        for i in range(start, len(rows)):
            df = baostock_rows_to_dataframe(rows[:i + 1])
            if df.empty or len(df) < 20:
                continue
            level, signals = ob_level(strat, scorer, df)
            if abs(level - OB_TARGET) > 1e-6:
                continue
            entry = _f(rows[i][1])
            if not entry or entry <= 0:
                continue
            fwd = {}
            for h in FWD:
                v = None
                if i + h < len(rows):
                    c1 = _f(rows[i + h][1])
                    if c1 and c1 > 0:
                        v = round((c1 / entry - 1) * 100, 2)
                fwd[h] = v
            picks.append((rows[i][0], code, entry,
                          json.dumps(signals, ensure_ascii=False),
                          fwd[1], fwd[3], fwd[5], fwd[10]))
    return bars, picks


def cmd_seed(args):
    from multiprocessing import Pool
    if not os.path.exists(INDEX_PATH):
        print("缺少 cache/klines/_code_index.json, 先跑 backfill_missing_klines.py")
        return 1
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        codes = json.load(f)
    conn = get_conn()
    init_db(conn)
    chunks = [codes[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    t0 = time.time()
    bars_all, picks_all = [], []
    print(f"[{datetime.now():%H:%M:%S}] 种子启动: {len(codes)} 只 × 近{SEED_DAYS}交易日 "
          f"({args.workers} workers)", flush=True)
    with Pool(args.workers, initializer=_worker_init) as pool:
        for i, (bars, picks) in enumerate(pool.imap_unordered(seed_chunk, chunks), 1):
            bars_all.extend(bars)
            picks_all.extend(picks)
            print(f"  chunk {i}/{len(chunks)}: +bars {len(bars)} +picks {len(picks)} "
                  f"累计 picks {len(picks_all)}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        "INSERT OR REPLACE INTO bars_recent VALUES (?,?,?,?,?,?,?,?,?)", bars_all)
    conn.executemany(
        "INSERT OR REPLACE INTO picks(date,code,entry_close,signals,d1,d3,d5,d10,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], now) for p in picks_all])
    set_meta(conn, "seeded_at", now)
    conn.commit()
    n_bars = conn.execute("SELECT COUNT(*) FROM bars_recent").fetchone()[0]
    n_picks = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
    print(f"种子完成: bars_recent {n_bars} 行 | picks {n_picks} 行 "
          f"| 耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    validate_against_panel(conn)
    summary(conn)
    return 0


def validate_against_panel(conn):
    """与面板 (date,code,oversold_rebound) 精确比对 72.5 档集合。

    注: 种子窗口按每只股票"最近75根K线"取, 短历史股票的窗口起点较早,
    这些早期日期影子只部分覆盖(属正常), 校验重点:
      - 影子不存在面板外的假阳性;
      - 双方都 >=30 只的日期, 面板独有比例 <2%。
    """
    import pandas as pd
    if not os.path.exists(PANEL_PATH):
        print("校验跳过: 面板不存在")
        return
    panel = pd.read_csv(PANEL_PATH, usecols=["date", "code", "oversold_rebound"])
    panel = panel[panel["oversold_rebound"] == OB_TARGET]
    pv = {d: set(g["code"]) for d, g in panel.groupby("date")}
    sv = {}
    for d, c in conn.execute("SELECT date, code FROM picks"):
        sv.setdefault(d, set()).add(c)
    common = sorted(set(pv) & set(sv))
    false_pos = sum(len(sv[d] - pv[d]) for d in common)
    full = [(d, len(pv[d]), len(sv[d])) for d in common if min(len(pv[d]), len(sv[d])) >= 30]
    bad = [(d, len(pv[d]), len(sv[d])) for d, p, s in full
           if len(pv[d] & sv[d]) < len(pv[d]) * 0.98]
    print(f"面板校验: 交集日期 {len(common)} 天 (双方均>=30只的 {len(full)} 天) | "
          f"影子假阳性(不在面板内) {false_pos} 只次 | 未达98%天数 {len(bad)}", flush=True)
    if bad:
        print("  偏差日(日期/面板/影子): " + "; ".join(
            f"{d} {p}/{s}" for d, p, s in bad[:10]), flush=True)
    else:
        print("  全部足量日期交集率 >=98%, 复刻可信", flush=True)


# ---------------- 每日 ----------------
def fetch_spot(timeout=120):
    """全市场快照: 东财优先, 失败回退新浪。

    返回 (source, df)。注意两端成交量单位不同(东财手/新浪股),
    由 parse_* 分别归一化。akshare 内部无超时, 用线程兜底。
    """
    import concurrent.futures as cf
    import akshare as ak
    last_err = None
    for src, fn in (("em", ak.stock_zh_a_spot_em), ("sina", ak.stock_zh_a_spot)):
        ex = cf.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        try:
            df = fut.result(timeout=timeout)
            return src, df
        except Exception as e:
            last_err = e
            print(f"  快照源 {src} 失败: {type(e).__name__} {str(e)[:80]}", flush=True)
        finally:
            ex.shutdown(wait=False)
    raise RuntimeError(f"全部快照源失败: {last_err}")


def _code_ok(mkt, num6):
    """交易所前缀 + 代码段白名单(防指数误入: sh000xxx 是指数, sz000xxx 是股票)"""
    if mkt == "sh":
        return num6[:2] in ("60", "68")
    if mkt == "sz":
        return num6[:2] in ("00", "30")
    return False


def parse_spot(src, df):
    """快照 → [(code, close, vol股, turn, pctchg, open, high, low)]"""
    rows = []
    for _, r in df.iterrows():
        raw = str(r.get("代码", "")).strip().lower()
        if raw.startswith(("sh", "sz")) and len(raw) >= 8:
            mkt, num6 = raw[:2], raw[2:]
        elif len(raw) == 6 and raw.isdigit():
            num6 = raw
            mkt = "sh" if num6.startswith("6") else "sz"
        else:
            continue
        if not _code_ok(mkt, num6):
            continue
        close = _f(r.get("最新价"))
        vol = _f(r.get("成交量"))
        if close is None or close <= 0 or vol is None or vol <= 0:
            continue
        if src == "em":
            vol *= 100  # 手 → 股
        rows.append((market_code(num6), close, vol, _f(r.get("换手率")),
                     _f(r.get("涨跌幅")), _f(r.get("今开")),
                     _f(r.get("最高")), _f(r.get("最低"))))
    return rows


def resolve_target_date():
    """收盘后运行 → 返回快照对应的交易日; 无法确定时返回 None"""
    import akshare as ak
    now = datetime.now()
    d = now.date()
    cal = ak.tool_trade_date_hist_sina()
    days = sorted(str(x) for x in cal["trade_date"])
    ds = str(d)
    if ds in days:
        if now.hour * 60 + now.minute < 15 * 60 + 10:
            print("尚未收盘(当前为交易日盘中), 影子追踪必须收盘后运行", flush=True)
            return None
        return ds
    prev = [x for x in days if x < ds]
    return prev[-1] if prev else None


def cmd_daily(args):
    conn = get_conn()
    init_db(conn)
    if conn.execute("SELECT COUNT(*) FROM bars_recent").fetchone()[0] == 0:
        print("bars_recent 为空: 请先运行 --seed", flush=True)
        return 1
    target = resolve_target_date()
    if target is None:
        return 1

    src, spot_df = fetch_spot()
    rows = parse_spot(src, spot_df)
    if len(rows) < SPOT_MIN_ROWS:
        print(f"快照有效行仅 {len(rows)}, 低于 {SPOT_MIN_ROWS}, 数据异常中止", flush=True)
        return 1

    # 快照落库(REPLACE 幂等; 亦会覆盖缓存里午间抓到的半日bar)
    conn.executemany(
        "INSERT OR REPLACE INTO bars_recent VALUES (?,?,?,?,?,?,?,?,?)",
        [(c, target, close, vol, turn, pc, o, h, l)
         for c, close, vol, turn, pc, o, h, l in rows])
    prev = get_meta(conn, "last_spot_date")
    set_meta(conn, "last_spot_date", target)
    conn.commit()
    print(f"[{datetime.now():%H:%M:%S}] 快照源 {src}: {len(rows)} 只 → {target} "
          f"(上次 {prev})", flush=True)
    if prev and prev < target:
        gap_msg = f"注意: 自 {prev} 后存在未采集交易日, 请检查漏跑日并重跑 --seed 补窗"
        print(gap_msg, flush=True)

    # 修剪: 每只股票保留最近 PRUNE_KEEP 根
    conn.execute(
        "DELETE FROM bars_recent WHERE (code, date) IN ("
        " SELECT code, date FROM ("
        "  SELECT code, date, ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) rn"
        "  FROM bars_recent) WHERE rn > ?)", (PRUNE_KEEP,))
    conn.commit()

    # 当日 picks 重算 + 前向收益补齐
    bars_by_code = {}
    for row in conn.execute(
            "SELECT code, date, close, volume, turn, pctchg, open, high, low "
            "FROM bars_recent ORDER BY code, date"):
        bars_by_code.setdefault(row[0], []).append(row[1:])
    from src.engine.backtest_factors import baostock_rows_to_dataframe
    _worker_init()
    strat, scorer = _WORKER_S

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_picks = []
    for code, bars in bars_by_code.items():
        if bars[-1][0] != target or len(bars) < 20:
            continue
        df = baostock_rows_to_dataframe(rows_to_kline(code, bars))
        if df.empty or len(df) < 20:
            continue
        level, signals = ob_level(strat, scorer, df)
        if abs(level - OB_TARGET) > 1e-6:
            continue
        today_picks.append((target, code, bars[-1][1],
                            json.dumps(signals, ensure_ascii=False),
                            None, None, None, None, now))
    # 先删后插: 保证当日集合 = 最新计算(None/部分数据下的旧行不得残留)
    conn.execute("DELETE FROM picks WHERE date=?", (target,))
    conn.executemany(
        "INSERT OR REPLACE INTO picks(date,code,entry_close,signals,d1,d3,d5,d10,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)", today_picks)
    conn.commit()
    print(f"当日 72.5 档: {len(today_picks)} 只", flush=True)

    filled = update_forward(conn, bars_by_code, now)
    print(f"前向收益补齐: {filled} 个字段", flush=True)
    summary(conn)
    return 0


def update_forward(conn, bars_by_code, now):
    """按K线位置重算 picks 的 d1/d3/d5/d10(口径: 第 N 根收盘价)。

    每次重算所有可计算字段并覆写 —— 幂等且可修复数据修订
    (如午间抓取的半日bar被收盘数据替换); 不可计算的老记录保留原值。
    """
    series = {code: [(b[0], b[1]) for b in bars]
              for code, bars in bars_by_code.items()}
    updates = []
    for date, code, entry, d1, d3, d5, d10 in conn.execute(
            "SELECT date, code, entry_close, d1, d3, d5, d10 FROM picks"):
        s = series.get(code)
        if not s or not entry or entry <= 0:
            continue
        dates = [x[0] for x in s]
        try:
            idx = dates.index(date)
        except ValueError:
            continue
        vals = {1: d1, 3: d3, 5: d5, 10: d10}
        changed = False
        for h in FWD:
            if idx + h < len(s):
                c1 = s[idx + h][1]
                if c1 and c1 > 0:
                    new = round((c1 / entry - 1) * 100, 2)
                    if new != vals[h]:
                        vals[h] = new
                        changed = True
        if changed:
            updates.append((vals[1], vals[3], vals[5], vals[10], now, date, code))
    if updates:
        conn.executemany(
            "UPDATE picks SET d1=?,d3=?,d5=?,d10=?,updated_at=? WHERE date=? AND code=?",
            updates)
        conn.commit()
    return len(updates)


def summary(conn, write_json=True):
    total = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
    if not total:
        print("picks 为空", flush=True)
        return
    stat = conn.execute(
        "SELECT COUNT(d1), AVG(d1), AVG(d3), AVG(d5), AVG(d10), COUNT(d10) FROM picks"
    ).fetchone()
    recent = conn.execute(
        "SELECT date, COUNT(*), ROUND(AVG(d5),2) FROM picks "
        "GROUP BY date ORDER BY date DESC LIMIT 12").fetchall()
    print("=" * 60, flush=True)
    print(f"picks 总数 {total} | 已备 d1 {stat[0]} | 平均 d1 {stat[1] or 0:+.2f}% "
          f"d3 {stat[2] or 0:+.2f}% d5 {stat[3] or 0:+.2f}% d10 {stat[4] or 0:+.2f}% "
          f"(d10 备 {stat[5]})", flush=True)
    print("最近交易日 (只数/均d5):", flush=True)
    for d, n, m in recent:
        print(f"  {d}: {n:>3} 只, d5 {m if m is not None else '-'}", flush=True)
    if write_json:
        today = recent[0][0] if recent else None
        today_codes = [r[0] for r in conn.execute(
            "SELECT code FROM picks WHERE date=? ORDER BY code", (today,))] if today else []
        out = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_picks": total,
            "filled": {"d1": stat[0], "d10": stat[5]},
            "mean": {"d1": stat[1], "d3": stat[2], "d5": stat[3], "d10": stat[4]},
            "recent": [{"date": d, "n": n, "mean_d5": m} for d, n, m in recent],
            "today": today,
            "today_codes": today_codes,
        }
        with open(LATEST_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"摘要已写: {LATEST_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="首次种子(近75交易日重算+面板校验)")
    ap.add_argument("--report", action="store_true", help="只读摘要")
    ap.add_argument("--validate", action="store_true", help="只做面板交叉校验")
    ap.add_argument("--workers", type=int, default=6, help="种子并行进程数")
    args = ap.parse_args()
    if args.seed:
        return cmd_seed(args)
    if args.report:
        conn = get_conn()
        init_db(conn)
        summary(conn)
        return 0
    if args.validate:
        conn = get_conn()
        init_db(conn)
        validate_against_panel(conn)
        return 0
    return cmd_daily(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
