# -*- coding: utf-8 -*-
"""K线缓存补抓(akshare/东财源) —— 为 rebuild_market_panel 补齐全市场日K。

背景(2026-10-16 终审):
  - baostock 持续限流(login 半死, 服务端 10002007/WinError 10054 频发),
    已不可作为批量数据源
  - 旧 fetch_1year_data 按 baostock 列表顺序抓取, 而该列表把指数(sh.000xxx)
    与 ETF(sh.51xxxx) 排在最前, 且过滤只排除 bj/688/300/301 —— 结果"成功"
    的 616 个缓存全是指数与 ETF, 真股票为 0 (缓存文件名 md5 反推证实)
  - akshare(stock_zh_a_hist, 东财源) 实测 0.1s/只, 同时给出换手率与涨跌幅,
    adjust="" 为不复权, 与既有缓存(baostock adjustflag=3)口径一致

产出格式(与 fetch_1year_data 缓存严格一致, rebuild_market_panel 消费):
  cache/klines/1year_{md5(code)[:16]}.pkl
  内容: list[list[str]], 每行 = [date, close, volume, turn, pctChg, open, high, low]
  - volume 单位"股"(东财返回"手", ×100); turn/pctChg 为百分比数值
  - code 格式 "sh.600519" / "sz.000001"
  运行结束生成 cache/klines/_code_index.json(rebuild 的股票清单);
  该文件此前全项目无生成方, 是 rebuild 读不到 1year 缓存的断点。

抓取范围: 60/00/30/68 开头共约 5200 只(排除指数/ETF/北交所), 口径接近
旧面板(5123 只, 含 454 创板), 不再混入 ETF。

用法: python -u scripts/backfill_missing_klines.py
可反复执行, 已有缓存(>=200行)自动跳过; 计划任务每 4 小时调用一次。
akshare 内部无请求超时, 用线程 + result(timeout) 兜底, 挂起不阻塞主流程。
"""
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

# 绕过系统残留代理(AGENTS.md 运维项): 东财会因不存在代理而连接失败
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "klines")
INDEX_PATH = os.path.join(CACHE, "_code_index.json")

# 股票代码白名单前缀(排除指数 sh.000xxx / ETF sh.5xxxxx / 北交所)
ALLOW_PREFIXES = ("60", "00", "30", "68")

DONE_MIN_ROWS = 200
SLEEP_PER_STOCK = 0.12      # 东财实测 0.1s/只, 留少量间隙避免触发限流
COOLDOWN_EVERY = 8          # 连续失败这么多只就冷却
COOLDOWN_SECS = 60.0
MAX_TIMEOUTS = 30           # 线程超时累计上限, 超了说明接口异常, 退出
FETCH_TIMEOUT = 30          # 单只抓取线程超时(秒)


def to_market_code(num6):
    """6位数字代码 → 'sh.600519' / 'sz.000001' 格式"""
    return ("sh." if num6.startswith("6") else "sz.") + num6


def cache_path(code):
    import hashlib
    key = hashlib.md5(code.encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"1year_{key}.pkl")


def has_enough_cache(code):
    p = cache_path(code)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return bool(data) and len(data) >= DONE_MIN_ROWS
    except Exception:
        return False


def get_stock_list():
    """东财全量列表 → 白名单过滤 → ['sh.600519', ...] 排序"""
    import akshare as ak
    lst = ak.stock_info_a_code_name()
    codes = []
    for c in lst.iloc[:, 0].astype(str):
        c = c.strip().zfill(6)
        if len(c) == 6 and c[:2] in ALLOW_PREFIXES:
            codes.append(to_market_code(c))
    return sorted(set(codes))


def fetch_hist_with_timeout(num6, start, end):
    """线程 + result(timeout) 兜底 akshare(其内部 requests 无超时)。

    超时后线程泄漏但不阻塞(shutdown wait=False); 调用方负责累计超时数
    并在超过 MAX_TIMEOUTS 时中止, 避免线程耗尽。
    返回 (status, df): status = ok / timeout / err
    """
    import concurrent.futures as cf
    import akshare as ak

    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ak.stock_zh_a_hist, symbol=num6, period="daily",
                    start_date=start, end_date=end, adjust="")
    try:
        df = fut.result(timeout=FETCH_TIMEOUT)
        return ("ok", df)
    except cf.TimeoutError:
        return ("timeout", None)
    except Exception as e:
        return ("err", f"{type(e).__name__} {e}")
    finally:
        ex.shutdown(wait=False)


def _num(v, fmt, scale=1.0):
    """转格式化字符串; NaN/None → ''(与 baostock 缺失表示一致)"""
    try:
        f = float(v) * scale
        if f != f:  # NaN
            return ""
        if fmt == "d":
            return str(int(f))
        return format(f, fmt)
    except (TypeError, ValueError):
        return ""


def convert_to_rows(df):
    """akshare DataFrame → 缓存行格式 [date, close, volume, turn, pctChg, open, high, low]

    volume 东财单位为"手", ×100 转股; 数值格式对齐既有缓存样例。
    """
    rows = []
    for _, r in df.iterrows():
        d = r["日期"]
        date_s = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
        rows.append([
            date_s,
            _num(r["收盘"], ".4f"),
            _num(r["成交量"], "d", scale=100),   # 手 → 股
            _num(r["换手率"], ".6f"),
            _num(r["涨跌幅"], ".6f"),
            _num(r["开盘"], ".4f"),
            _num(r["最高"], ".4f"),
            _num(r["最低"], ".4f"),
        ])
    return rows


def write_index(codes):
    """写 _code_index.json: 当前已有缓存的股票清单(rebuild 消费)

    门槛 60 行, 与 rebuild_market_panel 的加载条件(>=60)一致;
    低于 60 行的新股缓存保留但不入索引(因子需 >=60 根K线)。
    """
    cached = []
    for c in codes:
        p = cache_path(c)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d and len(d) >= 60:
                cached.append(c)
        except Exception:
            continue
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False)
    os.replace(tmp, INDEX_PATH)
    return len(cached)


def main():
    os.makedirs(CACHE, exist_ok=True)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=370)).strftime("%Y%m%d")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取股票列表(akshare/东财)...", flush=True)
    try:
        codes = get_stock_list()
    except Exception as e:
        print(f"  获取股票列表失败: {type(e).__name__} {e}", flush=True)
        return
    print(f"  候选股票 {len(codes)} 只, 日期 {start}~{end}", flush=True)

    todo = [c for c in codes if not has_enough_cache(c)]
    print(f"  待补抓 {len(todo)} 只 (其余已有缓存)", flush=True)

    ok = fail = timeouts = 0
    consecutive_fail = 0
    t0 = time.time()
    try:
        for i, code in enumerate(todo, 1):
            num6 = code.split(".")[1]
            status, payload = fetch_hist_with_timeout(num6, start, end)
            if status == "ok" and payload is not None and len(payload) >= 60:
                rows = convert_to_rows(payload)
                try:
                    with open(cache_path(code), "wb") as f:
                        pickle.dump(rows, f)
                    ok += 1
                    consecutive_fail = 0
                except Exception as e:
                    print(f"    {code} 写缓存失败: {e}", flush=True)
                    fail += 1
            else:
                fail += 1
                consecutive_fail += 1
                if status == "timeout":
                    timeouts += 1
                    print(f"    {code} 超时({FETCH_TIMEOUT}s)", flush=True)
                elif status == "err":
                    print(f"    {code} 失败: {payload}", flush=True)
                else:
                    print(f"    {code} 数据不足: "
                          f"{0 if payload is None else len(payload)}行", flush=True)
                time.sleep(SLEEP_PER_STOCK * 2)

            if consecutive_fail >= COOLDOWN_EVERY:
                print(f"  连续{consecutive_fail}只失败, 冷却 {COOLDOWN_SECS}s ...", flush=True)
                time.sleep(COOLDOWN_SECS)
                consecutive_fail = 0

            if timeouts >= MAX_TIMEOUTS:
                print(f"  累计超时 {timeouts} 次, 接口异常, 提前退出(下轮重试)", flush=True)
                break

            time.sleep(SLEEP_PER_STOCK)

            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                print(f"  进度 {i}/{len(todo)} 成功{ok} 失败{fail} "
                      f"耗时{el/60:.1f}分钟 {rate:.1f}只/秒", flush=True)
    finally:
        n_cached = write_index(codes)
        print(f"  索引已更新: _code_index.json ({n_cached} 只)", flush=True)

    print(f"\n完成: 成功 {ok}, 失败 {fail}, 总耗时 {(time.time()-t0)/60:.1f}分钟", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
