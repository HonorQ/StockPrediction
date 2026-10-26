"""
拉取1年历史数据脚本

使用baostock拉取A股股票的历史K线数据，用于训练和回测。

[2026-10-16 终审] baostock 已持续限流(login 半死/10002007), 批量抓取请改用
scripts/backfill_missing_klines.py (akshare/东财源, 0.1s/只, 含换手率)。
本脚本保留为历史入口, 过滤逻辑已修复(不再把指数/ETF当股票)。
"""
import logging
import os
import sys
import time
import pickle
import hashlib
import baostock as bs
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# K线缓存目录
KLINE_CACHE_DIR = os.path.join(ROOT, "cache", "klines")
os.makedirs(KLINE_CACHE_DIR, exist_ok=True)


def login_baostock():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock登录失败: {lg.error_msg}")
        return False
    print("baostock登录成功")
    return True


def get_stock_list():
    """获取A股股票列表（排除ST、北交所、科创板）"""
    # 尝试多个日期获取股票列表
    dates_to_try = [
        datetime.now().strftime("%Y-%m-%d"),
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    ]
    
    for today in dates_to_try:
        try:
            rs = bs.query_all_stock(day=today)
            stocks = []
            while rs.next():
                row = rs.get_row_data()
                code = row[0]
                name = row[1] if len(row) > 1 else ""
                
                # 白名单: 仅沪主板60/深主板00 才是股票。
                # 2026-10-16 终审: 旧逻辑只排除 bj/688/300/301, 漏排指数
                # (sh.000xxx)与ETF/LOF基金(sh.5xxxxx/sz.1xxxxx); baostock 列表
                # 按代码排序, 这些代码排最前, 顺序抓取先抓完非股票段就被限流
                # —— 曾产出的"成功"缓存 616 个全是指数与 ETF, 真股票为 0。
                if not code.startswith(("sh.60", "sz.00")):
                    continue

                stocks.append((code, name))
            
            if stocks:
                print(f"获取到 {len(stocks)} 只股票 (日期: {today})")
                return stocks
        except Exception as e:
            print(f"尝试日期 {today} 失败: {e}")
            continue
    
    # 如果都失败，使用硬编码的主要股票列表
    print("使用备用股票列表")
    return [
        ("sh.600519", "贵州茅台"), ("sh.601318", "中国平安"), ("sh.600036", "招商银行"),
        ("sh.600276", "恒瑞医药"), ("sh.601166", "兴业银行"), ("sh.600030", "中信证券"),
        ("sh.600900", "长江电力"), ("sh.601888", "中国中免"), ("sh.600809", "山西汾酒"),
        ("sh.603259", "药明康德"), ("sz.000858", "五粮液"), ("sz.000333", "美的集团"),
        ("sz.000651", "格力电器"), ("sz.002714", "牧原股份"), ("sz.002475", "立讯精密"),
        ("sz.000001", "平安银行"), ("sz.002594", "比亚迪"), ("sz.300750", "宁德时代"),
    ]


def fetch_kline(code, start, end):
    """拉取单只股票K线"""
    rs = bs.query_history_k_data_plus(
        code, "date,close,volume,turn,pctChg,open,high,low",
        start_date=start, end_date=end, frequency="d", adjustflag="3"
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return rows


def cache_path(code):
    """缓存文件路径"""
    key = hashlib.md5(code.encode()).hexdigest()[:16]
    return os.path.join(KLINE_CACHE_DIR, f"1year_{key}.pkl")


def fetch_and_cache(code, start, end, use_cache=True, force=False):
    """拉取并缓存K线"""
    cp = cache_path(code)
    if use_cache and not force:
        if os.path.exists(cp):
            try:
                with open(cp, "rb") as f:
                    data = pickle.load(f)
                    # 检查数据日期范围
                    if data and len(data) > 0:
                        first_date = data[0][0]
                        last_date = data[-1][0]
                        # 如果数据覆盖1年以上，直接使用
                        if first_date <= start and last_date >= end:
                            return data
            except Exception as e:
                logger.warning("[fetch_1year_data.py:117] fetch_and_cache 失败(忽略): %s", e)
    
    rows = fetch_kline(code, start, end)
    if rows:
        try:
            cp = cache_path(code)
            with open(cp, "wb") as f:
                pickle.dump(rows, f)
        except Exception as e:
            logger.warning("[fetch_1year_data.py:126] fetch_and_cache 失败(忽略): %s", e)
    return rows


def main():
    """主函数"""
    print("=" * 60)
    print("  拉取1年历史数据")
    print("=" * 60)
    
    # 登录baostock
    if not login_baostock():
        return
    
    try:
        # 计算日期范围（1年）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        print(f"日期范围: {start_date} ~ {end_date}")
        
        # 获取股票列表
        stocks = get_stock_list()
        if not stocks:
            print("获取股票列表失败")
            return
        
        # 拉取K线数据
        print(f"\n开始拉取 {len(stocks)} 只股票的K线数据...")
        t0 = time.time()
        success_count = 0
        fail_count = 0
        cached_count = 0
        
        for i, (code, name) in enumerate(stocks, 1):
            try:
                # 检查是否已有完整缓存
                cp = cache_path(code)
                if os.path.exists(cp):
                    try:
                        with open(cp, "rb") as f:
                            data = pickle.load(f)
                        if data and len(data) >= 200:  # 至少200天数据
                            cached_count += 1
                            success_count += 1
                            continue
                    except Exception as e:
                        logger.warning("[fetch_1year_data.py:172] main 失败(忽略): %s", e)
                
                rows = fetch_and_cache(code, start_date, end_date, force=True)
                if rows and len(rows) > 0:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning("捕获异常已忽略(scripts/fetch_1year_data.py:180): %s", e)
                fail_count += 1
            
            # 进度显示
            if i % 100 == 0 or i == len(stocks):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(stocks) - i) / speed if speed > 0 else 0
                print(f"  进度: {i}/{len(stocks)} (成功: {success_count}, 失败: {fail_count}, 缓存: {cached_count}) "
                      f"速度: {speed:.1f}只/秒 预计剩余: {eta/60:.1f}分钟")
            
            # 控制请求频率，避免被限流
            time.sleep(0.1)
        
        print(f"\n完成! 成功: {success_count}, 失败: {fail_count}, 缓存: {cached_count}")
        print(f"总耗时: {(time.time() - t0)/60:.1f}分钟")
        
        # 统计缓存
        cached = len([f for f in os.listdir(KLINE_CACHE_DIR) if f.startswith("1year_")])
        print(f"缓存文件数: {cached}")
        
    finally:
        bs.logout()
        print("baostock已登出")


if __name__ == "__main__":
    main()
