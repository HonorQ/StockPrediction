"""
工具函数
"""
import pandas as pd
import numpy as np


def get_limit_pct(code: str = "", name: str = "") -> float:
    """按板块与ST标记返回涨跌停幅度(%)。

    A股各板块口径不同，此前多处硬编码 ±9.5/1.1 导致
    20cm(创业/科创)、30cm(北交)、5%(ST) 全部误判。

    规则:
      - 北交所(bj前缀 或 4/8开头): 30%
      - 创业板(300/301/302): 20%
      - 科创板(688/689): 20%
      - 名称含 ST/*ST: 5% (优先级高于板块)
      - 其他主板: 10%
    """
    name_upper = (name or "").upper()
    if "ST" in name_upper and name_upper:
        return 5.0

    code_str = str(code or "").strip()
    # 去掉交易所前缀 (sz/sh/bj)
    raw = code_str[-6:] if len(code_str) >= 6 else code_str
    if code_str.lower().startswith("bj") or (raw and raw[0] in ("4", "8")) or raw.startswith("92"):
        return 30.0
    if raw.startswith(("300", "301", "302", "688", "689")):
        return 20.0
    return 10.0


def limit_up_price(pre_close: float, pct: float) -> float:
    """涨停价（按百分比四舍五入到分）"""
    return round(float(pre_close) * (1 + pct / 100), 2)


def limit_down_price(pre_close: float, pct: float) -> float:
    """跌停价（按百分比四舍五入到分）"""
    return round(float(pre_close) * (1 - pct / 100), 2)


def safe_divide(a, b, default=0.0):
    """安全除法，避免除零"""
    if b == 0 or pd.isna(b):
        return default
    return a / b


def normalize_to_range(value: float, min_val: float, max_val: float) -> float:
    """将值归一化到 [0, 1]"""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    截面Rank标准化，映射到 [-1, 1]
    用于消除不同因子的量级差异
    """
    ranked = series.rank(pct=True)
    return ranked * 2 - 1  # 映射到 [-1, 1]


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """截尾处理，去除极端值"""
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(q_low, q_high)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR（真实波幅）"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = safe_divide_series(gain, loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def safe_divide_series(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    """安全除法（Series版）"""
    result = a / b
    result = result.fillna(default)
    result = result.replace([np.inf, -np.inf], default)
    return result


def format_number(value, decimals: int = 2) -> str:
    """格式化数字显示"""
    if pd.isna(value) or value is None:
        return "-"
    if abs(value) >= 1e8:
        return f"{value / 1e8:.{decimals}f}亿"
    elif abs(value) >= 1e4:
        return f"{value / 1e4:.{decimals}f}万"
    return f"{value:.{decimals}f}"
