"""
工具函数
"""
import pandas as pd
import numpy as np
from typing import List, Optional


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
